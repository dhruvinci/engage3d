"""E16 detect: end-to-end 2D on the test views with no ground-truth boxes (GPU, .venv-rfdetr). Experimental.
Pre-registration: `docs/engagement/results/rfdetr_prereg.md`.

Stage A: Roboflow's COCO-pretrained RF-DETR detector (Medium) on the full 4K frame finds the pair: the most confident
person and every person box touching it (its box grown by half its size on each side), merged. Stage B: the square
crop around that pair box (the rule of `engage_rfdetr_data.py`, no jitter) goes to the fine-tuned keypoint and
segmentation models, and to the released keypoint model for comparison. Each model's two most confident people are
kept and mapped back to frame pixels. The truth is used only to score: the detections are matched to the two athletes
by box overlap (IoU >= 0.5, one-to-one). The click arm pairs each segmentation detection with the keypoint detection
whose box overlaps it most, without the truth.

  PYTORCH_ALLOC_CONF=expandable_segments:True ../.venv-rfdetr/bin/python engage_rfdetr_eval.py \
      --views heldout=F1.npz,mma_new_people=F2.npz --kp CKPT --seg CKPT --out D [--smoke]
  (F: E19's pose files, whose `rows` are E18's test views; --smoke: 6 development views from 025 / 035)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_rfdetr_data import OUT, PEOPLE, pair_crop, seq_for     # noqa: E402
from engage_rfdetr_train import WEIGHTS                            # noqa: E402

PERSON_COCO, STAGE_A_THRESHOLD, GROW, MATCH_IOU = 1, 0.3, 0.5, 0.5


def iou(a, b):
    """[N,4] x [M,4] xyxy -> [N,M]."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def match_two(det, gt):
    """Index of the detection matched to each of the 2 athletes (-1: none), one-to-one, IoU >= MATCH_IOU."""
    M = iou(gt, det) if len(det) else np.zeros((2, 0))
    ok = lambda n, d: d < 0 or M[n, d] >= MATCH_IOU
    options = [(d0, d1) for d0 in range(-1, len(det)) for d1 in range(-1, len(det))
               if (d0 < 0 or d0 != d1) and ok(0, d0) and ok(1, d1)]
    best = max(options, key=lambda o: sum(M[n, d] for n, d in enumerate(o) if d >= 0))
    return np.array(best)


def pair_box(dets):
    """Union of the most confident person and every person box touching it (grown by GROW), or None."""
    keep = dets.class_id == PERSON_COCO
    boxes, conf = dets.xyxy[keep], dets.confidence[keep]
    if not len(boxes):
        return None
    top = boxes[np.argmax(conf)]
    w, h = top[2] - top[0], top[3] - top[1]
    grown = np.array([[top[0] - GROW * w, top[1] - GROW * h, top[2] + GROW * w, top[3] + GROW * h]])
    near = iou(grown, boxes)[0] > 0
    sel = boxes[near]
    return np.array([sel[:, 0].min(), sel[:, 1].min(), sel[:, 2].max(), sel[:, 3].max()])


def to_frame(A, xy):
    """Crop pixels -> frame pixels for the pair-crop affine A (scale + shift)."""
    return (xy - A[:, 2]) / A[0, 0]


def top2(boxes, conf):
    return np.argsort(-conf)[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--views', required=True)
    ap.add_argument('--kp', required=True, help='fine-tuned keypoint checkpoint')
    ap.add_argument('--seg', required=True, help='fine-tuned segmentation checkpoint')
    ap.add_argument('--queries', type=int, default=30)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    import rfdetr
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.smoke:   # development views only (the validation sequences)
        T = np.load(Path(__file__).resolve().parents[1]/'benchmarks/engage_h4d/vitpose_ft/targets.npz', allow_pickle=True)
        rows = [tuple(r[:3]) for r, s in zip(T['rows'][::2], T['split'][::2]) if s == 'val'][:6]
        sets = {'smoke_dev': rows}
    else:
        sets = {}
        for item in a.views.split(','):
            name, path = item.split('=')
            sets[name] = [tuple(r) for r in np.load(path, allow_pickle=True)['rows']]
    q = dict(num_queries=a.queries, num_select=a.queries)
    stage_a = rfdetr.RFDETRMedium(pretrain_weights=str(WEIGHTS/'rf-detr-medium.pth'))
    models = {'kp_ft': rfdetr.RFDETRKeypointPreview(pretrain_weights=a.kp, **q),
              'kp_released': rfdetr.RFDETRKeypointPreview(pretrain_weights=str(WEIGHTS/'rf-detr-keypoint-preview-xlarge.pth')),
              'seg_ft': rfdetr.RFDETRSegMedium(pretrain_weights=a.seg, **q)}
    report = {}
    for set_name, rows in sets.items():
        V = len(rows)
        R = {'rows': np.array(rows, dtype=object), 'affine': np.full((V, 2, 3), np.nan), 'pair_box': np.full((V, 4), np.nan),
             'gt_box': np.full((V, 2, 4), np.nan), 'seg_box': np.full((V, 2, 4), np.nan), 'seg_conf': np.full((V, 2), np.nan),
             'seg_iou': np.full((V, 2), np.nan), 'click_kp': np.full((V, 2, 17, 3), np.nan)}
        for m in ('kp_ft', 'kp_released'):
            R[m] = np.full((V, 2, 17, 3), np.nan)       # frame pixels + per-keypoint confidence, matched to athletes
            R[m + '_iou'] = np.full((V, 2), np.nan)
        masks = [[None, None] for _ in range(V)]          # crop-pixel RLE of the matched segmentation detections
        t0 = time.time()
        for k, (s, c, f) in enumerate(rows):
            f = int(f)
            seq = seq_for(s)
            img = cv2.cvtColor(cv2.imread(str(seq.image_path(c, f))), cv2.COLOR_BGR2RGB)
            b = np.load(seq.root/'processed_data/bbox'/c/f'{f:05d}.npy', allow_pickle=True).item()
            gt = np.array([np.asarray(b[p], float) for p in PEOPLE])
            R['gt_box'][k] = gt
            pb = pair_box(stage_a.predict(img, threshold=STAGE_A_THRESHOLD))
            if pb is None:
                continue
            A = pair_crop(pb[None])
            R['pair_box'][k], R['affine'][k] = pb, A
            crop = cv2.warpAffine(img, A, (OUT, OUT), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
            kp_boxes = {}
            for m in ('kp_ft', 'kp_released'):
                kp = models[m].predict(crop, threshold=0.0)
                sel = top2(kp.data['xyxy'], kp.detection_confidence)
                boxes = to_frame(A, kp.data['xyxy'][sel].reshape(-1, 2)).reshape(-1, 4)
                pts = to_frame(A, kp.xy[sel].reshape(-1, 2)).reshape(-1, 17, 2)
                conf = kp.keypoint_confidence[sel]
                kp_boxes[m] = (boxes, pts, conf)
                mt = match_two(boxes, gt)
                for n in range(2):
                    if mt[n] >= 0:
                        R[m][k, n] = np.c_[pts[mt[n]], conf[mt[n]]]
                        R[m + '_iou'][k, n] = iou(gt[n:n + 1], boxes[mt[n]:mt[n] + 1])[0, 0]
            det = models['seg_ft'].predict(crop, threshold=0.0)
            sel = top2(det.xyxy, det.confidence)
            boxes = to_frame(A, det.xyxy[sel].reshape(-1, 2)).reshape(-1, 4)
            mt = match_two(boxes, gt)
            kb, kpts, kconf = kp_boxes['kp_ft']
            for n in range(2):
                if mt[n] < 0:
                    continue
                d = mt[n]
                R['seg_box'][k, n], R['seg_conf'][k, n] = boxes[d], det.confidence[sel][d]
                R['seg_iou'][k, n] = iou(gt[n:n + 1], boxes[d:d + 1])[0, 0]
                rle = mask_utils.encode(np.asfortranarray(det.mask[sel][d].astype(np.uint8)))
                masks[k][n] = rle['counts'].decode()
                if len(kb):    # keypoints for clicks: the keypoint detection overlapping this one most (no truth)
                    j = int(np.argmax(iou(boxes[d:d + 1], kb)[0]))
                    R['click_kp'][k, n] = np.c_[kpts[j], kconf[j]]
            if k % 100 == 0:
                print(f'{set_name} {k}/{V} {time.time() - t0:.0f}s', flush=True)
        np.savez(out/f'detect_{set_name}.npz', **R)
        json.dump(masks, open(out/f'masks_{set_name}.json', 'w'))
        found = np.isfinite(R['seg_iou'])
        report[set_name] = {'views': V, 'stage_a_found_pair': float(np.isfinite(R['pair_box'][:, 0]).mean()),
                            'athletes_found_seg': float(found.mean()), 'both_found_seg': float(found.all(1).mean()),
                            'athletes_found_kp_ft': float(np.isfinite(R['kp_ft_iou']).mean()),
                            'athletes_found_kp_released': float(np.isfinite(R['kp_released_iou']).mean()),
                            'median_box_iou_seg': float(np.nanmedian(R['seg_iou'])),
                            'seconds_per_view': round((time.time() - t0) / max(V, 1), 2)}
        print(set_name, json.dumps(report[set_name]), flush=True)
    json.dump(report, open(out/'detect.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
