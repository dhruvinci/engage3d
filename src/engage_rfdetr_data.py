"""E16 data: COCO-format pair crops of Harmony4D contact for RF-DETR keypoint and segmentation fine-tuning (CPU,
.venv-rfdetr). Experimental. Pre-registration: `docs/engagement/results/rfdetr_prereg.md`.

People are small in Harmony4D's 4K frames (about 250 x 600 px), so RF-DETR runs on a square crop around the pair (the
union of both athletes' boxes, side 1.3 x its longer edge), saved at 640 x 640. At test the pair box comes from a
detector on the full frame (`engage_rfdetr_eval.py`); in training the crop is jittered (side x 0.85-1.2, centre
+-8% of the side) to cover that detector's slack.
Per athlete: the ground-truth box (processed_data/bbox, the projected SMPL vertices), the 12 body joints projected
through the true fisheye camera (v=2 where the athlete owns that pixel, v=1 where the partner covers it; face points
unlabelled, v=0, as in E19), and the visible mask: both SMPL meshes painted far-to-near in crop pixels, as
`tangle_h4d.Sequence.gt`. Views: exactly E19's (8,000 training views from 10 development sequences, 500 validation
views from 025 / 035), read from its `targets.npz`.

  ../.venv-rfdetr/bin/python engage_rfdetr_data.py --views V/targets.npz --out D [--workers 4] [--smoke]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_paths import H4D                  # noqa: E402
from tangle_h4d import FACES, PEOPLE, Sequence  # noqa: E402

OUT, BASE, JITTER_SIDE, JITTER_SHIFT = 640, 1.3, (0.85, 1.2), 0.08
SMPL_FOR_COCO = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
COCO_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
              'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
              'right_knee', 'left_ankle', 'right_ankle']
SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
            [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
CATEGORY = {'id': 1, 'name': 'person', 'supercategory': 'person', 'keypoints': COCO_NAMES, 'skeleton': SKELETON}
_SEQ = {}


def seq_for(name):
    if name not in _SEQ:
        _SEQ[name] = Sequence(next(H4D.glob(f'*/{name}')))
    return _SEQ[name]


def pair_crop(boxes, rng=None):
    """2x3 affine from image pixels to the OUT x OUT crop around the union of boxes [N,4] (x1, y1, x2, y2)."""
    x1, y1 = boxes[:, 0].min(), boxes[:, 1].min()
    x2, y2 = boxes[:, 2].max(), boxes[:, 3].max()
    side = BASE * max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if rng is not None:
        side *= rng.uniform(*JITTER_SIDE)
        cx += rng.uniform(-JITTER_SHIFT, JITTER_SHIFT) * side
        cy += rng.uniform(-JITTER_SHIFT, JITTER_SHIFT) * side
    k = OUT / side
    return np.array([[k, 0, OUT / 2 - k * cx], [0, k, OUT / 2 - k * cy]])


def render_labels(seq, cam, frame, A):
    """[OUT,OUT] uint8 ownership in crop pixels: 0 background, 1 aria01, 2 aria02 (the nearer body wins)."""
    d = np.load(seq.root/'processed_data/smpl'/f'{frame:05d}.npy', allow_pickle=True).item()
    tris, deps, who = [], [], []
    for n, person in enumerate(PEOPLE, start=1):
        uv, z = seq.project(d[person]['vertices'], cam)
        uv = uv @ A[:, :2].T + A[:, 2]
        t, zt = uv[FACES], z[FACES]
        keep = (zt > 0).all(1) & (t[..., 0].max(1) >= 0) & (t[..., 1].max(1) >= 0) & \
               (t[..., 0].min(1) < OUT) & (t[..., 1].min(1) < OUT)
        tris.append(t[keep]); deps.append(zt[keep].mean(1)); who.append(np.full(keep.sum(), n, np.uint8))
    tris, deps, who = np.concatenate(tris), np.concatenate(deps), np.concatenate(who)
    out = np.zeros((OUT, OUT), np.uint8)
    pts = np.round(tris * 4).astype(np.int32)
    for i in np.argsort(-deps):
        cv2.fillConvexPoly(out, pts[i], int(who[i]), lineType=cv2.LINE_8, shift=2)
    return out


def truth_2d(seq, cam, frame):
    """[2,17,2] image pixels of each athlete's COCO body joints (face points left at 0)."""
    d = np.load(seq.root/'processed_data/smpl'/f'{frame:05d}.npy', allow_pickle=True).item()
    out = np.zeros((2, 17, 2))
    for n, person in enumerate(PEOPLE):
        uv, _ = seq.project(np.asarray(d[person]['joints'][:24], float), cam)
        for c, s in SMPL_FOR_COCO.items():
            out[n, c] = uv[s]
    return out


def make_example(args):
    """One crop: writes the JPEG, returns (image record, [annotations], meta)."""
    image_id, (s, c, f), out_dir, jitter_seed = args
    seq = seq_for(s)
    img = cv2.imread(str(seq.image_path(c, f)))
    b = np.load(seq.root/'processed_data/bbox'/c/f'{f:05d}.npy', allow_pickle=True).item()
    boxes = np.array([np.asarray(b[p], float) for p in PEOPLE])
    A = pair_crop(boxes, None if jitter_seed is None else np.random.default_rng(jitter_seed))
    crop = cv2.warpAffine(img, A, (OUT, OUT), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
    name = f'{s}__{c}__{f:05d}.jpg'
    cv2.imwrite(str(out_dir/name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    lab = render_labels(seq, c, f, A)
    kp = truth_2d(seq, c, f) @ A[:, :2].T + A[:, 2]
    anns = []
    for n in range(2):
        m = np.asfortranarray((lab == n + 1).astype(np.uint8))
        rle = mask_utils.encode(m)
        rle['counts'] = rle['counts'].decode()
        bx = np.r_[boxes[n, :2], 1.0] @ A.T, np.r_[boxes[n, 2:], 1.0] @ A.T
        x1, y1 = np.clip(bx[0], 0, OUT)
        x2, y2 = np.clip(bx[1], 0, OUT)
        k = []
        for j in range(17):
            if j not in SMPL_FOR_COCO:
                k += [0, 0, 0]
                continue
            u, v = kp[n, j]
            if not (0 <= u < OUT and 0 <= v < OUT):
                k += [0, 0, 0]
                continue
            owned = lab[int(v), int(u)] == n + 1
            k += [round(float(u), 2), round(float(v), 2), 2 if owned else 1]
        anns.append({'image_id': image_id, 'category_id': 1, 'iscrowd': 0, 'segmentation': rle,
                     'area': float(m.sum()), 'bbox': [round(float(x1), 2), round(float(y1), 2),
                                                      round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                     'keypoints': k, 'num_keypoints': int(sum(v > 0 for v in k[2::3]))})
    rec = {'id': image_id, 'file_name': name, 'width': OUT, 'height': OUT}
    return rec, anns, {'image_id': image_id, 'seq': s, 'cam': c, 'frame': int(f), 'affine': A.round(6).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--views', required=True, help="E19's targets.npz (rows and split)")
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    T = np.load(a.views, allow_pickle=True)
    rows, split = T['rows'][::2], T['split'][::2]           # one row per view (the file has one per athlete)
    out = Path(a.out)
    for part, tag in (('train', 'train'), ('valid', 'val')):
        idx = np.where(split == tag)[0]
        if a.smoke:
            idx = idx[:20] if os.environ.get('SMOKE_N') is None else idx[:int(os.environ['SMOKE_N'])]
        d = out/part
        d.mkdir(parents=True, exist_ok=True)
        jobs = [(k + 1, (rows[i][0], rows[i][1], int(rows[i][2])), d, (1000 + k) if part == 'train' else None)
                for k, i in enumerate(idx)]
        t0 = time.time()
        images, anns, meta = [], [], []
        with Pool(a.workers) as pool:
            for k, (rec, aa, m) in enumerate(pool.imap(make_example, jobs, chunksize=8)):
                images.append(rec); meta.append(m)
                for ann in aa:
                    ann['id'] = len(anns) + 1
                    anns.append(ann)
                if k % 500 == 0:
                    print(f'{part} {k}/{len(jobs)} {time.time() - t0:.0f}s', flush=True)
        json.dump({'images': images, 'annotations': anns, 'categories': [CATEGORY]}, open(d/'_annotations.coco.json', 'w'))
        json.dump(meta, open(out/f'{part}_meta.json', 'w'))
        hidden = np.mean([v == 1 for ann in anns for v in ann['keypoints'][2::3] if v > 0])
        print(f'{part}: {len(images)} crops, {len(anns)} athletes, {hidden:.1%} of labelled joints hidden by the partner,'
              f' {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
