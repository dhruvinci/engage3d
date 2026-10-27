"""E16 3D: SAM 3D Body on the test views from ground-truth vs detected boxes, with and without person masks, and with
clicks from the contact-trained RF-DETR keypoints (GPU, .venv-s3d). Experimental.
Pre-registration: `docs/engagement/results/rfdetr_prereg.md`.

Arms per athlete, measured as E18 (the 19 canonical points aligned rigidly onto the truth over CORE14; median error of
the 8 limb joints inside a tangle and of the others):
  gtbox            ground-truth box, no mask (E18's `original`)
  gtbox_truemask   ground-truth box + the true visible mask (both SMPL bodies painted far-to-near; the nearer one owns
                   a pixel): the ceiling of the mask route
  detbox           the box of the fine-tuned segmentation model (`engage_rfdetr_eval.py`), no mask: the cost of real
                   detection
  detbox_mask      that box + that model's mask: end to end, no ground truth
  detbox_mask_kp2  + 2 clicks (E18's loop) from the fine-tuned RF-DETR keypoints paired with that detection
Arms with the same box share one backbone pass (the mask enters after the backbone). 2D is scored too: the fine-tuned
and released RF-DETR keypoints against E19's ViTPose+ (released and fine-tuned) on the same views.

  PYTORCH_ALLOC_CONF=expandable_segments:True ../.venv-s3d/bin/python engage_rfdetr_s3d.py --feats F --det D \
      [--sets heldout,mma_new_people] [--every N] [--smoke]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_h4d_clicks import COCO_TO_MHR, truth_2d                 # noqa: E402
from engage_relations import CANON                                  # noqa: E402
from engage_s3d_finetune import (CORE14, JOINT8, canon_matrix, crops_from_feats, kabsch_t,  # noqa: E402
                                 load_estimator, prepare_batch, recursive_to, seq_root)
from tangle_h4d import Sequence                                     # noqa: E402

ARMS = ['gtbox', 'gtbox_truemask', 'detbox', 'detbox_mask', 'detbox_mask_kp2']
COCO_CANON = {5: 'l_shoulder', 6: 'r_shoulder', 7: 'l_elbow', 8: 'r_elbow', 9: 'l_wrist', 10: 'r_wrist', 11: 'l_hip',
              12: 'r_hip', 13: 'l_knee', 14: 'r_knee', 15: 'l_ankle', 16: 'r_ankle'}
LIMB8 = [7, 8, 9, 10, 13, 14, 15, 16]
CROP = 640
E19 = Path(__file__).resolve().parents[1]/'benchmarks/engage_h4d/vitpose_ft'


def coco_tangle(tang_view):
    """[2,19] canonical tangle mask -> [2,17] for the COCO joints (as E19)."""
    out = np.zeros((2, 17), bool)
    for c, name in COCO_CANON.items():
        out[:, c] = tang_view[:, CANON.index(name)]
    return out


def cache_backbone(m):
    """Make the backbone return its last output while state['reuse'] is set (same box, so the same crop)."""
    orig, state = m.backbone.forward, {'reuse': False, 'out': None}

    def forward(x, *args, **kw):
        if not (state['reuse'] and state['out'] is not None):
            state['out'] = orig(x, *args, **kw)
        return state['out']
    m.backbone.forward = forward
    return state


def run(m, est, img, box, mask, state, reuse, pose=None, clicks=2):
    """{'base': kp3d [70,3]} and, with a 2D pose [17,3], {'clicks': kp3d after `clicks` clicks}."""
    masks = None if mask is None else mask[None, :, :, None].astype(np.uint8)
    batch = recursive_to(prepare_batch(img, est.transform, box.astype(np.float32).reshape(1, 4), masks), 'cuda')
    m._initialize_batch(batch)
    state['reuse'] = reuse
    res = {}
    with torch.inference_mode():
        out0 = m.forward_step(batch, decoder_type='body')
        res['base'] = out0['mhr']['pred_keypoints_3d'][0].float().cpu().numpy()
        if pose is not None:
            full = np.zeros((1, 70, 2), np.float32)
            conf = np.zeros((1, 70, 1), np.float32)
            full[0, COCO_TO_MHR] = pose[:, :2]
            conf[0, COCO_TO_MHR, 0] = pose[:, 2]
            crop = m._full_to_crop(batch, torch.from_numpy(full).cuda())
            batch['keypoints_2d'] = torch.cat([crop, torch.from_numpy(conf).cuda()], -1).view(1, 1, 70, 3)
            out, prev = out0, []
            for _ in range(clicks):
                out, p = m._one_prompt_iter(batch, out, prev, None)
                prev.append(p)
            res['clicks'] = out['mhr']['pred_keypoints_3d'][0].float().cpu().numpy()
    state['reuse'] = False
    return res


def frame_mask(rle_counts, A, size):
    """Crop-pixel RLE -> frame-size uint8 mask through the pair-crop affine A."""
    m = mask_utils.decode({'size': [CROP, CROP], 'counts': rle_counts.encode()})
    return cv2.warpAffine(m, A, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)


def boot(a, b, t, n=1000):
    """95% interval of median(a)/median(b) - 1 over the points in t, resampling views."""
    rng, V, r = np.random.default_rng(1), len(t), []
    for _ in range(n):
        s = rng.integers(0, V, V)
        r.append(np.median(a[s][t[s]]) / np.median(b[s][t[s]]) - 1)
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def score_2d(D, views, rows, tang, seqs, pose_file):
    """Median 4K-pixel error of the 8 limb joints, inside tangles and elsewhere, per 2D model, on matched athletes."""
    tr = np.stack([truth_2d(seqs.setdefault(rows[v][0], Sequence(seq_root(rows[v][0]))), rows[v][1], rows[v][2])
                   for v in views])
    t = np.stack([coco_tangle(tang[v]) for v in views])[..., LIMB8]
    kp = {'rfdetr_ft': D['kp_ft'][..., :2], 'rfdetr_released': D['kp_released'][..., :2]}
    if pose_file and pose_file.exists():
        z = np.load(pose_file, allow_pickle=True)
        idx = {tuple(map(str, r)): i for i, r in enumerate(z['rows'])}
        j = [idx.get((rows[v][0], rows[v][1], str(rows[v][2]))) for v in views]
        if all(x is not None for x in j):
            kp['vitpose_ft_gtbox'], kp['vitpose_released_gtbox'] = z['kp'][j], z['kp_released'][j]
    both = np.isfinite(kp['rfdetr_ft'][..., 0, 0]) & np.isfinite(kp['rfdetr_released'][..., 0, 0])   # [V,2]
    out = {}
    for name, k in kp.items():
        e = np.linalg.norm(k - tr[..., :2], axis=-1)[..., LIMB8]
        ok = both[..., None] & np.ones_like(t)
        out[name] = {'tangle_limb_median_px': float(np.median(e[ok & t])), 'other_limb_median_px': float(np.median(e[ok & ~t]))}
    out['athletes_scored'] = int(both.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', required=True)
    ap.add_argument('--det', required=True, help='output folder of engage_rfdetr_eval.py')
    ap.add_argument('--sets', default='heldout,mma_new_people')
    ap.add_argument('--every', default='heldout=1,mma_new_people=2', help='score every Nth view of a set')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    det = Path(a.det)
    rows, G, tang = crops_from_feats(a.feats)
    where = {r: i for i, r in enumerate(rows)}
    names = ['smoke_dev'] if a.smoke else a.sets.split(',')
    every = dict(x.split('=') for x in a.every.split(','))
    est = load_estimator()
    m = est.model
    decoder = m.forward_decoder   # the click loop passes full_output=, which forward_decoder no longer takes (E05)
    m.forward_decoder = lambda *args, full_output=None, **kw: decoder(*args, **kw)
    state = cache_backbone(m)
    Acanon = torch.tensor(canon_matrix(), dtype=torch.float32)
    report, seqs = {}, {}
    if (det/'s3d.json').exists():
        report = json.load(open(det/'s3d.json'))
    for set_name in names:
        D = dict(np.load(det/f'detect_{set_name}.npz', allow_pickle=True))
        masks = json.load(open(det/f'masks_{set_name}.json'))
        keep = np.arange(len(D['rows']))[::int(every.get(set_name, 1))]
        views = [where[(r[0], r[1], int(r[2]))] for r in D['rows'][keep]]
        E = {n: np.full((len(views), 2, 19), np.nan) for n in ARMS}
        t0 = time.time()
        for k, (i, v) in enumerate(zip(keep, views)):
            s, c, f = rows[v]
            seq = seqs.setdefault(s, Sequence(seq_root(s)))
            img = cv2.cvtColor(cv2.imread(str(seq.image_path(c, f))), cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]
            lab = seq.gt(c, f, size=(W, H))
            Gt = torch.tensor(G[v], dtype=torch.float32)
            for n in range(2):
                res = {}
                gt_box = D['gt_box'][i, n]
                r = run(m, est, img, gt_box, None, state, reuse=False)
                res['gtbox'] = r['base']
                r = run(m, est, img, gt_box, (lab == n + 1), state, reuse=True)
                res['gtbox_truemask'] = r['base']
                if np.isfinite(D['seg_iou'][i, n]):
                    box = D['seg_box'][i, n]
                    res['detbox'] = run(m, est, img, box, None, state, reuse=False)['base']
                    pose = D['click_kp'][i, n] if np.isfinite(D['click_kp'][i, n, 0, 0]) else None
                    r = run(m, est, img, box, frame_mask(masks[i][n], D['affine'][i], (W, H)), state, reuse=True,
                            pose=pose)
                    res['detbox_mask'] = r['base']
                    if 'clicks' in r:
                        res['detbox_mask_kp2'] = r['clicks']
                for name, kp in res.items():
                    C = torch.einsum('jk,kd->jd', Acanon, torch.tensor(kp))[None]
                    E[name][k, n] = (kabsch_t(C, Gt[n][None], CORE14) - Gt[n][None]).norm(dim=-1)[0].numpy()
            torch.cuda.empty_cache()
            if k % 50 == 0:
                print(f'{set_name} {k}/{len(views)} {time.time() - t0:.0f}s', flush=True)
        tv = tang[np.array(views)]
        Ev = {n: E[n][..., JOINT8] for n in ARMS}
        t8 = tv[..., JOINT8]
        r = {'views': len(views), 'seconds_per_view': round((time.time() - t0) / max(len(views), 1), 2),
             'athletes_found': float(np.isfinite(E['detbox'][..., 0]).mean())}
        for n in ARMS:
            ok = np.isfinite(Ev[n][..., 0])[..., None] & np.ones_like(t8)
            r[n] = {'athletes': int(np.isfinite(Ev[n][..., 0]).sum()),
                    'tangle_joint8_median_cm': float(np.median(Ev[n][ok & t8]) * 100) if (ok & t8).any() else None,
                    'other_joint8_median_cm': float(np.median(Ev[n][ok & ~t8]) * 100) if (ok & ~t8).any() else None}
        # paired comparisons on the athletes every arm scored (detected, with keypoints for the clicks)
        paired = np.all([np.isfinite(Ev[n][..., 0]) for n in ARMS], 0)                 # [V,2]
        tp = t8 & paired[..., None]
        r['paired_athletes'] = int(paired.sum())
        r['paired'] = {n: float(np.median(Ev[n][tp]) * 100) for n in ARMS} if tp.any() else {}
        for x, y in (('gtbox_truemask', 'gtbox'), ('detbox', 'gtbox'), ('detbox_mask', 'gtbox'),
                     ('detbox_mask_kp2', 'detbox_mask')):
            if tp.any():
                r[f'{x}_vs_{y}_tangle_95ci'] = boot(np.nan_to_num(Ev[x]), np.nan_to_num(Ev[y]), tp)
        r['2d'] = score_2d({k: D[k][keep] for k in ('kp_ft', 'kp_released')}, views, rows, tang, seqs,
                           None if a.smoke else E19/f'pose_ft_{set_name}.npz')
        report[set_name] = r
        np.savez(det/f'errors_{set_name}.npz', views=np.array(views), tang=tv, **E)
        print(set_name, json.dumps(r), flush=True)
        json.dump(report, open(det/'s3d.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
