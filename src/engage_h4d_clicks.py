"""E18: how much tangled-limb error does SAM 3D Body remove when given true 2D keypoints as clicks? (GPU, .venv-s3d).
Pre-registration: `docs/engagement/results/clicks_prereg.md`.

The ceiling of the input route. Per athlete crop (ground-truth box) of the test views: SAM 3D Body runs once (the
released model, no clicks), then its click loop (`_one_prompt_iter`, the model's own prompt sampler: it clicks the body
keypoint whose current 2D estimate is furthest from the given 2D pose, as in E05) reuses that image embedding with:
  vitpose2  ViTPose+ 2D keypoints (the pipeline's current 2D model), 2 clicks (the model's trained maximum)
  oracle2   the true joints projected through the true fisheye camera (12 body joints: shoulders, elbows, wrists,
            hips, knees, ankles; the face keypoints get no click), 2 clicks
  oracle4   the same with 4 clicks (beyond what the model was trained for; exploratory)
Measure as E11: the 19 canonical points aligned rigidly onto the truth over CORE14; median error of the 8 limb joints,
for tangle points (within 20 cm of a bone of the other athlete) and the rest.

  PYTORCH_ALLOC_CONF=expandable_segments:True ../.venv-s3d/bin/python engage_h4d_clicks.py --feats F --out D [--smoke]
  ... --pose-file D/pose_ft_<set>.npz --name vitposeft2   (E19: 2 clicks from another 2D model's keypoints, on the
      views of that file only; the E18 variants are skipped)
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_bench_s3d import fisheye, seq_root                      # noqa: E402
from engage_s3d_finetune import (CORE14, JOINT8, canon_matrix, crops_from_feats, kabsch_t,  # noqa: E402
                                 load_estimator, prepare_batch, recursive_to)
from engage_limb_head import MMA, TEST, TEST_CAMS                   # noqa: E402
from tangle_h4d import Sequence                                     # noqa: E402

COCO_TO_MHR = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
SMPL_FOR_COCO = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
VARIANTS = [('vitpose2', 'vitpose', 2), ('oracle2', 'oracle', 2), ('oracle4', 'oracle', 4)]
REF = Path(__file__).resolve().parents[1]/'benchmarks/engage_h4d/s3d_box'


def truth_2d(seq, cam, f, cache={}):
    """[2,17,3] COCO-17 pixel positions of each athlete's true joints (face points zero, no click) in this camera."""
    key = (seq.root, cam)
    if key not in cache:
        Rc, tc, cid = seq.poses[cam]
        _, _, Pc = seq.cams[cid]
        K = np.array([[Pc[0], 0, Pc[2]], [0, Pc[1], Pc[3]], [0, 0, 1]])
        Dk = np.array(Pc[4:8]).reshape(1, 4)
        A4 = seq.world_to_colmap
        sc = np.cbrt(np.linalg.det(A4[:3, :3]))
        cache[key] = (lambda X: ((X @ A4[:3, :3].T + A4[:3, 3]) @ Rc.T + tc)/sc, K, Dk)
    tocam, K, Dk = cache[key]
    gt = np.load(seq.root/'processed_data/smpl'/f'{f:05d}.npy', allow_pickle=True).item()
    out = np.zeros((2, 17, 3))
    for n, k in enumerate(sorted(gt)):
        J = tocam(np.asarray(gt[k]['joints'][:24], float))
        uv = fisheye(J, K, Dk)
        for c, s in SMPL_FOR_COCO.items():
            out[n, c] = [uv[s, 0], uv[s, 1], 1.0]
    return out


def vitpose_2d(seq_name, cam, f, cache={}):
    key = (seq_name, cam)
    if key not in cache:
        z = np.load(REF/seq_name/f'{cam}_vitpose.npz')
        cache[key] = ({int(fr): j for j, fr in enumerate(z['frames'])}, z['kp'], z['score'])
    idx, kp, sc = cache[key]
    j = idx.get(f)
    return None if j is None else np.concatenate([kp[j], sc[j][..., None]], -1)


def run_crop(m, est, img, box, poses):
    """Released output and one output per variant, all from one backbone pass. Returns {name: kp3d [70,3]}."""
    batch = recursive_to(prepare_batch(img, est.transform, box, None, None), 'cuda')
    m._initialize_batch(batch)
    res = {}
    with torch.inference_mode():
        out0 = m.forward_step(batch, decoder_type='body')
        res['original'] = out0['mhr']['pred_keypoints_3d'][0].float().cpu().numpy()
        for name, src, clicks in VARIANTS:
            pose = poses[src]
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
            res[name] = out['mhr']['pred_keypoints_3d'][0].float().cpu().numpy()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true', help='a few development views (code check)')
    ap.add_argument('--pose-file', default='', help='E19: npz with rows, kp [V,2,17,2], score [V,2,17]')
    ap.add_argument('--name', default='filepose2')
    a = ap.parse_args()
    global VARIANTS
    filepose = {}
    if a.pose_file:
        z = np.load(a.pose_file, allow_pickle=True)
        filepose = {tuple(r): np.concatenate([z['kp'][i], z['score'][i][..., None]], -1) for i, r in enumerate(z['rows'])}
        VARIANTS = [(a.name, 'file', 2)]
    rows, G, tang = crops_from_feats(a.feats)
    if a.smoke:   # development views only: the test views are not looked at before the run
        dev = sorted({r[0] for r in rows if r[0].endswith('_grappling2') and r[0] not in TEST})
        sets = {'smoke_dev': [i for i, r in enumerate(rows) if r[0] == dev[0] and r[1] in TEST_CAMS][:4]}
    else:
        sets = {'heldout': [i for i, r in enumerate(rows) if r[0] in TEST and r[1] in TEST_CAMS],
                'mma_new_people': [i for i, r in enumerate(rows) if r[0] == MMA]}
    if filepose:
        sets = {k: [i for i in v if tuple(rows[i]) in filepose] for k, v in sets.items()}
        sets = {k: v for k, v in sets.items() if v}
    est = load_estimator()
    m = est.model
    decoder = m.forward_decoder   # the click loop passes full_output=, which forward_decoder no longer takes (E05)
    m.forward_decoder = lambda *args, full_output=None, **kw: decoder(*args, **kw)
    A = torch.tensor(canon_matrix(), dtype=torch.float32)
    names = ['original'] + [v[0] for v in VARIANTS]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    report, seqs = {}, {}
    for set_name, views in sets.items():
        E = {n: np.full((len(views), 2, 19), np.nan) for n in names}
        t0 = time.time()
        for k, v in enumerate(views):
            s, c, f = rows[v]
            seq = seqs.setdefault(s, Sequence(seq_root(s)))
            vp = vitpose_2d(s, c, f)
            b = np.load(seq.root/'processed_data/bbox'/c/f'{f:05d}.npy', allow_pickle=True).item()
            if vp is None or len(b) != 2:
                continue
            img = cv2.cvtColor(cv2.imread(str(seq.image_path(c, f))), cv2.COLOR_BGR2RGB)
            tr = truth_2d(seq, c, f)
            for n, who in enumerate(sorted(b)):
                box = np.asarray(b[who], np.float32).reshape(1, 4)
                poses = {'vitpose': vp[n], 'oracle': tr[n]}
                if filepose:
                    poses['file'] = filepose[(s, c, f)][n]
                res = run_crop(m, est, img, box, poses)
                Gt = torch.tensor(G[v, n][None], dtype=torch.float32)
                for name, kp in res.items():
                    C = torch.einsum('jk,nkd->njd', A, torch.tensor(kp[None]))
                    E[name][k, n] = (kabsch_t(C, Gt, CORE14) - Gt).norm(dim=-1)[0].numpy()
            torch.cuda.empty_cache()
            if k % 100 == 0:
                print(f'{set_name} {k}/{len(views)} {time.time() - t0:.0f}s', flush=True)
        ok = ~np.isnan(E['original'][:, 0, 0])
        t = tang[np.array(views)][ok][..., JOINT8]
        Ev = {n: E[n][ok][..., JOINT8] for n in names}
        r = {n: {'tangle_joint8_median_cm': float(np.median(Ev[n][t]) * 100),
                 'other_joint8_median_cm': float(np.median(Ev[n][~t]) * 100)} for n in names}
        r['views'], r['tangle_points'] = int(ok.sum()), int(t.sum())
        brng, V = np.random.default_rng(1), int(ok.sum())
        for n in names[1:]:
            ratios = []
            for _ in range(1000):
                bb = brng.integers(0, V, V)
                mk = t[bb]
                ratios.append(np.median(Ev[n][bb][mk]) / np.median(Ev['original'][bb][mk]) - 1)
            r[f'{n}_vs_original_tangle_95ci'] = [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))]
        report[set_name] = r
        np.savez(out/f'errors_{set_name}.npz', views=np.array(views)[ok], tang=tang[np.array(views)][ok],
                 **{n: E[n][ok] for n in names})
        print(set_name, json.dumps(r), flush=True)
    json.dump(report, open(out/'clicks.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
