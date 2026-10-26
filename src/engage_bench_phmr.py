"""Engagement benchmark, predicted side: PromptHMR on Harmony4D camera views (runs in .venv-phmr, GPU). Experimental.

The comparison model for engage_bench_s3d.py (SAM 3D Body). PromptHMR encodes the whole image once and decodes every
person against it (both ground-truth boxes in one call, cross-person attention on), so the two athletes are placed in
one shared view of the scene; SAM 3D Body reconstructs each athlete from their own crop (movement-research E13).
Same views (the frames of a reference engage_bench_s3d.py output), same ground-truth boxes, and the same lens
handling: the model gets the pinhole part of the true intrinsics, then each body's translation is re-solved so that
its vertices land through the true fisheye where the model's own pinhole projection put them.
Not trained on Harmony4D (repo configs and README list BEDLAM 1/2, H36M, 3DPW, MPII, BEHAVE, CHI3D, Hi4D, COCO).

Writes <out>/<seq>/<cam>.npz in engage_bench_s3d.py's format with SMPL topology: frames [T], kp3d [T,2,24,3] (SMPL
joints) and verts [T,2,6890,3] (float16), both relative to t [T,2,3] (re-solved translation, camera frame, metres),
t_model [T,2,3], resid_px [T,2]. Athlete order = sorted subject names. Resumable per (seq, cam).

  PYTORCH_ALLOC_CONF=expandable_segments:True ../.venv-phmr/bin/python engage_bench_phmr.py --ref DIR --out DIR [seq ...]
"""
import argparse
import glob
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_bench_s3d import NV, fisheye, seq_root   # noqa: E402
from tangle_h4d import Sequence                        # noqa: E402

from engage_paths import PHMR as _PHMR   # noqa: E402
PHMR = str(_PHMR)
IMG_SIZE = 896
AHEAD = 6


def resolve_t(V, K, D):
    """Translation shift that puts V (camera frame, pinhole-K placement) on its own pinhole projection through the
    true fisheye."""
    Vs = V[np.linspace(0, len(V) - 1, NV).astype(int)].astype(np.float64)
    uv_model = Vs[:, :2]/Vs[:, 2:3]*np.diag(K)[:2] + K[:2, 2]

    def res(dt):
        return (fisheye(Vs + dt, K, D) - uv_model).ravel()
    s = least_squares(res, np.zeros(3), method='lm', max_nfev=400)
    return s.x, float(np.median(np.linalg.norm(res(s.x).reshape(-1, 2), axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True, help='engage_bench_s3d.py output whose views to reproduce')
    ap.add_argument('--out', required=True)
    ap.add_argument('seqs', nargs='*')
    a = ap.parse_args()
    out = Path(a.out)
    files = sorted(p for p in glob.glob(f'{a.ref}/*/cam??.npz') if not a.seqs or p.split('/')[-2] in a.seqs)
    here = os.getcwd()
    sys.path.insert(0, PHMR)
    os.chdir(PHMR)   # the repo loads data/ by relative path
    from torch.amp import autocast
    from prompt_hmr import load_model_from_folder
    from prompt_hmr.models.inference import prepare_batch
    model = load_model_from_folder('data/pretrain/phmr')
    os.chdir(here)
    pool = ThreadPoolExecutor(3)   # JPEG decoding of 4K frames overlaps the GPU
    n_oom = 0
    for p in files:
        sname, cam = p.split('/')[-2], Path(p).stem
        dst = out/sname/f'{cam}.npz'
        if dst.exists():
            continue
        seq = Sequence(seq_root(sname))
        _, _, cid = seq.poses[cam]
        W, H, P = seq.cams[cid]
        K = np.array([[P[0], 0, P[2]], [0, P[1], P[3]], [0, 0, 1]])
        D = np.array(P[4:8]).reshape(1, 4)
        frames = np.load(p)['frames']
        load = lambda f: cv2.cvtColor(cv2.imread(str(seq.image_path(cam, int(f)))), cv2.COLOR_BGR2RGB)
        ahead = [pool.submit(load, f) for f in frames[:AHEAD]]   # bounded read-ahead: a 4K frame is 25 MB
        rec = {k: [] for k in ('frames', 'kp3d', 'verts', 't', 't_model', 'resid_px')}
        started = time.time()
        for j, f in enumerate(frames):
            img = ahead[j].result()
            ahead[j] = None
            if j + AHEAD < len(frames):
                ahead.append(pool.submit(load, frames[j + AHEAD]))
            b = np.load(seq.root/'processed_data/bbox'/cam/f'{f:05d}.npy', allow_pickle=True).item()
            if len(b) != 2:
                continue
            boxes = torch.tensor(np.array([np.asarray(b[w], np.float32).ravel()[:4] for w in sorted(b)]))
            Kt = torch.tensor(K, dtype=torch.float32)[None]
            try:
                with torch.no_grad(), autocast('cuda'):
                    batch = prepare_batch([{'image_cv': img, 'boxes': boxes, 'text': None, 'masks': None,
                                            'cam_int': Kt}], img_size=IMG_SIZE, interaction=True)
                    o = model(batch, use_mean_hands=True)[0]
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                n_oom += 1
                continue
            J = o['smpl_j3d'].float().cpu().numpy().astype(np.float64)
            V = o['smpl_vertices'].float().cpu().numpy().astype(np.float64)
            tm = o['transl'].float().cpu().numpy().astype(np.float64)
            if len(J) != 2:
                continue
            shifts = [resolve_t(V[n], K, D) for n in range(2)]
            rec['frames'].append(f)
            rec['kp3d'].append(J - tm[:, None])
            rec['verts'].append((V - tm[:, None]).astype(np.float16))
            rec['t'].append(tm + np.array([s for s, _ in shifts]))
            rec['t_model'].append(tm)
            rec['resid_px'].append([r for _, r in shifts])
            del batch, o
        torch.cuda.empty_cache()
        if not rec['frames']:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, **{k: np.asarray(v) for k, v in rec.items()})
        print(f'{sname}/{cam}: {len(rec["frames"])}/{len(frames)} frames {time.time() - started:.0f}s '
              f'resid median {np.median(rec["resid_px"]):.1f}px, peak GPU '
              f'{torch.cuda.max_memory_allocated()/1e9:.2f} GB, OOM skips so far {n_oom}', flush=True)


if __name__ == '__main__':
    main()
