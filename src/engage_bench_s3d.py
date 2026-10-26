"""Engagement benchmark, predicted side: SAM 3D Body on Harmony4D camera views (runs in .venv-s3d, GPU). Experimental.

Per (sequence, camera, frame), each athlete's ground-truth box -> SAM 3D Body (body decoder, one athlete at a time, as
the production stage autolabel_body3d.py runs it) -> body (MHR mesh + 70 keypoints, root-relative, metres) whose 3D
position is then re-solved through the camera's true fisheye lens: the body is frozen and only its translation moves,
until its vertices land where the model's own pinhole projection put them (movement-research p2_reanchor "arm1":
median placement error 6.36 m -> 0.095 m over 2,590 person-views; the model assumes a default focal length).
Ground-truth boxes isolate 3D quality from detection / separation, which fail separately under contact.

Writes <out>/<seq>/<cam>.npz: frames [T], kp3d [T,2,70,3] and verts [T,2,n,3] (float16, root-relative), t [T,2,3]
(camera frame after re-solving, metres), t_model [T,2,3], f_used [T,2], resid_px [T,2] (median reprojection residual
of the re-solve). Athlete order = sorted Harmony4D subject names (aria01, aria02). Resumable per (seq, cam).

  PYTORCH_ALLOC_CONF=expandable_segments:True ../.venv-s3d/bin/python engage_bench_s3d.py --out DIR [--step 4] [--cams cam01,..] [seq ...]
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autolabel_body3d import load_estimator   # noqa: E402
from tangle_h4d import Sequence               # noqa: E402

from engage_paths import H4D   # noqa: E402


def seq_root(name):
    """<activity>/<sequence> directory of a Harmony4D sequence name (e.g. 036_grappling2, 016_mma4)."""
    return next(H4D.glob(f'*/{name}'))
NV = 128   # vertices used in the translation re-solve


def fisheye(X, K, D):
    uv, _ = cv2.fisheye.projectPoints(X.reshape(-1, 1, 3).astype(np.float64), np.zeros(3), np.zeros(3), K, D)
    return uv.reshape(-1, 2)


def resolve_t(V, t_model, f_used, c, K, D):
    """Translation that puts V (root-relative) on the model's own pinhole projection, through the true fisheye."""
    idx = np.linspace(0, len(V) - 1, NV).astype(int)
    Vs = V[idx].astype(np.float64)
    X = Vs + t_model
    uv_model = f_used*X[:, :2]/X[:, 2:3] + c

    def res(t):
        return (fisheye(Vs + t, K, D) - uv_model).ravel()
    t0 = t_model*(K[0, 0]/f_used)
    s = least_squares(res, t0, method='lm', max_nfev=400)
    r = np.linalg.norm(res(s.x).reshape(-1, 2), axis=1)
    return s.x, float(np.median(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--step', type=int, default=4)
    ap.add_argument('--cams', default='', help='comma-separated cameras (default: every camera with boxes)')
    ap.add_argument('--save-tokens', action='store_true',
                    help='also keep each person\'s final pose token (the input of the pose head) and MHR parameters')
    ap.add_argument('seqs', nargs='*')
    a = ap.parse_args()
    import torch
    out = Path(a.out)
    seqs = a.seqs or sorted(os.path.basename(p) for p in glob.glob(str(H4D/'03_grappling2/*')))
    est = None
    for sname in seqs:
        seq = Sequence(seq_root(sname))
        cams = sorted(c for c in os.listdir(seq.root/'exo') if (seq.root/'processed_data/bbox'/c).exists()
                      and (not a.cams or c in a.cams.split(',')))
        for cam in cams:
            dst = out/sname/f'{cam}.npz'
            if dst.exists():
                continue
            if est is None:
                est = load_estimator()
                last_tok = {}
                if a.save_tokens:   # the decoder's final pose token: the input of the pose head (last call wins)
                    est.model.head_pose.register_forward_hook(
                        lambda mod, inp, out: last_tok.__setitem__('t', inp[0].detach().float().cpu().numpy()))
            R, t, cid = seq.poses[cam]
            W, H, P = seq.cams[cid]
            K = np.array([[P[0], 0, P[2]], [0, P[1], P[3]], [0, 0, 1]])
            D = np.array(P[4:8]).reshape(1, 4)
            c = np.array([W/2, H/2])
            rec = {k: [] for k in ('frames', 'kp3d', 'verts', 't', 't_model', 'f_used', 'resid_px')
                   + (('token', 'mhr_params') if a.save_tokens else ())}
            started = time.time()
            for f in seq.frames[::a.step]:
                bp = seq.root/'processed_data/bbox'/cam/f'{f:05d}.npy'
                ip = seq.image_path(cam, f)
                if not bp.exists() or not ip.exists():
                    continue
                boxes = np.load(bp, allow_pickle=True).item()
                if len(boxes) != 2:
                    continue
                img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
                per = []
                for who in sorted(boxes):
                    box = np.asarray(boxes[who], np.float32).reshape(1, 4)
                    try:
                        with torch.inference_mode():
                            o = est.process_one_image(img, bboxes=box, inference_type='body')
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        o = None
                    est.batch = est.image_embeddings = est.output = None
                    if not o:
                        break
                    o = o[0]
                    V = np.asarray(o['pred_vertices'], np.float64)
                    tm = np.asarray(o['pred_cam_t'], np.float64)
                    fu = float(np.asarray(o['focal_length']).ravel()[0])
                    ts, rp = resolve_t(V, tm, fu, c, K, D)
                    extra = ((last_tok['t'].reshape(-1).astype(np.float16), np.asarray(o['mhr_model_params'], np.float32))
                             if a.save_tokens else ())
                    per.append((np.asarray(o['pred_keypoints_3d'], np.float32), V.astype(np.float16), ts, tm, fu, rp, *extra))
                torch.cuda.empty_cache()
                if len(per) != 2:
                    continue
                rec['frames'].append(f)
                for k, i in ((('kp3d', 0), ('verts', 1), ('t', 2), ('t_model', 3), ('f_used', 4), ('resid_px', 5))
                             + ((('token', 6), ('mhr_params', 7)) if a.save_tokens else ())):
                    rec[k].append([p[i] for p in per])
            if not rec['frames']:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dst, **{k: np.asarray(v) for k, v in rec.items()})
            print(f'{sname}/{cam}: {len(rec["frames"])} frames {time.time() - started:.0f}s '
                  f'resid median {np.median(rec["resid_px"]):.1f}px', flush=True)


if __name__ == '__main__':
    main()
