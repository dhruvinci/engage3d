"""How much does a second (third, fourth) phone buy? Fuse SAM 3D Body across 1-4 Harmony4D cameras (CPU, .venv-depth).
Development sequences only; no test view. Experimental; exploratory (not pre-registered).

Per moment (sequence, frame) and athlete, each camera's SAM 3D Body points (place_feats2, camera frame, camera at the
origin) define a viewing ray per joint (the model's own 2D evidence in that camera). Cameras are registered to the
first one by the rigid transform that maps their ground-truth points onto its own (calibrated phones). Variants per
camera subset:
  single      the first camera's SAM 3D Body (no fusion)
  average     every camera's body mapped into the first camera's frame and averaged
  triangulate each joint at the least-squares closest point to its rays from all cameras in the subset
  robust      triangulation that drops, per joint, the camera whose ray misses the others most (subsets of 3+)
  vitpose     ViTPose+ 2D keypoints of the 12 body joints triangulated across the cameras (undistorted through each
              camera's true fisheye, each ray weighted by ViTPose's confidence squared, so a joint hidden in one view is
              carried by the views that see it); the averaged SAM 3D Body body is placed rigidly onto the triangulated
              shoulders and hips, then its 8 limb joints are replaced by the triangulated ones. The standard multi-view
              recipe with a stock 2D model.
Measures: median error of the 8 limb joints inside a tangle and elsewhere (rigid alignment over CORE14, as E14 / E18),
and the error of the distance between the two athletes' pelvises (placement), in the first camera's frame.

  ../.venv-depth/bin/python engage_multiview_sim.py --feats DIR --out DIR [--smoke]
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engage_limb_head as L                                            # noqa: E402
from engage_limb_head import CORE14, JOINT8, kabsch, tangle_mask       # noqa: E402
from engage_temporal import DEV, load_tracks                           # noqa: E402
from engage_bench_s3d import seq_root                                  # noqa: E402
from tangle_h4d import Sequence                                         # noqa: E402

REF = Path(__file__).resolve().parents[1]/'benchmarks/engage_h4d/s3d_box'
LIMB_COCO_CANON = {7: 13, 8: 14, 9: 15, 10: 16, 13: 5, 14: 6, 15: 7, 16: 8}   # COCO elbows, wrists, knees, ankles
TORSO_COCO_CANON = {5: 11, 6: 12, 11: 3, 12: 4}                                 # shoulders, hips


def vitpose_rays(s, c, f, cache={}):
    """[2,17,3] unit viewing rays (camera frame) of ViTPose+ keypoints through the true fisheye, and [2,17] scores."""
    if (s, c) not in cache:
        z = np.load(REF/s/f'{c}_vitpose.npz')
        seq = Sequence(seq_root(s))
        _, _, cid = seq.poses[c]
        Pc = seq.cams[cid][2]
        K = np.array([[Pc[0], 0, Pc[2]], [0, Pc[1], Pc[3]], [0, 0, 1]])
        cache[(s, c)] = ({int(fr): j for j, fr in enumerate(z['frames'])}, z['kp'], z['score'], K, np.array(Pc[4:8]))
    idx, kp, sc, K, D = cache[(s, c)]
    j = idx.get(int(f))
    if j is None:
        return None, None
    uv = cv2.fisheye.undistortPoints(kp[j].reshape(-1, 1, 2).astype(np.float64), K, D.reshape(1, 4)).reshape(2, 17, 2)
    d = np.concatenate([uv, np.ones((2, 17, 1))], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True), sc[j].astype(np.float64)


def triangulate_weighted(origins, dirs, w):
    """origins, dirs [C,J,3], weights [C,J] -> [J,3]."""
    I = np.eye(3)
    M = (I[None, None] - dirs[..., :, None] * dirs[..., None, :]) * w[..., None, None]
    A = M.sum(0); b = (M @ origins[..., None]).sum(0)[..., 0]
    return np.linalg.solve(A + 1e-9 * I, b[..., None])[..., 0]

L.ALIGN = CORE14
SUBSETS = {'1 cam (01)': ['cam01'], '2 cams 90deg (01,06)': ['cam01', 'cam06'], '2 cams opposite (01,11)': ['cam01', 'cam11'],
           '3 cams (01,06,11)': ['cam01', 'cam06', 'cam11'], '4 cams (01,06,11,16)': ['cam01', 'cam06', 'cam11', 'cam16']}


def rigid(A, B):
    """R, t with B ~ A @ R.T + t (A, B [N,3])."""
    a, b = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - a).T @ (B - b))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, b - a @ R.T


def triangulate(origins, dirs, drop_worst=False):
    """origins, dirs [C,J,3] -> [J,3] least-squares point closest to all rays (optionally dropping the worst ray)."""
    def solve(o, d):
        I = np.eye(3)
        M = I[None, None] - d[..., :, None] * d[..., None, :]          # [C,J,3,3]
        A = M.sum(0); b = (M @ o[..., None]).sum(0)[..., 0]
        return np.linalg.solve(A + 1e-9 * I, b[..., None])[..., 0]
    X = solve(origins, dirs)
    if not drop_worst or len(origins) < 3:
        return X
    r = np.linalg.norm(np.cross(X[None] - origins, dirs), axis=-1)     # [C,J] distance of X to each ray
    worst = r.argmax(0)
    keep = np.ones(r.shape, bool); keep[worst, np.arange(r.shape[1])] = False
    out = np.zeros_like(X)
    for j in range(X.shape[0]):
        out[j] = solve(origins[keep[:, j], j][:, None], dirs[keep[:, j], j][:, None])[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--seqs', default='', help='comma-separated sequences (default: the development grappling)')
    ap.add_argument('--ref', default='', help='folder with <seq>/<cam>_vitpose.npz (default: s3d_box)')
    a = ap.parse_args()
    global REF
    if a.ref:
        REF = Path(a.ref)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cams = sorted({c for s in SUBSETS.values() for c in s})
    seqs = a.seqs.split(',') if a.seqs else (DEV[:2] if a.smoke else DEV)
    P, G, _, key, tr = load_tracks(a.feats, seqs, set(cams))
    where = defaultdict(dict)
    for i, (s, c, f) in enumerate(key):
        where[(s, f)][c] = i
    # register every camera to cam01 per sequence from the ground truth (calibrated phones)
    reg = {}
    for s in sorted({k[0] for k in key}):
        moments = [m for m in where if m[0] == s and 'cam01' in where[m]]
        for c in cams:
            both = [m for m in moments if c in where[m]]
            if not both:
                continue
            A = np.concatenate([G[where[m][c]].reshape(-1, 3) for m in both])
            B = np.concatenate([G[where[m]['cam01']].reshape(-1, 3) for m in both])
            R, t = rigid(A, B)
            reg[(s, c)] = (R, t, float(np.median(np.linalg.norm(A @ R.T + t - B, axis=-1))))
    rep = {'registration_residual_mm_median': float(np.median([v[2] for v in reg.values()]) * 1000), 'subsets': {}}
    for name, sub in SUBSETS.items():
        moments = [m for m in where if all(c in where[m] for c in sub)]
        E = {v: [] for v in ('single', 'average', 'triangulate', 'robust', 'vitpose')}
        D = {v: [] for v in E}
        T = []
        for m in moments:
            s = m[0]
            g = G[where[m]['cam01']]                                       # [2,19,3] truth in cam01's frame
            bodies, origins, dirs = [], [], []
            for c in sub:
                R, t, _ = reg[(s, c)]
                p = P[where[m][c]] @ R.T + t                               # SAM 3D Body of camera c, in cam01's frame
                o = t                                                      # camera c's centre in cam01's frame
                bodies.append(p); origins.append(np.broadcast_to(o, p.shape))
                d = p - o; dirs.append(d / np.linalg.norm(d, axis=-1, keepdims=True))
            bodies, origins, dirs = np.array(bodies), np.array(origins), np.array(dirs)   # [C,2,19,3]
            vp = []
            for c in sub:
                R, t, _ = reg[(s, c)]
                d, sc = vitpose_rays(s, c, m[1])
                vp.append(None if d is None else (np.broadcast_to(t, d.shape), d @ R.T, sc))
            fused = {'single': bodies[0], 'average': bodies.mean(0),
                     'triangulate': np.stack([triangulate(origins[:, n], dirs[:, n]) for n in range(2)]) if len(sub) > 1 else bodies[0],
                     'robust': np.stack([triangulate(origins[:, n], dirs[:, n], True) for n in range(2)]) if len(sub) > 2 else
                     (np.stack([triangulate(origins[:, n], dirs[:, n]) for n in range(2)]) if len(sub) > 1 else bodies[0])}
            V = bodies.mean(0).copy()
            ok = [v for v in vp if v is not None]
            if len(ok) >= 2:
                o = np.array([v[0] for v in ok]); d = np.array([v[1] for v in ok]); w = np.array([v[2] for v in ok])
                for n in range(2):
                    X = triangulate_weighted(o[:, n], d[:, n], np.clip(w[:, n], 1e-3, None) ** 2)
                    ti = list(TORSO_COCO_CANON.values())
                    R2, t2 = rigid(V[n, ti], X[list(TORSO_COCO_CANON)])
                    V[n] = V[n] @ R2.T + t2
                    for ci, ki in LIMB_COCO_CANON.items():
                        V[n, ki] = X[ci]
            fused['vitpose'] = V
            for v, Q in fused.items():
                E[v].append(np.stack([np.linalg.norm(kabsch(Q[n][None], g[n][None])[0] - g[n], axis=-1) for n in range(2)]))
                D[v].append(abs(np.linalg.norm(Q[1, 0] - Q[0, 0]) - np.linalg.norm(g[1, 0] - g[0, 0])))
            T.append(g)
        tang = tangle_mask(np.array(T))[..., JOINT8]
        r = {'moments': len(moments)}
        for v in E:
            e = np.array(E[v])[..., JOINT8]
            r[v] = {'tangle_limb_cm': float(np.median(e[tang]) * 100), 'other_limb_cm': float(np.median(e[~tang]) * 100),
                    'pair_distance_err_cm': float(np.median(D[v]) * 100)}
        rep['subsets'][name] = r
        print(name, json.dumps(r), flush=True)
    json.dump(rep, open(out/'multiview.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
