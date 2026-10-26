"""Joint two-person rigid fit of predicted 3D bodies (CPU; .venv-pose: numpy + scipy). Experimental.

Pre-registration: docs/engagement/results/jointfit_prereg.md. For each view, both SAM 3D Body predictions may rotate
(about their own pelvis) and translate as a whole (12 parameters; limbs are not re-bent), minimising together:
  2d     each body's COCO-17 keypoints projected through the camera vs independent 2D keypoints of that person
         (confidence-weighted, in units of 2% of the person's 2D size; robust soft-L1 loss)
  pen    no interpenetration: a subset of each body's vertices inside the other body's limb capsules (radius from
         that body's own mesh, per bone), in cm
  prior  stay near the model's estimate: translation 5 cm laterally / 15 cm in depth, rotation 0.15 rad
Arms: full, no_pen (2d + prior), no_2d (pen + prior).

  ../.venv-pose/bin/python engage_jointfit.py h4d   S3D_DIR OUT_DIR ARM      (2D: <seq>/<cam>_vitpose.npz)
  ../.venv-pose/bin/python engage_jointfit.py vicos VICOS_NPZ OUT_DIR ARM   (2D: the dataset's own poses)
Output has the input's format (root-relative kp3d / verts and translation t), so the existing scorers read it.
"""
import glob
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')   # many tiny numpy ops: threads only contend (2 s -> 0.08 s per view)
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_relations import BONES, canon_from_mhr  # noqa: E402

COCO_TO_MHR = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
NSUB = 150
SIG_T = np.array([0.05, 0.05, 0.15])
SIG_R = 0.15
from engage_paths import VICOS   # noqa: E402
_SUB = None


def fisheye_fn(K, D):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    k = np.asarray(D).ravel()

    def proj(X):
        z = np.maximum(X[:, 2], 1e-3)
        x, y = X[:, 0]/z, X[:, 1]/z
        r = np.sqrt(x*x + y*y)
        th = np.arctan(r)
        thd = th*(1 + k[0]*th**2 + k[1]*th**4 + k[2]*th**6 + k[3]*th**8)
        s = np.where(r > 1e-8, thd/np.maximum(r, 1e-8), 1.0)
        return np.stack([fx*s*x + cx, fy*s*y + cy], 1)
    return proj


def pinhole_fn(f, c):
    def proj(X):
        z = np.maximum(X[:, 2], 1e-3)
        return f*X[:, :2]/z[:, None] + c
    return proj


def bones_of(C):
    return np.array([C[a] for _, a, _ in BONES]), np.array([C[b] for _, _, b in BONES])


def rodrigues(r):
    th = np.linalg.norm(r)
    if th < 1e-9:
        return np.eye(3)
    k = r/th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th)*Kx + (1 - np.cos(th))*(Kx @ Kx)


def seg_dist(P, A, B):
    """[n, m] distance of points P [n,3] to segments A->B [m,3]."""
    AB = B - A
    AP = P[:, None, :] - A[None]
    t = np.clip((AP*AB[None]).sum(-1)/np.maximum((AB*AB).sum(-1), 1e-12)[None], 0, 1)
    return np.linalg.norm(AP - t[..., None]*AB[None], axis=-1)


def radii(V, C):
    """Capsule radius per bone: 0.8 x the median distance of the vertices nearest that bone (inner capsule)."""
    A, B = bones_of(C)
    D = seg_dist(V[::4], A, B)
    near = D.argmin(1)
    return np.array([0.8*np.median(D[near == b, b]) if (near == b).sum() > 5 else 0.02 for b in range(len(A))])


def fit_view(kp3d, verts, t, kp2, sc, proj, arm):
    """kp3d [2,70,3], verts [2,n,3] (root-relative), t [2,3]; kp2 [2,17,2], sc [2,17]. Returns new (kp3d, verts, t)
    in the same convention, and fit diagnostics."""
    global _SUB
    if _SUB is None or _SUB.max() >= verts.shape[1]:
        _SUB = np.linspace(0, verts.shape[1] - 1, NSUB).astype(int)
    K = [kp3d[n].astype(np.float64) + t[n] for n in range(2)]
    V = [verts[n][_SUB].astype(np.float64) + t[n] for n in range(2)]
    Vfull = [verts[n].astype(np.float64) + t[n] for n in range(2)]
    C = [canon_from_mhr(K[n]) for n in range(2)]
    P = [C[n]['pelvis'] for n in range(2)]
    rad = [radii(Vfull[n], C[n]) for n in range(2)]
    w2 = 0.0 if arm == 'no_2d' else 1.0
    wp = 0.0 if arm == 'no_pen' else 1.0
    size = [max(np.ptp(kp2[n][sc[n] > 0.3], 0).max() if (sc[n] > 0.3).sum() >= 3 else 100.0, 20.0) for n in range(2)]
    canon_keys = list(C[0])
    Ck = [np.array([C[n][k] for k in canon_keys]) for n in range(2)]
    ia = np.array([canon_keys.index(a) for _, a, _ in BONES])
    ib = np.array([canon_keys.index(b) for _, _, b in BONES])
    Kc = [K[n][COCO_TO_MHR] for n in range(2)]
    m2 = [sc[n] > 0.3 for n in range(2)]

    def tf(x, n, X):
        return (X - P[n]) @ rodrigues(x[6*n:6*n + 3]).T + P[n] + x[6*n + 3:6*n + 6]

    def res(x):
        out = []
        for n in range(2):
            if w2:
                m = m2[n]
                q = proj(tf(x, n, Kc[n][m]))
                out.append((w2*sc[n][m, None]*(q - kp2[n][m])/(0.02*size[n])).ravel())
            out.append(x[6*n + 3:6*n + 6]/SIG_T)
            out.append(x[6*n:6*n + 3]/SIG_R)
        if wp:
            for a, b in ((0, 1), (1, 0)):
                Va = tf(x, a, V[a])
                cb = tf(x, b, Ck[b])
                pen = np.maximum(rad[b][None] - seg_dist(Va, cb[ia], cb[ib]), 0).max(1)
                out.append(wp*pen/0.01)
        return np.concatenate(out)
    x0 = np.zeros(12)
    r0 = res(x0)
    s = least_squares(res, x0, loss='soft_l1', f_scale=1.0, max_nfev=40, x_scale=np.r_[[0.1]*3, [0.05]*3, [0.1]*3, [0.05]*3])
    new_k, new_v, new_t = [], [], []
    for n in range(2):
        tn = t[n] + s.x[6*n + 3:6*n + 6]
        new_k.append(tf(s.x, n, K[n]) - tn)
        new_v.append((tf(s.x, n, Vfull[n]) - tn).astype(np.float16))
        new_t.append(tn)
    diag = {'cost0': float(0.5*(r0**2).sum()), 'cost': float(s.cost), 'dt_cm': [float(np.linalg.norm(s.x[6*n + 3:6*n + 6])*100)
                                                                             for n in range(2)]}
    return np.array(new_k, np.float32), np.array(new_v), np.array(new_t, np.float32), diag


def h4d_job(args):
    p, out, arm = args
    from engage_bench_s3d import seq_root
    from tangle_h4d import Sequence
    sname, cam = p.split('/')[-2], Path(p).stem
    dst = Path(out)/sname/f'{cam}.npz'
    if dst.exists():
        return f'{sname}/{cam}: exists'
    seq = Sequence(seq_root(sname))
    _, _, cid = seq.poses[cam]
    W, H, Pp = seq.cams[cid]
    Kc = np.array([[Pp[0], 0, Pp[2]], [0, Pp[1], Pp[3]], [0, 0, 1]])
    proj = fisheye_fn(Kc, Pp[4:8])
    z = dict(np.load(p))   # load each array once (indexing an NpzFile decompresses it every time)
    v = dict(np.load(p[:-4] + '_vitpose.npz'))
    assert (v['frames'] == z['frames']).all()
    out_k, out_v, out_t, diags = [], [], [], []
    t0 = time.time()
    for i in range(len(z['frames'])):
        k, vv, tt, dg = fit_view(z['kp3d'][i], z['verts'][i], z['t'][i], v['kp'][i], v['score'][i], proj, arm)
        out_k.append(k); out_v.append(vv); out_t.append(tt); diags.append(dg)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, frames=z['frames'], kp3d=np.array(out_k), verts=np.array(out_v), t=np.array(out_t))
    (Path(out)/sname/f'{cam}_fit.json').write_text(json.dumps(diags))
    dt = np.median([d['dt_cm'] for d in diags])
    return f'{sname}/{cam}: {len(diags)} views {time.time() - t0:.0f}s, median move {dt:.1f} cm'


def vicos_run(npz, out, arm, workers):
    ann = {x['image']: x for x in json.load(open(VICOS/'annotations.json'))}
    z = dict(np.load(npz))   # load each array once: indexing an NpzFile decompresses the whole array every time
    jobs = []
    for i in range(len(z['image'])):
        a = ann[str(z['image'][i])]
        kp = np.array([np.asarray(a['pose1'])[:, :2], np.asarray(a['pose2'])[:, :2]], float)
        sc = np.array([np.asarray(a['pose1'])[:, 2], np.asarray(a['pose2'])[:, 2]], float)
        W, H = z['wh'][i]
        jobs.append((z['kp3d'][i], z['verts'][i], z['t'][i], kp, sc, float(z['f_used'][i, 0]), (W/2, H/2), arm))
    with Pool(workers) as pool:
        res = pool.map(_vicos_one, jobs, chunksize=16)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    keep = {k: z[k] for k in ('image', 'position', 'wh', 'f_used')}
    np.savez_compressed(out/'vicos_s3d.npz', kp3d=np.array([r[0] for r in res]), verts=np.array([r[1] for r in res]),
                        t=np.array([r[2] for r in res]), **keep)
    (out/'fit.json').write_text(json.dumps([r[3] for r in res]))
    print('vicos', arm, len(res), 'frames; median move', np.median([r[3]['dt_cm'] for r in res]), 'cm', flush=True)


def _vicos_one(j):
    kp3d, verts, t, kp, sc, f, c, arm = j
    return fit_view(kp3d, verts, t, kp, sc, pinhole_fn(f, np.array(c)), arm)


if __name__ == '__main__':
    mode, src, out, arm = sys.argv[1:5]
    workers = int(os.environ.get('WORKERS', 3))
    if mode == 'h4d':
        files = sorted(glob.glob(f'{src}/*/cam??.npz'))
        with Pool(workers) as pool:
            for msg in pool.imap_unordered(h4d_job, [(p, out, arm) for p in files]):
                print(msg, flush=True)
    else:
        vicos_run(src, out, arm, workers)
