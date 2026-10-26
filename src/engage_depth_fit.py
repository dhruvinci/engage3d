"""Turn each predicted body to match its surface in a monocular depth map (CPU, .venv-depth). Experimental.

The oracle (`jointfit_results.md`) shows that whole-body orientation, not only depth, separates good from bad
relations: with true position + orientation, SAM 3D Body's own limbs give body lock 0.93 and legs 0.74. The depth map
shows the 3D shape of each athlete's visible surface, and that shape fixes which way the body faces.

Input: depth maps saved by `engage_depth_probe.py --save-maps` and a placement to start from (a variant directory,
e.g. `depth_probe/moge/pred_blend`). Per view and athlete: the visible pixels from the predicted render minus a band
around the other body (no masks); the map's points there, shifted along their rays so that their median depth matches
the body's (depth stays the placement's); then point-to-point ICP between those points and the body's visible core
vertices (head, chest, back, hips, thighs: limbs are not trusted to fit), solving a small rotation about the pelvis
plus a translation, with a prior on the rotation (SIGMA_ROT) and robust trimming. Variants:
  rot      the rotation only (the pelvis stays where the placement put it)
  rot_lat  the rotation plus the translation across the line of sight
Writes <out>/<variant>/<seq>/<cam>.npz (score with engage_bench_oracle.py) and <out>/fit.jsonl (per view and athlete:
rotation applied, points used, torso orientation error against the truth before / after).

  ../.venv-depth/bin/python engage_depth_fit.py --maps DIR/maps --place DIR --out DIR [--relief 1.0] [--sigma-rot 15]
"""
import argparse
import glob
import json
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_bench_s3d import seq_root                       # noqa: E402
from engage_depth_probe import BAND, FACES, render           # noqa: E402
from engage_relations import REGIONS, canon_from_pred, canon_from_smpl, vertex_regions  # noqa: E402
from tangle_h4d import Sequence                              # noqa: E402

CORE = [REGIONS.index(r) for r in ('head', 'chest', 'back', 'hips', 'l_thigh', 'r_thigh')]
TORSO = ['pelvis', 'neck', 'l_hip', 'r_hip', 'l_shoulder', 'r_shoulder']
MAX_PTS = 3000
ITERS = 20
SIGMA_PT = 0.03   # metres: point noise
VARIANTS = ['rot', 'rot_lat']


def rodrigues(w):
    return cv2.Rodrigues(np.asarray(w, np.float64).reshape(3, 1))[0]


def kabsch_angle(P, Q):
    """Angle (deg) of the best rotation taking P to Q (rows = points)."""
    P, Q = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1)/2, -1, 1))))


def icp(W, Q, c, sigma_rot):
    """Small rotation (about c) + translation taking points W toward Q (nearest-neighbour association, trimmed), with
    a Gaussian prior of sigma_rot radians on the rotation. Returns R, t, inliers."""
    R, t = np.eye(3), np.zeros(3)
    prior = np.diag([1/sigma_rot**2]*3 + [0.0]*3)
    n_in = 0
    for _ in range(ITERS):
        X = (W - c) @ R.T + c + t
        d, j = cKDTree(X).query(Q, k=1)
        keep = d < max(0.03, np.percentile(d, 80))
        n_in = int(keep.sum())
        if n_in < 30:
            break
        x, q = X[j[keep]], Q[keep]
        r = (x - q).reshape(-1)
        a = x - c - t
        S = np.zeros((len(a), 3, 3))
        S[:, 0, 1], S[:, 0, 2], S[:, 1, 2] = -a[:, 2], a[:, 1], -a[:, 0]
        S[:, 1, 0], S[:, 2, 0], S[:, 2, 1] = a[:, 2], -a[:, 1], a[:, 0]
        J = np.concatenate([-S, np.broadcast_to(np.eye(3), S.shape)], 2).reshape(-1, 6)
        # prior on the total rotation: approximate by the current rotation vector
        w_cur = cv2.Rodrigues(R)[0].ravel()
        A = J.T @ J/SIGMA_PT**2 + prior
        b = -J.T @ r/SIGMA_PT**2 - prior @ np.r_[w_cur, np.zeros(3)]
        delta = np.linalg.solve(A, b)
        R = rodrigues(delta[:3]) @ R
        t = t + delta[3:]
        if np.linalg.norm(delta) < 1e-4:
            break
    return R, t, n_in


def fit_file(job):
    p, maps, out, relief, sigma_rot = job
    sname, cam = p.split('/')[-2], Path(p).stem
    if all((Path(out)/v/sname/f'{cam}.npz').exists() for v in VARIANTS):
        return []
    faces = np.load(FACES)['mhr']
    seq = Sequence(seq_root(sname))
    Rc, tc, _ = seq.poses[cam]
    A4 = seq.world_to_colmap
    sc = np.cbrt(np.linalg.det(A4[:3, :3]))
    tocam = lambda X: ((X @ A4[:3, :3].T + A4[:3, 3]) @ Rc.T + tc)/sc
    z = dict(np.load(p))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*BAND + 1, 2*BAND + 1))
    res = {v: {'kp3d': z['kp3d'].astype(np.float32).copy(), 'verts': z['verts'].copy(), 't': z['t'].copy()}
           for v in VARIANTS}
    recs = []
    for i, f in enumerate(z['frames']):
        mp = Path(maps)/sname/cam/f'{int(f):05d}.npz'
        if not mp.exists():
            continue
        m = np.load(mp)
        dm, Rv, Pv = m['depth'].astype(np.float32), m['R'], m['P']
        h, w = dm.shape
        Vc = [z['verts'][i, n].astype(np.float64) + z['t'][i, n] for n in range(2)]
        Jc = [z['kp3d'][i, n].astype(np.float64) + z['t'][i, n] for n in range(2)]
        own, dep, sil = render([(V @ Rv.T, faces) for V in Vc], Pv, w, h)
        gt = np.load(seq.root/'processed_data/smpl'/f'{int(f):05d}.npy', allow_pickle=True).item()
        Cg = [canon_from_smpl(tocam(gt[k]['joints'][:24])) for k in sorted(gt)]
        rec = {'seq': sname, 'cam': cam, 'frame': int(f), 'athletes': []}
        for n in range(2):
            C = canon_from_pred(Jc[n])
            st = {'err_before': kabsch_angle(np.array([C[k] for k in TORSO]), np.array([Cg[n][k] for k in TORSO]))}
            pix = (own == n + 1) & np.isfinite(dm) & ~cv2.dilate(sil[1 - n].astype(np.uint8), kernel).astype(bool)
            vv, uu = np.nonzero(pix)
            if len(uu) < 200:
                st['skipped'] = 'few pixels'
                rec['athletes'].append(st)
                continue
            if len(uu) > MAX_PTS:
                s = np.random.default_rng(int(f)*2 + n).choice(len(uu), MAX_PTS, replace=False)
                vv, uu = vv[s], uu[s]
            rays = np.c_[(uu + 0.5 - Pv[0, 2])/Pv[0, 0], (vv + 0.5 - Pv[1, 2])/Pv[1, 1], np.ones(len(uu))]
            dz = dm[vv, uu] - np.median(dm[vv, uu] - dep[vv, uu])      # depth offset handled by the placement
            zc = np.median(dz)
            Q = rays*(zc + relief*(dz - zc))[:, None]                   # virtual camera frame
            V = Vc[n] @ Rv.T
            uvz = V[:, :2]/V[:, 2:3]*Pv[[0, 1], [0, 1]] + Pv[:2, 2] - 0.5
            ui, vi = np.round(uvz).astype(int).T
            inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            vis = np.zeros(len(V), bool)
            vis[inside] = (own[vi[inside], ui[inside]] == n + 1) & \
                (np.abs(V[inside, 2] - dep[vi[inside], ui[inside]]) < 0.02)
            reg = vertex_regions(Vc[n], C)
            core = vis & np.isin(reg, CORE)
            if core.sum() < 100:
                st['skipped'] = 'few core vertices'
                rec['athletes'].append(st)
                continue
            cv_ = C['pelvis'] @ Rv.T
            Rf, tf, n_in = icp(V[core], Q, cv_, np.radians(sigma_rot))
            # back to the camera frame: rotation about the pelvis, translation split along / across the ray
            Rcam = Rv.T @ Rf @ Rv
            tcam = Rv.T @ tf
            u = C['pelvis']/np.linalg.norm(C['pelvis'])
            t_lat = tcam - (tcam @ u)*u
            for v, shift in (('rot', np.zeros(3)), ('rot_lat', t_lat)):
                Jn = (Jc[n] - C['pelvis']) @ Rcam.T + C['pelvis'] + shift
                Vn = (Vc[n] - C['pelvis']) @ Rcam.T + C['pelvis'] + shift
                tn = z['t'][i, n] + shift
                res[v]['kp3d'][i, n] = Jn - tn
                res[v]['verts'][i, n] = (Vn - tn).astype(np.float16)
                res[v]['t'][i, n] = tn
            Cn = canon_from_pred((Jc[n] - C['pelvis']) @ Rcam.T + C['pelvis'])
            st.update(rot_deg=float(np.degrees(np.linalg.norm(cv2.Rodrigues(Rf)[0]))), lat_cm=float(np.linalg.norm(t_lat)*100),
                      n_pts=int(len(Q)), n_in=n_in, n_core=int(core.sum()),
                      err_after=kabsch_angle(np.array([Cn[k] for k in TORSO]), np.array([Cg[n][k] for k in TORSO])))
            rec['athletes'].append(st)
        recs.append(rec)
    keep = [i for i, f in enumerate(z['frames']) if (Path(maps)/sname/cam/f'{int(f):05d}.npz').exists()]
    for v in VARIANTS:
        dst = Path(out)/v/sname/f'{cam}.npz'
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, frames=z['frames'][keep], **{k: res[v][k][keep] for k in ('kp3d', 'verts', 't')})
    print(f'{sname}/{cam}: {len(recs)} views', flush=True)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', required=True)
    ap.add_argument('--place', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--relief', type=float, default=1.0, help='stretch of the map\'s depth relief within a body')
    ap.add_argument('--sigma-rot', type=float, default=15.0, help='prior on the rotation (degrees)')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    files = sorted(glob.glob(f'{a.place}/*/cam??.npz'))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as pool:
        recs = [r for part in pool.map(fit_file, [(p, a.maps, a.out, a.relief, a.sigma_rot) for p in files])
                for r in part]
    with open(Path(a.out)/'fit.jsonl', 'a') as fh:
        for r in recs:
            fh.write(json.dumps(r) + '\n')
    st = [s for r in recs for s in r['athletes'] if 'err_after' in s]
    if st:
        b, f = np.array([s['err_before'] for s in st]), np.array([s['err_after'] for s in st])
        print(f'{len(st)} bodies fitted: torso orientation error median {np.median(b):.1f} -> {np.median(f):.1f} deg, '
              f'better in {np.mean(f < b)*100:.0f}%; rotation applied median {np.median([s["rot_deg"] for s in st]):.1f} deg')


if __name__ == '__main__':
    main()
