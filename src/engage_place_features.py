"""Features and targets for the learned two-person placement model (CPU, .venv-depth). Experimental.

Per view of the engagement benchmark (SAM 3D Body bodies from `s3d_box`, MoGe-2 maps from `engage_depth_probe.py
--save-maps`): the bodies are first placed with the confirmed combined depth (each moved along its ray by 0.7 x (map
offset - mean offset); `depthblend_prereg.md`), then described in the crop's virtual camera frame (z = viewing
direction of the pair), centred on the pair's midpoint:
  P       [2,19,3] both bodies' canonical points (the model corrects these)
  G       [2,19,3] the truth's canonical points in the same frame (targets)
  X       per-view features:
          - per athlete and body region (16): median(map - render) relative to the athlete's own median, and the
            region's share of the athlete's visible pixels (depth relief: where the map disagrees with the body's
            shape);
          - per athlete: 17 COCO keypoints' 2D residual (ViTPose+ - projected body, through the true fisheye; in units
            of the athlete's 2D size) and ViTPose confidence;
          - the map's gap minus SAM 3D Body's gap (before the combination), visible pixel counts (log), the pair's
            distance from the camera.
Also stored to write corrected bodies back: the view's ids, Rv, the midpoint m, and the combined-depth translation.
Writes <out>/<seq>__<cam>.npz per file (resumable).

  ../.venv-depth/bin/python engage_place_features.py --maps DIR/maps --ref s3d_box --out DIR [--workers 6] [--overlap]
  ../.venv-depth/bin/python engage_place_features.py --attach-tokens TOKDIR --out DIR
      adds `tok` [N,2,1024] (SAM 3D Body's final pose token per athlete, from engage_bench_s3d.py --save-tokens
      outputs in TOKDIR, matched by sequence, camera and frame) to the feature files that lack it
"""
import argparse
import glob
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_bench_s3d import fisheye, seq_root                         # noqa: E402
from engage_depth_probe import BAND, FACES, MIN_PIX, render           # noqa: E402
from engage_relations import CANON, REGIONS, canon_from_pred, canon_from_smpl, vertex_regions  # noqa: E402
from tangle_h4d import Sequence                                        # noqa: E402

WEIGHT = 0.7
COCO_TO_MHR = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
NR = len(REGIONS)


def overlap_feats(Vc, Rv, Pv, w, h, dm, delta, faces):
    """Where both predicted bodies cover the same pixel, which one the depth map says is visible there.
    Per athlete n (after removing its own map offset delta_n): median of (map - n's surface) over the overlap, and
    the share of overlap pixels where the map is closer to n's surface than to the other's. Then log(1 + overlap
    pixels) and the share where the map disagrees with the predicted front-to-back order. Zeros without overlap."""
    deps = [render([(V @ Rv.T, faces)], Pv, w, h)[1] for V in Vc]
    ov = np.isfinite(deps[0]) & np.isfinite(deps[1]) & np.isfinite(dm)
    if ov.sum() < 20 or not all(np.isfinite(delta)):
        return np.zeros(7)
    r = [dm[ov] - deps[n][ov] - delta[n] for n in range(2)]
    vis0 = np.abs(r[0]) < np.abs(r[1])                 # the map shows athlete 0's surface here
    front0 = deps[0][ov] < deps[1][ov]                 # the prediction puts athlete 0 in front here
    return np.array([np.median(r[0]), vis0.mean(), np.median(r[1]), 1 - vis0.mean(), np.log1p(ov.sum()),
                     float(np.mean(vis0 != front0)), 0.0])


def feats_file(job):
    p, maps, out, overlap = job
    sname, cam = p.split('/')[-2], Path(p).stem
    dst = Path(out)/f'{sname}__{cam}.npz'
    if dst.exists():
        return
    faces = np.load(FACES)['mhr']
    seq = Sequence(seq_root(sname))
    Rc, tc, cid = seq.poses[cam]
    _, _, Pc = seq.cams[cid]
    K = np.array([[Pc[0], 0, Pc[2]], [0, Pc[1], Pc[3]], [0, 0, 1]])
    Dk = np.array(Pc[4:8]).reshape(1, 4)
    A4 = seq.world_to_colmap
    sc = np.cbrt(np.linalg.det(A4[:3, :3]))
    tocam = lambda X: ((X @ A4[:3, :3].T + A4[:3, 3]) @ Rc.T + tc)/sc
    z = dict(np.load(p))
    vp = dict(np.load(p[:-4] + '_vitpose.npz'))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*BAND + 1, 2*BAND + 1))
    rows = {k: [] for k in ('i', 'frame', 'P', 'G', 'X', 'R', 'm', 't_blend')}
    for i, f in enumerate(z['frames']):
        mp = Path(maps)/sname/cam/f'{int(f):05d}.npz'
        j = np.nonzero(vp['frames'] == f)[0]
        if not mp.exists() or not len(j):
            continue
        j = j[0]
        m = np.load(mp)
        dm, Rv, Pv = m['depth'].astype(np.float32), m['R'], m['P']
        h, w = dm.shape
        Vc = [z['verts'][i, n].astype(np.float64) + z['t'][i, n] for n in range(2)]
        Jc = [z['kp3d'][i, n].astype(np.float64) + z['t'][i, n] for n in range(2)]
        own, dep, sil, fid = render([(V @ Rv.T, faces) for V in Vc], Pv, w, h, faces=True)
        C = [canon_from_pred(J) for J in Jc]
        delta, relief, npix = [], [], []
        for n in range(2):
            pix = (own == n + 1) & np.isfinite(dm) & ~cv2.dilate(sil[1 - n].astype(np.uint8), kernel).astype(bool)
            d = float(np.median(dm[pix] - dep[pix])) if pix.sum() >= MIN_PIX else np.nan
            freg = vertex_regions(Vc[n], C[n])[faces[:, 0]]
            reg = np.full(pix.shape, -1)
            reg[pix] = freg[fid[pix]]
            rr = np.zeros((NR, 2))
            for r in range(NR):
                sel = reg == r
                if sel.sum() >= 10 and np.isfinite(d):
                    rr[r] = [np.median(dm[sel] - dep[sel]) - d, sel.sum()/max(pix.sum(), 1)]
            delta.append(d)
            relief.append(rr)
            npix.append(int(pix.sum()))
        ok = all(np.isfinite(delta))
        shifts = [WEIGHT*(delta[n] - np.mean(delta)) if ok else 0.0 for n in range(2)]
        t = z['t'][i].astype(np.float64).copy()
        for n in range(2):
            u = C[n]['pelvis']/np.linalg.norm(C[n]['pelvis'])
            t[n] += shifts[n]/(u @ Rv[2])*u
        Jb = [z['kp3d'][i, n].astype(np.float64) + t[n] for n in range(2)]
        Cb = [canon_from_pred(J) for J in Jb]
        gt = np.load(seq.root/'processed_data/smpl'/f'{int(f):05d}.npy', allow_pickle=True).item()
        Cg = [canon_from_smpl(tocam(gt[k]['joints'][:24])) for k in sorted(gt)]
        Pb = np.array([[Cb[n][k] for k in CANON] for n in range(2)]) @ Rv.T
        Pg = np.array([[Cg[n][k] for k in CANON] for n in range(2)]) @ Rv.T
        mid = Pb.reshape(-1, 3).mean(0)
        kp2d = []
        for n in range(2):
            uv = fisheye(Jb[n][COCO_TO_MHR], K, Dk)
            obs, conf = vp['kp'][j, n].astype(np.float64), vp['score'][j, n].astype(np.float64)
            size = max(np.ptp(obs[conf > 0.3], 0).max() if (conf > 0.3).sum() >= 2 else 1.0, 20.0)
            kp2d.append(np.c_[(obs - uv)/size, conf])
        gap_map = delta[0] - delta[1] if ok else 0.0
        x = np.concatenate([np.array(relief).ravel(), np.array(kp2d).ravel(),
                            [gap_map, float(ok), np.log1p(npix[0]), np.log1p(npix[1]), float(np.linalg.norm(mid))]])
        if overlap:
            x = np.concatenate([x, overlap_feats(Vc, Rv, Pv, w, h, dm, delta, faces)])
        rows['i'].append(i)
        rows['frame'].append(int(f))
        rows['P'].append(Pb - mid)
        rows['G'].append(Pg - mid)
        rows['X'].append(x)
        rows['R'].append(Rv)
        rows['m'].append(mid)
        rows['t_blend'].append(t)
    if rows['i']:
        np.savez_compressed(dst, **{k: np.array(v) for k, v in rows.items()})
    print(f'{sname}/{cam}: {len(rows["i"])} views', flush=True)


def attach_tokens(tokdir, out):
    n = 0
    for p in sorted(glob.glob(f'{out}/*__cam??.npz')):
        d = dict(np.load(p))
        if 'tok' in d:
            continue
        sname, cam = Path(p).stem.split('__')
        src = Path(tokdir)/sname/f'{cam}.npz'
        if not src.exists():
            continue
        t = np.load(src)
        if 'token' not in t.files:
            continue
        idx = {int(f): j for j, f in enumerate(t['frames'])}
        if not all(int(f) in idx for f in d['frame']):
            print(f'{sname}/{cam}: token frames missing, skipped', flush=True)
            continue
        d['tok'] = t['token'][[idx[int(f)] for f in d['frame']]]
        np.savez_compressed(p, **d)
        n += 1
    print(f'tokens attached to {n} files', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attach-tokens', default='')
    ap.add_argument('--maps', default='')
    ap.add_argument('--ref', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--overlap', action='store_true', help='append the 7 overlap (front-to-back) features')
    ap.add_argument('--cams', default='', help='only these cameras (comma-separated)')
    a = ap.parse_args()
    if a.attach_tokens:
        return attach_tokens(a.attach_tokens, a.out)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(glob.glob(f'{a.ref}/*/cam??.npz')) if (Path(a.maps)/p.split('/')[-2]/Path(p).stem).exists()
             and (not a.cams or Path(p).stem in a.cams.split(','))]
    with Pool(a.workers) as pool:
        pool.map(feats_file, [(p, a.maps, a.out, a.overlap) for p in files])


if __name__ == '__main__':
    main()
