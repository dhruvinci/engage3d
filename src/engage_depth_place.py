"""Place the two athletes along their camera rays from saved depth maps (CPU, .venv-depth). Experimental.

`engage_depth_probe.py` measures each athlete's depth offset (depth map - predicted render) over all of its visible
pixels. The map's error grows with the size of the gap it measures (`depthmap_results.md`: 4.5 cm for gaps under
10 cm, 12 cm over 50 cm), so this measures the offsets only where the two athletes meet in the picture: the pixels of
each athlete within --zone cm of the other's predicted silhouette (outside the BAND kept clear of it). Fewer than
MIN_PIX pixels there: all of its visible pixels. Each body moves along its ray by weight x (offset - mean offset),
i.e. gap = weight x map + (1 - weight) x SAM 3D Body.

Writes <out>/place/<seq>/<cam>.npz (score with engage_bench_oracle.py) and <out>/place.jsonl (per view: per athlete
offset, pixels used, ray displacement for weight 1, predicted and true pelvis), and prints the relative pelvis depth
error over a grid of weights (in-sample: for development only; fix the weight before a test).

  ../.venv-depth/bin/python engage_depth_place.py --maps DIR/maps --ref s3d_box --out DIR [--zone 0] [--weight 0.7] \
      [--erode 0] [--smooth 0]
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_bench_s3d import seq_root                 # noqa: E402
from engage_depth_probe import BAND, FACES, MIN_PIX, render  # noqa: E402
from engage_relations import canon_from_pred, canon_from_smpl  # noqa: E402
from tangle_h4d import Sequence                        # noqa: E402


def place_file(job):
    p, maps, out, zone, weight, erode, smooth = job
    sname, cam = p.split('/')[-2], Path(p).stem
    faces = np.load(FACES)['mhr']
    seq = Sequence(seq_root(sname))
    Rc, tc, _ = seq.poses[cam]
    A4 = seq.world_to_colmap
    sc = np.cbrt(np.linalg.det(A4[:3, :3]))
    tocam = lambda X: ((X @ A4[:3, :3].T + A4[:3, 3]) @ Rc.T + tc)/sc
    z = dict(np.load(p))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*BAND + 1, 2*BAND + 1))
    keep, T, recs = [], [], []
    for i, f in enumerate(z['frames']):
        mp = Path(maps)/sname/cam/f'{int(f):05d}.npz'
        if not mp.exists():
            continue
        m = np.load(mp)
        dm, Rv, Pv = m['depth'].astype(np.float32), m['R'], m['P']
        h, w = dm.shape
        Vc = [z['verts'][i, n].astype(np.float64) + z['t'][i, n] for n in range(2)]
        own, dep, sil = render([(V @ Rv.T, faces) for V in Vc], Pv, w, h)
        C = [canon_from_pred(z['kp3d'][i, n].astype(np.float64) + z['t'][i, n]) for n in range(2)]
        gt = np.load(seq.root/'processed_data/smpl'/f'{int(f):05d}.npy', allow_pickle=True).item()
        Cg = [canon_from_smpl(tocam(gt[k]['joints'][:24])) for k in sorted(gt)]
        rec = {'seq': sname, 'cam': cam, 'frame': int(f), 'athletes': []}
        disp = []
        for n in range(2):
            other = sil[1 - n].astype(np.uint8)
            base = (own == n + 1) & np.isfinite(dm) & ~cv2.dilate(other, kernel).astype(bool)
            if erode > 0:   # interior only: the map blurs across every edge
                inner = cv2.erode((own == n + 1).astype(np.uint8), np.ones((2*erode + 1, 2*erode + 1), np.uint8))
                if (base & inner.astype(bool)).sum() >= MIN_PIX:
                    base = base & inner.astype(bool)
            pix = base
            if zone > 0:
                dist = cv2.distanceTransform(1 - other, cv2.DIST_L2, 5)
                zone_px = zone*Pv[0, 0]/max(float(np.nanmedian(dep[own == n + 1])), 0.5)
                near = base & (dist <= BAND + zone_px)
                if near.sum() >= MIN_PIX:
                    pix = near
            ok = int(pix.sum()) >= MIN_PIX
            delta = float(np.median(dm[pix] - dep[pix])) if ok else None
            rec['athletes'].append({'delta': delta, 'n_pix': int(pix.sum()), 'zone_used': bool(pix is not base),
                                    'pelvis': C[n]['pelvis'].tolist(), 'pelvis_true': Cg[n]['pelvis'].tolist()})
        dl = [a['delta'] for a in rec['athletes']]
        rec['shift'] = [np.nan, np.nan] if None in dl else [dl[0] - np.mean(dl), dl[1] - np.mean(dl)]
        rec['zaxis'] = Rv[2].tolist()
        keep.append(i)
        recs.append(rec)
    S = np.array([r['shift'] for r in recs], float).reshape(-1, 2)
    if smooth > 0 and len(recs):   # median of each athlete's shift over neighbouring views (+-smooth, <= 4 frames apart each)
        fr = np.array([r['frame'] for r in recs])
        S = np.array([np.nanmedian(S[np.abs(fr - fr[j]) <= 4*smooth], 0) for j in range(len(recs))]).reshape(-1, 2)
    for j, (i, rec) in enumerate(zip(keep, recs)):
        t = z['t'][i].astype(np.float64).copy()
        for n in range(2):
            pel = np.array(rec['athletes'][n]['pelvis'])
            u = pel/np.linalg.norm(pel)
            shift = 0.0 if np.isnan(S[j, n]) else S[j, n]
            d = shift/(u @ np.array(rec['zaxis']))*u
            rec['athletes'][n]['disp'] = d.tolist()
            t[n] += weight*d
        T.append(t)
    if keep:
        dst = Path(out)/'place'/sname/f'{cam}.npz'
        dst.parent.mkdir(parents=True, exist_ok=True)
        ii = np.array(keep)
        np.savez_compressed(dst, frames=z['frames'][ii], kp3d=z['kp3d'][ii], verts=z['verts'][ii],
                            t=np.array(T, np.float32))
    return recs


def rel_depth_err(recs, w):
    """Median relative pelvis depth error (along the ray to athlete A) with displacement weight w."""
    e = []
    for r in recs:
        a, b = r['athletes']
        pa, pb = np.array(a['pelvis']) + w*np.array(a['disp']), np.array(b['pelvis']) + w*np.array(b['disp'])
        d = (pb - pa) - (np.array(b['pelvis_true']) - np.array(a['pelvis_true']))
        e.append(abs(d @ (pa/np.linalg.norm(pa))))
    return float(np.median(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--zone', type=float, default=0.0, help='metres from the other athlete (0 = all visible pixels)')
    ap.add_argument('--weight', type=float, default=0.7)
    ap.add_argument('--smooth', type=int, default=0, help='median of each athlete\'s shift over +- this many views')
    ap.add_argument('--erode', type=int, default=0, help='keep only pixels this far (px) inside the athlete\'s render')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    seqs = sorted({p.split('/')[-1] for p in glob.glob(f'{a.maps}/*')})
    files = [p for p in sorted(glob.glob(f'{a.ref}/*/cam??.npz'))
             if p.split('/')[-2] in seqs and (Path(a.maps)/p.split('/')[-2]/Path(p).stem).exists()]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as pool:
        recs = [r for part in pool.map(place_file, [(p, a.maps, a.out, a.zone, a.weight, a.erode, a.smooth) for p in files]) for r in part]
    with open(Path(a.out)/'place.jsonl', 'w') as fh:
        for r in recs:
            fh.write(json.dumps(r) + '\n')
    g = [r for r in recs if 'grappling' in r['seq']]
    print(f'{len(recs)} views ({len(g)} grappling); zone {a.zone} m; zone used for '
          f'{np.mean([x["zone_used"] for r in g for x in r["athletes"]])*100:.0f}% of grappling athletes')
    print('grappling rel pelvis depth error by weight: ' +
          ', '.join(f'{w:.1f}: {rel_depth_err(g, w)*100:.1f} cm' for w in np.arange(0, 1.01, 0.1)))


if __name__ == '__main__':
    main()
