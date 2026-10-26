"""Hybrid predicted bodies: one model's bodies placed where another model put them (CPU, .venv-pose). Experimental.

Exploratory (not pre-registered), after phmr_prereg.md: PromptHMR placed the athletes better than SAM 3D Body
(distance error 6.1 vs 10.2 cm) but resolved wrapping limbs worse (body lock recall 0.38 vs 0.81). Here each body of
--pose (orientation and limbs kept) is translated so its pelvis sits on the pelvis of the same athlete from --place.
Both inputs must cover the same views (engage_bench_s3d.py / engage_bench_phmr.py outputs).

--anchor centroid uses the mean of all mesh vertices instead of the pelvis (the models put their hip joints at
different places on the body, so a pelvis anchor can shift a body by a pose-dependent offset).
  ../.venv-pose/bin/python engage_bench_hybrid.py --pose DIR --place DIR --out DIR [--anchor pelvis|centroid]
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_relations import canon_from_pred  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pose', required=True)
    ap.add_argument('--place', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--anchor', default='pelvis', choices=['pelvis', 'centroid'])
    a = ap.parse_args()
    for p in sorted(glob.glob(f'{a.pose}/*/cam??.npz')):
        rel = Path(p).relative_to(a.pose)
        z, q = dict(np.load(p)), dict(np.load(Path(a.place)/rel))   # load each array once
        common = np.intersect1d(z['frames'], q['frames'])
        iz = np.searchsorted(z['frames'], common)
        iq = np.searchsorted(q['frames'], common)
        t = z['t'][iz].astype(np.float64)
        for j, (i, k) in enumerate(zip(iz, iq)):
            for n in range(2):
                if a.anchor == 'pelvis':
                    own = canon_from_pred(z['kp3d'][i, n].astype(np.float64) + z['t'][i, n])['pelvis']
                    tgt = canon_from_pred(q['kp3d'][k, n].astype(np.float64) + q['t'][k, n])['pelvis']
                else:
                    own = z['verts'][i, n].astype(np.float64).mean(0) + z['t'][i, n]
                    tgt = q['verts'][k, n].astype(np.float64).mean(0) + q['t'][k, n]
                t[j, n] += tgt - own
        dst = Path(a.out)/rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, frames=common, kp3d=z['kp3d'][iz], verts=z['verts'][iz], t=t.astype(np.float32))
        print(f'{rel}: {len(common)} views', flush=True)


if __name__ == '__main__':
    main()
