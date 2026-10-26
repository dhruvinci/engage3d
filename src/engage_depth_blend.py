"""Blend SAM 3D Body's and the depth map's relative depth (CPU, .venv-pose). Experimental, exploratory (after
depthmap_prereg.md; not pre-registered).

On Harmony4D grappling the depth map shrinks the depth gap between the two athletes (0.84 x the truth) while SAM 3D
Body stretches it (1.20 x), so their errors oppose and a weighted mean beats both. The weight is fitted per held-out
sequence (leave one sequence out, on the visible-surface gaps in engage_depth_probe.py's views.jsonl), and each body
is moved along its ray by that fraction of the `*_rel` variant's move: gap = w x map + (1 - w) x SAM 3D Body.
Writes <dir>/{gt_blend,pred_blend}/<seq>/<cam>.npz (score with engage_bench_oracle.py) and <dir>/blend_weights.json.

  ../.venv-pose/bin/python engage_depth_blend.py DIR/<model> [--weight W]   (fixed weight: no fitting, for a fresh test)
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np


def main():
    d = Path(sys.argv[1])
    fixed = float(sys.argv[sys.argv.index('--weight') + 1]) if '--weight' in sys.argv else None
    rows = [json.loads(l) for l in open(d/'views.jsonl')]
    rows = [r for r in rows if all('true' in s for s in r['athletes'])]
    gap = lambda r, k: r['athletes'][0][k] - r['athletes'][1][k]
    seq = np.array([r['seq'] for r in rows])
    T, M, S = (np.array([gap(r, k) for r in rows]) for k in ('true', 'map', 's3d'))
    grid = np.linspace(0, 1, 21)
    w = {}
    for s in sorted(set(seq)) if fixed is None else []:
        tr = (seq != s) & np.array(['grappling' in x for x in seq])   # fitted on grappling only
        w[s] = float(grid[np.argmin([np.median(np.abs(x*M[tr] + (1 - x)*S[tr] - T[tr])) for x in grid])])
    if fixed is None:
        (d/'blend_weights.json').write_text(json.dumps(w, indent=1))
    for p in sorted(glob.glob(str(d/'pred/*/cam??.npz'))):
        sname, cam = p.split('/')[-2], Path(p).stem
        base = dict(np.load(p))
        for src, dst in (('gt_rel', 'gt_blend'), ('pred_rel', 'pred_blend')):
            moved = np.load(d/src/sname/f'{cam}.npz')['t']
            out = d/dst/sname/f'{cam}.npz'
            out.parent.mkdir(parents=True, exist_ok=True)
            ws = w[sname] if fixed is None else fixed
            np.savez_compressed(out, **{**base, 't': (base['t'] + ws*(moved - base['t'])).astype(np.float32)})
    print('weights', w)


if __name__ == '__main__':
    main()
