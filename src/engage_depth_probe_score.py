"""Summary of engage_depth_probe.py's direct measurement (CPU, any venv with numpy). Experimental.

For each view, the depth gap between the two athletes' visible surfaces (median depth over the pixels where each is
visible in both the true and the predicted render) by the truth, the depth map and SAM 3D Body's render, on the same
pixels. Reports the median error of the gap, how often each gets the order (who is nearer) right when the true gap
is >= 10 cm, and the error by size of the true gap (an error that grows with the gap, not with the distance, is the
reason to use a depth map). Pre-registration: docs/engagement/results/depthmap_prereg.md (D0).

  python engage_depth_probe_score.py DIR/<model>/views.jsonl
"""
import json
import sys
from collections import defaultdict

import numpy as np


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    by = defaultdict(list)
    for r in rows:
        a = r['athletes']
        if not all('true' in s for s in a):
            continue
        g = {k: a[0][k] - a[1][k] for k in ('true', 'map', 's3d')}
        by[r['seq'].split('_', 1)[1]].append(g)
        by['pooled'].append(g)
    print(f'{sys.argv[1]}: {len(rows)} views, gap measured on {len(by["pooled"])}')
    for act, gs in by.items():
        t = np.array([g['true'] for g in gs])
        line = [f'=== {act} ({len(gs)} views; median |true gap| {np.median(np.abs(t))*100:.1f} cm)']
        for k in ('map', 's3d'):
            e = np.abs(np.array([g[k] for g in gs]) - t)
            big = np.abs(t) >= 0.10
            order = np.mean(np.sign([g[k] for g in gs])[big] == np.sign(t[big]))
            line.append(f'  {k:4s} gap error median {np.median(e)*100:5.1f} cm, p75 {np.percentile(e, 75)*100:5.1f} cm; '
                        f'nearer athlete right {order*100:4.0f}% (true gap >= 10 cm, n={big.sum()})')
        for lo, hi in ((0, .1), (.1, .25), (.25, .5), (.5, 9)):
            sel = (np.abs(t) >= lo) & (np.abs(t) < hi)
            if sel.sum() >= 10:
                em = np.median(np.abs(np.array([g['map'] for g in gs])[sel] - t[sel]))
                es = np.median(np.abs(np.array([g['s3d'] for g in gs])[sel] - t[sel]))
                line.append(f'  true gap {lo*100:3.0f}-{hi*100:3.0f} cm (n={sel.sum():4d}): map {em*100:5.1f} cm, '
                            f's3d {es*100:5.1f} cm')
        print('\n'.join(line))


if __name__ == '__main__':
    main()
