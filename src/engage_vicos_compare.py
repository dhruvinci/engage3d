"""Compare two scored ViCoS runs (E05: SAM 3D Body from boxes; and from boxes + masks), and test whether geometry can
arbitrate the 2D position model's mount / side control confusion (CPU). Experimental; exploratory.

Per position: the share of frames where both bodies are the same body (duplicate), and the share showing the
position's defining relation (mount and closed guard: the other athlete's torso through the dominant one's leg loop;
half guard: a leg through the leg loop; back: a torso through a leg loop; others: any leg-loop threading, which should
be rare). Arbitration: on frames whose 2D prediction (blocked cross-validation, smoothed) is mount or side control,
predict mount when a torso threads either athlete's leg loop, else side control; accuracy against the experts.

  python3 engage_vicos_compare.py --a DIR_A/records.jsonl --b DIR_B/records.jsonl --pred DIR/predictions.npz
"""
import argparse
import json
from collections import defaultdict

import numpy as np

DEFINING = {'mount': 'O.torso->D.leg_loop', 'closed_guard': 'O.torso->D.leg_loop', 'half_guard': 'O.leg->D.leg_loop',
            'back': 'torso_leg_loop_either'}


def load(path):
    return {r['image']: r for r in map(json.loads, open(path))}


def table(recs):
    by = defaultdict(list)
    for r in recs.values():
        by[r['base']].append(r)
    out = {}
    for b, rs in sorted(by.items()):
        key = DEFINING.get(b, 'any_leg_loop')
        out[b] = {'n': len(rs), 'duplicate': float(np.mean([r['f']['duplicate'] for r in rs])),
                  'relation': key, 'relation_rate': float(np.mean([r['f'][key] for r in rs])),
                  'relation_rate_no_dup': float(np.mean([r['f'][key] for r in rs if not r['f']['duplicate']] or [np.nan]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--pred', required=True)
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    ta, tb = table(A), table(B)
    print(f"{'position':14s} {'n':>4s}  duplicate A -> B    defining relation A -> B   (relation)")
    for k in ta:
        x, y = ta[k], tb.get(k, {})
        print(f"{k:14s} {x['n']:4d}  {x['duplicate']:.2f} -> {y.get('duplicate', float('nan')):.2f}        "
              f"{x['relation_rate']:.2f} -> {y.get('relation_rate', float('nan')):.2f}   ({x['relation']})")
    z = np.load(a.pred)
    classes = [str(c) for c in z['classes']]
    pred = {f'{s}{int(f):05d}': classes[p] for s, f, p in zip(z['stream'], z['frame'], z['pred_smoothed'])}
    for name, R in (('A (boxes)', A), ('B (boxes + masks)', B)):
        n = ok2d = okgeo = 0
        for im, r in R.items():
            p = pred.get(im)
            if p is None or r['base'] not in ('mount', 'side_control') or p.rstrip('12') not in ('mount', 'side_control'):
                continue
            n += 1
            ok2d += p.rstrip('12') == r['base']
            geo = 'mount' if r['f']['torso_leg_loop_either'] and not r['f']['duplicate'] else 'side_control'
            okgeo += geo == r['base']
        print(f'arbitration {name}: {n} mount/side-control frames; 2D alone {ok2d / max(n, 1):.2f}; '
              f'geometry rule {okgeo / max(n, 1):.2f}')
    json.dump({'A': ta, 'B': tb}, open(a.b.rsplit('/', 1)[0] + '/compare.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
