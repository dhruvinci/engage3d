"""Score engagement relations on ViCoS against the human position labels (CPU, .venv-pose). Experimental.

Input: engage_vicos_s3d.py output. Pre-registered hypotheses H1-H6: docs/engagement/results/vicos_prereg.md.
Roles: D = the numbered athlete (top in mount / side control / turtle, the guard player in closed / half / open guard,
unknown in back), O = the other; 'standing' and '5050_guard' have no number (relations counted in either direction).
Relation rates per position on predicted bodies (margin >= 5 cm), with 95% intervals from a cluster bootstrap over the
16 video streams (streams of one session share moments, so the intervals are optimistic).

  ../.venv-pose/bin/python engage_vicos_score.py DIR/vicos_s3d.npz [--out DIR/score]
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_relations import canon_from_pred, relations  # noqa: E402

MARGIN = 0.05
POS = ['standing', 'takedown', 'open_guard', 'half_guard', 'closed_guard', '5050_guard', 'side_control', 'mount',
       'back', 'turtle']


def roles(position):
    base = position.rstrip('12')
    num = position[len(base):]
    return base, ({'1': 'A', '2': 'B'}[num] if num else None)


def features(thr, D):
    """Role-mapped relation flags. thr: {'B.torso through A.leg_loop': [segment, margin]} (A = pose1, B = pose2)."""
    s = {k for k, (_, m) in thr.items() if m >= MARGIN}

    def has(chain_owner, chains, loop_owner, loop):
        return any(f'{chain_owner}.{c} through {loop_owner}.{loop}' in s for c in chains)
    legs, torso = ('l_leg', 'r_leg'), ('torso',)
    if D is None:   # symmetric classes: either direction
        both = lambda c, l: has('A', c, 'B', l) or has('B', c, 'A', l)
        return {'O.torso->D.leg_loop': both(torso, 'leg_loop'), 'D.torso->O.leg_loop': both(torso, 'leg_loop'),
                'O.leg->D.leg_loop': both(legs, 'leg_loop'), 'D.leg->O.leg_loop': both(legs, 'leg_loop'),
                'mutual_leg': has('A', legs, 'B', 'leg_loop') and has('B', legs, 'A', 'leg_loop'),
                'any_leg_loop': both(torso + legs, 'leg_loop'), 'torso_leg_loop_either': both(torso, 'leg_loop'),
                'any_arm_loop': any(k.endswith('arm_loop') for k in s)}
    O = 'B' if D == 'A' else 'A'
    f = {'O.torso->D.leg_loop': has(O, torso, D, 'leg_loop'), 'D.torso->O.leg_loop': has(D, torso, O, 'leg_loop'),
         'O.leg->D.leg_loop': has(O, legs, D, 'leg_loop'), 'D.leg->O.leg_loop': has(D, legs, O, 'leg_loop')}
    f['mutual_leg'] = f['O.leg->D.leg_loop'] and f['D.leg->O.leg_loop']
    f['any_leg_loop'] = any(has(x, torso + legs, y, 'leg_loop') for x, y in ((O, D), (D, O)))
    f['torso_leg_loop_either'] = f['O.torso->D.leg_loop'] or f['D.torso->O.leg_loop']
    f['any_arm_loop'] = any(k.endswith('arm_loop') for k in s)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    out = Path(a.out or Path(a.npz).parent/'score')
    out.mkdir(parents=True, exist_ok=True)
    z = dict(np.load(a.npz))   # load each array once (indexing an NpzFile decompresses it every time)
    rows = []
    for i in range(len(z['image'])):
        C = [canon_from_pred(z['kp3d'][i, n].astype(np.float64) + z['t'][i, n]) for n in range(2)]
        thr = relations(C[0], C[1])['threads']
        base, D = roles(str(z['position'][i]))
        f = features(thr, D)
        # both predicted bodies on one athlete: corresponding canonical points a median < 15 cm apart
        f['duplicate'] = float(np.median([np.linalg.norm(C[0][k] - C[1][k]) for k in C[0]])) < 0.15
        rows.append({'image': str(z['image'][i]), 'stream': str(z['image'][i])[:2], 'position': str(z['position'][i]),
                     'base': base, 'D': D, 'threads': thr, 'f': f})
    with open(out/'records.jsonl', 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    feats = list(rows[0]['f'])
    by = defaultdict(list)
    for r in rows:
        by[r['base']].append(r)
    rng = np.random.default_rng(0)

    def rate(rs, k):
        return float(np.mean([r['f'][k] for r in rs])) if rs else float('nan')

    def ci(rs, k, B=1000):
        st = defaultdict(list)
        for r in rs:
            st[r['stream']].append(r)
        keys = sorted(st)
        v = [rate([r for s in rng.choice(keys, len(keys)) for r in st[s]], k) for _ in range(B)]
        return np.nanpercentile(v, [2.5, 97.5])
    table = {p: {k: rate(by[p], k) for k in feats} for p in POS}
    lines = [f'{len(rows)} frames; relation rates on predicted bodies (margin >= 5 cm)', '',
             f"{'position':14s} {'n':>4s} " + ' '.join(f'{k[:13]:>13s}' for k in feats)]
    for p in POS:
        lines.append(f'{p:14s} {len(by[p]):4d} ' + ' '.join(f'{table[p][k]:13.2f}' for k in feats))
    # pre-registered hypotheses
    def rc(p, k):
        lo, hi = ci(by[p], k)
        return table[p][k], lo, hi
    H = {}
    m, lo, hi = rc('mount', 'O.torso->D.leg_loop'); s, slo, shi = rc('side_control', 'O.torso->D.leg_loop')
    H['H1 mount O.torso->D.leg_loop >= 0.60; side control <= 0.20'] = (m >= 0.6 and s <= 0.2, f'mount {m:.2f} [{lo:.2f},{hi:.2f}], side control {s:.2f} [{slo:.2f},{shi:.2f}]')
    g, lo, hi = rc('closed_guard', 'O.torso->D.leg_loop')
    H['H2 closed guard O.torso->D.leg_loop >= 0.60'] = (g >= 0.6, f'{g:.2f} [{lo:.2f},{hi:.2f}]')
    h, lo, hi = rc('half_guard', 'O.leg->D.leg_loop'); hc = table['closed_guard']['O.leg->D.leg_loop']
    H['H3 half guard O.leg->D.leg_loop >= 0.40 and > closed guard'] = (h >= 0.4 and h > hc, f'half {h:.2f} [{lo:.2f},{hi:.2f}] vs closed {hc:.2f}')
    mu = {p: table[p]['mutual_leg'] for p in POS}
    H['H4 50/50 highest mutual leg threading'] = (max(mu, key=mu.get) == '5050_guard', ', '.join(f'{p} {v:.2f}' for p, v in sorted(mu.items(), key=lambda x: -x[1])[:4]))
    lowp = ['standing', 'takedown', 'turtle', 'side_control']
    H['H5 standing/takedown/turtle/side control any leg-loop threading <= 0.20'] = (all(table[p]['any_leg_loop'] <= 0.2 for p in lowp), ', '.join(f'{p} {table[p]["any_leg_loop"]:.2f}' for p in lowp))
    b, lo, hi = rc('back', 'torso_leg_loop_either')
    H['H6 back torso through either leg loop >= 0.40'] = (b >= 0.4, f'{b:.2f} [{lo:.2f},{hi:.2f}]')
    lines += ['', 'pre-registered hypotheses:'] + [f"  {'PASS' if ok else 'FAIL'}  {k}: {v}" for k, (ok, v) in H.items()]
    (out/'summary.txt').write_text('\n'.join(lines) + '\n')
    (out/'summary.json').write_text(json.dumps({'table': table, 'hypotheses': {k: [bool(ok), v] for k, (ok, v) in H.items()},
                                                'n': {p: len(by[p]) for p in POS}}, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
