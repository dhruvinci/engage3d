"""Does combining views or frames fix monocular relation errors? (CPU, .venv-pose). Experimental.
Reads engage_bench_score.py records and re-scores threading and loop balance after fusing predictions:
  single     one camera, one frame (as scored)
  temporal   one camera, majority over the frame and its 2 neighbours on each side (sampled frames)
  2 cameras  majority over a pair of cameras (a relation counts if both agree... reported as 'both' and 'either')
  4 cameras  majority over all 4 cameras of the same moment (ties count as present)
Truth: present with margin >= 5 cm; ambiguous left out (as in the scorer).
  ../.venv-pose/bin/python engage_bench_fuse.py RECORDS.jsonl"""
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_bench_score import MARGIN, family, prf  # noqa: E402

R = [json.loads(l) for l in open(sys.argv[1])]
conf = lambda thr: {k for k, (_, m) in thr.items() if m >= 0.0}


def score(items):
    """items: list of (gt_threads, predicted key set). Returns F1 per family and loop-balance sign agreement."""
    C = defaultdict(lambda: np.zeros(3, int))
    bal = []
    for gt, pr in items:
        amb = {k for k, (_, m) in gt.items() if m < MARGIN}
        pos = {k for k, (_, m) in gt.items() if m >= MARGIN}
        for k in (pos | pr) - amb:
            C[family(k)] += [k in pos and k in pr, k not in pos and k in pr, k in pos and k not in pr]
        g = sum(1 if k.split(' through ')[1][0] == 'A' else -1 for k in pos)
        p = sum(1 if k.split(' through ')[1][0] == 'A' else -1 for k in pr)
        if g != 0:
            bal.append(np.sign(g) == np.sign(p))
    return {f: prf(*C[f])[2] for f in C}, float(np.mean(bal)) if bal else float('nan'), len(bal)


by_frame = defaultdict(dict)
by_cam = defaultdict(list)
for r in R:
    by_frame[(r['seq'], r['frame'])][r['cam']] = r
    by_cam[(r['seq'], r['cam'])].append(r)
arms = {}
arms['single'] = [(r['gt'], conf(r['pred'])) for r in R]
tmp = []
for rs in by_cam.values():
    rs.sort(key=lambda r: r['frame'])
    for i, r in enumerate(rs):
        win = rs[max(0, i - 2):i + 3]
        keys = set().union(*[conf(w['pred']) for w in win])
        tmp.append((r['gt'], {k for k in keys if sum(k in conf(w['pred']) for w in win) * 2 > len(win)}))
arms['temporal (5 frames)'] = tmp
two_both, two_either, four = [], [], []
for v in by_frame.values():
    cams = sorted(v)
    for a, b in itertools.combinations(cams, 2):
        pa, pb = conf(v[a]['pred']), conf(v[b]['pred'])
        two_both.append((v[a]['gt'], pa & pb))
        two_either.append((v[a]['gt'], pa | pb))
    if len(cams) >= 3:
        keys = set().union(*[conf(v[c]['pred']) for c in cams])
        four.append((v[cams[0]]['gt'], {k for k in keys if sum(k in conf(v[c]['pred']) for c in cams) * 2 >= len(cams)}))
arms['2 cameras, both'] = two_both
arms['2 cameras, either'] = two_either
arms['4 cameras, majority'] = four
fams = ['torso->arm_loop', 'arm->arm_loop', 'leg->leg_loop', 'leg->arm_loop', 'head->arm_loop']
print(f"{'arm':22s} {'n':>6s} " + ' '.join(f'{f:>16s}' for f in fams) + f" {'balance sign':>13s}")
for name, items in arms.items():
    F, b, nb = score(items)
    print(f"{name:22s} {len(items):6d} " + ' '.join(f'{F.get(f, float("nan")):16.2f}' for f in fams) + f" {b:9.0%} (n={nb})")
