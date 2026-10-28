"""Demo stage 4: which threading relations survive which 3D errors? (CPU, .venv-depth). Experimental; exploratory.

The true Harmony4D bodies (place_feats2, one camera per moment, camera frame) are given the errors our pipeline makes,
and the spec's threading relations (`engage_relations.relations`, margin >= MARGIN as in the ViCoS scorer) are compared
with the truth's:
  depth      athlete B moved along the camera axis by d cm (random sign): relative-depth error between the two people
  limbs      Gaussian noise on both athletes' 8 limb joints, scaled so the median 3D error is s cm
  duplicate  athlete B replaced by a copy of athlete A (the identity failure)
Per relation family (torso / arm / leg / head through a leg loop / arm loop): precision, recall and F1 of its presence,
and the share of moments whose loop balance (relations A holds minus relations B holds) keeps its sign.

  ../.venv-depth/bin/python engage_relation_noise.py --feats DIR --out DIR
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_relations import CANON, relations                  # noqa: E402
from engage_temporal import DEV, HELD, load_tracks              # noqa: E402

MARGIN = 0.05
LIMB = [CANON.index(k) for k in ('l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle')]
DEPTHS, LIMB_CM = [0, 2, 5, 10, 15, 20], [0, 2, 4, 6, 8]


def family(name):
    """'B.l_arm through A.leg_loop' -> 'arm through leg_loop'."""
    chain, loop = name.split(' through ')
    c = chain.split('.')[1]
    c = 'arm' if 'arm' in c else 'leg' if 'leg' in c else c
    return f'{c} through {loop.split(".")[1]}'


def rels(A, B):
    thr = relations({k: A[i] for i, k in enumerate(CANON)}, {k: B[i] for i, k in enumerate(CANON)})['threads']
    return {k for k, (_, m) in thr.items() if m >= MARGIN}


def balance(R):
    return sum(1 for k in R if k.endswith('A.leg_loop') or k.endswith('A.arm_loop')) - \
           sum(1 for k in R if k.endswith('B.leg_loop') or k.endswith('B.arm_loop'))


def score(truth, pred):
    fam = defaultdict(lambda: [0, 0, 0])                          # tp, fp, fn
    for T, Pp in zip(truth, pred):
        for k in T | Pp:
            f = family(k)
            fam[f][0] += k in T and k in Pp
            fam[f][1] += k in Pp and k not in T
            fam[f][2] += k in T and k not in Pp
    out = {}
    for f, (tp, fp, fn) in sorted(fam.items()):
        if tp + fn >= 10:
            out[f] = {'f1': 2 * tp / max(2 * tp + fp + fn, 1), 'truth_count': tp + fn}
    agree = [np.sign(balance(T)) == np.sign(balance(Pp)) for T, Pp in zip(truth, pred) if balance(T) != 0]
    out['loop_balance_sign_kept'] = float(np.mean(agree)) if agree else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seqs', default='')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    _, G, _, key, _ = load_tracks(a.feats, a.seqs.split(',') if a.seqs else DEV + HELD + ['016_mma4'], {'cam01'})
    rng = np.random.default_rng(0)
    truth = [rels(g[0], g[1]) for g in G]
    rep = {'moments': len(G), 'moments_with_a_relation': int(sum(bool(t) for t in truth)), 'depth_cm': {}, 'limbs_cm': {}}
    for d in DEPTHS:
        sign = rng.choice([-1, 1], len(G))
        pred = []
        for g, s in zip(G, sign):
            B = g[1].copy(); B[:, 2] += s * d / 100
            pred.append(rels(g[0], B))
        rep['depth_cm'][d] = score(truth, pred)
        print('depth', d, json.dumps(rep['depth_cm'][d]), flush=True)
    for s in LIMB_CM:
        sigma = s / 100 / 1.5382                                   # median radius of a 3D Gaussian = 1.5382 sigma
        pred = []
        for g in G:
            A, B = g[0].copy(), g[1].copy()
            A[LIMB] += rng.normal(0, sigma, (len(LIMB), 3)); B[LIMB] += rng.normal(0, sigma, (len(LIMB), 3))
            pred.append(rels(A, B))
        rep['limbs_cm'][s] = score(truth, pred)
        print('limbs', s, json.dumps(rep['limbs_cm'][s]), flush=True)
    rep['duplicate'] = score(truth, [rels(g[0], g[0]) for g in G])
    print('duplicate', json.dumps(rep['duplicate']), flush=True)
    json.dump(rep, open(out/'relation_noise.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
