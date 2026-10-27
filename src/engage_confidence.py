"""E17: can the pipeline tell, per frame, when its own measurement is wrong? (CPU, .venv-depth). Experimental.
Pre-registration: `docs/engagement/results/confidence_prereg.md`.

Targets (from the truth, used only to train and score the flag): the distance error between the two people after the
learned placement, and the error of the limb joints in contact (E11's measure, released model).
Signals (no truth): the disagreement of the placement ensemble's 3 members; the predicted closest approach of the two
bodies and their distance; the overlap block of the features (mutual occlusion); ViTPose+ confidence and 2D residuals
of the limb joints; the depth map's evidence (gap, valid flag, visible pixels); how far the correction moved the bodies.
A logistic flag per target is trained on out-of-fold predictions of the 12 development sequences (4-fold CV by
sequence, E08's placement settings) and tested on held-out grappling, MMA, hugging, ballroom and sword fighting.

  TORCH_THREADS=8 ../.venv-depth/bin/python engage_confidence.py --feats DIR --out DIR [--smoke]
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_limb_head import CORE14, JOINT8, tangle_mask       # noqa: E402
from engage_place_train import train                             # noqa: E402
from engage_relations import CANON                               # noqa: E402

DEV = ['025_grappling2', '028_grappling2', '030_grappling2', '032_grappling2', '033_grappling2', '034_grappling2',
       '035_grappling2', '036_grappling2', '038_grappling2', '039_grappling2', '040_grappling2', '041_grappling2']
HELD = ['031_grappling2', '037_grappling2', '042_grappling2', '043_grappling2']
CAMS4 = {'cam01', 'cam06', 'cam11', 'cam16'}
OTHER4 = {'cam03', 'cam08', 'cam13', 'cam18'}
ALL20 = {f'cam{i:02d}' for i in range(1, 21)}
# confirmatory sets first: views a smoke run never scored (it scored the CAMS4 sets below by mistake; see the
# pre-registration); the CAMS4 sets and the other activities are secondary
TESTS = {'C_heldout_grappling_cams_03_08_13_18': (HELD, OTHER4), 'C_mma_other_16_cams': (['016_mma4'], ALL20 - CAMS4),
         'heldout_grappling': (HELD, CAMS4), 'mma': (['016_mma4'], CAMS4), 'hugging': (['002_hugging'], CAMS4),
         'ballroom': (['007_ballroom2', '008_ballroom2', '009_ballroom2', '010_ballroom2'], CAMS4),
         'sword': (['009_sword2', '010_sword2'], CAMS4)}
DIST_BAD, LIMB_BAD = 0.05, 0.06
SETTINGS = dict(epochs=60, width=512, seeds=3, wd=1e-3, drop=0.2)
PEL = CANON.index('pelvis')
LIMB_COCO = [7, 8, 9, 10, 13, 14, 15, 16]


def load(feats, seqs, cams):
    D = {k: [] for k in ('P', 'G', 'X', 'seq')}
    for p in sorted(glob.glob(f'{feats}/*__cam??.npz')):
        s, c = Path(p).stem.split('__')
        if s not in seqs or c not in cams:
            continue
        d = np.load(p)
        if d['X'].shape[1] < 178:
            continue
        D['P'].append(d['P'].astype(np.float64)); D['G'].append(d['G'].astype(np.float64)); D['X'].append(d['X'])
        D['seq'] += [s] * len(d['P'])
    return {k: (np.concatenate(v) if k != 'seq' else np.array(v)) for k, v in D.items()}


def pair_dist(A):
    return np.linalg.norm(A[:, 1, PEL] - A[:, 0, PEL], axis=-1)


def limb_err(P, G):
    """[V] median error of the limb joints inside a tangle (NaN when none), rigid-aligned over CORE14."""
    t = tangle_mask(G)
    out = np.full(len(P), np.nan)
    e = np.zeros(P.shape[:3])
    for n in range(2):
        A, B = P[:, n], G[:, n]
        a, b = A[:, CORE14].mean(1, keepdims=True), B[:, CORE14].mean(1, keepdims=True)
        U, _, Vt = np.linalg.svd(np.swapaxes(A[:, CORE14] - a, 1, 2) @ (B[:, CORE14] - b))
        d = np.sign(np.linalg.det(np.swapaxes(Vt, 1, 2) @ np.swapaxes(U, 1, 2)))
        Dm = np.zeros((len(A), 3, 3)); Dm[:, 0, 0] = 1; Dm[:, 1, 1] = 1; Dm[:, 2, 2] = d
        R = np.swapaxes(Vt, 1, 2) @ Dm @ np.swapaxes(U, 1, 2)
        e[:, n] = np.linalg.norm((A - a) @ np.swapaxes(R, 1, 2) + b - B, axis=-1)
    m = t & JOINT8[None, None]
    for v in range(len(P)):
        if m[v].any():
            out[v] = np.median(e[v][m[v]])
    return out


def signals(P, X, predict):
    """[V,F] truth-free signals, and the learned placement's output Y."""
    Y = predict(P, X[:, :171])
    members = np.stack([f(P, X[:, :171]) for f in predict.members])                 # [3,V,2,19,3]
    dist_spread = pair_dist(members.reshape(-1, *members.shape[2:])).reshape(len(members), -1).std(0)
    point_spread = members.std(0).mean((1, 2, 3))
    closest = np.linalg.norm(Y[:, 0, :, None] - Y[:, 1, None], axis=-1).min((1, 2))
    shift = np.linalg.norm(Y.mean(2) - P.mean(2), axis=-1).mean(1)
    kp = X[:, 64:166].reshape(len(X), 2, 17, 3)
    conf = kp[:, :, LIMB_COCO, 2]
    resid = np.linalg.norm(kp[:, :, LIMB_COCO, :2], axis=-1)
    S = np.c_[dist_spread, point_spread, closest, pair_dist(Y), shift, conf.mean((1, 2)), conf.min((1, 2)),
              resid.mean((1, 2)), resid.max((1, 2)), X[:, 166:171], X[:, 171:178]]
    return S, Y


def fit_logistic(S, y, l2=1e-2, steps=600):
    mu, sd = S.mean(0), S.std(0) + 1e-6
    Z = torch.tensor((S - mu) / sd, dtype=torch.float32)
    t = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(Z.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Z @ w + b, t) + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    w, b = w.detach().numpy(), b.detach().numpy()
    return lambda S_: 1 / (1 + np.exp(-(((S_ - mu) / sd) @ w + b)))


def auroc(score, bad):
    """Probability that a bad view scores higher than a good one (ties count half)."""
    from scipy.stats import rankdata
    pos, neg = score[bad], score[~bad]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    ranks = rankdata(np.r_[pos, neg])                 # average ranks over ties
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def report(risk, err, bad_thr):
    ok = np.isfinite(err)
    risk, err = risk[ok], err[ok]
    keep = risk <= np.median(risk)
    return {'views': int(len(err)), 'bad_share': float((err > bad_thr).mean()),
            'auroc': auroc(risk, err > bad_thr),
            'median_err_all_cm': float(np.median(err) * 100), 'median_err_trusted_half_cm': float(np.median(err[keep]) * 100),
            'median_err_flagged_half_cm': float(np.median(err[~keep]) * 100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    kw = dict(SETTINGS, epochs=2) if a.smoke else SETTINGS
    dev = load(a.feats, DEV, {f'cam{i:02d}' for i in range(1, 21)})
    if a.smoke:
        keep = np.random.default_rng(0).choice(len(dev['P']), 3000, replace=False)
        dev = {k: v[keep] for k, v in dev.items()}
    # out-of-fold placement and signals on the development sequences (4 folds by sequence)
    S_dev, d_err, l_err = None, np.zeros(len(dev['P'])), np.zeros(len(dev['P']))
    for k in range(4):
        te = np.isin(dev['seq'], DEV[k::4]); tr = ~te
        pred = train(dev['P'][tr], dev['G'][tr], dev['X'][tr, :171], seed=k, **kw)
        S, Y = signals(dev['P'][te], dev['X'][te], pred)
        if S_dev is None:
            S_dev = np.zeros((len(dev['P']), S.shape[1]))
        S_dev[te] = S
        d_err[te] = np.abs(pair_dist(Y) - pair_dist(dev['G'][te]))
        l_err[te] = limb_err(dev['P'][te], dev['G'][te])
        print(f'fold {k}: {tr.sum()} train / {te.sum()} out-of-fold views', flush=True)
    flag_d = fit_logistic(S_dev, d_err > DIST_BAD)
    ok = np.isfinite(l_err)
    flag_l = fit_logistic(S_dev[ok], l_err[ok] > LIMB_BAD)
    rep = {'settings': {**kw, 'dist_bad_m': DIST_BAD, 'limb_bad_m': LIMB_BAD}, 'dev_views': int(len(d_err)),
           'dev_bad_share_distance': float((d_err > DIST_BAD).mean())}
    full = train(dev['P'], dev['G'], dev['X'][:, :171], seed=9, **kw)
    for name, (seqs, cams) in TESTS.items():
        T = load(a.feats, seqs, cams)
        if not len(T['P']):
            continue
        S, Y = signals(T['P'], T['X'], full)
        de = np.abs(pair_dist(Y) - pair_dist(T['G']))
        le = limb_err(T['P'], T['G'])
        r = {'distance': report(flag_d(S), de, DIST_BAD), 'limbs': report(flag_l(S), le, LIMB_BAD),
             'single_signals_auroc_distance': {'ensemble_spread': auroc(S[:, 0], de > DIST_BAD),
                                               'closest_approach_inverted': auroc(-S[:, 2], de > DIST_BAD)}}
        rep[name] = r
        print(name, json.dumps(r), flush=True)
    json.dump(rep, open(out/'confidence.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
