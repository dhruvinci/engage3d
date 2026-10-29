"""E17b: making the trust flag usable (CPU, .venv-depth). Pre-registration: engage3d experiments/E17b-trust-flag-deeper/PREREG.md.

Re-creates E17 (same features, settings, seeds; `engage_confidence.py`) and saves every view: sequence, camera, frame,
the truth-free signals, the predicted and true distance between the two people, the limb error and both flags' risk,
for the development out-of-fold predictions and every test set. Then, with every rule fitted on the development
predictions only:
  B1  error bars by normalised split conformal (90% and 80%), against a fixed-width interval;
  B2  flagged views (risk above the development median) repaired from unflagged neighbours within 3 samples;
  B3  clip summaries (mean distance per sequence-camera series): all views, unflagged only, (1 - risk)-weighted;
  B4  the lowest-risk camera of 3 drawn at random (20 draws per frame) against a random pick and the best of 3;
  B5  (exploratory) a small MLP flag; B6 (exploratory) sword fighting, relative error above 4%; B7 (exploratory) B1 and
      B4 for the limb flag.

  TORCH_THREADS=6 ../.venv-depth/bin/python engage_trust_b.py --feats ../benchmarks/engage_h4d/place_feats2 \
      --out ../benchmarks/engage_h4d/trust_b [--smoke]
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
from engage_confidence import (DEV, DIST_BAD, LIMB_BAD, SETTINGS, TESTS, auroc, fit_logistic, limb_err,  # noqa: E402
                               pair_dist, report, signals)
from engage_place_train import train                                                                     # noqa: E402

CONFIRM = ('C_heldout_grappling_cams_03_08_13_18', 'C_mma_other_16_cams')
E17_JSON = Path(__file__).resolve().parents[1]/'benchmarks/engage_h4d/confidence/confidence.json'
B = 2000
SEED = 2610


def load(feats, seqs, cams):
    """As engage_confidence.load, keeping the camera and frame of every view."""
    D = {k: [] for k in ('P', 'G', 'X', 'seq', 'cam', 'frame')}
    for p in sorted(glob.glob(f'{feats}/*__cam??.npz')):
        s, c = Path(p).stem.split('__')
        if s not in seqs or c not in cams:
            continue
        d = np.load(p)
        if d['X'].shape[1] < 178:
            continue
        D['P'].append(d['P'].astype(np.float64)); D['G'].append(d['G'].astype(np.float64)); D['X'].append(d['X'])
        D['seq'] += [s] * len(d['P']); D['cam'] += [c] * len(d['P']); D['frame'].append(d['frame'])
    return {k: (np.concatenate(v) if k not in ('seq', 'cam') else np.array(v)) for k, v in D.items()}


def ridge_fit(S, y, lam=1.0):
    mu, sd = S.mean(0), S.std(0) + 1e-6
    Z = np.c_[(S - mu) / sd, np.ones(len(S))]
    w = np.linalg.solve(Z.T @ Z + lam * np.diag(np.r_[np.ones(S.shape[1]), 0]), Z.T @ y)
    return lambda S_: np.c_[(S_ - mu) / sd, np.ones(len(S_))] @ w


def mlp_flag(S, y, seed=0, steps=1500):
    torch.manual_seed(seed)
    mu, sd = S.mean(0), S.std(0) + 1e-6
    Z = torch.tensor((S - mu) / sd, dtype=torch.float32); t = torch.tensor(y, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(S.shape[1], 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Z)[:, 0], t); loss.backward(); opt.step()
    net.eval()
    return lambda S_: torch.sigmoid(net(torch.tensor((S_ - mu) / sd, dtype=torch.float32))[:, 0]).detach().numpy()


def boot_ci(stat_fn, groups, rng):
    """95% interval over B resamples of the groups (with repeats): stat_fn gets the row indices of the drawn groups."""
    u, inv = np.unique(groups, return_inverse=True)
    rows = [np.where(inv == g)[0] for g in range(len(u))]
    vals = [stat_fn(np.concatenate([rows[g] for g in rng.integers(0, len(u), len(u))])) for _ in range(B)]
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def series_keys(V):
    return np.array([f'{s}|{c}' for s, c in zip(V['seq'], V['cam'])])


def repair(V, flagged, k=3):
    """B2: a flagged view's predicted distance -> (1 - risk)-weighted mean of unflagged views within k samples."""
    out = V['pred'].copy()
    keys = series_keys(V)
    for key in np.unique(keys):
        idx = np.where(keys == key)[0]
        idx = idx[np.argsort(V['frame'][idx])]
        for j, i in enumerate(idx):
            if not flagged[i]:
                continue
            nb = [idx[q] for q in range(max(0, j - k), min(len(idx), j + k + 1)) if q != j and not flagged[idx[q]]]
            if nb:
                w = 1 - V['risk_d'][nb]
                out[i] = float((w * V['pred'][nb]).sum() / w.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    kw = dict(SETTINGS, epochs=2) if a.smoke else SETTINGS
    # ---- E17, re-created (engage_confidence.main), keeping every view
    dev = load(a.feats, DEV, {f'cam{i:02d}' for i in range(1, 21)})
    if a.smoke:
        keep = np.sort(np.random.default_rng(0).choice(len(dev['P']), 3000, replace=False))
        dev = {k: v[keep] for k, v in dev.items()}
    S_dev, pred_dev, d_err, l_err = None, np.zeros(len(dev['P'])), np.zeros(len(dev['P'])), np.zeros(len(dev['P']))
    for k in range(4):
        te = np.isin(dev['seq'], DEV[k::4]); tr = ~te
        pred = train(dev['P'][tr], dev['G'][tr], dev['X'][tr, :171], seed=k, **kw)
        S, Y = signals(dev['P'][te], dev['X'][te], pred)
        if S_dev is None:
            S_dev = np.zeros((len(dev['P']), S.shape[1]))
        S_dev[te] = S
        pred_dev[te] = pair_dist(Y)
        d_err[te] = np.abs(pair_dist(Y) - pair_dist(dev['G'][te]))
        l_err[te] = limb_err(dev['P'][te], dev['G'][te])
        print(f'fold {k}: {tr.sum()} train / {te.sum()} out-of-fold views', flush=True)
    flag_d = fit_logistic(S_dev, d_err > DIST_BAD)
    okl = np.isfinite(l_err)
    flag_l = fit_logistic(S_dev[okl], l_err[okl] > LIMB_BAD)
    full = train(dev['P'], dev['G'], dev['X'][:, :171], seed=9, **kw)
    views = {'dev': {'seq': dev['seq'], 'cam': dev['cam'], 'frame': dev['frame'], 'S': S_dev, 'pred': pred_dev,
                     'true': pair_dist(dev['G']), 'd_err': d_err, 'l_err': l_err, 'risk_d': flag_d(S_dev), 'risk_l': flag_l(S_dev)}}
    rep = {'settings': {**kw, 'dist_bad_m': DIST_BAD, 'limb_bad_m': LIMB_BAD}, 'dev_views': int(len(d_err)), 'sets': {}}
    for name, (seqs, cams) in TESTS.items():
        T = load(a.feats, seqs, cams)
        if not len(T['P']):
            continue
        S, Y = signals(T['P'], T['X'], full)
        views[name] = {'seq': T['seq'], 'cam': T['cam'], 'frame': T['frame'], 'S': S, 'pred': pair_dist(Y), 'true': pair_dist(T['G']),
                       'd_err': np.abs(pair_dist(Y) - pair_dist(T['G'])), 'l_err': limb_err(T['P'], T['G']),
                       'risk_d': flag_d(S), 'risk_l': flag_l(S)}
        V = views[name]
        rep['sets'][name] = {'e17_rerun': {'distance': report(V['risk_d'], V['d_err'], DIST_BAD), 'limbs': report(V['risk_l'], V['l_err'], LIMB_BAD)}}
        print(name, 'rerun distance AUROC', round(rep['sets'][name]['e17_rerun']['distance']['auroc'], 4), flush=True)
    np.savez_compressed(out/'views.npz', **{f'{n}__{k}': v for n, V in views.items() for k, v in V.items()})
    # ---- rules fitted on the development predictions only
    Vd = views['dev']
    tau = float(np.median(Vd['risk_d']))
    scale_d = ridge_fit(Vd['S'], np.log(Vd['d_err'] + 0.005))
    sc = np.sort(Vd['d_err'] / np.exp(scale_d(Vd['S'])))
    n = len(sc)
    q = {a_: float(sc[min(n - 1, int(np.ceil((n + 1) * (1 - a_))) - 1)]) for a_ in (0.1, 0.2)}
    fixed = {a_: float(np.quantile(Vd['d_err'], 1 - a_)) for a_ in (0.1, 0.2)}
    scale_l = ridge_fit(Vd['S'][okl], np.log(Vd['l_err'][okl] + 0.005))
    scl = np.sort(Vd['l_err'][okl] / np.exp(scale_l(Vd['S'][okl])))
    ql = float(scl[min(len(scl) - 1, int(np.ceil((len(scl) + 1) * 0.9)) - 1)])
    mlp_d = mlp_flag(Vd['S'], (Vd['d_err'] > DIST_BAD).astype(float))
    rep['rules'] = {'tau_risk': tau, 'conformal_q': q, 'fixed_halfwidth_cm': {k: v * 100 for k, v in fixed.items()}, 'limb_conformal_q90': ql}
    rng = np.random.default_rng(SEED)
    for name in rep['sets']:
        V = views[name]; R = rep['sets'][name]
        keys = series_keys(V)
        sig = np.exp(scale_d(V['S']))
        # B1
        R['B1'] = {}
        for a_ in (0.1, 0.2):
            cov = V['d_err'] <= q[a_] * sig
            R['B1'][f'{int(100 * (1 - a_))}'] = {'coverage': float(cov.mean()), 'median_halfwidth_cm': float(np.median(q[a_] * sig) * 100),
                                                 'fixed_coverage': float((V['d_err'] <= fixed[a_]).mean()), 'fixed_halfwidth_cm': fixed[a_] * 100,
                                                 'coverage_ci': boot_ci(lambda ix: np.mean(cov[ix]), keys, rng)}
        # B2
        flagged = V['risk_d'] > tau
        rp = repair(V, flagged)
        e_new = np.abs(rp - V['true'])
        def fall(ix):
            f = ix[flagged[ix]]
            return (np.median(V['d_err'][f]) - np.median(e_new[f])) / np.median(V['d_err'][f]) if len(f) else np.nan
        R['B2'] = {'flagged_share': float(flagged.mean()),
                   'flagged_median_cm_before': float(np.median(V['d_err'][flagged]) * 100) if flagged.any() else None,
                   'flagged_median_cm_after': float(np.median(e_new[flagged]) * 100) if flagged.any() else None,
                   'flagged_relative_fall': float(fall(np.arange(len(keys)))) if flagged.any() else None,
                   'flagged_relative_fall_ci': boot_ci(fall, keys, rng) if flagged.any() else None,
                   'all_median_cm_before': float(np.median(V['d_err']) * 100), 'all_median_cm_after': float(np.median(e_new) * 100),
                   'repaired_views': int((flagged & (rp != V['pred'])).sum())}
        # B3
        ser = {}
        for key in np.unique(keys):
            ix = np.where(keys == key)[0]
            t_ = V['true'][ix].mean(); w = 1 - V['risk_d'][ix]; u = ~flagged[ix]
            ser[key] = {'all': abs(V['pred'][ix].mean() - t_), 'unflagged': abs(V['pred'][ix][u].mean() - t_) if u.any() else np.nan,
                        'weighted': abs((w * V['pred'][ix]).sum() / w.sum() - t_)}
        R['B3_series'] = {k: {m: float(v * 100) for m, v in d.items()} for k, d in ser.items()}
        # B4
        fr = {}
        for i, (s, f) in enumerate(zip(V['seq'], V['frame'])):
            fr.setdefault((s, int(f)), []).append(i)
        picked, randm, best, fid = [], [], [], []
        rng4 = np.random.default_rng(SEED + 4)
        for key_, ix in fr.items():
            if len(ix) < 3:
                continue
            ix = np.array(ix)
            for _ in range(20):
                tri = rng4.choice(ix, 3, replace=False)
                picked.append(V['d_err'][tri[np.argmin(V['risk_d'][tri])]]); randm.append(V['d_err'][tri].mean()); best.append(V['d_err'][tri].min())
                fid.append(f'{key_[0]}|{key_[1]}')
        if picked:
            picked, randm, best, fid = map(np.array, (picked, randm, best, fid))
            red = lambda ix: 1 - np.median(picked[ix]) / np.median(randm[ix])
            R['B4'] = {'frames': int(len(np.unique(fid))), 'picked_median_cm': float(np.median(picked) * 100),
                       'random_median_cm': float(np.median(randm) * 100), 'best_of_3_median_cm': float(np.median(best) * 100),
                       'reduction_vs_random': float(red(np.arange(len(fid)))), 'reduction_ci': boot_ci(red, fid, rng)}
        # B5-B7 (exploratory)
        R['B5_mlp_auroc_distance'] = auroc(mlp_d(V['S']), V['d_err'] > DIST_BAD)
        R['B5_logistic_auroc_distance'] = auroc(V['risk_d'], V['d_err'] > DIST_BAD)
        rel = V['d_err'] / np.maximum(V['true'], 1e-6)
        R['B6_auroc_relative_error_over_4pct'] = auroc(V['risk_d'], rel > 0.04)
        okv = np.isfinite(V['l_err'])
        if okv.any():
            R['B7_limb_coverage90'] = float((V['l_err'][okv] <= ql * np.exp(scale_l(V['S'][okv]))).mean())
        print(name, json.dumps({k: v for k, v in R.items() if k != 'B3_series'})[:1500], flush=True)
    # B3 criterion (pooled confirmatory series)
    pool = {k: v for c in CONFIRM if c in rep['sets'] for k, v in rep['sets'][c]['B3_series'].items()}
    if pool:
        ks = np.array(sorted(pool))
        dv = np.array([pool[k]['all'] - pool[k]['weighted'] for k in ks])
        rep['B3_confirmatory'] = {'series': int(len(ks)), 'mean_err_all_cm': float(np.mean([pool[k]['all'] for k in ks])),
                                  'mean_err_weighted_cm': float(np.mean([pool[k]['weighted'] for k in ks])),
                                  'mean_err_unflagged_cm': float(np.nanmean([pool[k]['unflagged'] for k in ks])),
                                  'all_minus_weighted_cm': float(dv.mean()), 'ci': boot_ci(lambda ix: dv[ix].mean(), ks, rng)}
    # criteria
    H, M = rep['sets'].get(CONFIRM[0], {}), rep['sets'].get(CONFIRM[1], {})
    e17 = json.loads(E17_JSON.read_text())
    rep['criteria'] = {
        'rerun_check': {c: {'e17': e17[c]['distance']['auroc'], 'rerun': rep['sets'][c]['e17_rerun']['distance']['auroc'],
                            'within_0.01': abs(e17[c]['distance']['auroc'] - rep['sets'][c]['e17_rerun']['distance']['auroc']) <= 0.01}
                        for c in CONFIRM if c in rep['sets'] and c in e17},
        'C1': H and 0.85 <= H['B1']['90']['coverage'] <= 0.95,
        'C2': M and M['B1']['90']['coverage'] >= 0.85,
        'C3': H and H['B2']['flagged_relative_fall'] is not None and H['B2']['flagged_relative_fall'] >= 0.20
        and H['B2']['flagged_relative_fall_ci'][0] > 0 and H['B2']['all_median_cm_after'] <= H['B2']['all_median_cm_before'],
        'C4': 'B3_confirmatory' in rep and rep['B3_confirmatory']['ci'][0] > 0,
        'C5': H and 'B4' in H and H['B4']['reduction_vs_random'] >= 0.20 and H['B4']['reduction_ci'][0] > 0}
    rep['criteria'] = {k: (bool(v) if not isinstance(v, dict) else v) for k, v in rep['criteria'].items()}
    print('criteria', rep['criteria'], flush=True)
    json.dump(rep, open(out/'trust_b.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
