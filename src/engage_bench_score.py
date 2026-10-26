"""Score predicted engagement relations against Harmony4D ground truth (CPU, .venv-pose). Experimental.

Input: engage_bench_s3d.py outputs (<dir>/<seq>/<cam>.npz). Ground truth (SMPL) and predictions (MHR) are both mapped
to the canonical skeleton of engage_relations.py and measured with the same definitions in the camera frame (metres).

Reported (cluster bootstrap over sequences for 95% intervals; frames within a sequence are not independent):
  placement      pelvis error, root-relative canonical-point error, inter-person pelvis-distance error
  threading      per family (torso / head / arm / leg through the other's arm / leg loop): precision, recall, F1.
                 Truth: present with margin >= 5 cm; present with a smaller margin is ambiguous and left out.
  loop balance   loops A holds on B minus loops B holds on A: sign agreement on frames where the truth is not tied
  contact        bodies within 2 cm (truth) vs predicted minimum distance under 2 / 5 / 10 cm; region-pair F1
  nulls          base rate (a random guess at the true rate has F1 = rate); shuffled pairing (each athlete's predicted
                 body kept, the partner taken from a frame >= 1 s away), which keeps poses but breaks relative placement
  cross-camera   agreement of predicted threading between cameras viewing the same frame (truth agrees by definition)

  ../.venv-pose/bin/python engage_bench_score.py DIR [--out DIR/score]
"""
import argparse
import glob
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')   # many tiny numpy ops: threads only contend (2 s -> 0.08 s per view)
import json
import sys
import zlib
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_relations import CANON, REGIONS, canon_from_pred, canon_from_smpl, relations, vertex_regions  # noqa: E402
from tangle_h4d import Sequence  # noqa: E402

from engage_bench_s3d import seq_root  # noqa: E402
MARGIN = 0.05
TAUS = (0.02, 0.05, 0.10)


def family(key):
    chain, loop = key.split(' through ')
    c = chain.split('.')[1]
    c = 'arm' if c.endswith('_arm') else 'leg' if c.endswith('_leg') else c
    return f"{c}->{loop.split('.')[1]}"


def region_pair_dist(va, Ca, vb, Cb):
    ra, rb = vertex_regions(va, Ca), vertex_regions(vb, Cb)
    d, idx = cKDTree(vb).query(va, k=1)
    M = np.full((len(REGIONS), len(REGIONS)), np.inf)
    np.minimum.at(M, (ra, rb[idx]), d)
    return M


def balance(thr, need_margin=MARGIN):
    """Loops A holds on B minus loops B holds on A (A's loops threaded by B's chains)."""
    n = {'A': 0, 'B': 0}
    for k, (_, m) in thr.items():
        if m >= need_margin:
            n[k.split(' through ')[1].split('.')[0]] += 1
    return n['A'] - n['B']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    out = Path(a.out or Path(a.dir)/'score')
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(f'{a.dir}/*/cam??.npz'))
    with Pool(int(os.environ.get('WORKERS', 6))) as pool:
        recs = [r for part in pool.map(score_file, files) for r in part]
    with open(out/'records.jsonl', 'w') as fh:
        for r in recs:
            fh.write(json.dumps(r) + '\n')
    for act in sorted({r['seq'].split('_', 1)[1] for r in recs}):
        sub = [r for r in recs if r['seq'].endswith('_' + act)]
        if len({r['seq'] for r in sub}) < len({r['seq'] for r in recs}):
            (out/act).mkdir(exist_ok=True)
            print(f'=== {act}')
            summarize(sub, out/act)
    print('=== pooled')
    summarize(recs, out)


def score_file(p):
    if True:
        recs = []
        rng = np.random.default_rng(zlib.crc32(p.encode()))   # fixed per file (str hash is salted per process)
        sname, cam = p.split('/')[-2], Path(p).stem
        seq = Sequence(seq_root(sname))
        A4 = seq.world_to_colmap
        sc = np.cbrt(np.linalg.det(A4[:3, :3]))
        R, t, _ = seq.poses[cam]
        tocam = lambda X: ((X @ A4[:3, :3].T + A4[:3, 3]) @ R.T + t)/sc
        z = dict(np.load(p))   # load each array once (indexing an NpzFile decompresses it every time)
        F = z['frames']
        pred = []
        for i in range(len(F)):
            Cp = [canon_from_pred(z['kp3d'][i, n].astype(np.float64) + z['t'][i, n]) for n in range(2)]
            Vp = [z['verts'][i, n].astype(np.float64) + z['t'][i, n] for n in range(2)]
            pred.append((Cp, Vp))
        for i, f in enumerate(F):
            d = np.load(seq.root/'processed_data/smpl'/f'{f:05d}.npy', allow_pickle=True).item()
            who = sorted(d)
            Cg = [canon_from_smpl(tocam(d[w]['joints'][:24])) for w in who]
            Vg = [tocam(d[w]['vertices']) for w in who]
            Cp, Vp = pred[i]
            rg = relations(Cg[0], Cg[1])
            rp = relations(Cp[0], Cp[1])
            # shuffled pairing: partner from a frame >= 30 label frames (~1 s) away, same camera
            far = [j for j in range(len(F)) if abs(int(F[j]) - int(f)) >= 30]
            rs = relations(Cp[0], pred[rng.choice(far)][0][1])['threads'] if far else None
            Mg, Mp = region_pair_dist(Vg[0], Cg[0], Vg[1], Cg[1]), region_pair_dist(Vp[0], Cp[0], Vp[1], Cp[1])
            pel = [float(np.linalg.norm(Cg[n]['pelvis'] - Cp[n]['pelvis'])) for n in range(2)]
            rr = [float(np.median([np.linalg.norm((Cg[n][k] - Cg[n]['pelvis']) - (Cp[n][k] - Cp[n]['pelvis']))
                                   for k in CANON])) for n in range(2)]
            dg = float(np.linalg.norm(Cg[0]['pelvis'] - Cg[1]['pelvis']))
            dp = float(np.linalg.norm(Cp[0]['pelvis'] - Cp[1]['pelvis']))
            recs.append({'seq': sname, 'cam': cam, 'frame': int(f), 'gt': rg['threads'], 'pred': rp['threads'],
                         'shuf': rs, 'gt_min': float(Mg.min()), 'pred_min': float(Mp.min()),
                         'gt_pairs': np.argwhere(Mg < 0.02).tolist(),
                         'pred_pairs': {str(tau): np.argwhere(Mp < tau).tolist() for tau in TAUS},
                         'pelvis_err': pel, 'rootrel_err': rr, 'pair_dist_gt': dg, 'pair_dist_pred': dp})
        print(f'{sname}/{cam}: {len(F)} views', flush=True)
        return recs


def prf(tp, fp, fn):
    p = tp/max(tp + fp, 1)
    r = tp/max(tp + fn, 1)
    return p, r, 2*p*r/max(p + r, 1e-9)


def thread_counts(recs, key='pred', need=0.0):
    """Per family: tp, fp, fn, n_pos, n_neg over views (truth ambiguous keys excluded)."""
    C = defaultdict(lambda: np.zeros(5, int))
    fams = set()
    for r in recs:
        for k in list(r['gt']) + list(r[key] or {}):
            fams.add(family(k))
    for r in recs:
        if r[key] is None:
            continue
        keys = set(r['gt']) | set(r[key])
        amb = {k for k, (_, m) in r['gt'].items() if m < MARGIN}
        pos = {k for k, (_, m) in r['gt'].items() if m >= MARGIN}
        pr = {k for k, (_, m) in r[key].items() if m >= need}
        seen = set()
        for k in keys - amb:
            f = family(k)
            seen.add(f)
            C[f] += [k in pos and k in pr, k not in pos and k in pr, k in pos and k not in pr, 0, 0]
        for f in fams:
            # per-view prevalence: does this view hold a confident true relation of this family
            C[f][3] += any(family(k) == f for k in pos)
            C[f][4] += 1
    return C


def boot(recs, fn, B=500):
    seqs = sorted({r['seq'] for r in recs})
    by = defaultdict(list)
    for r in recs:
        by[r['seq']].append(r)
    rng = np.random.default_rng(1)
    vals = []
    for _ in range(B):
        pick = rng.choice(seqs, len(seqs))
        vals.append(fn([r for s in pick for r in by[s]]))
    vals = np.array(vals, float)
    return np.nanpercentile(vals, 2.5, axis=0), np.nanpercentile(vals, 97.5, axis=0)


def summarize(recs, out):
    S = {'n_views': len(recs), 'n_frames': len({(r['seq'], r['frame']) for r in recs}),
         'n_sequences': len({r['seq'] for r in recs})}
    pe = np.array([r['pelvis_err'] for r in recs]).ravel()
    re_ = np.array([r['rootrel_err'] for r in recs]).ravel()
    de = np.array([abs(r['pair_dist_pred'] - r['pair_dist_gt']) for r in recs])
    S['placement'] = {'pelvis_err_median_m': float(np.median(pe)), 'rootrel_err_median_m': float(np.median(re_)),
                      'pair_dist_err_median_m': float(np.median(de))}
    lines = [f"{S['n_views']} views, {S['n_frames']} frames, {S['n_sequences']} sequences",
             f"placement: pelvis {np.median(pe)*100:.1f} cm, pose (root-relative) {np.median(re_)*100:.1f} cm, "
             f"distance between athletes {np.median(de)*100:.1f} cm (medians)", '',
             'threading (truth margin >= 5 cm; ambiguous left out)',
             f"{'family':22s} {'views w/ truth':>14s} {'P':>5s} {'R':>5s} {'F1':>5s} {'F1 95% CI':>13s} "
             f"{'F1 pred m>=5cm':>14s} {'F1 shuffled':>11s} {'F1 base rate':>12s}"]
    S['threading'] = {}
    C = thread_counts(recs)
    Cm = thread_counts(recs, need=MARGIN)
    Cs = thread_counts([r for r in recs if r['shuf'] is not None], key='shuf')
    for f in sorted(C, key=lambda f: -C[f][3]):
        tp, fp, fn, npos, nv = C[f]
        p, r, f1 = prf(tp, fp, fn)
        f1m = prf(*Cm[f][:3])[2]
        f1s = prf(*Cs[f][:3])[2]
        rate = npos/max(nv, 1)
        lo, hi = boot(recs, lambda rr, f=f: prf(*thread_counts(rr)[f][:3])[2], B=200)
        S['threading'][f] = dict(views_with_truth=int(npos), P=p, R=r, F1=f1, F1_ci=[float(lo), float(hi)],
                                 F1_pred_margin=f1m, F1_shuffled=f1s, base_rate=rate)
        lines.append(f"{f:22s} {npos:14d} {p:5.2f} {r:5.2f} {f1:5.2f} {f'[{lo:.2f}, {hi:.2f}]':>13s} {f1m:14.2f} "
                     f"{f1s:11.2f} {rate:12.2f}")
    # loop balance sign agreement where the truth is not tied
    bal = [(balance(r['gt']), balance(r['pred'], 0.0)) for r in recs]
    nz = [(g, p) for g, p in bal if g != 0]
    agree = float(np.mean([np.sign(g) == np.sign(p) for g, p in nz])) if nz else float('nan')
    S['loop_balance'] = {'views_truth_not_tied': len(nz), 'sign_agreement': agree,
                         'exact_match_all_views': float(np.mean([g == p for g, p in bal]))}
    lines += ['', f"loop balance: truth not tied in {len(nz)} views; predicted sign agrees {agree:.0%}; "
                  f"exact match over all views {S['loop_balance']['exact_match_all_views']:.0%}"]
    # contact
    gt_c = np.array([r['gt_min'] < 0.02 for r in recs])
    lines.append('')
    lines.append(f'contact (truth: bodies within 2 cm in {gt_c.mean():.0%} of views)')
    S['contact'] = {'truth_rate': float(gt_c.mean())}
    for tau in TAUS:
        pc = np.array([r['pred_min'] < tau for r in recs])
        p, r_, f1 = prf(int((pc & gt_c).sum()), int((pc & ~gt_c).sum()), int((~pc & gt_c).sum()))
        tp = fp = fn = 0
        for rr in recs:
            g = {tuple(x) for x in rr['gt_pairs']}
            q = {tuple(x) for x in rr['pred_pairs'][str(tau)]}
            tp += len(g & q); fp += len(q - g); fn += len(g - q)
        rp_, rr_, rf = prf(tp, fp, fn)
        S['contact'][str(tau)] = dict(P=p, R=r_, F1=f1, region_pair_P=rp_, region_pair_R=rr_, region_pair_F1=rf)
        lines.append(f'  predicted min distance < {tau*100:.0f} cm: in-contact P {p:.2f} R {r_:.2f} F1 {f1:.2f} | '
                     f'region pairs P {rp_:.2f} R {rr_:.2f} F1 {rf:.2f}')
    # cross-camera agreement of predicted confident threading
    by = defaultdict(dict)
    for r in recs:
        by[(r['seq'], r['frame'])][r['cam']] = {k for k, (_, m) in r['pred'].items() if m >= MARGIN}
    jac = []
    for v in by.values():
        cams = sorted(v)
        for i in range(len(cams)):
            for j in range(i + 1, len(cams)):
                u = v[cams[i]] | v[cams[j]]
                if u:
                    jac.append(len(v[cams[i]] & v[cams[j]])/len(u))
    S['cross_camera_jaccard'] = float(np.mean(jac)) if jac else float('nan')
    lines += ['', f'cross-camera agreement of predicted threading (Jaccard, camera pairs where either has one): '
                  f'{S["cross_camera_jaccard"]:.2f} over {len(jac)} pairs (truth: 1.00)']
    (out/'summary.json').write_text(json.dumps(S, indent=1))
    (out/'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
