"""What a second (third, fourth) camera buys: late fusion of per-camera predicted bodies (CPU, .venv-pose). Experimental.

Input: an engage_bench_s3d.py output directory with several cameras per sequence (Harmony4D: cam01/06/11/16, about
90 degrees apart). Per frame, for every subset S of the cameras and every camera c in S, camera c's predicted bodies
are corrected with the other cameras in S (no ground truth used) and scored in camera c with engage_bench_score.py's
definitions:
  place        each athlete's pelvis = the fusion of the cameras' pelvis estimates, each trusted SIG_LAT across its
               line of sight and SIG_DEPTH along it; camera c's own orientation and limbs kept
  place_orient as place, and each body turned to the mean of the cameras' whole-body orientations (limbs kept)
|S| = 1 is the single-camera baseline. Pair geometry is reported by the angle between the two cameras' lines of sight.

  ../.venv-pose/bin/python engage_bench_multiview.py DIR [--out DIR/multiview]
"""
import argparse
import glob
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import json
import sys
from collections import defaultdict
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_relations import CANON, canon_from_pred, canon_from_smpl, relations, vertex_regions  # noqa: E402
from engage_bench_oracle import kabsch, moved, pairs  # noqa: E402
from engage_bench_score import balance, prf, thread_counts  # noqa: E402
from engage_bench_s3d import seq_root  # noqa: E402
from tangle_h4d import Sequence  # noqa: E402

SIG_DEPTH, SIG_LAT = 0.15, 0.05   # the single-camera relative errors measured by engage_bench_oracle.py
CHUNK = 40
FAMS = ['torso->arm_loop', 'leg->leg_loop', 'arm->arm_loop']
VARIANTS = ['place', 'place_orient']


def proj_so3(M):
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def fuse_point(ps, us):
    """Weighted least-squares point from estimates ps[k] with lines of sight us[k] (unit)."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for p, u in zip(ps, us):
        W = np.outer(u, u)/SIG_DEPTH**2 + (np.eye(3) - np.outer(u, u))/SIG_LAT**2
        A += W
        b += W @ p
    return np.linalg.solve(A, b)


def job(args):
    sname, src, f0 = args
    seq = Sequence(seq_root(sname))
    A4 = seq.world_to_colmap
    sc = np.cbrt(np.linalg.det(A4[:3, :3]))
    cams = sorted(Path(p).stem for p in glob.glob(f'{src}/{sname}/cam??.npz'))
    M = {}   # world -> camera (metric, rigid) as 4x4
    for c in cams:
        R, t, _ = seq.poses[c]
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R @ A4[:3, :3]/sc, (R @ A4[:3, 3] + t)/sc
        M[c] = T
    Z = {c: dict(np.load(f'{src}/{sname}/{c}.npz')) for c in cams}   # load each array once
    idx = {c: {int(f): i for i, f in enumerate(Z[c]['frames'])} for c in cams}
    frames = sorted(set.intersection(*[set(idx[c]) for c in cams]))[f0:f0 + CHUNK]
    recs = []
    for f in frames:
        d = np.load(seq.root/'processed_data/smpl'/f'{f:05d}.npy', allow_pickle=True).item()
        who = sorted(d)
        Jw = [np.asarray(d[w]['joints'][:24], float) for w in who]
        Vw = [np.asarray(d[w]['vertices'], float) for w in who]
        tocam = lambda X, c: ((X @ A4[:3, :3].T + A4[:3, 3]) @ seq.poses[c][0].T + seq.poses[c][1])/sc
        body = {}
        for c in cams:
            i = idx[c][f]
            Cp = [canon_from_pred(Z[c]['kp3d'][i, n].astype(np.float64) + Z[c]['t'][i, n]) for n in range(2)]
            Vp = [Z[c]['verts'][i, n].astype(np.float64) + Z[c]['t'][i, n] for n in range(2)]
            body[c] = [(Cp[n], Vp[n], vertex_regions(Vp[n], Cp[n])) for n in range(2)]
        for c in cams:
            Cg = [canon_from_smpl(tocam(Jw[n], c)) for n in range(2)]
            Vg = [tocam(Vw[n], c) for n in range(2)]
            rg = [vertex_regions(Vg[n], Cg[n]) for n in range(2)]
            gt = relations(Cg[0], Cg[1])['threads']
            gmin, gpairs = pairs(Vg[0], rg[0], Vg[1], rg[1], 0.02)
            dg = float(np.linalg.norm(Cg[0]['pelvis'] - Cg[1]['pelvis']))
            others = [k for k in cams if k != c]
            for r in range(len(cams)):
                for S in combinations(others, r):
                    to_c = {k: M[c] @ np.linalg.inv(M[k]) for k in S}
                    out = {v: [] for v in VARIANTS}
                    for n in range(2):
                        Cc, Vc, rc = body[c][n]
                        ps, us, Rs = [Cc['pelvis']], [Cc['pelvis']/np.linalg.norm(Cc['pelvis'])], [np.eye(3)]
                        Bc = np.array([Cc[q] for q in CANON]) - Cc['pelvis']
                        for k in S:
                            T = to_c[k]
                            Ck = {q: T[:3, :3] @ x + T[:3, 3] for q, x in body[k][n][0].items()}
                            ps.append(Ck['pelvis'])
                            us.append((Ck['pelvis'] - T[:3, 3])/np.linalg.norm(Ck['pelvis'] - T[:3, 3]))
                            Rk, _ = kabsch(np.array([Ck[q] for q in CANON]) - Ck['pelvis'], Bc)
                            Rs.append(Rk)
                        p = fuse_point(ps, us)
                        Q = proj_so3(sum(R.T for R in Rs))   # turns camera c's body to the mean orientation
                        out['place'].append((*moved(Cc, Vc, t=p - Cc['pelvis']), rc))
                        out['place_orient'].append((*moved(Cc, Vc, Q, p - Cc['pelvis'] @ Q.T), rc))
                    rec = {'seq': sname, 'frame': int(f), 'cam': c, 'others': list(S), 'gt': gt, 'gt_min': gmin,
                           'gt_pairs': gpairs, 'pair_dist_gt': dg,
                           'angles': [float(np.degrees(np.arccos(np.clip(us[0] @ u, -1, 1)))) for u in us[1:]]}
                    for v, ((Ca, Va, ra), (Cb, Vb, rb)) in out.items():
                        rec[v] = relations(Ca, Cb)['threads']
                        rec[v + '_min'], rec[v + '_pairs'] = pairs(Va, ra, Vb, rb)
                        rec[v + '_dist'] = float(np.linalg.norm(Ca['pelvis'] - Cb['pelvis']))
                    recs.append(rec)
    return recs


def summarize(recs):
    gt_c = np.array([r['gt_min'] < 0.02 for r in recs])
    gt_pairs = [{tuple(x) for x in r['gt_pairs']} for r in recs]
    row = {}
    for v in VARIANTS:
        C = thread_counts(recs, key=v)
        x = {f: prf(*C[f][:3])[2] for f in FAMS}
        bal = [(balance(r['gt']), balance(r[v])) for r in recs]   # matched: confident crossings on both sides
        nz = [(g, q) for g, q in bal if g != 0]
        x['balance'] = float(np.mean([np.sign(g) == np.sign(q) for g, q in nz])) if nz else float('nan')
        x['dist_err'] = float(np.median([abs(r[v + '_dist'] - r['pair_dist_gt']) for r in recs]))
        pc = np.array([r[v + '_min'] < 0.05 for r in recs])
        x['contact'] = prf(int((pc & gt_c).sum()), int((pc & ~gt_c).sum()), int((~pc & gt_c).sum()))[2]
        tp = fp = fn = 0
        for r, g in zip(recs, gt_pairs):
            q = {tuple(y) for y in r[v + '_pairs']}
            tp += len(g & q); fp += len(q - g); fn += len(g - q)
        x['region_pairs'] = prf(tp, fp, fn)[2]
        row[v] = x
    return row


def report(recs):
    cols = [('dist_err', 'dist err cm', 100), ('region_pairs', 'region pairs', 1), ('contact', 'in contact', 1)] + \
           [(f, f, 1) for f in FAMS] + [('balance', 'balance matched', 1)]
    head = f"{'cameras':26s} {'views':>6s} " + ' '.join(f'{c:>15s}' for _, c, _ in cols)
    groups = defaultdict(list)
    for r in recs:
        k = len(r['others']) + 1
        groups[f'{k} camera' + ('s' if k > 1 else '')].append(r)
        if k == 2:
            a = r['angles'][0]
            groups['2 cameras, ' + ('< 60 deg apart' if a < 60 else '60-120 deg' if a < 120 else '> 120 deg')].append(r)
    S, lines = {}, []
    for v in VARIANTS:
        lines += [f'--- {v}', head]
        for g in sorted(groups):
            row = summarize(groups[g])[v]
            S.setdefault(g, {})[v] = row
            lines.append(f'{g:26s} {len(groups[g]):6d} ' + ' '.join(f'{row[k]*s:15.2f}' for k, _, s in cols))
    return S, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    out = Path(a.out or Path(a.dir)/'multiview')
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for sd in sorted(glob.glob(f'{a.dir}/*/')):
        sname = Path(sd).name
        cams = sorted(glob.glob(f'{sd}/cam??.npz'))
        if len(cams) < 2:
            continue
        n = len(set.intersection(*[set(np.load(p)['frames'].tolist()) for p in cams]))
        jobs += [(sname, a.dir, f0) for f0 in range(0, n, CHUNK)]
    with Pool(int(os.environ.get('WORKERS', 6))) as pool:
        recs = []
        for part in pool.imap_unordered(job, jobs):
            recs += part
            print(f'{len(recs)} records', flush=True)
    with open(out/'records.jsonl', 'w') as fh:
        for r in recs:
            fh.write(json.dumps(r) + '\n')
    S, lines = {}, [f'SIG_DEPTH {SIG_DEPTH} m, SIG_LAT {SIG_LAT} m; truth contact at 2 cm, predicted at 5 cm', '']
    acts = sorted({r['seq'].split('_', 1)[1] for r in recs})
    for name, sub in [(x, [r for r in recs if r['seq'].endswith('_' + x)]) for x in acts] + [('pooled', recs)]:
        S[name], ls = report(sub)
        lines += [f'=== {name}'] + ls + ['']
    (out/'summary.json').write_text(json.dumps(S, indent=1))
    (out/'summary.txt').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
