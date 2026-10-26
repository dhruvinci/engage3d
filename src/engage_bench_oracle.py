"""Which error breaks the relations: oracle ablation on the engagement benchmark (CPU, .venv-pose). Experimental.

Input: an engage_bench_s3d.py (or engage_jointfit.py) output directory. Each predicted body is corrected with ground
truth in one respect at a time, and the relations are re-scored with engage_bench_score.py's definitions:
  pred         as predicted
  fix_depth    each body moved along its camera ray by its pelvis depth error (lateral placement error kept)
  fix_lateral  each body moved across its camera ray by its pelvis lateral error (depth error kept)
  fix_place    both of the above: pelvis on the truth, predicted pose and orientation kept
  fix_rigid    each body rotated + translated onto the truth (best rigid fit of its 19 canonical points): the ceiling
               of any whole-body fit (engage_jointfit.py), predicted limbs kept
  fix_limbs    the true body (orientation and limbs) placed at the predicted pelvis: placement error kept
  fix_rot      fix_rigid's rotation only, about the predicted pelvis (placement kept)
  fix_center   moved so the centroid of its 19 canonical points lands on the truth's (no rotation)
  truth        the truth itself (threading F1 1; contact and region pairs show the cost of the 5 cm threshold alone)

  ../.venv-pose/bin/python engage_bench_oracle.py DIR [--out DIR/oracle]
"""
import argparse
import glob
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_relations import CANON, REGIONS, canon_from_pred, canon_from_smpl, relations, vertex_regions  # noqa: E402
from engage_bench_score import balance, prf, thread_counts  # noqa: E402
from engage_bench_s3d import seq_root  # noqa: E402
from tangle_h4d import Sequence  # noqa: E402

VARIANTS = ['pred', 'fix_depth', 'fix_lateral', 'fix_place', 'fix_rigid', 'fix_limbs', 'fix_rot', 'fix_center', 'truth']
TAU = 0.05
FAMS = ['torso->arm_loop', 'leg->leg_loop', 'arm->arm_loop']


def kabsch(P, Q):
    """R, t minimising |P @ R.T + t - Q| (rows = points)."""
    mp, mq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, mq - mp @ R.T


def moved(C, V, R=np.eye(3), t=np.zeros(3)):
    return {k: v @ R.T + t for k, v in C.items()}, V @ R.T + t


def pairs(Va, ra, Vb, rb, tau=TAU):
    d, idx = cKDTree(Vb).query(Va, k=1)
    M = np.full((len(REGIONS), len(REGIONS)), np.inf)
    np.minimum.at(M, (ra, rb[idx]), d)
    return float(M.min()), np.argwhere(M < tau).tolist()


def score_file(p):
    sname, cam = p.split('/')[-2], Path(p).stem
    seq = Sequence(seq_root(sname))
    A4 = seq.world_to_colmap
    sc = np.cbrt(np.linalg.det(A4[:3, :3]))
    R, t, _ = seq.poses[cam]
    tocam = lambda X: ((X @ A4[:3, :3].T + A4[:3, 3]) @ R.T + t)/sc
    z = dict(np.load(p))   # load each array once (indexing an NpzFile decompresses it every time)
    recs = []
    for i, f in enumerate(z['frames']):
        d = np.load(seq.root/'processed_data/smpl'/f'{f:05d}.npy', allow_pickle=True).item()
        who = sorted(d)
        Cg = [canon_from_smpl(tocam(d[w]['joints'][:24])) for w in who]
        Vg = [tocam(d[w]['vertices']) for w in who]
        Cp = [canon_from_pred(z['kp3d'][i, n].astype(np.float64) + z['t'][i, n]) for n in range(2)]
        Vp = [z['verts'][i, n].astype(np.float64) + z['t'][i, n] for n in range(2)]
        # regions follow the body's own skeleton, so they survive rigid moves: compute once per body
        rp = [vertex_regions(Vp[n], Cp[n]) for n in range(2)]
        rg = [vertex_regions(Vg[n], Cg[n]) for n in range(2)]
        bodies = {v: [] for v in VARIANTS}
        for n in range(2):
            e = Cg[n]['pelvis'] - Cp[n]['pelvis']
            u = Cp[n]['pelvis']/np.linalg.norm(Cp[n]['pelvis'])
            ed = (e @ u)*u
            Rr, tr = kabsch(np.array([Cp[n][k] for k in CANON]), np.array([Cg[n][k] for k in CANON]))
            bodies['pred'].append((Cp[n], Vp[n], rp[n]))
            bodies['fix_depth'].append((*moved(Cp[n], Vp[n], t=ed), rp[n]))
            bodies['fix_lateral'].append((*moved(Cp[n], Vp[n], t=e - ed), rp[n]))
            bodies['fix_place'].append((*moved(Cp[n], Vp[n], t=e), rp[n]))
            bodies['fix_rigid'].append((*moved(Cp[n], Vp[n], Rr, tr), rp[n]))
            bodies['fix_limbs'].append((*moved(Cg[n], Vg[n], t=-e), rg[n]))
            pel = Cp[n]['pelvis']
            bodies['fix_rot'].append((*moved(Cp[n], Vp[n], Rr, pel - pel @ Rr.T), rp[n]))
            cen = np.mean([Cg[n][k] - Cp[n][k] for k in CANON], 0)
            bodies['fix_center'].append((*moved(Cp[n], Vp[n], t=cen), rp[n]))
            bodies['truth'].append((Cg[n], Vg[n], rg[n]))
        rec = {'seq': sname, 'cam': cam, 'frame': int(f), 'gt': relations(Cg[0], Cg[1])['threads'],
               'pair_dist_gt': float(np.linalg.norm(Cg[0]['pelvis'] - Cg[1]['pelvis']))}
        rec['gt_min'], rec['gt_pairs'] = pairs(Vg[0], rg[0], Vg[1], rg[1], 0.02)
        for v, ((Ca, Va, ra), (Cb, Vb, rb)) in bodies.items():
            rec[v] = relations(Ca, Cb)['threads']
            rec[v + '_min'], rec[v + '_pairs'] = pairs(Va, ra, Vb, rb)
            rec[v + '_dist'] = float(np.linalg.norm(Ca['pelvis'] - Cb['pelvis']))
            # relative placement error of B seen from A, split along / across the ray to A
            e = (Cb['pelvis'] - Ca['pelvis']) - (Cg[1]['pelvis'] - Cg[0]['pelvis'])
            u = Ca['pelvis']/np.linalg.norm(Ca['pelvis'])
            rec[v + '_rel'] = [float(abs(e @ u)), float(np.linalg.norm(e - (e @ u)*u))]
        recs.append(rec)
    print(f'{sname}/{cam}: {len(recs)} views', flush=True)
    return recs


def summarize(recs):
    gt_c = np.array([r['gt_min'] < 0.02 for r in recs])
    gt_pairs = [{tuple(x) for x in r['gt_pairs']} for r in recs]
    rows = {}
    for v in VARIANTS:
        C = thread_counts(recs, key=v)
        row = {f: prf(*C[f][:3])[2] for f in FAMS}
        bal = [(balance(r['gt']), balance(r[v], 0.0)) for r in recs]
        nz = [(g, q) for g, q in bal if g != 0]
        row['balance'] = float(np.mean([np.sign(g) == np.sign(q) for g, q in nz]))
        row['dist_err'] = float(np.median([abs(r[v + '_dist'] - r['pair_dist_gt']) for r in recs]))
        rel = np.array([r[v + '_rel'] for r in recs])
        row['rel_depth'], row['rel_lat'] = np.median(rel, 0)
        pc = np.array([r[v + '_min'] < TAU for r in recs])
        row['contact'] = prf(int((pc & gt_c).sum()), int((pc & ~gt_c).sum()), int((~pc & gt_c).sum()))[2]
        tp = fp = fn = 0
        for r, g in zip(recs, gt_pairs):
            q = {tuple(x) for x in r[v + '_pairs']}
            tp += len(g & q); fp += len(q - g); fn += len(g - q)
        row['region_pairs'] = prf(tp, fp, fn)[2]
        rows[v] = row
    cols = [('dist_err', 'dist err cm', 100), ('rel_depth', 'rel depth cm', 100), ('rel_lat', 'rel lateral cm', 100),
            ('region_pairs', 'region pairs', 1), ('contact', 'in contact', 1)] + [(f, f, 1) for f in FAMS] + \
           [('balance', 'balance sign', 1)]
    lines = [f"{len(recs)} views, {len({r['seq'] for r in recs})} sequences; truth pairs and contact at 2 cm, "
             f"predicted at {TAU*100:.0f} cm; F1 except distances (medians)",
             f"{'variant':12s} " + ' '.join(f'{c:>15s}' for _, c, _ in cols)]
    for v, row in rows.items():
        lines.append(f'{v:12s} ' + ' '.join(f'{row[k]*s:15.2f}' for k, _, s in cols))
    return rows, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    out = Path(a.out or Path(a.dir)/'oracle')
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(f'{a.dir}/*/cam??.npz'))
    with Pool(int(os.environ.get('WORKERS', 6))) as pool:
        recs = [r for part in pool.map(score_file, files) for r in part]
    with open(out/'records.jsonl', 'w') as fh:
        for r in recs:
            fh.write(json.dumps(r) + '\n')
    S, lines = {}, []
    groups = {act: [r for r in recs if r['seq'].endswith('_' + act)] for act in sorted({r['seq'].split('_', 1)[1] for r in recs})}
    for name, sub in list(groups.items()) + [('pooled', recs)]:
        S[name], ls = summarize(sub)
        lines += [f'=== {name}'] + ls + ['']
    (out/'summary.json').write_text(json.dumps(S, indent=1))
    (out/'summary.txt').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
