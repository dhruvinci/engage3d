"""E20 demo figures: a typical hugging moment and a typical ballroom moment (runs in .venv-depth, CPU).
For each activity, among the views where the two people touch (true bodies within 5 cm), the one whose learned
placement has the median distance error, so the picture shows the usual case, not the best one. Writes
<out>/easy_<activity>_overlay.png (photo, truth, SAM 3D Body + combined depth + learned placement) and
<out>/easy_<activity>_topdown.png (seen from above).
  ../.venv-depth/bin/python engage_demo_easy.py --out DIR
"""
import argparse
import glob
from pathlib import Path

import numpy as np

import engage_demo as D
from engage_relations import canon_from_pred, canon_from_smpl

E = D.B/'easy'


def bodies(sname, cam, f):
    """As engage_demo.bodies, from E20's outputs: truth, SAM 3D Body, combined depth, learned placement."""
    if sname not in D._SEQ:
        D._SEQ[sname] = D.Sequence(D.seq_root(sname))
    seq = D._SEQ[sname]
    K, Dk, tocam = D.cam_of(seq, cam)
    gt = np.load(seq.root/'processed_data/smpl'/f'{f:05d}.npy', allow_pickle=True).item()
    out = {'truth': [(tocam(gt[k]['vertices']), canon_from_smpl(tocam(gt[k]['joints'][:24]))) for k in sorted(gt)]}
    for name, d in (('SAM 3D Body', E/'s3d'), ('combined depth', E/'place/blend'), ('learned', E/'place/model')):
        p = d/sname/f'{cam}.npz'
        if not p.exists():
            continue
        z = np.load(p)
        i = np.nonzero(z['frames'] == f)[0]
        if len(i):
            i = int(i[0])
            out[name] = [(z['verts'][i, n].astype(float) + z['t'][i, n],
                          canon_from_pred(z['kp3d'][i, n].astype(float) + z['t'][i, n])) for n in range(2)]
    return seq, K, Dk, out


def pick(activity):
    cands = []
    for p in sorted(glob.glob(str(E/f'place/model/*_{activity}/cam??.npz'))):
        sname, cam = p.split('/')[-2], Path(p).stem
        for f in np.load(p)['frames'][::2]:
            _, _, _, bb = bodies(sname, cam, int(f))
            if 'learned' not in bb:
                continue
            gap = np.min(np.linalg.norm(bb['truth'][0][0][::50, None] - bb['truth'][1][0][None, ::50], axis=-1))
            if gap < 0.05:
                cands.append((D.dist_err(bb['learned'], bb['truth']), sname, cam, int(f)))
    cands.sort()
    return cands[len(cands) // 2], len(cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    plt = D.plt_setup()
    for activity, label in (('hugging', 'Hugging'), ('ballroom2', 'Ballroom dancing')):
        (err, sname, cam, f), n = pick(activity)
        seq, K, Dk, bb = bodies(sname, cam, f)
        D.overlay(out/f'easy_{activity}_overlay.png', seq, K, Dk, cam, f, bb, ['truth', 'learned'])
        D.topdown(plt, out/f'easy_{activity}_topdown.png', bb,
                  f'{label}: a typical moment in contact (median of {n}), seen from above')
        print(activity, sname, cam, f, f'learned distance error {err * 100:.1f} cm', flush=True)


if __name__ == '__main__':
    main()
