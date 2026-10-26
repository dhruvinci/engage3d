"""E12: ViCoS ground work without 3D truth, using the synchronised phones (CPU; .venv-depth). Experimental.
Pre-registration: `docs/engagement/results/vicos_sync_prereg.md`.

ViCoS's 16 streams are 6 sessions filmed by 2-3 phones at once (`vicos_results.md`); at the same frame number the
phones of a session show the same moment. If the measured relations are about the bodies, two phones of one moment
should agree more than two phones of different moments with the same position label.

  select   synchronised moments per session (both athletes posed in every phone, the same label), spread over the
           positions and >= 30 frames apart -> <out>/ids.txt (image ids) and <out>/moments.json
  place    the learned placement correction (E06's v1 model) on ViCoS: features as engage_place_features.py (depth
           relief per body region from the saved maps, 2D residuals against ViCoS's own 2D poses, gap, sizes; the
           camera is SAM 3D Body's default, no fisheye) -> <out>/place/vicos_s3d.npz
  agree    threading relations (margin >= 5 cm, as E05) per image for each variant; agreement between phones of the
           same moment vs the same phones at another moment with the same label -> <out>/agree.json

  ../.venv-depth/bin/python engage_vicos_sync.py select --out DIR [--per-session 100]
  ../.venv-depth/bin/python engage_vicos_sync.py place --out DIR --bodies CLICKS/vicos_s3d.npz --blend BLEND/vicos_s3d.npz
      --maps BLEND/maps --model benchmarks/engage_h4d/models/placement_v1.pt
  ../.venv-depth/bin/python engage_vicos_sync.py agree --out DIR --variant NAME=NPZ [--variant ...]
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_depth_probe import BAND, FACES, MIN_PIX, render              # noqa: E402
from engage_paths import VICOS                                          # noqa: E402
from engage_relations import CANON, canon_from_pred, relations, vertex_regions  # noqa: E402

# stream 11 runs at an offset from 12/13 (labels agree only 84-85%), so that session uses 12 and 13
SESSIONS = [('00', '01', '02'), ('03', '04', '05'), ('06', '07', '08'), ('09', '10'), ('12', '13'), ('14', '15')]
COCO_TO_MHR = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]   # as engage_place_features.py
WEIGHT, MARGIN, SEED, NR = 0.7, 0.05, 0, 16
COARSE = {'torso': 'torso', 'head': 'head', 'l_leg': 'leg', 'r_leg': 'leg', 'l_arm': 'arm', 'r_arm': 'arm'}


def cmd_select(a):
    ann = json.load(open(VICOS/'annotations.json'))
    by = {(x['image'][:2], x['frame']): x for x in ann}
    rng = random.Random(SEED)
    moments = []
    for s in SESSIONS:
        frames = sorted(set.intersection(*[{f for (c, f) in by if c == cam} for cam in s]))
        ok = [f for f in frames if all(by[(c, f)].get('pose1') and by[(c, f)].get('pose2') for c in s)
              and len({by[(c, f)]['position'] for c in s}) == 1]
        bypos = defaultdict(list)
        for f in ok:
            bypos[by[(s[0], f)]['position']].append(f)
        for v in bypos.values():
            rng.shuffle(v)
        chosen = []
        order = sorted(bypos)
        while len(chosen) < a.per_session and any(bypos[p] for p in order):
            for p in order:
                while bypos[p]:
                    f = bypos[p].pop()
                    if all(abs(f - g) >= 30 for g in chosen):
                        chosen.append(f)
                        moments.append({'session': '+'.join(s), 'frame': f, 'position': by[(s[0], f)]['position'],
                                        'images': [by[(c, f)]['image'] for c in s]})
                        break
                if len(chosen) >= a.per_session:
                    break
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(moments, open(out/'moments.json', 'w'), indent=0)
    ids = [i for m in moments for i in m['images']]
    open(out/'ids.txt', 'w').write('\n'.join(ids) + '\n')
    print(f'{len(moments)} moments, {len(ids)} images', flush=True)


def cmd_place(a):
    import torch  # noqa: F401  (engage_place_train needs it)
    from engage_place_train import load_model
    ann = {x['image']: x for x in json.load(open(VICOS/'annotations.json'))}
    z0 = dict(np.load(a.bodies))           # SAM 3D Body (clicks), before the combined depth
    zb = dict(np.load(a.blend))            # the same bodies after the combined depth
    assert list(z0['image']) == list(zb['image'])
    faces = np.load(FACES)['mhr']
    predict = load_model(a.model)
    P_all, X_all, keep, mids = [], [], [], []
    for i, img_id in enumerate(map(str, z0['image'])):
        mp = Path(a.maps)/f'{img_id}.npz'
        if not mp.exists():
            continue
        m = np.load(mp)
        dm, Pv = m['depth'].astype(np.float32), m['P']
        h, w = dm.shape
        Vc = [z0['verts'][i, n].astype(np.float64) + z0['t'][i, n] for n in range(2)]
        Jc = [z0['kp3d'][i, n].astype(np.float64) + z0['t'][i, n] for n in range(2)]
        own, dep, sil, fid = render([(V, faces) for V in Vc], Pv, w, h, faces=True)
        C = [canon_from_pred(J) for J in Jc]
        band = max(2, round(BAND*max(w, h)/768))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*band + 1, 2*band + 1))
        delta, relief, npix = [], [], []
        for n in range(2):
            pix = (own == n + 1) & np.isfinite(dm) & ~cv2.dilate(sil[1 - n].astype(np.uint8), kernel).astype(bool)
            d = float(np.median(dm[pix] - dep[pix])) if pix.sum() >= MIN_PIX else np.nan
            freg = vertex_regions(Vc[n], C[n])[faces[:, 0]]
            reg = np.full(pix.shape, -1)
            reg[pix] = freg[fid[pix]]
            rr = np.zeros((NR, 2))
            for r in range(NR):
                sel = reg == r
                if sel.sum() >= 10 and np.isfinite(d):
                    rr[r] = [np.median(dm[sel] - dep[sel]) - d, sel.sum()/max(pix.sum(), 1)]
            delta.append(d); relief.append(rr); npix.append(int(pix.sum()))
        ok = all(np.isfinite(delta))
        Jb = [zb['kp3d'][i, n].astype(np.float64) + zb['t'][i, n] for n in range(2)]
        Cb = np.array([[canon_from_pred(J)[k] for k in CANON] for J in Jb])
        mid = Cb.reshape(-1, 3).mean(0)
        x_ann = ann[img_id]
        kp2d = []
        for n, key in enumerate(('pose1', 'pose2')):
            J = Jb[n][COCO_TO_MHR]
            uv = (J @ Pv.T)[:, :2] / (J @ Pv.T)[:, 2:]
            obs = np.asarray(x_ann[key], np.float64)
            pts, conf = obs[:, :2], obs[:, 2]
            size = max(np.ptp(pts[conf > 0.3], 0).max() if (conf > 0.3).sum() >= 2 else 1.0, 20.0)
            kp2d.append(np.c_[(pts - uv)/size, conf])
        gap = delta[0] - delta[1] if ok else 0.0
        X_all.append(np.concatenate([np.array(relief).ravel(), np.array(kp2d).ravel(),
                                     [gap, float(ok), np.log1p(npix[0]), np.log1p(npix[1]), float(np.linalg.norm(mid))]]))
        P_all.append(Cb - mid)
        keep.append(i)
        mids.append(mid)
    P_all, X_all = np.array(P_all), np.array(X_all)
    Y = predict(P_all, X_all)
    out = {k: v[keep] if isinstance(v, np.ndarray) and len(v) == len(z0['image']) else v for k, v in zb.items()}
    kp, vs, ts = [], [], []
    for j, i in enumerate(keep):
        kq, vq, tq = [], [], []
        for n in range(2):
            A, B = P_all[j, n], Y[j, n]
            ca, cb = A.mean(0), B.mean(0)
            U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
            Rn = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
            off = cb - Rn @ (ca + mids[j]) + mids[j]                 # virtual camera = image camera here (Rv = I)
            kq.append(zb['kp3d'][i, n].astype(np.float64) @ Rn.T)
            vq.append((zb['verts'][i, n].astype(np.float64) @ Rn.T).astype(np.float16))
            tq.append(Rn @ zb['t'][i, n].astype(np.float64) + off)
        kp.append(kq); vs.append(vq); ts.append(tq)
    out.update(kp3d=np.array(kp, np.float32), verts=np.array(vs), t=np.array(ts, np.float32))
    dst = Path(a.out)/'place'; dst.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst/'vicos_s3d.npz', **out)
    shift = np.linalg.norm((Y - P_all[..., :, :]).mean(2), axis=-1)
    print(f'placed {len(keep)} images; median body shift {np.median(shift)*100:.1f} cm', flush=True)


def relation_sets(npz):
    """Per image: identity-free and identity-aware sets of threading relations with margin >= MARGIN."""
    z = np.load(npz)
    out = {}
    for i, img in enumerate(map(str, z['image'])):
        Ca, Cb = (canon_from_pred(z['kp3d'][i, n].astype(np.float64) + z['t'][i, n]) for n in range(2))
        thr = relations(Ca, Cb)['threads']
        aware = {k for k, (_, m) in thr.items() if m >= MARGIN}
        free = set()
        for k in aware:
            chain, loop = k.split(' through ')
            free.add(f"{COARSE[chain.split('.')[1]]} through {loop.split('.')[1]}")
        out[img] = (free, aware)
    return out


def dice(a, b):
    return 2*len(a & b)/(len(a) + len(b)) if (a or b) else None


def cmd_agree(a):
    moments = json.load(open(Path(a.out)/'moments.json'))
    variants = dict(v.split('=', 1) for v in a.variant)
    rng = random.Random(SEED)
    # baseline partner for each moment: another moment of the same session and label, >= 300 frames away if possible
    partner = {}
    groups = defaultdict(list)
    for k, m in enumerate(moments):
        groups[(m['session'], m['position'])].append(k)
    for k, m in enumerate(moments):
        cands = [j for j in groups[(m['session'], m['position'])] if j != k]
        far = [j for j in cands if abs(moments[j]['frame'] - m['frame']) >= 300]
        partner[k] = rng.choice(far or cands) if cands else None
    report = {'moments': len(moments), 'variants': {}}
    for name, npz in variants.items():
        R = relation_sets(npz)
        rows = []   # (moment, kind, sync dice, baseline dice) per phone pair
        for k, m in enumerate(moments):
            imgs = m['images']
            if partner[k] is None or not all(i in R for i in imgs):
                continue
            other = moments[partner[k]]['images']
            if not all(i in R for i in other):
                continue
            for p in range(len(imgs)):
                for q in range(p + 1, len(imgs)):
                    for kind in (0, 1):
                        s = dice(R[imgs[p]][kind], R[imgs[q]][kind])
                        b = dice(R[imgs[p]][kind], R[other[q]][kind])
                        rows.append((k, kind, s, b))
        res = {}
        for kind, label in ((0, 'identity_free'), (1, 'identity_aware')):
            rk = [(k, s, b) for k, kd, s, b in rows if kd == kind and s is not None and b is not None]
            if not rk:
                continue
            ks = np.array([r[0] for r in rk]); S = np.array([r[1] for r in rk]); B = np.array([r[2] for r in rk])
            uniq = np.unique(ks)
            brng = np.random.default_rng(1)
            diffs = []
            for _ in range(1000):
                pick = brng.choice(uniq, len(uniq))
                idx = np.concatenate([np.where(ks == u)[0] for u in pick])
                diffs.append(S[idx].mean() - B[idx].mean())
            res[label] = {'sync_dice': float(S.mean()), 'baseline_dice': float(B.mean()),
                          'difference': float(S.mean() - B.mean()),
                          'difference_95ci': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                          'pairs': len(rk), 'moments': int(len(uniq))}
        none_both = sum(1 for k, kd, s, b in rows if kd == 0 and s is None)
        res['pairs_with_no_relation_in_either_view'] = int(none_both)
        report['variants'][name] = res
        print(name, json.dumps(res), flush=True)
    # paired difference between variants (identity-free sync dice), for the reliability ranking
    json.dump(report, open(Path(a.out)/'agree.json', 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['select', 'place', 'agree'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--per-session', type=int, default=100)
    ap.add_argument('--bodies', default='')
    ap.add_argument('--blend', default='')
    ap.add_argument('--maps', default='')
    ap.add_argument('--model', default='')
    ap.add_argument('--variant', action='append', default=[])
    a = ap.parse_args()
    {'select': cmd_select, 'place': cmd_place, 'agree': cmd_agree}[a.cmd](a)


if __name__ == '__main__':
    main()
