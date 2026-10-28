"""E16 stage 0: how far can perfect prompts take SAM 3D Body's hidden limbs? (GPU, .venv-s3d). Experimental.
Development views only (E19's 500 validation views, sequences 025 / 035); no test view is touched. It sizes the E16
run: if a perfect mask gains little, the mask route does not deserve days of training.

Per athlete, from the ground-truth box (one backbone pass shared by all arms):
  gtbox                  no prompt (E18's `original`)
  truemask               + the true visible mask (both SMPL bodies painted far-to-near)
  trueclicks2            + 2 clicks from the true 2D joints (E18's `oracle2`)
  truemask_trueclicks2   both
Measure as E18: median error of the 8 limb joints inside a tangle and of the others, rigid-aligned over 14 points.

  PYTORCH_ALLOC_CONF=expandable_segments:True ../.venv-s3d/bin/python engage_s3d_ceiling.py --feats F --out D [--smoke]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_h4d_clicks import truth_2d                                  # noqa: E402
from engage_rfdetr_s3d import boot, cache_backbone, run                 # noqa: E402
from engage_s3d_finetune import (CORE14, JOINT8, canon_matrix, crops_from_feats, kabsch_t,  # noqa: E402
                                 load_estimator, seq_root)
from tangle_h4d import PEOPLE, Sequence                                 # noqa: E402

ARMS = ['gtbox', 'truemask', 'trueclicks2', 'truemask_trueclicks2']
VIEWS = Path(__file__).resolve().parents[1]/'benchmarks/engage_h4d/vitpose_ft/targets.npz'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    T = np.load(VIEWS, allow_pickle=True)
    val = [(r[0], r[1], int(r[2])) for r, s in zip(T['rows'][::2], T['split'][::2]) if s == 'val']
    rows, G, tang = crops_from_feats(a.feats)
    where = {r: i for i, r in enumerate(rows)}
    views = [where[r] for r in val if r in where][:4 if a.smoke else None]
    est = load_estimator()
    m = est.model
    decoder = m.forward_decoder   # the click loop passes full_output=, which forward_decoder no longer takes (E05)
    m.forward_decoder = lambda *args, full_output=None, **kw: decoder(*args, **kw)
    state = cache_backbone(m)
    A = torch.tensor(canon_matrix(), dtype=torch.float32)
    E = {n: np.full((len(views), 2, 19), np.nan) for n in ARMS}
    seqs, t0 = {}, time.time()
    for k, v in enumerate(views):
        s, c, f = rows[v]
        seq = seqs.setdefault(s, Sequence(seq_root(s)))
        img = cv2.cvtColor(cv2.imread(str(seq.image_path(c, f))), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        lab = seq.gt(c, f, size=(W, H))
        b = np.load(seq.root/'processed_data/bbox'/c/f'{f:05d}.npy', allow_pickle=True).item()
        tr = truth_2d(seq, c, f)
        for n, who in enumerate(PEOPLE):
            box = np.asarray(b[who], float)
            r0 = run(m, est, img, box, None, state, reuse=False, pose=tr[n])
            r1 = run(m, est, img, box, lab == n + 1, state, reuse=True, pose=tr[n])
            res = {'gtbox': r0['base'], 'trueclicks2': r0['clicks'], 'truemask': r1['base'],
                   'truemask_trueclicks2': r1['clicks']}
            Gt = torch.tensor(G[v, n][None], dtype=torch.float32)
            for name, kp in res.items():
                C = torch.einsum('jk,nkd->njd', A, torch.tensor(kp[None]))
                E[name][k, n] = (kabsch_t(C, Gt, CORE14) - Gt).norm(dim=-1)[0].numpy()
        torch.cuda.empty_cache()
        if k % 50 == 0:
            print(f'{k}/{len(views)} {time.time() - t0:.0f}s', flush=True)
    t8 = tang[np.array(views)][..., JOINT8]
    Ev = {n: E[n][..., JOINT8] for n in ARMS}
    r = {'views': len(views), 'tangle_points': int(t8.sum()), 'seconds_per_view': round((time.time() - t0) / len(views), 2)}
    for n in ARMS:
        r[n] = {'tangle_joint8_median_cm': float(np.median(Ev[n][t8]) * 100),
                'other_joint8_median_cm': float(np.median(Ev[n][~t8]) * 100)}
    for n in ARMS[1:]:
        r[f'{n}_vs_gtbox_tangle_95ci'] = boot(Ev[n], Ev['gtbox'], t8)
        r[f'{n}_vs_gtbox_other_95ci'] = boot(Ev[n], Ev['gtbox'], ~t8)
    np.savez(out/'errors.npz', views=np.array(views), tang=tang[np.array(views)], **E)
    json.dump(r, open(out/'ceiling.json', 'w'), indent=1)
    print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
