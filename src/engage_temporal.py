"""E14: can time fill in limbs hidden at a contact? A temporal head over SAM 3D Body's per-frame points (CPU,
.venv-depth). Experimental. Pre-registration: `docs/engagement/results/temporal_prereg.md`.

A limb hidden behind the partner now was usually visible a moment earlier or later. Per athlete and frame, the head
reads a window of 2k+1 frames of that camera (every 2nd ground-truth frame, about 15 per second): the athlete's own 19
points and the other athlete's (in the camera frame, centred on the athlete's centre-frame centroid, so the camera's
own frame stays fixed over time), plus ViTPose+ confidence of the 17 keypoints and a "present" flag per frame. It
outputs a correction of the athlete's centre-frame pose; the target is the truth aligned rigidly onto the prediction
over CORE14 (as E11: pose, not placement). Window 1 is the per-frame head (E10's points-only head on E11's measure).

Masked-gap protocol: the window-15 head is also trained with random masking of the centre frame and up to 7 frames
around it (inputs zeroed, present flag off); at test the centre +-g frames are hidden (g = 0, 1, 3, 7: up to about
1 s) to measure how far time alone carries the limbs.

  TORCH_THREADS=8 ../.venv-depth/bin/python engage_temporal.py --feats DIR --out DIR [--smoke]
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_limb_head import CORE14, JOINT8, kabsch, tangle_mask   # noqa: E402

torch.set_num_threads(int(os.environ.get('TORCH_THREADS', 8)))
DEV = ['025_grappling2', '028_grappling2', '030_grappling2', '032_grappling2', '033_grappling2', '034_grappling2',
       '035_grappling2', '036_grappling2', '038_grappling2', '039_grappling2', '040_grappling2', '041_grappling2']
HELD = ['031_grappling2', '037_grappling2', '042_grappling2', '043_grappling2']
CAMS4 = {'cam01', 'cam06', 'cam11', 'cam16'}
WINDOWS = [1, 5, 15, 31]
GAPS = [0, 1, 3, 7]
WIDTH, EPOCHS, SEEDS, DROP, WD, LR, BATCH, SEED = 512, 40, 2, 0.2, 1e-3, 1e-3, 512, 0
import engage_limb_head as L   # noqa: E402
L.ALIGN = CORE14               # targets and errors aligned over the 14 points both body models define alike


def load_tracks(feats, seqs, cams):
    """{(seq, cam): dict(frame -> index)} and stacked arrays P, G (camera frame), conf [V,2,17]."""
    P, G, C, key = [], [], [], []
    for p in sorted(glob.glob(f'{feats}/*__cam??.npz')):
        s, c = Path(p).stem.split('__')
        if s not in seqs or c not in cams:
            continue
        d = np.load(p)
        R = d['R'].astype(np.float64)                     # camera -> virtual, per view
        m = d['m'].astype(np.float64)
        for name, out in (('P', P), ('G', G)):
            A = d[name].astype(np.float64) + m[:, None, None]
            out.append(np.einsum('vij,vnpi->vnpj', R, A))   # back to the (fixed) camera frame: X_cam = R^T X_virt
        C.append(d['X'][:, 64:166].reshape(len(d['X']), 2, 17, 3)[..., 2])
        key += [(s, c, int(f)) for f in d['frame']]
    P, G, C = np.concatenate(P), np.concatenate(G), np.concatenate(C)
    tracks = defaultdict(dict)
    for i, (s, c, f) in enumerate(key):
        tracks[(s, c)][f] = i
    return P, G, C, key, tracks


def windows_for(key, tracks, k):
    """[V, 2k+1] view index per offset (-1 = missing), stepping through that camera's frames in order."""
    idx = np.full((len(key), 2 * k + 1), -1)
    for (s, c), fr in tracks.items():
        frames = sorted(fr)
        pos = {f: j for j, f in enumerate(frames)}
        for f in frames:
            j = pos[f]
            for o in range(-k, k + 1):
                if 0 <= j + o < len(frames):
                    idx[fr[f], o + k] = fr[frames[j + o]]
    return idx


def inputs(P, C, idx, n, mask=None):
    """Per athlete n: [V, (2k+1) * (57 + 57 + 17 + 1)] window features, centred on the centre frame's own centroid."""
    V, W = idx.shape
    c0 = P[:, n].mean(1)                                   # [V,3]
    feats = []
    for o in range(W):
        j = idx[:, o]
        ok = (j >= 0) if mask is None else (j >= 0) & ~mask[:, o]
        jj = np.where(j >= 0, j, 0)
        own = (P[jj, n] - c0[:, None]).reshape(V, -1)
        other = (P[jj, 1 - n] - c0[:, None]).reshape(V, -1)
        f = np.concatenate([own, other, C[jj, n], np.ones((V, 1))], 1)
        feats.append(np.where(ok[:, None], f, 0.0).astype(np.float32))
    return np.concatenate(feats, 1)


def target(P, G, n):
    return (kabsch(G[:, n], P[:, n]) - P[:, n]).reshape(len(P), -1)


class Head(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d_in, WIDTH), nn.GELU(), nn.Dropout(DROP), nn.Linear(WIDTH, WIDTH), nn.GELU(),
                               nn.Dropout(DROP), nn.Linear(WIDTH, 57))
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)

    def forward(self, x):
        return self.f(x)


def train_head(X, T, seed, epochs, mask_fn=None, Xbuild=None):
    """X: [N,D] (or rebuilt each epoch by Xbuild(mask) for masked training); T: [N,57]."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X0 = X if Xbuild is None else Xbuild(None)
    mu, sd = X0.mean(0), X0.std(0) + 1e-6
    net = Head(X0.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Tt = torch.tensor(T, dtype=torch.float32)
    for ep in range(epochs):
        Xe = X0 if Xbuild is None else Xbuild(mask_fn(rng))
        Xt = torch.tensor((Xe - mu) / sd, dtype=torch.float32)
        net.train()
        for b in torch.randperm(len(Xt)).split(BATCH):
            loss = torch.linalg.norm((net(Xt[b]) - Tt[b]).view(-1, 19, 3), dim=-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    net.eval()
    return lambda Xq: net(torch.tensor((Xq - mu) / sd, dtype=torch.float32)).detach().numpy().astype(np.float64)


def errors(P, G, corr):
    """[V,2,19] rigid-aligned (CORE14) error of corrected points."""
    Q = P + corr
    e = np.zeros(P.shape[:3])
    for n in range(2):
        e[:, n] = np.linalg.norm(kabsch(Q[:, n], G[:, n]) - G[:, n], axis=-1)
    return e


def summarise(e, tang):
    t = tang[..., JOINT8]
    ee = e[..., JOINT8]
    return {'tangle_joint8_median_cm': float(np.median(ee[t]) * 100), 'other_joint8_median_cm': float(np.median(ee[~t]) * 100),
            'tangle_points': int(t.sum())}


def boot(e_a, e_b, tang, n=1000):
    """95% interval of median(a)/median(b) - 1 over tangle limb joints, resampling views."""
    rng = np.random.default_rng(1)
    t = tang[..., JOINT8]
    a, b = e_a[..., JOINT8], e_b[..., JOINT8]
    V = len(t)
    r = []
    for _ in range(n):
        s = rng.integers(0, V, V)
        r.append(np.median(a[s][t[s]]) / np.median(b[s][t[s]]) - 1)
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    epochs = 2 if a.smoke else EPOCHS
    allcams = {f'cam{i:02d}' for i in range(1, 21)}
    Pd, Gd, Cd, kd, td = load_tracks(a.feats, DEV[:3] if a.smoke else DEV, allcams)
    tests = {'heldout': load_tracks(a.feats, HELD, CAMS4), 'mma_new_people': load_tracks(a.feats, ['016_mma4'], CAMS4)}
    if a.smoke:   # code check on development data only
        tests = {'smoke_dev': load_tracks(a.feats, DEV[3:4], CAMS4)}
    rep = {'settings': dict(WIDTH=WIDTH, EPOCHS=epochs, SEEDS=SEEDS, WINDOWS=WINDOWS, GAPS=GAPS), 'dev_views': int(len(Pd))}
    Ttr = np.concatenate([target(Pd, Gd, n) for n in range(2)])
    E = {name: {} for name in tests}
    for k in [w // 2 for w in WINDOWS]:
        idx = windows_for(kd, td, k)
        Xtr = np.concatenate([inputs(Pd, Cd, idx, n) for n in range(2)])
        heads = [train_head(Xtr, Ttr, SEED + s, epochs) for s in range(SEEDS)]
        for name, (P, G, C, key, tr) in tests.items():
            ti = windows_for(key, tr, k)
            corr = np.stack([np.mean([h(inputs(P, C, ti, n)) for h in heads], 0).reshape(-1, 19, 3) for n in range(2)], 1)
            E[name][f'w{2 * k + 1}'] = errors(P, G, corr)
        print(f'window {2 * k + 1} done', flush=True)
    # masked-gap: window 15, trained with random hiding of the centre and up to 7 frames around it
    k = 7
    idx = windows_for(kd, td, k)
    V = len(Pd)

    def mask_fn(rng):
        g = rng.integers(0, 8, V)
        offs = np.arange(-k, k + 1)[None]
        return np.abs(offs) <= g[:, None]
    build = lambda msk: np.concatenate([inputs(Pd, Cd, idx, n, msk) for n in range(2)])
    heads = [train_head(None, Ttr, SEED + 10 + s, epochs, mask_fn=mask_fn, Xbuild=build) for s in range(SEEDS)]
    for name, (P, G, C, key, tr) in tests.items():
        ti = windows_for(key, tr, k)
        for g in GAPS:
            msk = np.abs(np.arange(-k, k + 1))[None].repeat(len(P), 0) <= g if g > 0 else np.zeros((len(P), 2 * k + 1), bool)
            if g == 0:
                msk[:] = False
            corr = np.stack([np.mean([h(inputs(P, C, ti, n, msk)) for h in heads], 0).reshape(-1, 19, 3) for n in range(2)], 1)
            E[name][f'masked_gap{g}'] = errors(P, G, corr)
        E[name]['none'] = errors(P, G, np.zeros_like(P))
    for name, (P, G, C, key, tr) in tests.items():
        tang = tangle_mask(G)
        r = {v: summarise(e, tang) for v, e in E[name].items()}
        r['w15_vs_w1_tangle_95ci'] = boot(E[name]['w15'], E[name]['w1'], tang)
        r['masked_gap3_vs_w1_tangle_95ci'] = boot(E[name]['masked_gap3'], E[name]['w1'], tang)
        rep[name] = r
        print(name, json.dumps(r), flush=True)
    json.dump(rep, open(out/'temporal.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
