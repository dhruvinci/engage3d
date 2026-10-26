"""E10: how much limb error inside tangles can a small head remove, reading SAM 3D Body's final pose token (CPU,
.venv-depth). Pre-registration: `docs/engagement/results/limbhead_prereg.md`.

Input: `engage_place_features.py` files that carry `tok` (SAM 3D Body's final pose token per athlete). Per athlete,
the head sees its own 19 canonical points (centred on its centroid), the other athlete's points in the same frame,
and (variant `token`) its token reduced to 64 principal components fitted on the training rows only. It outputs a
per-point correction of the athlete's own pose. The target is the truth rigidly aligned onto the prediction (Kabsch
over the 19 points), so the head learns pose, not placement. Evaluation: align the corrected body rigidly onto the
truth and measure each point's error (a rigid "PA" error, no scaling).

Variants: `none` (no correction), `ponly` (points only: also learns any fixed difference between the two body models'
point definitions), `token` (points + token), `shuffled` (points + a token from another view of the same split).
Tangle points: limb points (elbows, wrists, hands, knees, ankles, toes) whose true position is within 20 cm of a bone
of the other athlete's true skeleton.

Splits: development = grappling2 sequences other than the held-out four (all cameras); test = held-out sequences 031 /
037 / 042 / 043 on cameras 01/06/11/16 (the pre-registered E06 test views), and MMA 016 (new people, never trained on).

  ../.venv-depth/bin/python engage_limb_head.py --feats DIR --out DIR --cv        # 4-fold CV by sequence, development
  ../.venv-depth/bin/python engage_limb_head.py --feats DIR --out DIR --test      # train on development, score tests
  --align core14 --measure joint8: E11's measure (align over the 14 points SMPL and MHR define alike; score elbows,
  wrists, knees, ankles), for the E11 baseline "original + points-only head".
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_relations import BONES, CANON                              # noqa: E402

torch.set_num_threads(int(os.environ.get('TORCH_THREADS', 8)))
TEST = ['031_grappling2', '037_grappling2', '042_grappling2', '043_grappling2']
TEST_CAMS = {'cam01', 'cam06', 'cam11', 'cam16'}
MMA = '016_mma4'
LIMB = np.array([any(k in n for k in ('elbow', 'wrist', 'hand', 'knee', 'ankle', 'toe')) for n in CANON])
SEGS = [(CANON.index(a), CANON.index(b)) for _, a, b in BONES]
TANGLE_M = 0.20
K_TOK, WIDTH, EPOCHS, SEEDS, DROP, WD, LR, BATCH = 64, 512, 60, 3, 0.2, 1e-3, 1e-3, 512
VARIANTS = ['none', 'ponly', 'token', 'shuffled']
ALIGN = None          # indices for the rigid alignment (None: all 19); --align core14 sets E11's 14 points
CORE14 = [CANON.index(n) for n in ('pelvis', 'neck', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle',
                                   'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist')]
JOINT8 = np.array([n in ('l_knee', 'r_knee', 'l_ankle', 'r_ankle', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist')
                   for n in CANON])


def load(feats):
    rows = {k: [] for k in ('P', 'G', 'tok')}
    meta = []
    for p in sorted(glob.glob(f'{feats}/*__cam??.npz')):
        seq, cam = Path(p).stem.split('__')
        d = np.load(p)
        if 'tok' not in d.files:
            continue
        for k in rows:
            rows[k].append(d[k])
        meta += [(seq, cam, int(f)) for f in d['frame']]
    return {k: np.concatenate(v).astype(np.float64) for k, v in rows.items()}, meta


def kabsch(A, B):
    """A rigidly aligned onto B (rotation + translation, no scale), batched over the leading axis: [N,J,3]."""
    idx = ALIGN if ALIGN is not None else slice(None)
    a, b = A[:, idx].mean(1, keepdims=True), B[:, idx].mean(1, keepdims=True)
    U, _, Vt = np.linalg.svd(np.swapaxes(A[:, idx] - a, 1, 2) @ (B[:, idx] - b))
    d = np.sign(np.linalg.det(np.swapaxes(Vt, 1, 2) @ np.swapaxes(U, 1, 2)))
    D = np.zeros((len(A), 3, 3)); D[:, 0, 0] = 1; D[:, 1, 1] = 1; D[:, 2, 2] = d
    R = np.swapaxes(Vt, 1, 2) @ D @ np.swapaxes(U, 1, 2)
    return (A - a) @ np.swapaxes(R, 1, 2) + b


def tangle_mask(G):
    """[V,2,19] True where a limb point is within TANGLE_M of a bone of the other athlete (both from the truth)."""
    out = np.zeros(G.shape[:3], bool)
    for n in range(2):
        p = G[:, n][:, :, None]                                    # [V,19,1,3]
        A = np.stack([G[:, 1 - n, a] for a, _ in SEGS], 1)[:, None]  # [V,1,S,3]
        B = np.stack([G[:, 1 - n, b] for _, b in SEGS], 1)[:, None]
        ab = B - A
        t = np.clip(((p - A) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
        d = np.linalg.norm(A + t[..., None] * ab - p, axis=-1).min(-1)  # [V,19]
        out[:, n] = (d < TANGLE_M) & LIMB[None]
    return out


def bodies(D, idx, tok_rows=None):
    """Per-athlete samples for views idx: inputs (own points, other's points, token rows) and the pose target."""
    P, G = D['P'][idx], D['G'][idx]
    X, T, own = [], [], []
    for n in range(2):
        c = P[:, n].mean(1, keepdims=True)
        X.append(np.concatenate([(P[:, n] - c).reshape(len(idx), -1), (P[:, 1 - n] - c).reshape(len(idx), -1)], 1))
        T.append((kabsch(G[:, n], P[:, n]) - P[:, n]).reshape(len(idx), -1))
        own.append((tok_rows if tok_rows is not None else D['tok'][idx])[:, n])
    return np.concatenate(X), np.concatenate(T), np.concatenate(own)


class Head(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d_in, WIDTH), nn.GELU(), nn.Dropout(DROP), nn.Linear(WIDTH, WIDTH), nn.GELU(),
                               nn.Dropout(DROP), nn.Linear(WIDTH, 57))
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)

    def forward(self, x):
        return self.f(x)


def fit_predict(Xtr, Ttr, Xte, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(Ttr, dtype=torch.float32)
    net = Head(xt.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    for _ in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), BATCH):
            b = perm[i:i + BATCH]
            loss = torch.linalg.norm((net(xt[b]) - yt[b]).view(-1, 19, 3), dim=-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor((Xte - mu) / sd, dtype=torch.float32)).numpy().astype(np.float64)


def token_pca(tok_train, tok_eval):
    mu = tok_train.mean(0)
    _, _, Vt = np.linalg.svd(tok_train - mu, full_matrices=False)
    W = Vt[:K_TOK].T
    z_tr = (tok_train - mu) @ W
    s = z_tr.std(0) + 1e-6
    return z_tr / s, ((tok_eval - mu) @ W) / s


def run(D, tr, te, rng):
    """Corrected points [len(te),2,19,3] per variant, trained on views tr."""
    Xtr, Ttr, Ktr = bodies(D, tr)
    Xte, _, Kte = bodies(D, te)
    shuf_tr = D['tok'][tr][rng.permutation(len(tr))]
    shuf_te = D['tok'][te][rng.permutation(len(te))]
    _, _, KStr = bodies(D, tr, shuf_tr)
    _, _, KSte = bodies(D, te, shuf_te)
    out = {'none': np.zeros_like(Xte[:, :57])}
    for name, (a, b) in {'ponly': (Xtr, Xte),
                         'token': tuple(np.concatenate([x, z], 1) for x, z in zip((Xtr, Xte), token_pca(Ktr, Kte))),
                         'shuffled': tuple(np.concatenate([x, z], 1) for x, z in zip((Xtr, Xte), token_pca(KStr, KSte)))
                         }.items():
        out[name] = np.mean([fit_predict(a, Ttr, b, s) for s in range(SEEDS)], 0)
    P = D['P'][te]
    res = {}
    for name, delta in out.items():
        d = delta.reshape(2, len(te), 19, 3).transpose(1, 0, 2, 3)
        res[name] = P + d
    return res


def errors(Q, G):
    """Per-point rigid-aligned error [V,2,19] of corrected bodies Q against the truth G."""
    e = np.zeros(G.shape[:3])
    for n in range(2):
        e[:, n] = np.linalg.norm(kabsch(Q[:, n], G[:, n]) - G[:, n], axis=-1)
    return e


def summarise(E, tang, boot_rng=None, n_boot=1000):
    limb = np.broadcast_to(LIMB[None, None], tang.shape)
    sets = {'all': np.ones_like(tang), 'limb': limb, 'tangle_limb': tang, 'other_limb': limb & ~tang}
    s = {v: {k: {'median_cm': float(np.median(E[v][m]) * 100), 'mean_cm': float(E[v][m].mean() * 100),
                 'points': int(m.sum())} for k, m in sets.items()} for v in E}
    s['per_limb_tangle_median_cm'] = {v: {CANON[j]: float(np.median(E[v][:, :, j][tang[:, :, j]]) * 100)
                                          for j in np.where(LIMB)[0] if tang[:, :, j].any()} for v in E}
    if boot_rng is not None:
        V = len(tang)
        diffs = {'token_vs_ponly': [], 'token_vs_none': [], 'token_vs_shuffled': []}
        for _ in range(n_boot):
            b = boot_rng.integers(0, V, V)
            m = tang[b]
            med = {v: np.median(E[v][b][m]) for v in ('none', 'ponly', 'token', 'shuffled')}
            diffs['token_vs_ponly'].append(med['token'] / med['ponly'] - 1)
            diffs['token_vs_none'].append(med['token'] / med['none'] - 1)
            diffs['token_vs_shuffled'].append(med['token'] / med['shuffled'] - 1)
        s['bootstrap_tangle_median_rel_change_95ci'] = {k: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
                                                        for k, x in diffs.items()}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', required=True)
    ap.add_argument('--out', required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--cv', action='store_true')
    g.add_argument('--test', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='development CV on a few views with 2 epochs (code check)')
    ap.add_argument('--align', choices=['all', 'core14'], default='all')
    ap.add_argument('--measure', choices=['limb12', 'joint8'], default='limb12')
    a = ap.parse_args()
    global ALIGN, LIMB
    if a.align == 'core14':
        ALIGN = CORE14
    if a.measure == 'joint8':
        LIMB = JOINT8
    D, meta = load(a.feats)
    if a.smoke:
        global EPOCHS
        EPOCHS = 2
        keep = np.where(np.array([m[1] for m in meta]) == 'cam01')[0]
        D = {k: v[keep] for k, v in D.items()}
        meta = [meta[i] for i in keep]
    seq = np.array([m[0] for m in meta]); cam = np.array([m[1] for m in meta])
    dev_seqs = sorted({s for s in seq if s.endswith('_grappling2') and s not in TEST})
    dev = np.where(np.isin(seq, dev_seqs))[0]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    tang_all = tangle_mask(D['G'])
    report = {'dev_sequences': dev_seqs, 'dev_views': int(len(dev)), 'settings': dict(
        K_TOK=K_TOK, WIDTH=WIDTH, EPOCHS=EPOCHS, SEEDS=SEEDS, DROP=DROP, WD=WD, LR=LR, BATCH=BATCH, TANGLE_M=TANGLE_M)}
    if a.cv:
        E = {v: np.zeros((len(dev), 2, 19)) for v in VARIANTS}
        pos = {i: k for k, i in enumerate(dev)}
        for f in range(4):
            te_s = dev_seqs[f::4]
            te = dev[np.isin(seq[dev], te_s)]
            tr = dev[~np.isin(seq[dev], te_s)]
            Q = run(D, tr, te, rng)
            for v in VARIANTS:
                E[v][[pos[i] for i in te]] = errors(Q[v], D['G'][te])
            print(f'fold {f}: {len(tr)} train / {len(te)} eval views', flush=True)
        report['cv'] = summarise(E, tang_all[dev])
        (out/'cv.json').write_text(json.dumps(report, indent=1))
    else:
        tests = {'heldout': np.where(np.isin(seq, TEST) & np.isin(cam, list(TEST_CAMS)))[0],
                 'mma_new_people': np.where(seq == MMA)[0]}
        report['test_views'] = {k: int(len(v)) for k, v in tests.items()}
        te_all = np.concatenate(list(tests.values()))
        Q = run(D, dev, te_all, rng)                       # one training on development, both tests scored
        start = 0
        for name, te in tests.items():
            sl = slice(start, start + len(te)); start += len(te)
            E = {v: errors(Q[v][sl], D['G'][te]) for v in VARIANTS}
            report[name] = summarise(E, tang_all[te], boot_rng=np.random.default_rng(1))
            print(f'{name}: {len(te)} views', flush=True)
        (out/'test.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k in ('cv', 'heldout', 'mma_new_people')}, indent=1)[:4000])


if __name__ == '__main__':
    main()
