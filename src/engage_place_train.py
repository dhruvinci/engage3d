"""Learned two-person placement: train, cross-validate, and write corrected bodies (CPU, .venv-depth). Experimental.

Input: `engage_place_features.py` output. The model sees both bodies after the combined depth (19 canonical points
each, in the crop's virtual camera frame, centred on the pair) plus the per-view features, and outputs a centre shift
and a small rotation (about its centroid) for each body. Loss: the corrected pair's configuration against the truth's,
both centred on their own pair midpoint (relations depend only on where the bodies are relative to each other).
Augmentation: the two athletes swapped, and the view mirrored left-right. An ensemble of --seeds nets is averaged.
The last layer starts at zero, so an untrained model is the combined depth.

Sequences: grappling 031 / 037 / 042 / 043 are the held-out test (`placemodel_prereg.md`); they are used only with
--test. Development: 4-fold cross-validation by sequence over the other 12 grappling sequences (MMA always in
training). Writes <out>/<variant>/<seq>/<cam>.npz (score with engage_bench_oracle.py) for `blend` (the combined
depth on the same views) and `model` (out-of-fold predictions, or the test predictions with --test), plus
<out>/summary.json.

Options for the overnight data experiments: --tokens K adds SAM 3D Body's final pose token per athlete (from feature
files that carry `tok`), reduced to K principal components fitted on each training set only (mirroring is then off:
a learned token cannot be mirrored). --train-cams / --eval-cams restrict the cameras used for training and for
evaluation (the pre-registered test is cam01/06/11/16). --ref takes several directories (comma-separated), searched
in order.

  ../.venv-depth/bin/python engage_place_train.py --feats DIR --ref s3d_box[,DIR2] --out DIR [--test] [--epochs 60]
      [--tokens 32] [--no-mirror] [--train-cams cam01,..] [--eval-cams cam01,..]
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

TEST = ['031_grappling2', '037_grappling2', '042_grappling2', '043_grappling2']
NR, NK = 16, 17
CANON = ['pelvis', 'neck', 'head', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle', 'l_toe', 'r_toe',
         'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hand', 'r_hand']   # engage_relations
REGIONS = ['head', 'chest', 'back', 'hips', 'l_upper_arm', 'r_upper_arm', 'l_forearm', 'r_forearm', 'l_hand', 'r_hand',
           'l_thigh', 'r_thigh', 'l_shin', 'r_shin', 'l_foot', 'r_foot']
lr_perm = lambda names: [names.index(('r_' + n[2:]) if n.startswith('l_') else ('l_' + n[2:]) if n.startswith('r_')
                                     else n) for n in names]
CANON_LR, REG_LR = lr_perm(CANON), lr_perm(REGIONS)
COCO_LR = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', 8)))   # keep threads x concurrent jobs <= cores


def load(feats, tokens=False):
    D = {k: [] for k in ('P', 'G', 'X', 'R', 'm', 't_blend', 'i', 'frame') + (('tok',) if tokens else ())}
    meta = []
    for p in sorted(glob.glob(f'{feats}/*__cam??.npz')):
        sname, cam = Path(p).stem.split('__')
        d = np.load(p)
        if tokens and 'tok' not in d.files:
            continue
        for k in D:
            D[k].append(d[k])
        meta += [(sname, cam)]*len(d['i'])
    return {k: np.concatenate(v) for k, v in D.items()}, meta


def swap(P, G, X, ntok=0):
    """The same view with the two athletes' order swapped (the last 2 x ntok columns: per-athlete token components)."""
    X = X.copy()
    if ntok:
        X[:, -2*ntok:] = np.concatenate([X[:, -ntok:], X[:, -2*ntok:-ntok]], 1)
    if X.shape[1] - 2*ntok >= 178:   # overlap block (engage_place_features.py --overlap): per-athlete pairs at 171-174
        X[:, [171, 172, 173, 174]] = X[:, [173, 174, 171, 172]]
    r, k = NR*2, NK*3
    X[:, :2*r] = np.concatenate([X[:, r:2*r], X[:, :r]], 1)
    X[:, 2*r:2*r + 2*k] = np.concatenate([X[:, 2*r + k:2*r + 2*k], X[:, 2*r:2*r + k]], 1)
    o = 2*r + 2*k
    X[:, o] = -X[:, o]
    X[:, [o + 2, o + 3]] = X[:, [o + 3, o + 2]]
    return P[:, ::-1].copy(), G[:, ::-1].copy(), X


def mirror(P, G, X):
    """The same view mirrored left-right (x -> -x in the virtual frame; left and right body parts swapped). The 2D
    residuals are in image axes, which are close to the virtual camera's near the crop centre."""
    P, G, X = P[:, :, CANON_LR].copy(), G[:, :, CANON_LR].copy(), X.copy()
    P[..., 0] *= -1
    G[..., 0] *= -1
    r, k = NR*2, NK*3
    rel = X[:, :2*r].reshape(-1, 2, NR, 2)[:, :, REG_LR]
    kp = X[:, 2*r:2*r + 2*k].reshape(-1, 2, NK, 3)[:, :, COCO_LR].copy()
    kp[..., 0] *= -1
    X[:, :2*r], X[:, 2*r:2*r + 2*k] = rel.reshape(len(X), -1), kp.reshape(len(X), -1)
    return P, G, X


def rotmat(w):
    """Axis-angle [B,3] -> rotation [B,3,3]."""
    S = torch.zeros(w.shape[0], 3, 3, dtype=w.dtype)
    S[:, 0, 1], S[:, 0, 2], S[:, 1, 2] = -w[:, 2], w[:, 1], -w[:, 0]
    S[:, 1, 0], S[:, 2, 0], S[:, 2, 1] = w[:, 2], -w[:, 1], w[:, 0]
    return torch.matrix_exp(S)


def apply(P, out):
    """P [B,2,19,3] corrected by out [B,12] (per body: shift 3 in 0.1 m, rotation 3 in 0.1 rad)."""
    res = []
    for n in range(2):
        o = out[:, 6*n:6*n + 6]
        c = P[:, n].mean(1, keepdim=True)
        R = rotmat(0.1*o[:, 3:])
        res.append((P[:, n] - c) @ R.transpose(1, 2) + c + 0.1*o[:, None, :3])
    return torch.stack(res, 1)


def rel_loss(Y, G):
    Yc = Y - Y.reshape(Y.shape[0], -1, 3).mean(1)[:, None, None]
    Gc = G - G.reshape(G.shape[0], -1, 3).mean(1)[:, None, None]
    return torch.linalg.norm(Yc - Gc, dim=-1).mean()


class Net(nn.Module):
    def __init__(self, d_in, width=256, drop=0.1):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d_in, width), nn.GELU(), nn.Dropout(drop), nn.Linear(width, width), nn.GELU(),
                               nn.Dropout(drop), nn.Linear(width, 12))
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)

    def forward(self, x):
        return self.f(x)


def inputs(P, X):
    return np.concatenate([P.reshape(len(P), -1), X], 1)


def train(P, G, X, epochs, seed=0, wd=1e-3, drop=0.2, seeds=3, T=None, ntok=0, use_mirror=True, width=256):
    """An ensemble of `seeds` nets (outputs averaged), trained on the views plus their swapped (and, without tokens,
    mirrored) copies. Returns predict(P, X, T)."""
    pca = None
    if ntok:
        flat = T.reshape(-1, T.shape[-1]).astype(np.float64)
        mu = flat.mean(0)
        pca = (mu, np.linalg.svd(flat - mu, full_matrices=False)[2][:ntok])
        X = np.concatenate([X, ((T - mu) @ pca[1].T).reshape(len(T), -1)], 1)
    P2, G2, X2 = swap(P, G, X, ntok)
    P, G, X = np.concatenate([P, P2]), np.concatenate([G, G2]), np.concatenate([X, X2])
    if use_mirror and not ntok:
        P2, G2, X2 = mirror(P, G, X)
        P, G, X = np.concatenate([P, P2]), np.concatenate([G, G2]), np.concatenate([X, X2])
    fs = [train_one(P, G, X, epochs, seed*100 + k, wd, drop, width) for k in range(seeds)]

    def predict(P_, X_, T_=None):
        if pca is not None:
            X_ = np.concatenate([X_, ((T_ - pca[0]) @ pca[1].T).reshape(len(T_), -1)], 1)
        return fs[0](P_, X_, [f.net_out for f in fs])
    return predict


def train_one(P, G, X, epochs, seed, wd, drop, width=256):
    torch.manual_seed(seed)
    Z = inputs(P, X)
    mu, sd = Z.mean(0), Z.std(0) + 1e-6
    net = Net(Z.shape[1], width=width, drop=drop)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Zt = torch.tensor((Z - mu)/sd, dtype=torch.float32)
    Pt, Gt = torch.tensor(P, dtype=torch.float32), torch.tensor(G, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        for b in torch.randperm(len(Zt), generator=g).split(256):
            loss = rel_loss(apply(Pt[b], net(Zt[b])), Gt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    net.eval()

    def net_out(P_, X_):
        return net(torch.tensor((inputs(P_, X_) - mu)/sd, dtype=torch.float32))

    def f(P_, X_, outs=None):
        with torch.no_grad():
            o = torch.stack([g(P_, X_) for g in (outs or [net_out])]).mean(0)
            return apply(torch.tensor(P_, dtype=torch.float32), o).numpy()
    f.net_out = net_out
    return f


def metrics(Y, G):
    pi = 0   # CANON[0] = pelvis
    rel = lambda A: A[:, 1, pi] - A[:, 0, pi]
    Yc = Y - Y.reshape(len(Y), -1, 3).mean(1)[:, None, None]
    Gc = G - G.reshape(len(G), -1, 3).mean(1)[:, None, None]
    return {'config_err_cm': float(np.median(np.linalg.norm(Yc - Gc, axis=-1).mean((1, 2)))*100),
            'rel_depth_cm': float(np.median(np.abs(rel(Y)[:, 2] - rel(G)[:, 2]))*100),
            'rel_lateral_cm': float(np.median(np.linalg.norm((rel(Y) - rel(G))[:, :2], axis=1))*100)}


def write(out, variant, D, meta, idx, Y, ref):
    """Corrected bodies (rigid per body: from the combined-depth P to Y) in engage_bench_s3d.py's format."""
    by = {}
    for j in idx:
        by.setdefault(meta[j], []).append(j)
    for (sname, cam), js in by.items():
        src = next(f'{r}/{sname}/{cam}.npz' for r in ref.split(',') if os.path.exists(f'{r}/{sname}/{cam}.npz'))
        z = dict(np.load(src))   # load once (indexing an NpzFile decompresses every time)
        kp, vs, ts, fr = [], [], [], []
        for j in js:
            i, Rv, m = D['i'][j], D['R'][j], D['m'][j]
            kq, vq, tq = [], [], []
            for n in range(2):
                A, B = D['P'][j, n], Y[j, n]
                ca, cb = A.mean(0), B.mean(0)
                U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
                Rn = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
                Rcam = Rv.T @ Rn @ Rv
                off = Rv.T @ (cb - Rn @ (ca + m) + m)
                t_old = D['t_blend'][j, n]
                kq.append(z['kp3d'][i, n].astype(np.float64) @ Rcam.T)
                vq.append((z['verts'][i, n].astype(np.float64) @ Rcam.T).astype(np.float16))
                tq.append(Rcam @ t_old + off)
            kp.append(kq)
            vs.append(vq)
            ts.append(tq)
            fr.append(D['frame'][j])
        dst = Path(out)/variant/sname/f'{cam}.npz'
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, frames=np.array(fr), kp3d=np.array(kp, np.float32), verts=np.array(vs),
                            t=np.array(ts, np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--wd', type=float, default=1e-3)
    ap.add_argument('--drop', type=float, default=0.2)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--tokens', type=int, default=0)
    ap.add_argument('--no-mirror', action='store_true')
    ap.add_argument('--train-cams', default='')
    ap.add_argument('--eval-cams', default='')
    ap.add_argument('--no-write', action='store_true', help='summary only (no corrected bodies)')
    ap.add_argument('--drop-overlap', action='store_true', help='ignore the overlap block of --overlap features')
    ap.add_argument('--test-seqs', default='', help='override the held-out test sequences (comma-separated)')
    ap.add_argument('--exclude-seqs', default='', help='sequences used for nothing (comma-separated)')
    a = ap.parse_args()
    global TEST
    if a.test_seqs:
        TEST = a.test_seqs.split(',')
    D, meta = load(a.feats, tokens=a.tokens > 0)
    if a.exclude_seqs:
        keep = np.array([s not in a.exclude_seqs.split(',') for s, _ in meta])
        D = {k: v[keep] for k, v in D.items()}
        meta = [m for m, k in zip(meta, keep) if k]
    if a.drop_overlap:
        D['X'] = D['X'][:, :171]
    seq = np.array([s for s, _ in meta])
    cam = np.array([c for _, c in meta])
    grap = np.array(['grappling' in s for s in seq])
    test = np.isin(seq, TEST)
    trc = np.isin(cam, a.train_cams.split(',')) if a.train_cams else np.ones(len(seq), bool)
    evc = np.isin(cam, a.eval_cams.split(',')) if a.eval_cams else np.ones(len(seq), bool)
    dev_seqs = sorted(set(seq[grap & ~test]))
    T = D.get('tok')
    kw = dict(wd=a.wd, drop=a.drop, seeds=a.seeds, ntok=a.tokens, use_mirror=not a.no_mirror, width=a.width)
    sel = lambda m: (D['P'][m], D['G'][m], D['X'][m])
    tok = lambda m: T[m] if T is not None else None
    Y = D['P'].copy()
    if a.test:
        tr = ~test & trc
        f = train(*sel(tr), a.epochs, T=tok(tr), **kw)
        ev = np.nonzero(test & evc)[0]
        Y[ev] = f(D['P'][ev], D['X'][ev], tok(ev))
    else:
        folds = [dev_seqs[k::4] for k in range(4)]
        for k, fs in enumerate(folds):
            va = np.isin(seq, fs) & evc
            tr = ~test & ~np.isin(seq, fs) & trc
            f = train(*sel(tr), a.epochs, seed=k, T=tok(tr), **kw)
            Y[va] = f(D['P'][va], D['X'][va], tok(va))
            print(f'fold {k} {fs} (train {tr.sum()} views): blend {metrics(D["P"][va], D["G"][va])} -> '
                  f'model {metrics(Y[va], D["G"][va])}', flush=True)
        ev = np.nonzero(grap & ~test & evc)[0]
    S = {'blend': metrics(D['P'][ev], D['G'][ev]), 'model': metrics(Y[ev], D['G'][ev]), 'n': int(len(ev)),
         'n_train_views': int((~test & trc).sum()), 'width': a.width, 'epochs': a.epochs, 'wd': a.wd, 'drop': a.drop,
         'seeds': a.seeds, 'tokens': a.tokens, 'mirror': not a.no_mirror and not a.tokens,
         'train_cams': a.train_cams or 'all', 'eval_cams': a.eval_cams or 'all', 'test': a.test}
    acts = np.array([x.split('_', 1)[1] for x in seq])
    S['by_activity'] = {act: {'n': int((acts[ev] == act).sum()), 'blend': metrics(D['P'][ev][acts[ev] == act], D['G'][ev][acts[ev] == act]),
                              'model': metrics(Y[ev][acts[ev] == act], D['G'][ev][acts[ev] == act])} for act in sorted(set(acts[ev]))}
    print(json.dumps(S), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out)/'summary.json').write_text(json.dumps(S, indent=1))
    if not a.no_write:
        write(a.out, 'blend', D, meta, ev, D['P'], a.ref)
        write(a.out, 'model', D, meta, ev, Y, a.ref)


if __name__ == '__main__':
    main()
