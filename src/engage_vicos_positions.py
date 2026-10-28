"""Demo stage 5 feasibility on real BJJ: can the position, and who holds it, be read from both athletes' 2D poses?
(CPU, .venv-depth). Experimental; exploratory (not pre-registered).

ViCoS labels 18 classes per frame: 9 positions x the dominant athlete (suffix 1 / 2), plus standing and 50/50. Input
per frame: both athletes' 17 COCO keypoints (x, y, confidence; pose1 then pose2), centred on the pair and scaled by the
pair's size, plus the offset between the two athletes. A 2-layer MLP (with left/right mirroring and athlete swapping
as augmentation) is trained with blocked temporal cross-validation, then smoothed over time per stream with a Viterbi
pass over the label transitions seen in training (positions last seconds).

Why blocked in time: ViCoS was recorded position by position, so each of its 6 sessions drills one group of positions
(guards; 50/50 and half guard; mount and side control; turtle; standing and takedowns; back). Leaving a session out
leaves whole classes unseen. Its 2-3 phones per session are synced (same frame numbers), so a random frame split
leaks the same moment. Each session's timeline is cut into 5 blocks; fold k tests block k of every session on every
phone, and training drops frames within GAP frames of the test block. ViCoS is one pair of athletes, so this tests new
moments and viewpoints, not new people.
Reports per-frame accuracy and macro-F1 for the 18 classes, for the position alone (holder ignored), and for the holder
given the right position; and coverage (frames with both poses).

  TORCH_THREADS=8 ../.venv-depth/bin/python engage_vicos_positions.py --out DIR [--epochs 30] [--window]
With --window, each frame's input also holds the same features at offsets of -30, -15, -6, +6, +15 and +30 frames
(about +-1 s) of the same stream, with a present flag (zeros where that frame has no poses).
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_paths import VICOS   # noqa: E402

torch.set_num_threads(int(os.environ.get('TORCH_THREADS', 8)))
SESSIONS = [['00', '01', '02'], ['03', '04', '05'], ['06', '07', '08'], ['09', '10'], ['11', '12', '13'], ['14', '15']]
BLOCKS, GAP = 5, 150                                           # 150 frames = 5 s at 30 fps
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]   # COCO left/right swap


def base(label):
    return label[:-1] if label[-1] in '12' and label not in ('5050_guard',) else label


def features(p1, p2):
    """[N,2*17*3+4] both poses centred on the pair and scaled by its size, plus the athlete-to-athlete offset."""
    K = np.stack([p1, p2], 1).astype(np.float32)                    # [N,2,17,3]
    xy, c = K[..., :2], K[..., 2:]
    w = (c > 0.2).astype(np.float32)
    cen = (xy * w).sum((1, 2)) / np.maximum(w.sum((1, 2)), 1)        # [N,2]
    mn = np.where(w > 0, xy, np.inf).min((1, 2))
    mx = np.where(w > 0, xy, -np.inf).max((1, 2))
    scale = np.maximum((mx - mn).max(1), 1.0)[:, None]
    nx = (xy - cen[:, None, None]) / scale[:, None, None]
    own = [(nx[:, n] * w[:, n]).sum(1) / np.maximum(w[:, n].sum(1), 1) for n in range(2)]
    return np.concatenate([np.concatenate([nx, c], -1).reshape(len(K), -1), own[0], own[1]], 1)


def mirror(p):
    q = np.asarray(p, np.float32)[:, FLIP].copy()
    q[..., 0] = -q[..., 0]
    return q


def train(X, y, n_cls, epochs, seed=0):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(X.shape[1], 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, 512), nn.GELU(),
                        nn.Dropout(0.2), nn.Linear(512, n_cls))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt, yt = torch.tensor((X - mu) / sd), torch.tensor(y)
    for _ in range(epochs):
        net.train()
        for b in torch.randperm(len(Xt)).split(512):
            loss = nn.functional.cross_entropy(net(Xt[b]), yt[b], label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    net.eval()
    return lambda Q: torch.log_softmax(net(torch.tensor((Q - mu) / sd)), -1).detach().numpy()


def viterbi(logp, logT):
    T, K = logp.shape
    score, back = logp[0].copy(), np.zeros((T, K), int)
    for t in range(1, T):
        cand = score[:, None] + logT
        back[t] = cand.argmax(0)
        score = cand.max(0) + logp[t]
    path = [int(score.argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return np.array(path[::-1])


def scores(pred, true, classes):
    acc = float((pred == true).mean())
    f1 = []
    for k in range(len(classes)):
        tp = ((pred == k) & (true == k)).sum(); fp = ((pred == k) & (true != k)).sum(); fn = ((pred != k) & (true == k)).sum()
        if tp + fn:
            f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    pb = np.array([base(classes[k]) for k in pred]); tb = np.array([base(classes[k]) for k in true])
    same = pb == tb
    holder = float((pred[same] == true[same]).mean()) if same.any() else float('nan')
    return {'acc_18': acc, 'macro_f1_18': float(np.mean(f1)), 'acc_position': float(same.mean()),
            'holder_given_position': holder}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--window', action='store_true')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    A = json.load(open(VICOS/'annotations.json'))
    classes = sorted({x['position'] for x in A})
    cid = {c: i for i, c in enumerate(classes)}
    swap = np.array([cid.get(c[:-1] + {'1': '2', '2': '1'}[c[-1]], i) if c[-1] in '12' and c != '5050_guard' else i
                     for i, c in enumerate(classes)])             # label when the two athletes' order is swapped
    rows = [x for x in A if x.get('pose1') and x.get('pose2')]
    stream = np.array([x['image'][:2] for x in rows])
    frame = np.array([x['frame'] for x in rows])
    y = np.array([cid[x['position']] for x in rows])
    P1 = np.array([x['pose1'] for x in rows], np.float32); P2 = np.array([x['pose2'] for x in rows], np.float32)
    X = features(P1, P2)
    Xm = features(mirror(P1), mirror(P2))                          # left/right mirror: same label
    Xs = features(P2, P1)                                          # athletes swapped: holder label swapped
    if a.window:
        at = {(s, f): i for i, (s, f) in enumerate(zip(stream, frame))}
        def win(F):
            cols = [F]
            for o in (-30, -15, -6, 6, 15, 30):
                j = np.array([at.get((s, f + o), -1) for s, f in zip(stream, frame)])
                cols += [np.where(j[:, None] >= 0, F[np.maximum(j, 0)], 0.0), (j >= 0)[:, None].astype(np.float32)]
            return np.concatenate(cols, 1).astype(np.float32)
        X, Xm, Xs = win(X), win(Xm), win(Xs)
    rep = {'frames_total': len(A), 'frames_both_poses': len(rows), 'coverage': len(rows) / len(A),
           'coverage_by_position': {c: float(np.mean([bool(x.get('pose1') and x.get('pose2')) for x in A if base(x['position']) == c]))
                                    for c in sorted({base(x['position']) for x in A})}, 'folds': {}}
    allp, allp_s, ally, alli = [], [], [], []
    sess_of = {s: i for i, ss in enumerate(SESSIONS) for s in ss}
    sid = np.array([sess_of[s] for s in stream])
    lo = {i: frame[sid == i].min() for i in range(len(SESSIONS))}
    hi = {i: frame[sid == i].max() + 1 for i in range(len(SESSIONS))}
    blk = np.array([min(int((f - lo[i]) * BLOCKS / (hi[i] - lo[i])), BLOCKS - 1) for f, i in zip(frame, sid)])
    edges = {(i, b): (lo[i] + (hi[i] - lo[i]) * b / BLOCKS, lo[i] + (hi[i] - lo[i]) * (b + 1) / BLOCKS)
             for i in range(len(SESSIONS)) for b in range(BLOCKS)}
    for k in range(BLOCKS):
        te = blk == k
        near = np.array([edges[(i, k)][0] - GAP <= f < edges[(i, k)][1] + GAP for f, i in zip(frame, sid)])
        tr = ~near
        sess = sorted(set(stream[te]))
        Xtr = np.concatenate([X[tr], Xm[tr], Xs[tr]]); ytr = np.concatenate([y[tr], y[tr], swap[y[tr]]])
        f = train(Xtr, ytr, len(classes), a.epochs, seed=k)
        # transition prior from the training sessions' label sequences (per stream, consecutive frames)
        Tc = np.ones((len(classes), len(classes)))
        for s in np.unique(stream[tr]):
            i = sess_of[s]
            seqr = [x for x in A if x['image'][:2] == s and not (edges[(i, k)][0] - GAP <= x['frame'] < edges[(i, k)][1] + GAP)]
            seqr.sort(key=lambda x: x['frame'])
            for u, v in zip(seqr, seqr[1:]):
                if v['frame'] - u['frame'] == 1:
                    Tc[cid[u['position']], cid[v['position']]] += 1
        logT = np.log(Tc / Tc.sum(1, keepdims=True))
        lp = f(X[te])
        pf = lp.argmax(1)
        ps = np.zeros_like(pf)
        idx_te = np.where(te)[0]
        for s in sess:
            m = stream[idx_te] == s
            order = np.argsort(frame[idx_te][m])
            sel = np.where(m)[0][order]
            ps[sel] = viterbi(lp[sel], logT)
        rep['folds'][f'block{k}'] = {'frames': int(te.sum()), 'train_frames': int(tr.sum()),
                                     'per_frame': scores(pf, y[te], classes), 'smoothed': scores(ps, y[te], classes)}
        allp.append(pf); allp_s.append(ps); ally.append(y[te]); alli.append(np.where(te)[0])
        print(k, json.dumps(rep['folds'][f'block{k}']), flush=True)
    P, Ps, Y = np.concatenate(allp), np.concatenate(allp_s), np.concatenate(ally)
    I = np.concatenate(alli)
    np.savez_compressed(out/'predictions.npz', stream=stream[I], frame=frame[I], true=Y, pred=P, pred_smoothed=Ps,
                        classes=np.array(classes))
    rep['pooled'] = {'per_frame': scores(P, Y, classes), 'smoothed': scores(Ps, Y, classes)}
    cm = np.zeros((len(classes), len(classes)), int)
    for t, p in zip(Y, Ps):
        cm[t, p] += 1
    rep['per_class_recall_smoothed'] = {classes[i]: float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(len(classes))}
    rep['classes'] = classes
    json.dump(rep, open(out/'positions.json', 'w'), indent=1)
    np.save(out/'confusion_smoothed.npy', cm)
    print(json.dumps(rep['pooled'], indent=1))
    print(json.dumps(rep['per_class_recall_smoothed'], indent=1))


if __name__ == '__main__':
    main()
