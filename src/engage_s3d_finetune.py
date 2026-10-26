"""E11 pilot: fine-tune SAM 3D Body's body decoder on two-person contact (Harmony4D), backbone frozen (GPU, .venv-s3d).
Pre-registration: `docs/engagement/results/s3dft_prereg.md`. Plan: `docs/engagement/E11_PLAN.md`.

The released SAM 3D Body has no training code. This trains only the promptable body decoder (39.4M parameters); the
ViT-H backbone and the MHR head stay frozen. The backbone runs once per crop (`cache`): its image embedding (bf16,
1280 x 32 x 24) and the original 70 keypoints are stored, and training feeds the cached embedding back in (checked:
same keypoints as the normal path to 0.001 mm). Crops are the ground-truth person boxes of the views in
`engage_place_features.py` outputs, whose truth points `G` (19 canonical points per athlete) are the targets.

Loss, per athlete: the predicted 19 canonical points (a fixed linear map of the 70 keypoints, engage_relations) plus a
learned fixed offset per point in the body's own frame (SMPL and MHR place some points differently; without it the
decoder would bend poses to copy SMPL's definitions), aligned rigidly onto the truth over the 14 points both models
define alike (CORE14); mean error over CORE14, limb points inside a tangle weighted 3x; plus 0.1 x the mean distance
of the 70 keypoints from the original model's (keeps everything else in place).
Measure (as E10, rigid-aligned, no scaling): median error of the 8 limb joints (elbows, wrists, knees, ankles), for
tangle points (within 20 cm of a bone of the other athlete, from the truth) and the rest.

  PYTORCH_ALLOC_CONF=expandable_segments:True ../.venv-s3d/bin/python engage_s3d_finetune.py cache --feats F --out D
  ... engage_s3d_finetune.py train --out D        (writes D/decoder.pt, D/offsets.pt, D/train_log.json)
  ... engage_s3d_finetune.py eval --feats F --out D   (writes D/eval.json: original vs fine-tuned on the test views)
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autolabel_body3d import S3D, load_estimator                    # noqa: E402
from engage_bench_s3d import seq_root                              # noqa: E402
from engage_limb_head import MMA, TEST, TEST_CAMS, tangle_mask     # noqa: E402
from engage_relations import CANON, canon_from_mhr                 # noqa: E402
from tangle_h4d import Sequence                                    # noqa: E402

sys.path.insert(0, str(S3D))
from sam_3d_body.data.utils.prepare_batch import prepare_batch     # noqa: E402
from sam_3d_body.utils import recursive_to                         # noqa: E402

CORE14 = [CANON.index(n) for n in ('pelvis', 'neck', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle',
                                   'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist')]
JOINT8 = [CANON.index(n) for n in ('l_knee', 'r_knee', 'l_ankle', 'r_ankle', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist')]
N_VIEWS, N_VAL_VIEWS, SEED = 5000, 500, 0
EPOCHS, BATCH, LR, LR_OFF, WD, DISTILL, TANGLE_W = 3, 8, 1e-5, 1e-3, 0.01, 0.1, 3.0
EMB = (1280, 32, 24)
PERSON_KEYS = ('img', 'bbox', 'mask', 'mask_score', 'bbox_center', 'bbox_scale', 'ori_img_size', 'img_size',
               'affine_trans', 'person_valid')


def canon_matrix():
    """[19,70] A with canon_from_mhr(K) == A @ K (the canonical points are averages of keypoints)."""
    A = np.zeros((len(CANON), 70))
    for i in range(70):
        K = np.zeros((70, 3)); K[i, 0] = 1
        C = canon_from_mhr(K)
        A[:, i] = [C[n][0] for n in CANON]
    K = np.random.default_rng(0).normal(size=(70, 3))
    C = canon_from_mhr(K)
    assert np.allclose(A @ K, np.stack([C[n] for n in CANON])), 'canon_from_mhr is not linear'
    return A


def kabsch_t(A, B, idx):
    """A [N,J,3] rigidly aligned onto B over points idx (rotation + translation), applied to all of A."""
    a, b = A[:, idx].mean(1, keepdim=True), B[:, idx].mean(1, keepdim=True)
    U, _, Vh = torch.linalg.svd((A[:, idx] - a).transpose(1, 2) @ (B[:, idx] - b))
    d = torch.sign(torch.linalg.det(Vh.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    R = Vh.transpose(1, 2) @ D @ U.transpose(1, 2)
    return (A - a) @ R.transpose(1, 2) + b


def body_frame(C):
    """[N,3,3] columns: left (hips), up (neck - pelvis, orthogonalised), forward."""
    p, n = C[:, CANON.index('pelvis')], C[:, CANON.index('neck')]
    x = C[:, CANON.index('l_hip')] - C[:, CANON.index('r_hip')]
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    y = n - p
    y = y - (y * x).sum(-1, keepdim=True) * x
    y = y / y.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.stack([x, y, torch.cross(x, y, dim=-1)], -1)


def crops_from_feats(feats):
    """Every athlete crop of the feature views: (seq, cam, frame, n), truth G [19,3], tangle mask [19]."""
    rows, G = [], []
    for p in sorted(glob.glob(f'{feats}/*__cam??.npz')):
        seq, cam = Path(p).stem.split('__')
        d = np.load(p)
        if 'tok' not in d.files:      # the same views as E10 (files with SAM 3D Body tokens)
            continue
        rows += [(seq, cam, int(f)) for f in d['frame']]
        G.append(d['G'])
    G = np.concatenate(G).astype(np.float64)
    return rows, G, tangle_mask(G)


def select(rows, tang, seqs, n, rng):
    """n views from seqs, views with a tangle point weighted 3x (fixed seed)."""
    idx = np.array([i for i, r in enumerate(rows) if r[0] in seqs])
    w = np.where(tang[idx].any((1, 2)), 3.0, 1.0)
    return np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False, p=w / w.sum()))


class CachedBackbone(torch.nn.Module):
    """Stands in for the backbone: returns `feed` when set, otherwise runs the real one and keeps its output."""
    def __init__(self, real):
        super().__init__()
        self.real, self.feed, self.capture = real, None, None

    def forward(self, x, extra_embed=None):
        if self.feed is not None:
            return self.feed
        y = self.real(x, extra_embed=extra_embed)
        self.capture = (y[-1] if isinstance(y, tuple) else y).detach()
        return y


def setup():
    est = load_estimator()
    m = est.model
    for p in m.parameters():
        p.requires_grad_(False)
    m.backbone = CachedBackbone(m.backbone)
    return est, m


def merge(batches):
    out = dict(batches[0])
    for k in PERSON_KEYS:
        out[k] = torch.cat([b[k] for b in batches], 1)
    return out


def forward(m, batch):
    batch = recursive_to(batch, 'cuda')
    m._initialize_batch(batch)
    return m.forward_step(batch, decoder_type='body')['mhr']['pred_keypoints_3d']


def boxes_for(seq, cam, f):
    b = np.load(seq.root/'processed_data/bbox'/cam/f'{f:05d}.npy', allow_pickle=True).item()
    return [np.asarray(b[w], np.float32).reshape(1, 4) for w in sorted(b)] if len(b) == 2 else None


def run_backbone(est, m, crop_rows, emb_out, kp_out, box_out, log_every=200):
    """For each (seq, cam, frame, n): backbone embedding and original keypoints; images decoded once per view."""
    seqs = {}
    order = sorted(range(len(crop_rows)), key=lambda i: crop_rows[i][:3])
    pending = []

    def flush():
        if not pending:
            return
        with torch.no_grad():
            m.backbone.feed = None
            kp = forward(m, merge([b for _, b in pending])).float().cpu().numpy()
        cap = m.backbone.capture
        for j, (i, _) in enumerate(pending):
            emb_out[i] = cap[j].view(torch.int16).cpu().numpy()
            kp_out[i] = kp[j]
        pending.clear()

    t0 = time.time()
    last = None
    for k, i in enumerate(order):
        s, c, f, n = crop_rows[i]
        if (s, c, f) != last:
            seq = seqs.setdefault(s, Sequence(seq_root(s)))
            img = cv2.cvtColor(cv2.imread(str(seq.image_path(c, f))), cv2.COLOR_BGR2RGB)
            bx = boxes_for(seq, c, f)
            last = (s, c, f)
        box_out[i] = bx[n].reshape(4)
        pending.append((i, prepare_batch(img, est.transform, bx[n], None, None)))
        if len(pending) == BATCH:
            flush()
        if k % log_every == 0:
            print(f'backbone {k}/{len(order)} {time.time() - t0:.0f}s', flush=True)
    flush()


def cmd_cache(a):
    rows, G, tang = crops_from_feats(a.feats)
    seq_names = np.array([r[0] for r in rows])
    dev = sorted({s for s in seq_names if s.endswith('_grappling2') and s not in TEST})
    val_seqs = dev[::6]
    train_seqs = [s for s in dev if s not in val_seqs]
    rng = np.random.default_rng(SEED)
    n_views = 20 if a.smoke else N_VIEWS
    tr = select(rows, tang, set(train_seqs), n_views, rng)
    va = select(rows, tang, set(val_seqs), 4 if a.smoke else N_VAL_VIEWS, rng)
    views = np.concatenate([tr, va])
    crop_rows = [(*rows[v], n) for v in views for n in range(2)]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    N = len(crop_rows)
    emb = np.lib.format.open_memmap(out/'emb.npy', mode='w+', dtype=np.int16, shape=(N, *EMB))
    kp, box = np.zeros((N, 70, 3), np.float32), np.zeros((N, 4), np.float32)
    est, m = setup()
    run_backbone(est, m, crop_rows, emb, kp, box)
    emb.flush()
    g = np.stack([G[v, n] for v in views for n in range(2)])
    t = np.stack([tang[v, n] for v in views for n in range(2)])
    split = np.array(['train'] * (2 * len(tr)) + ['val'] * (2 * len(va)))
    np.savez(out/'crops.npz', rows=np.array(crop_rows, dtype=object), G=g, tang=t, kp_orig=kp, box=box, split=split)
    json.dump({'train_seqs': train_seqs, 'val_seqs': val_seqs, 'train_crops': int(2 * len(tr)),
               'val_crops': int(2 * len(va))}, open(out/'cache.json', 'w'), indent=1)
    print(f'cached {N} crops', flush=True)


class Trainable(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.offsets = torch.nn.Parameter(torch.zeros(len(CANON), 3))
        self.register_buffer('A', torch.tensor(canon_matrix(), dtype=torch.float32))

    def canon(self, kp):
        C = torch.einsum('jk,nkd->njd', self.A, kp.float())
        return C + torch.einsum('jd,nid->nji', self.offsets, body_frame(C))


def batch_inputs(est, blank, boxes):
    return merge([prepare_batch(blank, est.transform, b.reshape(1, 4), None, None) for b in boxes])


def losses(T, kp, kp_orig, G, tang):
    C = T.canon(kp)
    Ca = kabsch_t(C, G, CORE14)
    e = (Ca - G).norm(dim=-1)[:, CORE14]
    w = 1 + (TANGLE_W - 1) * tang[:, CORE14].float()
    pose = (e * w).sum() / w.sum()
    distill = (kp.float() - kp_orig).norm(dim=-1).mean()
    return pose + DISTILL * distill, pose.item(), distill.item(), e


def evaluate_crops(est, T, blank, idx, emb, data, use_finetuned=True):
    """Per-crop errors [n,19] (rigid-aligned over CORE14) for crops idx, from cached embeddings."""
    out = []
    T.eval()
    with torch.no_grad():
        for s in range(0, len(idx), BATCH):
            b = idx[s:s + BATCH]
            T.m.backbone.feed = torch.from_numpy(np.ascontiguousarray(emb[b])).view(torch.bfloat16).cuda()
            kp = forward(T.m, batch_inputs(est, blank, data['box'][b]))
            G = torch.tensor(data['G'][b], dtype=torch.float32, device='cuda')
            C = T.canon(kp) if use_finetuned else torch.einsum('jk,nkd->njd', T.A, kp.float())
            out.append((kabsch_t(C, G, CORE14) - G).norm(dim=-1).cpu().numpy())
    return np.concatenate(out)


def cmd_train(a):
    out = Path(a.out)
    data = dict(np.load(out/'crops.npz', allow_pickle=True))
    emb = np.load(out/'emb.npy', mmap_mode='r')
    tr, va = np.where(data['split'] == 'train')[0], np.where(data['split'] == 'val')[0]
    est, m = setup()
    blank = np.zeros((2160, 3840, 3), np.uint8)
    T = Trainable(m).cuda()
    for p in m.decoder.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW([{'params': m.decoder.parameters(), 'lr': LR},
                             {'params': [T.offsets], 'lr': LR_OFF, 'weight_decay': 0.0}], weight_decay=WD)
    epochs = 1 if a.smoke else EPOCHS
    steps = epochs * (len(tr) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR, LR_OFF], total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(SEED)
    log = {'val': []}

    def val(tag):
        e = evaluate_crops(est, T, blank, va, emb, data)
        tg = data['tang'][va][:, JOINT8]
        r = {'tag': tag, 'tangle_joint8_median_cm': float(np.median(e[:, JOINT8][tg]) * 100),
             'other_joint8_median_cm': float(np.median(e[:, JOINT8][~tg]) * 100)}
        log['val'].append(r)
        print('val', r, flush=True)
        T.train()

    val('start')
    step, t0 = 0, time.time()
    for ep in range(epochs):
        T.train()
        perm = rng.permutation(tr)
        for s in range(0, len(tr) - BATCH + 1, BATCH):
            b = np.sort(perm[s:s + BATCH])
            m.backbone.feed = torch.from_numpy(np.ascontiguousarray(emb[b])).view(torch.bfloat16).cuda()
            kp = forward(m, batch_inputs(est, blank, data['box'][b]))
            kp_orig = torch.tensor(data['kp_orig'][b], device='cuda')
            G = torch.tensor(data['G'][b], dtype=torch.float32, device='cuda')
            tang = torch.tensor(data['tang'][b], device='cuda')
            loss, pose, dist, _ = losses(T, kp, kp_orig, G, tang)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.decoder.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f'epoch {ep} step {step}/{steps} pose {pose * 100:.2f} cm distill {dist * 100:.2f} cm '
                      f'{time.time() - t0:.0f}s', flush=True)
        val(f'epoch {ep}')
    torch.save(m.decoder.state_dict(), out/'decoder.pt')
    torch.save(T.offsets.detach().cpu(), out/'offsets.pt')
    json.dump(log, open(out/'train_log.json', 'w'), indent=1)


def cmd_eval(a):
    """Original vs fine-tuned on the held-out views and MMA (the backbone runs once per crop; both decoders reuse it)."""
    out = Path(a.out)
    rows, G, tang = crops_from_feats(a.feats)
    sets = {'heldout': [i for i, r in enumerate(rows) if r[0] in TEST and r[1] in TEST_CAMS],
            'mma_new_people': [i for i, r in enumerate(rows) if r[0] == MMA]}
    if a.smoke:   # code check on development views only: the test views are never looked at before the run
        dev = sorted({r[0] for r in rows if r[0].endswith('_grappling2') and r[0] not in TEST})
        sets = {'smoke_dev': [i for i, r in enumerate(rows) if r[0] == dev[0]][:6]}
    est, m = setup()
    blank = np.zeros((2160, 3840, 3), np.uint8)
    T = Trainable(m).cuda()
    orig = {k: v.clone() for k, v in m.decoder.state_dict().items()}
    tuned = torch.load(out/'decoder.pt', map_location='cuda')
    T.offsets.data = torch.load(out/'offsets.pt').cuda()
    report = {}
    for name, views in sets.items():
        crop_rows = [(*rows[v], n) for v in views for n in range(2)]
        N, CH = len(crop_rows), 256          # chunks keep the embeddings in memory small (0.5 GB)
        kp, box = np.zeros((N, 70, 3), np.float32), np.zeros((N, 4), np.float32)
        Gc = np.stack([G[v, n] for v in views for n in range(2)])
        e0, e1 = np.zeros((N, len(CANON))), np.zeros((N, len(CANON)))
        for c0 in range(0, N, CH):
            sl = slice(c0, min(N, c0 + CH))
            emb = np.zeros((sl.stop - c0, *EMB), np.int16)
            kc, bc = kp[sl], box[sl]
            m.decoder.load_state_dict(orig)
            run_backbone(est, m, crop_rows[sl], emb, kc, bc, log_every=10 ** 9)
            data = {'box': bc, 'G': Gc[sl]}
            idx = np.arange(len(emb))
            e0[sl] = evaluate_crops(est, T, blank, idx, emb, data, use_finetuned=False)
            m.decoder.load_state_dict(tuned)
            e1[sl] = evaluate_crops(est, T, blank, idx, emb, data, use_finetuned=True)
            print(f'{name}: {sl.stop}/{N} crops', flush=True)
        t = np.stack([tang[v, n] for v in views for n in range(2)])
        np.savez(out/f'eval_{name}.npz', e_orig=e0, e_tuned=e1, tang=t, kp_orig=kp)
        tg = t[:, JOINT8]
        r = {}
        for v, e in (('original', e0), ('finetuned', e1)):
            r[v] = {'tangle_joint8_median_cm': float(np.median(e[:, JOINT8][tg]) * 100),
                    'other_joint8_median_cm': float(np.median(e[:, JOINT8][~tg]) * 100),
                    'tangle_points': int(tg.sum())}
        brng = np.random.default_rng(1)
        V = len(views)
        ratios = []
        for _ in range(1000):
            b = brng.integers(0, V, V)
            cb = np.concatenate([2 * b, 2 * b + 1])
            m_ = tg[cb]
            ratios.append(np.median(e1[cb][:, JOINT8][m_]) / np.median(e0[cb][:, JOINT8][m_]) - 1)
        r['finetuned_vs_original_tangle_95ci'] = [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))]
        report[name] = r
        print(name, json.dumps(r), flush=True)
    json.dump(report, open(out/'eval.json', 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['cache', 'train', 'eval'])
    ap.add_argument('--feats', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true', help='a few crops, one epoch (code check on development data)')
    a = ap.parse_args()
    {'cache': cmd_cache, 'train': cmd_train, 'eval': cmd_eval}[a.cmd](a)


if __name__ == '__main__':
    main()
