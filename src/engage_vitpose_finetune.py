"""E19: fine-tune ViTPose+ base on Harmony4D contact so its 2D keypoints can drive SAM 3D Body's clicks (GPU,
.venv-pose). Pre-registration: `docs/engagement/results/vitposeft_prereg.md`.

Training targets are the true joints projected through the true fisheye camera (12 body joints: shoulders, elbows,
wrists, hips, knees, ankles; the face keypoints are not trained), including joints hidden by the partner, which is what
the click loop needs inside tangles. Crops follow ViTPose's own top-down transform (box -> centre and scale, 1.25
padding, 256 x 192), heatmaps 64 x 48 with Gaussian targets (sigma 2); limb joints inside a tangle weigh 3x.
Same view split as E11: training views from 10 development sequences (contact-weighted, fixed seed), validation
from 025 / 035; tested on the E06 held-out views and MMA.

  ../.venv-pose/bin/python engage_vitpose_finetune.py cache   --feats F --out D [--smoke]
  ../.venv-pose/bin/python engage_vitpose_finetune.py train   --out D [--smoke]
  ../.venv-pose/bin/python engage_vitpose_finetune.py predict --feats F --out D [--smoke]
      2D keypoints of the released and the fine-tuned model on the test views (the pipeline's estimate() path),
      their 2D errors against the truth, and D/pose_ft_<set>.npz for engage_h4d_clicks.py --pose-file
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
from autolabel_pose2d import MODEL, estimate                         # noqa: E402
from engage_bench_s3d import fisheye, seq_root                       # noqa: E402
from engage_limb_head import MMA, TEST, TEST_CAMS, tangle_mask       # noqa: E402
from engage_relations import CANON                                   # noqa: E402
from tangle_h4d import Sequence                                      # noqa: E402

SMPL_FOR_COCO = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
COCO_CANON = {5: 'l_shoulder', 6: 'r_shoulder', 7: 'l_elbow', 8: 'r_elbow', 9: 'l_wrist', 10: 'r_wrist', 11: 'l_hip',
              12: 'r_hip', 13: 'l_knee', 14: 'r_knee', 15: 'l_ankle', 16: 'r_ankle'}
LIMB8 = [7, 8, 9, 10, 13, 14, 15, 16]          # COCO elbows, wrists, knees, ankles
IN_W, IN_H, HM_W, HM_H, SIGMA = 192, 256, 48, 64, 2.0
N_VIEWS, N_VAL_VIEWS, SEED = 8000, 500, 0
EPOCHS, BATCH, LR, WD, TANGLE_W = 3, 24, 5e-5, 0.01, 3.0
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', 8)))


def truth_2d(seq, cam, f, cache={}):
    """[2,17,3] COCO-17 pixel positions of each athlete's true joints (as engage_h4d_clicks.truth_2d)."""
    key = (seq.root, cam)
    if key not in cache:
        Rc, tc, cid = seq.poses[cam]
        _, _, Pc = seq.cams[cid]
        K = np.array([[Pc[0], 0, Pc[2]], [0, Pc[1], Pc[3]], [0, 0, 1]])
        Dk = np.array(Pc[4:8]).reshape(1, 4)
        A4 = seq.world_to_colmap
        sc = np.cbrt(np.linalg.det(A4[:3, :3]))
        cache[key] = (lambda X: ((X @ A4[:3, :3].T + A4[:3, 3]) @ Rc.T + tc)/sc, K, Dk)
    tocam, K, Dk = cache[key]
    gt = np.load(seq.root/'processed_data/smpl'/f'{f:05d}.npy', allow_pickle=True).item()
    out = np.zeros((2, 17, 3))
    for n, k in enumerate(sorted(gt)):
        uv = fisheye(tocam(np.asarray(gt[k]['joints'][:24], float)), K, Dk)
        for c, s in SMPL_FOR_COCO.items():
            out[n, c] = [uv[s, 0], uv[s, 1], 1.0]
    return out


def warp_for(box_xywh):
    """ViTPose's top-down crop: the 2x3 matrix from image pixels to the 192 x 256 input."""
    from transformers.models.vitpose.image_processing_vitpose import box_to_center_and_scale, get_warp_matrix
    c, s = box_to_center_and_scale(box_xywh, image_width=IN_W, image_height=IN_H)
    return get_warp_matrix(0, c * 2.0, np.array([IN_W - 1.0, IN_H - 1.0]), s * 200.0)


def rows_from_feats(feats):
    rows, G = [], []
    for p in sorted(glob.glob(f'{feats}/*__cam??.npz')):
        seq, cam = Path(p).stem.split('__')
        d = np.load(p)
        if 'tok' not in d.files:      # the same views as E10 / E11
            continue
        rows += [(seq, cam, int(f)) for f in d['frame']]
        G.append(d['G'])
    G = np.concatenate(G).astype(np.float64)
    return rows, G, tangle_mask(G)


def coco_tangle(tang_view):
    """[2,19] canonical tangle mask -> [2,17] for the COCO joints."""
    out = np.zeros((2, 17), bool)
    for c, name in COCO_CANON.items():
        out[:, c] = tang_view[:, CANON.index(name)]
    return out


def cmd_cache(a):
    rows, G, tang = rows_from_feats(a.feats)
    seqs = np.array([r[0] for r in rows])
    dev = sorted({s for s in seqs if s.endswith('_grappling2') and s not in TEST})
    val_seqs = dev[::6]
    rng = np.random.default_rng(SEED)

    def select(names, n):
        idx = np.where(np.isin(seqs, list(names)))[0]
        w = np.where(tang[idx].any((1, 2)), 3.0, 1.0)
        return np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False, p=w / w.sum()))
    tr = select([s for s in dev if s not in val_seqs], 40 if a.smoke else N_VIEWS)
    va = select(val_seqs, 8 if a.smoke else N_VAL_VIEWS)
    views = np.concatenate([tr, va])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    N = 2 * len(views)
    crops = np.lib.format.open_memmap(out/'crops.npy', mode='w+', dtype=np.uint8, shape=(N, IN_H, IN_W, 3))
    kp = np.zeros((N, 17, 2), np.float32)
    valid = np.zeros((N, 17), np.float32)
    tg = np.zeros((N, 17), bool)
    cache, t0 = {}, time.time()
    for k, v in enumerate(views):
        s, c, f = rows[v]
        seq = cache.setdefault(s, Sequence(seq_root(s)))
        img = cv2.cvtColor(cv2.imread(str(seq.image_path(c, f))), cv2.COLOR_BGR2RGB)
        b = np.load(seq.root/'processed_data/bbox'/c/f'{f:05d}.npy', allow_pickle=True).item()
        tr2 = truth_2d(seq, c, f)
        ct = coco_tangle(tang[v])
        for n, who in enumerate(sorted(b)):
            x1, y1, x2, y2 = np.asarray(b[who], float)
            M = warp_for([x1, y1, x2 - x1, y2 - y1])
            i = 2 * k + n
            crops[i] = cv2.warpAffine(img, M, (IN_W, IN_H), flags=cv2.INTER_LINEAR)
            uv = np.c_[tr2[n, :, :2], np.ones(17)] @ M.T                     # input-crop pixels
            kp[i] = uv * np.array([(HM_W - 1) / (IN_W - 1), (HM_H - 1) / (IN_H - 1)])
            valid[i] = tr2[n, :, 2]
            tg[i] = ct[n]
        if k % 500 == 0:
            print(f'cache {k}/{len(views)} {time.time() - t0:.0f}s', flush=True)
    crops.flush()
    split = np.array(['train'] * (2 * len(tr)) + ['val'] * (2 * len(va)))
    np.savez(out/'targets.npz', kp=kp, valid=valid, tang=tg, split=split,
             rows=np.array([(*rows[v], n) for v in views for n in range(2)], dtype=object))
    json.dump({'val_seqs': val_seqs, 'train_views': int(len(tr)), 'val_views': int(len(va))},
              open(out/'cache.json', 'w'), indent=1)
    print(f'cached {N} crops {time.time() - t0:.0f}s', flush=True)


def heatmaps(kp, valid):
    """Gaussian targets [B,17,64,48]."""
    ys, xs = torch.arange(HM_H, dtype=torch.float32), torch.arange(HM_W, dtype=torch.float32)
    d2 = (xs[None, None, None, :] - kp[..., 0, None, None])**2 + (ys[None, None, :, None] - kp[..., 1, None, None])**2
    return torch.exp(-d2 / (2 * SIGMA**2)) * valid[..., None, None]


def decode(hm):
    """Argmax with a quarter-pixel shift towards the higher neighbour (as standard top-down decoding), [B,17,2]."""
    B, J, H, W = hm.shape
    flat = hm.reshape(B, J, -1).argmax(-1)
    y, x = (flat // W).float(), (flat % W).float()
    xi, yi = x.long(), y.long()
    bi, ji = torch.meshgrid(torch.arange(B), torch.arange(J), indexing='ij')
    dx = hm[bi, ji, yi, (xi + 1).clamp(max=W - 1)] - hm[bi, ji, yi, (xi - 1).clamp(min=0)]
    dy = hm[bi, ji, (yi + 1).clamp(max=H - 1), xi] - hm[bi, ji, (yi - 1).clamp(min=0), xi]
    return torch.stack([x + 0.25 * torch.sign(dx), y + 0.25 * torch.sign(dy)], -1)


def cmd_train(a):
    from transformers import AutoProcessor, VitPoseForPoseEstimation
    out = Path(a.out)
    T = dict(np.load(out/'targets.npz', allow_pickle=True))
    crops = np.load(out/'crops.npy', mmap_mode='r')
    proc = AutoProcessor.from_pretrained(MODEL)
    mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1).cuda()
    std = torch.tensor(proc.image_std).view(1, 3, 1, 1).cuda()
    model = VitPoseForPoseEstimation.from_pretrained(MODEL).cuda()
    tr, va = np.where(T['split'] == 'train')[0], np.where(T['split'] == 'val')[0]
    epochs = 1 if a.smoke else EPOCHS
    steps = epochs * (len(tr) // BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 1), pct_start=0.05)
    rng = np.random.default_rng(SEED)
    log = {'val': []}

    def batch(idx):
        x = torch.from_numpy(np.ascontiguousarray(crops[idx])).cuda().permute(0, 3, 1, 2).float() / 255.0
        return (x - mean) / std

    def val(tag):
        model.eval()
        errs, tgs = [], []
        with torch.no_grad():
            for s in range(0, len(va), 64):
                b = va[s:s + 64]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    hm = model(pixel_values=batch(b), dataset_index=torch.zeros(len(b), dtype=torch.long).cuda()).heatmaps
                p = decode(hm.float().cpu())
                errs.append(((p - torch.from_numpy(T['kp'][b])).norm(dim=-1) * 4.0).numpy())   # input-crop pixels
                tgs.append(T['tang'][b])
        e, t = np.concatenate(errs)[:, LIMB8], np.concatenate(tgs)[:, LIMB8]
        r = {'tag': tag, 'tangle_limb_median_px_crop': float(np.median(e[t])), 'other_limb_median_px_crop': float(np.median(e[~t]))}
        log['val'].append(r)
        print('val', r, flush=True)
        model.train()

    val('start')
    model.train()
    step, t0 = 0, time.time()
    for ep in range(epochs):
        perm = rng.permutation(tr)
        for s in range(0, len(perm) - BATCH + 1, BATCH):
            b = np.sort(perm[s:s + BATCH])
            kp = torch.from_numpy(T['kp'][b]).cuda()
            valid = torch.from_numpy(T['valid'][b]).cuda()
            w = valid * (1 + (TANGLE_W - 1) * torch.from_numpy(T['tang'][b].astype(np.float32)).cuda())
            with torch.autocast('cuda', dtype=torch.bfloat16):
                hm = model(pixel_values=batch(b), dataset_index=torch.zeros(len(b), dtype=torch.long).cuda()).heatmaps
            target = heatmaps(kp.cpu(), valid.cpu()).cuda()
            loss = (((hm.float() - target) ** 2).mean((-1, -2)) * w).sum() / w.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f'epoch {ep} step {step}/{steps} loss {loss.item():.6f} {time.time() - t0:.0f}s', flush=True)
        val(f'epoch {ep}')
    model.save_pretrained(out/'model')
    json.dump(log, open(out/'train_log.json', 'w'), indent=1)


def cmd_predict(a):
    """Released and fine-tuned ViTPose+ through the pipeline's estimate() on the test views; 2D errors; pose files."""
    from transformers import AutoProcessor, VitPoseForPoseEstimation
    from PIL import Image
    out = Path(a.out)
    rows, G, tang = rows_from_feats(a.feats)
    if a.smoke:
        dev = sorted({r[0] for r in rows if r[0].endswith('_grappling2') and r[0] not in TEST})
        sets = {'smoke_dev': [i for i, r in enumerate(rows) if r[0] == dev[0] and r[1] in TEST_CAMS][:6]}
    else:
        sets = {'heldout': [i for i, r in enumerate(rows) if r[0] in TEST and r[1] in TEST_CAMS],
                'mma_new_people': [i for i, r in enumerate(rows) if r[0] == MMA]}
    proc = AutoProcessor.from_pretrained(MODEL)
    models = {'released': VitPoseForPoseEstimation.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval(),
              'finetuned': VitPoseForPoseEstimation.from_pretrained(out/'model', torch_dtype=torch.float16).cuda().eval()}
    report, cache = {}, {}
    for name, views in sets.items():
        kp = {m: np.zeros((len(views), 2, 17, 2)) for m in models}
        sc = {m: np.zeros((len(views), 2, 17)) for m in models}
        tr = np.zeros((len(views), 2, 17, 3))
        for k, v in enumerate(views):
            s, c, f = rows[v]
            seq = cache.setdefault(s, Sequence(seq_root(s)))
            b = np.load(seq.root/'processed_data/bbox'/c/f'{f:05d}.npy', allow_pickle=True).item()
            img = Image.open(seq.image_path(c, f)).convert('RGB')
            boxes = [[[float(x[0]), float(x[1]), float(x[2] - x[0]), float(x[3] - x[1])]
                      for x in (np.asarray(b[w], float) for w in sorted(b))]]
            for m, model in models.items():
                kk, ss = estimate(proc, model, [img], boxes)[0]     # [2,17,2], [2,17] for this image's two boxes
                kp[m][k], sc[m][k] = kk, ss
            tr[k] = truth_2d(seq, c, f)
        t = np.stack([coco_tangle(tang[v]) for v in views])[..., LIMB8]
        r = {}
        for m in models:
            e = np.linalg.norm(kp[m] - tr[..., :2], axis=-1)[..., LIMB8]
            r[m] = {'tangle_limb_median_px': float(np.median(e[t])), 'other_limb_median_px': float(np.median(e[~t]))}
        report[name] = r
        np.savez(out/f'pose_ft_{name}.npz', rows=np.array([rows[v] for v in views], dtype=object),
                 kp=kp['finetuned'], score=sc['finetuned'], kp_released=kp['released'], score_released=sc['released'],
                 truth=tr)
        print(name, json.dumps(r), flush=True)
    json.dump(report, open(out/'predict.json', 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['cache', 'train', 'predict'])
    ap.add_argument('--feats', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    {'cache': cmd_cache, 'train': cmd_train, 'predict': cmd_predict}[a.cmd](a)


if __name__ == '__main__':
    main()
