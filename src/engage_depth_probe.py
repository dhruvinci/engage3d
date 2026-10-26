"""Relative depth of the two athletes from a monocular depth model (runs in .venv-depth, GPU). Experimental.

SAM 3D Body places the two athletes 14.5 cm wrong relative to each other along the line of sight (median, Harmony4D
grappling), judging each one's distance from how big it looks: an error that grows with the distance to the camera
(about 3.5% of it). A depth-map model judges depth from the whole picture, including where the two bodies meet, an
error that should grow with the gap between them instead. This measures whether it gets their relative depth better,
and whether bodies moved to agree with it score better relations.

Per view (every --every-th view of a reference engage_bench_s3d.py output: same frames, same ground-truth boxes) the
fisheye frame is resampled to a pinhole crop around both boxes (virtual camera turned toward their centre, true lens),
the depth model runs on it with the true focal length, and the true (SMPL) and predicted (MHR) bodies are rendered into
the crop (which athlete is visible at each pixel, and its depth). For athlete n, on the pixels where it is visible in
its predicted render and in a mask:
  gt    the true render (who is really visible where): the depth map's ceiling with perfect masks
  pred  the predicted render, minus a band around the other predicted body: no masks needed
delta_n = median(depth map - predicted render depth). Variants (each body moved along its camera ray):
  pred      unchanged (the baseline on this subset)
  gt_rel    by delta_n - mean(delta), gt mask: the pair's mean depth stays SAM 3D Body's, so the map's offset cancels
  pred_rel  the same with the pred mask
  pred_abs  by delta_n, pred mask: trusts the map's metric depth
Writes <out>/<model>/<variant>/<seq>/<cam>.npz in engage_bench_s3d.py's format (score with engage_bench_oracle.py) and
<out>/<model>/views.jsonl: per view and athlete, depth medians of the truth / depth map / predicted render on the same
(gt-mask) pixels, the deltas and pixel counts.

  PYTORCH_ALLOC_CONF=expandable_segments:True ../.venv-depth/bin/python engage_depth_probe.py --ref DIR --out DIR \
      --model depthpro|moge [--every 8] [--offset 0] [--save-maps] [seq ...]
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_bench_s3d import seq_root     # noqa: E402
from engage_relations import canon_from_pred  # noqa: E402
from tangle_h4d import Sequence           # noqa: E402

SIDE = 768        # longest side of the crop (pixels)
MARGIN = 1.15     # crop around the union of the two boxes
BAND = 5          # pred mask: pixels kept this far (px) from the other predicted body's silhouette
MIN_PIX = 50
AHEAD = 4
FACES = Path(__file__).resolve().parent.parent/'benchmarks/engage_h4d/depth_probe/faces.npz'
VARIANTS = ['pred', 'gt_rel', 'pred_rel', 'pred_abs']


def load_model(name):
    """f(rgb uint8 [h,w,3], focal px) -> depth along the optical axis [h,w] (metres, nan where invalid)."""
    if name == 'depthpro':
        from dataclasses import replace
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        from huggingface_hub import hf_hub_download
        cfg = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=hf_hub_download('apple/DepthPro', 'depth_pro.pt'))
        model, tf = depth_pro.create_model_and_transforms(config=cfg, device=torch.device('cuda'), precision=torch.half)
        model.eval()

        def run(img, f):
            with torch.no_grad():
                return model.infer(tf(img), f_px=torch.tensor(float(f), device='cuda'))['depth'].float().cpu().numpy()
        return run
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained('Ruicheng/moge-2-vitl-normal').cuda().eval()

    def run(img, f):
        x = torch.from_numpy(img).cuda().permute(2, 0, 1).float()/255
        with torch.no_grad():
            d = model.infer(x, fov_x=float(np.degrees(2*np.arctan(img.shape[1]/2/f))))['depth'].float().cpu().numpy()
        d[~np.isfinite(d)] = np.nan
        return d
    return run


def virtual_cam(K, D, box):
    """Rotation (camera -> virtual), pinhole matrix and size of a camera turned toward the box centre, covering it."""
    x0, y0, x1, y1 = box
    xs, ys = np.linspace(x0, x1, 9), np.linspace(y0, y1, 9)
    edge = np.array([(x, y) for x in xs for y in (y0, y1)] + [(x, y) for y in ys for x in (x0, x1)], np.float64)

    def ray(p):
        return np.c_[cv2.fisheye.undistortPoints(p.reshape(-1, 1, 2), K, D).reshape(-1, 2), np.ones(len(p))]
    z = ray(np.array([[(x0 + x1)/2, (y0 + y1)/2]]))[0]
    z /= np.linalg.norm(z)
    x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z])
    q = ray(edge) @ R.T
    q = q[:, :2]/q[:, 2:3]
    mid, half = (q.min(0) + q.max(0))/2, (q.max(0) - q.min(0))/2*MARGIN
    f = min(K[0, 0], SIDE/(2*half.max()))
    w, h = np.ceil(2*half*f).astype(int)
    return R, np.array([[f, 0, w/2 - mid[0]*f], [0, f, h/2 - mid[1]*f], [0, 0, 1]]), (int(w), int(h))


def render(meshes, P, w, h, chunk=20000, faces=False):
    """Z-buffer render of meshes [(V [N,3] virtual camera frame, F [M,3])]: owner (1-based, 0 = none), depth (nan =
    none), and each mesh's own silhouette; with faces=True also the visible face index per pixel (-1 = none)."""
    idx_all, z_all, o_all, f_all = [], [], [], []
    for n, (V, F) in enumerate(meshes):
        uv = V[:, :2]/V[:, 2:3]*P[[0, 1], [0, 1]] + P[:2, 2] - 0.5   # pixel centres at integers
        for s in range(0, len(F), chunk):
            f = F[s:s + chunk]
            a, b, c = uv[f[:, 0]], uv[f[:, 1]], uv[f[:, 2]]
            lo = np.floor(np.minimum(np.minimum(a, b), c)).astype(int)
            size = (np.ceil(np.maximum(np.maximum(a, b), c)).astype(int) - lo).max(1) + 1
            for S in (4, 8, 16, 32, 64, 128, 256):   # triangles larger than 256 px (a body almost on the lens) are skipped
                sel = (size <= S) & (size > S//2) if S > 4 else size <= S
                if not sel.any():
                    continue
                oy, ox = np.mgrid[:S, :S]
                px = lo[sel, None, :] + np.stack([ox.ravel(), oy.ravel()], 1)[None]          # [T,S*S,2]
                A, B, C = a[sel, None], b[sel, None], c[sel, None]
                area = (B[..., 0] - A[..., 0])*(C[..., 1] - A[..., 1]) - (B[..., 1] - A[..., 1])*(C[..., 0] - A[..., 0])
                ok_area = np.abs(area) > 1e-9
                area = np.where(ok_area, area, 1.0)
                w0 = ((B[..., 0] - px[..., 0])*(C[..., 1] - px[..., 1]) - (B[..., 1] - px[..., 1])*(C[..., 0] - px[..., 0]))/area
                w1 = ((C[..., 0] - px[..., 0])*(A[..., 1] - px[..., 1]) - (C[..., 1] - px[..., 1])*(A[..., 0] - px[..., 0]))/area
                w2 = 1 - w0 - w1
                ins = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) & ok_area & (px[..., 0] >= 0) & (px[..., 0] < w) & \
                    (px[..., 1] >= 0) & (px[..., 1] < h)
                zt = V[f[sel], 2]
                zz = w0*zt[:, None, 0] + w1*zt[:, None, 1] + w2*zt[:, None, 2]
                idx_all.append((px[..., 1]*w + px[..., 0])[ins])
                z_all.append(zz[ins])
                o_all.append(np.full(int(ins.sum()), n + 1, np.uint8))
                f_all.append(np.broadcast_to((s + np.nonzero(sel)[0])[:, None], ins.shape)[ins])
    idx, zz, oo = np.concatenate(idx_all), np.concatenate(z_all), np.concatenate(o_all)
    sil = [np.zeros(h*w, bool) for _ in meshes]
    for n in range(len(meshes)):
        sil[n][idx[oo == n + 1]] = True
    order = np.lexsort((zz, idx))
    first = np.unique(idx[order], return_index=True)[1]
    own, dep = np.zeros(h*w, np.uint8), np.full(h*w, np.nan, np.float32)
    own[idx[order][first]] = oo[order][first]
    dep[idx[order][first]] = zz[order][first]
    if faces:
        fid = np.full(h*w, -1, np.int64)
        fid[idx[order][first]] = np.concatenate(f_all)[order][first]
        return own.reshape(h, w), dep.reshape(h, w), [s.reshape(h, w) for s in sil], fid.reshape(h, w)
    return own.reshape(h, w), dep.reshape(h, w), [s.reshape(h, w) for s in sil]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True, help='engage_bench_s3d.py output')
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', required=True, choices=['depthpro', 'moge'])
    ap.add_argument('--every', type=int, default=8)
    ap.add_argument('--offset', type=int, default=0, help='first view index (a different --offset gives unseen views)')
    ap.add_argument('--save-maps', action='store_true', help='keep each crop\'s depth map and camera in <out>/maps/')
    ap.add_argument('--maps-only', action='store_true', help='with --save-maps: only the maps (no variant files, no views.jsonl rows)')
    ap.add_argument('seqs', nargs='*')
    a = ap.parse_args()
    out = Path(a.out)/a.model
    files = sorted(p for p in glob.glob(f'{a.ref}/*/cam??.npz') if not a.seqs or p.split('/')[-2] in a.seqs)
    faces = np.load(FACES)
    Fs, Fm = faces['smpl'], faces['mhr']
    run = load_model(a.model)
    pool = ThreadPoolExecutor(2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*BAND + 1, 2*BAND + 1))
    out.mkdir(parents=True, exist_ok=True)
    views = open(out/'views.jsonl', 'a')
    t0, nv = time.time(), 0
    for p in files:
        sname, cam = p.split('/')[-2], Path(p).stem
        if (out/VARIANTS[-1]/sname/f'{cam}.npz').exists() or (a.maps_only and (out/'maps'/sname/cam).exists()):
            continue
        seq = Sequence(seq_root(sname))
        Rc, tc, cid = seq.poses[cam]
        _, _, Pc = seq.cams[cid]
        K = np.array([[Pc[0], 0, Pc[2]], [0, Pc[1], Pc[3]], [0, 0, 1]])
        Dk = np.array(Pc[4:8]).reshape(1, 4)
        A4 = seq.world_to_colmap
        sc = np.cbrt(np.linalg.det(A4[:3, :3]))
        tocam = lambda X: ((X @ A4[:3, :3].T + A4[:3, 3]) @ Rc.T + tc)/sc
        z = dict(np.load(p))
        keep = np.arange(a.offset, len(z['frames']), a.every)
        load = lambda f: cv2.cvtColor(cv2.imread(str(seq.image_path(cam, int(f)))), cv2.COLOR_BGR2RGB)
        imgs = [pool.submit(load, z['frames'][i]) for i in keep[:AHEAD]]   # bounded read-ahead: a 4K frame is 25 MB
        rec = {v: {'frames': [], 't': []} for v in VARIANTS}
        for j, i in enumerate(keep):
            f = int(z['frames'][i])
            img = imgs[j].result()
            imgs[j] = None
            if j + AHEAD < len(keep):
                imgs.append(pool.submit(load, z['frames'][keep[j + AHEAD]]))
            b = np.load(seq.root/'processed_data/bbox'/cam/f'{f:05d}.npy', allow_pickle=True).item()
            bx = np.array([np.asarray(b[w], np.float64).ravel()[:4] for w in sorted(b)])
            Rv, Pv, (w, h) = virtual_cam(K, Dk, np.r_[bx[:, :2].min(0), bx[:, 2:].max(0)])
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, Dk, Rv, Pv, (w, h), cv2.CV_32FC1)
            dm = run(cv2.remap(img, m1, m2, cv2.INTER_LINEAR), Pv[0, 0])
            if a.save_maps:
                mp = out/'maps'/sname/cam/f'{f:05d}.npz'
                mp.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(mp, depth=dm.astype(np.float16), R=Rv, P=Pv, i=i)
            if a.maps_only:   # the map is all that is needed: skip the renders and statistics
                nv += 1
                continue
            d = np.load(seq.root/'processed_data/smpl'/f'{f:05d}.npy', allow_pickle=True).item()
            Vg = [tocam(d[k]['vertices']) @ Rv.T for k in sorted(d)]
            Vp_cam = [z['verts'][i, n].astype(np.float64) + z['t'][i, n] for n in range(2)]
            own_g, dep_g, _ = render([(Vg[0], Fs), (Vg[1], Fs)], Pv, w, h)
            own_p, dep_p, sil_p = render([(V @ Rv.T, Fm) for V in Vp_cam], Pv, w, h)
            fin = np.isfinite(dm)
            r = {'seq': sname, 'cam': cam, 'frame': f, 'crop': [w, h], 'f': float(Pv[0, 0]), 'athletes': []}
            delta = {}
            for n in range(2):
                pg = (own_g == n + 1) & (own_p == n + 1) & fin
                pp = (own_p == n + 1) & fin & ~cv2.dilate(sil_p[1 - n].astype(np.uint8), kernel).astype(bool)
                st = {'n_gt': int(pg.sum()), 'n_pred': int(pp.sum())}
                if pg.sum() >= MIN_PIX:
                    st.update(true=float(np.median(dep_g[pg])), map=float(np.median(dm[pg])),
                              s3d=float(np.median(dep_p[pg])), delta_gt=float(np.median(dm[pg] - dep_p[pg])))
                if pp.sum() >= MIN_PIX:
                    st['delta_pred'] = float(np.median(dm[pp] - dep_p[pp]))
                r['athletes'].append(st)
            for v in VARIANTS:
                key = 'delta_gt' if v == 'gt_rel' else 'delta_pred'
                dl = [s.get(key) for s in r['athletes']]
                if v == 'pred' or None in dl:
                    shift = [0.0, 0.0]
                elif v.endswith('_rel'):
                    shift = [dl[0] - np.mean(dl), dl[1] - np.mean(dl)]
                else:
                    shift = dl
                t = z['t'][i].astype(np.float64).copy()
                for n in range(2):
                    pel = canon_from_pred(z['kp3d'][i, n].astype(np.float64) + z['t'][i, n])['pelvis']
                    u = pel/np.linalg.norm(pel)
                    t[n] += shift[n]/(u @ Rv[2])*u
                rec[v]['frames'].append(i)
                rec[v]['t'].append(t)
                delta[v] = [float(s) for s in shift]
            r['shift'] = delta
            views.write(json.dumps(r) + '\n')
            nv += 1
        views.flush()
        for v in VARIANTS if not a.maps_only else []:
            ii = np.array(rec[v]['frames'], int)
            dst = out/v/sname/f'{cam}.npz'
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dst, frames=z['frames'][ii], kp3d=z['kp3d'][ii], verts=z['verts'][ii],
                                t=np.array(rec[v]['t'], np.float32))
        print(f'{sname}/{cam}: {len(keep)} views, {nv} total, {time.time() - t0:.0f}s, peak GPU '
              f'{torch.cuda.max_memory_allocated()/1e9:.2f} GB', flush=True)


if __name__ == '__main__':
    main()
