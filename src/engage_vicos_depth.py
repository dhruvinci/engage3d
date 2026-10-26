"""Combined depth (depth map + SAM 3D Body) on ViCoS ground work (runs in .venv-depth, GPU). Experimental.

Pre-registration: docs/engagement/results/vicos_depth_prereg.md. Input: an engage_vicos_s3d.py output
(`vicos_s3d.npz`, SAM 3D Body bodies in the model's default camera: focal sqrt(W^2 + H^2), principal point at the
image centre). Per frame, MoGe-2 runs on the image with that camera's field of view, both bodies are rendered with it,
and each athlete's depth offset is the median of (map - render) over its rendered pixels minus a band around the
other body (as engage_depth_probe.py's `pred` mask). Each body moves along its ray by --weight x (offset - mean
offset): weight 1 trusts the map's gap, weight 0 keeps SAM 3D Body's.
Writes <out>/{blend,rel}/vicos_s3d.npz (weights --weight and 1.0; same format and fields as the input, t changed) and
<out>/views.jsonl.

  PYTORCH_ALLOC_CONF=expandable_segments:True ../.venv-depth/bin/python engage_vicos_depth.py IN/vicos_s3d.npz \
      --out DIR [--weight 0.7] [--save-maps]
"""
import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_depth_probe import FACES, load_model, render   # noqa: E402
from engage_relations import canon_from_pred              # noqa: E402

from engage_paths import VICOS   # noqa: E402
MIN_PIX = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('--out', required=True)
    ap.add_argument('--weight', type=float, default=0.7)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--save-maps', action='store_true', help='also keep each depth map and camera (out/maps/<image>.npz)')
    a = ap.parse_args()
    out = Path(a.out)
    z = dict(np.load(a.npz))
    n_all = len(z['image']) if not a.limit else a.limit
    faces = np.load(FACES)['mhr']
    run = load_model('moge')
    zf = zipfile.ZipFile(VICOS/'images.zip')
    out.mkdir(parents=True, exist_ok=True)
    t_rel, t_blend = z['t'].astype(np.float64).copy(), z['t'].astype(np.float64).copy()
    views = open(out/'views.jsonl', 'w')
    t0 = time.time()
    for i in range(n_all):
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(zf.read(str(z['image'][i]) + '.jpg'), np.uint8),
                                        cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        f = float(np.hypot(W, H))
        P = np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1]])
        dm = run(img, f)
        if a.save_maps:
            (out/'maps').mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out/'maps'/f'{z["image"][i]}.npz', depth=dm.astype(np.float16), P=P)
        Vc = [z['verts'][i, n].astype(np.float64) + z['t'][i, n] for n in range(2)]
        own, dep, sil = render([(V, faces) for V in Vc], P, W, H)
        band = max(2, round(5*max(W, H)/768))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*band + 1, 2*band + 1))
        rec = {'image': str(z['image'][i]), 'position': str(z['position'][i]), 'wh': [W, H], 'athletes': []}
        dl = []
        for n in range(2):
            pix = (own == n + 1) & np.isfinite(dm) & ~cv2.dilate(sil[1 - n].astype(np.uint8), kernel).astype(bool)
            d = float(np.median(dm[pix] - dep[pix])) if pix.sum() >= MIN_PIX else None
            dl.append(d)
            rec['athletes'].append({'delta': d, 'n_pix': int(pix.sum()),
                                    'map_med': float(np.median(dm[pix])) if pix.sum() >= MIN_PIX else None,
                                    'render_med': float(np.median(dep[pix])) if pix.sum() >= MIN_PIX else None})
        if None not in dl:
            for n in range(2):
                pel = canon_from_pred(z['kp3d'][i, n].astype(np.float64) + z['t'][i, n])['pelvis']
                u = pel/np.linalg.norm(pel)
                d = (dl[n] - np.mean(dl))/u[2]*u          # camera = virtual camera here: depth is z
                t_rel[i, n] += d
                t_blend[i, n] += a.weight*d
            rec['shift_rel'] = [dl[0] - np.mean(dl), dl[1] - np.mean(dl)]
        views.write(json.dumps(rec) + '\n')
        if i % 100 == 0:
            print(f'{i}/{n_all} {time.time() - t0:.0f}s peak GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB', flush=True)
    views.close()
    keep = np.arange(n_all)
    for name, t in (('blend', t_blend), ('rel', t_rel)):
        (out/name).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out/name/'vicos_s3d.npz', **{**{k: v[keep] for k, v in z.items()},
                                                         't': t[keep].astype(np.float32)})
    print('wrote', n_all, 'frames', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
