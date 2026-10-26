"""E12 part C: can MapAnything turn ViCoS's synchronised phones into usable 3D? (GPU, .venv-mapany). Experimental.
Pre-registration: `docs/engagement/results/vicos_sync_prereg.md` (C1).

For 60 three-phone moments (20 from each three-phone session, fixed seed) the three phone images of one moment go
through MapAnything together (Apache-licensed checkpoint `facebook/map-anything-apache`), which returns per-pixel 3D
points in one shared, metric world frame. Each athlete's 3D centre per phone = median of the points at that athlete's
ViCoS 2D keypoints (confidence >= 0.5). If the three phones see a coherent scene, (i) the distance between the two
athletes is the same from each phone and (ii) each athlete's centre lands in the same place.

  PYTORCH_ALLOC_CONF=expandable_segments:True ../.venv-mapany/bin/python engage_vicos_mapany.py --sync DIR
      [--blend DIR/depth/blend/vicos_s3d.npz]   -> DIR/mapany.json
"""
import argparse
import json
import random
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engage_paths import VICOS   # noqa: E402

THREE = ['00+01+02', '03+04+05', '06+07+08']
PER, SEED, CONF = 20, 0, 0.5


def pick(moments):
    rng = random.Random(SEED)
    out = []
    for s in THREE:
        ms = [m for m in moments if m['session'] == s]
        rng.shuffle(ms)
        out += ms[:PER]
    return out


def pixel_maps(paths):
    """Per image, the map from original pixels to MapAnything's processed pixels, as load_images does it: one target
    size per call (closest to the mean aspect ratio, 518 set), each image scaled by max(target / size) to cover it
    (size floored), then centre-cropped (mapanything.utils.cropping, no intrinsics given)."""
    import PIL.Image
    from mapanything.utils.image import find_closest_aspect_ratio
    sizes = [PIL.Image.open(p).size for p in paths]
    tw, th = find_closest_aspect_ratio(np.mean([w / h for w, h in sizes]), 518)
    maps = []
    for W, H in sizes:
        s = max(tw / W, th / H) + 1e-8
        w2, h2 = int(np.floor(W * s)), int(np.floor(H * s))
        maps.append((w2 / W, h2 / H, -((w2 - tw) // 2), -((h2 - th) // 2)))
    return maps


def centres(pts, kps, pmap):
    """[2,3] each athlete's median 3D point at its confident keypoints, or None."""
    h, w = pts.shape[:2]
    sx, sy, ox, oy = pmap
    out = []
    for kp in kps:
        k = np.asarray(kp, float)
        k = k[k[:, 2] >= CONF]
        if len(k) < 3:
            return None
        uv = np.round(k[:, :2] * np.array([sx, sy]) + np.array([ox, oy])).astype(int)
        uv[:, 0] = uv[:, 0].clip(0, w - 1)
        uv[:, 1] = uv[:, 1].clip(0, h - 1)
        out.append(np.median(pts[uv[:, 1], uv[:, 0]], 0))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sync', required=True)
    ap.add_argument('--blend', default='')
    a = ap.parse_args()
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images
    sync = Path(a.sync)
    moments = pick(json.load(open(sync/'moments.json')))
    ann = {x['image']: x for x in json.load(open(VICOS/'annotations.json'))}
    zf = zipfile.ZipFile(VICOS/'images.zip')
    model = MapAnything.from_pretrained('facebook/map-anything-apache').cuda().eval()
    rows, t0 = [], time.time()
    torch.cuda.reset_peak_memory_stats()
    for k, m in enumerate(moments):
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for img in m['images']:
                p = Path(td)/f'{img}.jpg'
                p.write_bytes(zf.read(f'{img}.jpg'))
                paths.append(str(p))
            pmaps = pixel_maps(paths)
            views = load_images(paths)
            with torch.no_grad():
                pred = model.infer(views, memory_efficient_inference=True, minibatch_size=1, use_amp=True,
                                   amp_dtype='bf16')
        per_view = []
        for v, img in enumerate(m['images']):
            pts = pred[v]['pts3d'][0].float().cpu().numpy()
            x = ann[img]
            per_view.append(centres(pts, [x['pose1'], x['pose2']], pmaps[v]))
        row = {'session': m['session'], 'frame': m['frame'], 'position': m['position'], 'images': m['images']}
        if all(c is not None for c in per_view):
            C = np.stack(per_view)                                  # [view, athlete, 3]
            d = np.linalg.norm(C[:, 0] - C[:, 1], axis=1)
            spread_d = float(d.max() - d.min())
            spread_c = float(max(np.linalg.norm(C[i, n] - C[j, n]) for n in range(2)
                                 for i in range(3) for j in range(i + 1, 3)))
            row.update(dist_per_view=d.tolist(), dist_spread=spread_d, centre_spread=spread_c,
                       coherent=bool(spread_d < 0.15 and spread_c < 0.25))
        rows.append(row)
        print(f'{k + 1}/{len(moments)} {time.time() - t0:.0f}s peak GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB '
              f'{row.get("dist_spread", "n/a")}', flush=True)
    ok = [r for r in rows if 'coherent' in r]
    rep = {'moments': len(rows), 'with_keypoints_in_all_views': len(ok),
           'coherent_share': float(np.mean([r['coherent'] for r in ok])) if ok else None,
           'median_dist_spread_m': float(np.median([r['dist_spread'] for r in ok])) if ok else None,
           'median_centre_spread_m': float(np.median([r['centre_spread'] for r in ok])) if ok else None,
           'peak_gpu_gb': torch.cuda.max_memory_allocated()/1e9, 'seconds': time.time() - t0}
    if a.blend and ok:
        from scipy.stats import spearmanr
        z = np.load(a.blend)
        idx = {str(im): i for i, im in enumerate(z['image'])}
        ours, theirs = [], []
        for r in ok:
            for v, img in enumerate(r['images']):
                if img in idx:
                    i = idx[img]
                    c = [z['kp3d'][i, n].astype(np.float64).mean(0) + z['t'][i, n] for n in range(2)]
                    ours.append(np.linalg.norm(c[0] - c[1]))
                    theirs.append(r['dist_per_view'][v])
        rep['spearman_dist_vs_blend'] = float(spearmanr(ours, theirs).correlation)
    json.dump({'summary': rep, 'moments': rows}, open(sync/'mapany.json', 'w'), indent=1)
    print(json.dumps(rep), flush=True)


if __name__ == '__main__':
    main()
