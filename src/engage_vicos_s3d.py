"""Engagement check on real BJJ ground work: SAM 3D Body on ViCoS frames (runs in .venv-s3d, GPU). Experimental.

ViCoS (Hudovernik & Skocaj 2022) labels 120,279 sparring frames with 18 position classes, assigned by people; each
frame carries auto-detected 2D poses of both athletes when the detector found them. Pre-registration:
docs/engagement/results/vicos_prereg.md. Up to --per frames per class with both poses, spread over the 16 video
streams and >= 30 frames apart within a stream; each athlete's box comes from its 2D keypoints (padded 20%); SAM 3D
Body runs per athlete with its default focal length (no intrinsics are known). Images are read from images.zip.

Writes <out>/vicos_s3d.npz: image ids, positions, kp3d [N,2,70,3] and verts [N,2,n,3] (float16, root-relative),
t [N,2,3] (the model's own camera frame), f_used [N,2]. Athlete order: pose1, pose2.
  PYTORCH_ALLOC_CONF=expandable_segments:True ../.venv-s3d/bin/python engage_vicos_s3d.py --out DIR [--per 80]
"""
import argparse
import json
import os
import random
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autolabel_body3d import load_estimator  # noqa: E402

from engage_paths import VICOS   # noqa: E402


def pick(ann, per, seed=0):
    rng = random.Random(seed)
    by = defaultdict(lambda: defaultdict(list))
    for x in ann:
        if x.get('pose1') and x.get('pose2'):
            by[x['position']][x['image'][:2]].append(x)
    out = []
    for pos, vids in sorted(by.items()):
        for v in vids.values():
            rng.shuffle(v)
        chosen, taken = [], defaultdict(list)
        order = sorted(vids)
        while len(chosen) < per and any(vids[v] for v in order):
            for v in order:
                while vids[v]:
                    x = vids[v].pop()
                    if all(abs(x['frame'] - f) >= 30 for f in taken[v]):
                        chosen.append(x)
                        taken[v].append(x['frame'])
                        break
                if len(chosen) >= per:
                    break
        out += chosen
    return out


def box_of(pose, W, H, pad=0.2):
    k = np.asarray(pose, float)
    k = k[k[:, 2] > 0.3] if (k[:, 2] > 0.3).sum() >= 4 else k
    x1, y1 = k[:, :2].min(0)
    x2, y2 = k[:, :2].max(0)
    p = pad*max(x2 - x1, y2 - y1)
    return np.array([[max(0, x1 - p), max(0, y1 - p), min(W, x2 + p), min(H, y2 + p)]], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--per', type=int, default=80)
    a = ap.parse_args()
    import torch
    ann = json.load(open(VICOS/'annotations.json'))
    sel = pick(ann, a.per)
    print(len(sel), 'frames', flush=True)
    z = zipfile.ZipFile(VICOS/'images.zip')
    est = load_estimator()
    rec = defaultdict(list)
    t0 = time.time()
    for n, x in enumerate(sel):
        img = cv2.imdecode(np.frombuffer(z.read(x['image'] + '.jpg'), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        per = []
        for pose in (x['pose1'], x['pose2']):
            try:
                with torch.inference_mode():
                    o = est.process_one_image(img, bboxes=box_of(pose, W, H), inference_type='body')
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                o = None
            est.batch = est.image_embeddings = est.output = None
            if not o:
                break
            o = o[0]
            per.append((np.asarray(o['pred_keypoints_3d'], np.float32), np.asarray(o['pred_vertices'], np.float16),
                        np.asarray(o['pred_cam_t'], np.float32), float(np.asarray(o['focal_length']).ravel()[0])))
        torch.cuda.empty_cache()
        if len(per) != 2:
            continue
        rec['image'].append(x['image'])
        rec['position'].append(x['position'])
        rec['wh'].append((W, H))
        for k, i in (('kp3d', 0), ('verts', 1), ('t', 2), ('f_used', 3)):
            rec[k].append([p[i] for p in per])
        if n % 100 == 0:
            print(f'{n}/{len(sel)} {time.time() - t0:.0f}s', flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(a.out)/'vicos_s3d.npz', **{k: np.asarray(v) for k, v in rec.items()})
    print('wrote', len(rec['image']), 'frames', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
