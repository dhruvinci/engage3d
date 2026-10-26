"""Engagement check on real BJJ ground work with PromptHMR (runs in .venv-phmr, GPU). Experimental.

The comparison model for engage_vicos_s3d.py: the same frames (its pick(), same seed) and the same boxes (from each
athlete's 2D keypoints), but both athletes in one call against one whole-image encoding with cross-person attention
on, where SAM 3D Body reconstructs each athlete from their own crop. No intrinsics are known: the model's own default.

Writes <out>/vicos_s3d.npz in engage_vicos_s3d.py's format with SMPL topology (kp3d [N,2,24,3], verts [N,2,6890,3]
relative to t [N,2,3]), so engage_vicos_score.py reads it unchanged. Athlete order: pose1, pose2.
  PYTORCH_ALLOC_CONF=expandable_segments:True ../.venv-phmr/bin/python engage_vicos_phmr.py --out DIR [--per 80]
"""
import argparse
import json
import os
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_bench_phmr import IMG_SIZE, PHMR  # noqa: E402
from engage_vicos_s3d import VICOS, box_of, pick  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--per', type=int, default=80)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    sel = pick(json.load(open(VICOS/'annotations.json')), a.per)
    print(len(sel), 'frames', flush=True)
    z = zipfile.ZipFile(VICOS/'images.zip')
    sys.path.insert(0, PHMR)
    os.chdir(PHMR)   # the repo loads data/ by relative path
    from torch.amp import autocast
    from prompt_hmr import load_model_from_folder
    from prompt_hmr.models.inference import prepare_batch
    model = load_model_from_folder('data/pretrain/phmr')
    rec = defaultdict(list)
    t0 = time.time()
    for n, x in enumerate(sel):
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(z.read(x['image'] + '.jpg'), np.uint8), cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        boxes = torch.tensor(np.concatenate([box_of(p, W, H) for p in (x['pose1'], x['pose2'])]))
        try:
            with torch.no_grad(), autocast('cuda'):
                batch = prepare_batch([{'image_cv': img, 'boxes': boxes, 'text': None, 'masks': None,
                                        'cam_int': None}], img_size=IMG_SIZE, interaction=True)
                o = model(batch, use_mean_hands=True)[0]
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        J = o['smpl_j3d'].float().cpu().numpy()
        if len(J) != 2:
            continue
        tm = o['transl'].float().cpu().numpy()
        rec['image'].append(x['image'])
        rec['position'].append(x['position'])
        rec['wh'].append((W, H))
        rec['kp3d'].append(J - tm[:, None])
        rec['verts'].append((o['smpl_vertices'].float().cpu().numpy() - tm[:, None]).astype(np.float16))
        rec['t'].append(tm)
        rec['f_used'].append(o['cam_int'][:, 0, 0].float().cpu().numpy())
        if n % 100 == 0:
            print(f'{n}/{len(sel)} {time.time() - t0:.0f}s peak GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB',
                  flush=True)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'vicos_s3d.npz', **{k: np.asarray(v) for k, v in rec.items()})
    print('wrote', len(rec['image']), 'frames', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
