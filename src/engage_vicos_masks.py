"""Demo stage 1 on real BJJ: do person masks stop SAM 3D Body rebuilding the same athlete twice? Experimental;
exploratory (not pre-registered). Same 1,426 ViCoS frames as E05 (`engage_vicos_s3d.py`).

  masks  (.venv-bmp, GPU)  each athlete's mask from BBoxMaskPose v2's SAM-pose2seg, prompted with that athlete's 3 most
                           confident ViCoS keypoints (BMPv2's own setting: no box, no negative points); pixels claimed
                           by both masks go to the higher-scoring one (BMPv2's exclusive masks)
  s3d    (.venv-s3d, GPU)  SAM 3D Body per athlete from the same box as E05 plus its mask (mask prompt), written in E05's
                           format so `engage_vicos_score.py` reports duplicates and relation rates for both runs

  ../.venv-bmp/bin/python engage_vicos_masks.py masks --ref DIR/vicos_s3d.npz --out DIR [--limit N]
  PYTORCH_ALLOC_CONF=expandable_segments:True ../.venv-s3d/bin/python engage_vicos_masks.py s3d --ref ... --out DIR
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_paths import VICOS   # noqa: E402

BMP = Path.home()/'Code/kakashi-research/external/BBoxMaskPose'
POSE2SEG = Path.home()/'.cache/torch/hub/checkpoints/SAM-pose2seg_hiera_b%2B.pt'
N_POS = 3


def annotations(images):
    want = set(images)
    return {x['image']: x for x in json.load(open(VICOS/'annotations.json')) if x['image'] in want}


def cmd_masks(a):
    sys.path.insert(0, str(BMP))
    import bboxmaskpose.sam2  # noqa: F401  (initialises SAM2's hydra config module)
    from bboxmaskpose.sam2_utils import prepare_model
    ref = np.load(a.ref, allow_pickle=False)
    images = [str(s) for s in ref['image']][:a.limit or None]
    ann = annotations(images)
    model = prepare_model(str(BMP/'bboxmaskpose/sam2/configs/sam-pose2seg/sam-pose2seg_hiera_b+.yaml'), str(POSE2SEG))
    z = zipfile.ZipFile(VICOS/'images.zip')
    bits, shapes, scores, t0 = [], [], [], time.time()
    for i, im in enumerate(images):
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(z.read(im + '.jpg'), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        model.set_image(img)
        ms, ss = [], []
        for pose in (ann[im]['pose1'], ann[im]['pose2']):
            k = np.asarray(pose, np.float32)
            top = k[np.argsort(-k[:, 2])[:N_POS]]
            m, s, _ = model.predict(point_coords=top[:, :2], point_labels=np.ones(len(top)), box=None, multimask_output=False)
            ms.append(m[0].astype(bool)); ss.append(float(s[0]))
        both = ms[0] & ms[1]
        ms[1 - int(np.argmax(ss))][both] = False                       # exclusive: overlap to the higher score
        bits.append(np.packbits(np.stack(ms), axis=None)); shapes.append(ms[0].shape); scores.append(ss)
        if i % 100 == 0:
            print(f'masks {i}/{len(images)} {time.time() - t0:.0f}s', flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lens = np.array([len(b) for b in bits])
    np.savez_compressed(out/'masks.npz', image=np.array(images), bits=np.concatenate(bits), offsets=np.r_[0, np.cumsum(lens)],
                        shapes=np.array(shapes), scores=np.array(scores))
    print('wrote', len(images), f'{time.time() - t0:.0f}s', flush=True)


def load_masks(path):
    z = np.load(path)
    out = {}
    for i, im in enumerate(z['image']):
        h, w = z['shapes'][i]
        flat = np.unpackbits(z['bits'][z['offsets'][i]:z['offsets'][i + 1]])[:2 * h * w]
        out[str(im)] = flat.reshape(2, h, w).astype(bool)
    return out


def cmd_s3d(a):
    import torch
    from autolabel_body3d import load_estimator
    from engage_vicos_s3d import box_of
    ref = np.load(a.ref, allow_pickle=False)
    masks = load_masks(Path(a.out)/'masks.npz')
    images = [str(s) for s in ref['image'] if str(s) in masks]
    ann = annotations(images)
    z = zipfile.ZipFile(VICOS/'images.zip')
    est = load_estimator()
    rec = {k: [] for k in ('image', 'position', 'wh', 'kp3d', 'verts', 't', 'f_used')}
    t0 = time.time()
    for i, im in enumerate(images):
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(z.read(im + '.jpg'), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        per = []
        for n, pose in enumerate((ann[im]['pose1'], ann[im]['pose2'])):
            m = masks[im][n]
            try:
                with torch.inference_mode():
                    o = est.process_one_image(img, bboxes=box_of(pose, W, H), masks=m[None].astype(np.uint8),
                                              use_mask=True, inference_type='body') if m.sum() > 50 else \
                        est.process_one_image(img, bboxes=box_of(pose, W, H), inference_type='body')
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
        rec['image'].append(im); rec['position'].append(ann[im]['position']); rec['wh'].append((W, H))
        for k, j in (('kp3d', 0), ('verts', 1), ('t', 2), ('f_used', 3)):
            rec[k].append([p[j] for p in per])
        if i % 100 == 0:
            print(f's3d {i}/{len(images)} {time.time() - t0:.0f}s', flush=True)
    np.savez_compressed(Path(a.out)/'vicos_s3d.npz', **{k: np.asarray(v) for k, v in rec.items()})
    print('wrote', len(rec['image']), f'{time.time() - t0:.0f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['masks', 's3d'])
    ap.add_argument('--ref', required=True, help="E05's vicos_s3d.npz (the frames to redo)")
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    {'masks': cmd_masks, 's3d': cmd_s3d}[a.cmd](a)


if __name__ == '__main__':
    main()
