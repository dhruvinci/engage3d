"""Arm 2 of the joint-fit pre-registration on ViCoS: SAM 3D Body refined by its own keypoint clicks (runs in .venv-s3d,
GPU). Experimental.

Pre-registration: docs/engagement/results/jointfit_prereg.md (Arm 2). Same frames and boxes as engage_vicos_s3d.py.
Per athlete, the body decoder runs once, then MAX_NUM_CLICKS times more with a keypoint click: the model's own
prompt sampler (eval mode) picks the body keypoint where its current 2D estimate is furthest from that athlete's 2D
pose (the dataset's own detections, COCO-17 mapped to MHR70, confidence >= 0.5, inside the crop) and the decoder is
re-run with every click so far, as in training. Limbs and the whole body may change, unlike the rigid Arm 1 fit.

Writes <out>/vicos_s3d.npz in engage_vicos_s3d.py's format, plus clicks [N,2,MAX_NUM_CLICKS] (MHR70 index of each
click, -1 = none / dummy).
  PYTORCH_ALLOC_CONF=expandable_segments:True ../.venv-s3d/bin/python engage_vicos_s3d_clicks.py --out DIR [--per 80]
      [--ids FILE]   (E12: a fixed list of image ids instead of the per-class pick)
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autolabel_body3d import load_estimator  # noqa: E402
from engage_vicos_s3d import VICOS, box_of, pick  # noqa: E402

COCO_TO_MHR = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]   # as engage_jointfit.py (not imported: it pins BLAS threads)


def refine(est, img, box, pose):
    """SAM 3D Body on one athlete, then its click iterations with that athlete's 2D pose."""
    import torch
    from sam_3d_body.data.utils.prepare_batch import prepare_batch
    from sam_3d_body.utils import recursive_to
    m = est.model
    batch = recursive_to(prepare_batch(img, est.transform, box, None, None), 'cuda')
    m._initialize_batch(batch)
    k = np.asarray(pose, np.float32)
    full = np.zeros((1, 70, 2), np.float32)
    conf = np.zeros((1, 70, 1), np.float32)
    full[0, COCO_TO_MHR] = k[:, :2]
    conf[0, COCO_TO_MHR, 0] = k[:, 2]
    with torch.inference_mode():
        out = m.forward_step(batch, decoder_type='body')
        crop = m._full_to_crop(batch, torch.from_numpy(full).cuda())
        batch['keypoints_2d'] = torch.cat([crop, torch.from_numpy(conf).cuda()], -1).view(1, 1, 70, 3)
        prev, clicks = [], []
        for _ in range(m.max_num_clicks):
            out, p = m._one_prompt_iter(batch, out, prev, None)
            prev.append(p)
            clicks.append(int(p[0, 0, 2]))
    o = recursive_to(recursive_to(out['mhr'], 'cpu'), 'numpy')
    # the click's third value is a prompt index, not an MHR70 index: map back
    inv = {v: kk for kk, v in m.keypoint_prompt_sampler.prompt_keypoints.items()}
    return (np.asarray(o['pred_keypoints_3d'][0], np.float32), np.asarray(o['pred_vertices'][0], np.float16),
            np.asarray(o['pred_cam_t'][0], np.float32), float(np.asarray(o['focal_length'][0]).ravel()[0]),
            [inv.get(c, -1) for c in clicks])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--per', type=int, default=80)
    ap.add_argument('--ids', default='', help='a file of image ids (one per line) to run instead of pick() (E12)')
    a = ap.parse_args()
    import torch
    ann = json.load(open(VICOS/'annotations.json'))
    if a.ids:
        by = {x['image']: x for x in ann}
        sel = [by[s.strip()] for s in open(a.ids) if s.strip()]
    else:
        sel = pick(ann, a.per)
    print(len(sel), 'frames', flush=True)
    z = zipfile.ZipFile(VICOS/'images.zip')
    est = load_estimator()
    # the training-time click loop (_one_prompt_iter) passes full_output=, which forward_decoder no longer takes
    decoder = est.model.forward_decoder
    est.model.forward_decoder = lambda *args, full_output=None, **kw: decoder(*args, **kw)
    print('clicks per athlete:', est.model.max_num_clicks, flush=True)
    rec = defaultdict(list)
    t0 = time.time()
    for n, x in enumerate(sel):
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(z.read(x['image'] + '.jpg'), np.uint8), cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        per = []
        for pose in (x['pose1'], x['pose2']):
            try:
                per.append(refine(est, img, box_of(pose, W, H), pose))
            except torch.OutOfMemoryError:
                break
            finally:
                torch.cuda.empty_cache()
        if len(per) != 2:
            continue
        rec['image'].append(x['image'])
        rec['position'].append(x['position'])
        rec['wh'].append((W, H))
        for k, i in (('kp3d', 0), ('verts', 1), ('t', 2), ('f_used', 3), ('clicks', 4)):
            rec[k].append([p[i] for p in per])
        if n % 100 == 0:
            print(f'{n}/{len(sel)} {time.time() - t0:.0f}s', flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(a.out)/'vicos_s3d.npz', **{k: np.asarray(v) for k, v in rec.items()})
    print('wrote', len(rec['image']), 'frames', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
