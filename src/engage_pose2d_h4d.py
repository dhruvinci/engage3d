"""Independent 2D keypoints for the engagement benchmark: ViTPose+ base (the production 2D stage's model) on each
athlete's ground-truth box, for every view engage_bench_s3d.py predicted (runs in .venv-pose, GPU). Experimental.
No masks: in contact the 2D model can take the partner's limbs, as it would without ground truth.

Writes <dir>/<seq>/<cam>_vitpose.npz: frames [T], kp [T,2,17,2] (COCO-17, full-image pixels), score [T,2,17];
athlete order = sorted subject names, as in engage_bench_s3d.py.
  ../.venv-pose/bin/python engage_pose2d_h4d.py DIR
"""
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autolabel_pose2d import estimate, load_model  # noqa: E402
from engage_bench_s3d import seq_root  # noqa: E402
from tangle_h4d import Sequence  # noqa: E402

BATCH = 8


def main():
    d = Path(sys.argv[1])
    processor, model = None, None
    for p in sorted(glob.glob(f'{d}/*/cam??.npz')):
        dst = Path(p[:-4] + '_vitpose.npz')
        if dst.exists():
            continue
        if model is None:
            processor, model = load_model()
        sname, cam = p.split('/')[-2], Path(p).stem
        seq = Sequence(seq_root(sname))
        frames = np.load(p)['frames']
        kp = np.zeros((len(frames), 2, 17, 2), np.float32)
        sc = np.zeros((len(frames), 2, 17), np.float32)
        t0 = time.time()
        for s in range(0, len(frames), BATCH):
            fs = frames[s:s + BATCH]
            imgs, boxes = [], []
            for f in fs:
                b = np.load(seq.root/'processed_data/bbox'/cam/f'{f:05d}.npy', allow_pickle=True).item()
                with Image.open(seq.image_path(cam, int(f))) as im:
                    imgs.append(im.convert('RGB'))
                boxes.append([[float(v[0]), float(v[1]), float(v[2] - v[0]), float(v[3] - v[1])]
                              for v in (np.asarray(b[w], float).ravel()[:4] for w in sorted(b))])
            for j, (k, c) in enumerate(estimate(processor, model, imgs, boxes)):
                kp[s + j], sc[s + j] = k, c
        np.savez_compressed(dst, frames=frames, kp=kp, score=sc)
        print(f'{sname}/{cam}: {len(frames)} frames {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
