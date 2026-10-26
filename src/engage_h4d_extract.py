"""Unpack more cameras of Harmony4D test-split sequences into the benchmark tree (CPU, any venv with numpy). Experimental.

The benchmark tree (`~/Code/shared/datasets/harmony4d/benchmark-4cam`, alias `kakashi-research/external/harmony4d`)
holds 4 of each sequence's 20 cameras. This adds other cameras straight from the test-split zips
(`~/Code/shared/datasets/harmony4d/test-split-full/test/*.zip`): for each sequence and camera, the images of every
--step-th ground-truth frame and that camera's person boxes, at the paths the benchmark scripts read
(`<seq>/exo/<cam>/images/<frame>.jpg`, `<seq>/processed_data/bbox/<cam>/<frame>.npy`). Existing files are left alone.

  python engage_h4d_extract.py --cams cam02,cam03 [--step 2] SEQ [SEQ ...]
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_bench_s3d import H4D, seq_root   # noqa: E402

from engage_paths import H4D_ZIPS as ZIPS   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cams', required=True)
    ap.add_argument('--step', type=int, default=2)
    ap.add_argument('seqs', nargs='+')
    a = ap.parse_args()
    cams = a.cams.split(',')
    zips = {}
    for sname in a.seqs:
        root = seq_root(sname)
        act = root.parent.name                       # e.g. 03_grappling2
        if act not in zips:
            zips[act] = zipfile.ZipFile(ZIPS/f'{act}.zip')
        z = zips[act]
        frames = sorted(int(f.stem) for f in (root/'processed_data/smpl').glob('*.npy'))[::a.step]
        n_img = n_box = 0
        for cam in cams:
            for f in frames:
                for rel in (f'exo/{cam}/images/{f:05d}.jpg', f'processed_data/bbox/{cam}/{f:05d}.npy'):
                    dst = root/rel
                    if dst.exists():
                        continue
                    try:
                        data = z.read(f'{act}/{sname}/{rel}')
                    except KeyError:
                        continue
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(data)
                    n_img += rel.endswith('.jpg')
                    n_box += rel.endswith('.npy')
        print(f'{sname}: {len(frames)} frames x {len(cams)} cameras: {n_img} images, {n_box} boxes added', flush=True)


if __name__ == '__main__':
    main()
