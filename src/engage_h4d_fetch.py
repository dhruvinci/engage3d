"""Pull selected sequences / cameras / frames out of Harmony4D zips on Hugging Face without downloading them (CPU,
.venv-depth). Experimental.

The training split holds new athletes (e.g. `train/02_grappling.zip`, 44 GB: the only grappling pair besides the
test split's). Zip members are read by HTTP range requests through `HfFileSystem`, so only the needed bytes are
fetched: per sequence its calibration (`colmap/workplace/{cameras,images}.txt` and the Aria transforms), and for every
--step-th ground-truth frame its ground-truth bodies and, for the chosen cameras, the image and the person boxes (so
downstream scripts run with --step 1 on what was fetched). The training split has no `scale.npy` (the test split's
world -> COLMAP similarity); it is written from `aria_from_colmap_transforms.pkl['aria01']`, which has the same form
(checked on 051_grappling: ground-truth joints project inside their own person boxes in 40/40 person-views). Files land in the
benchmark tree (`~/Code/shared/datasets/harmony4d/benchmark-4cam/<activity>/<seq>/...`, alias
`kakashi-research/external/harmony4d`), at the paths the benchmark scripts read. Existing files are skipped.

  ../.venv-depth/bin/python engage_h4d_fetch.py --zip train/02_grappling.zip --cams cam01,cam06,cam11,cam16 \
      [--step 4] [--threads 8] [SEQ ...]   (no SEQ: every sequence in the zip)
"""
import argparse
import os
import re
import sys
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_bench_s3d import H4D   # noqa: E402

REPO = 'datasets/Jyun-Ting/Harmony4D'


def open_zip(name):
    from huggingface_hub import HfFileSystem
    return zipfile.ZipFile(HfFileSystem().open(f'{REPO}/{name}', 'rb', block_size=2**18))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', required=True)
    ap.add_argument('--cams', required=True)
    ap.add_argument('--step', type=int, default=4)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('seqs', nargs='*')
    a = ap.parse_args()
    cams = a.cams.split(',')
    z0 = open_zip(a.zip)
    names = z0.namelist()
    act = names[0].split('/')[0]
    by_seq = defaultdict(list)
    for n in names:
        p = n.split('/')
        if len(p) > 2 and not n.endswith('/'):
            by_seq[p[1]].append(n)
    seqs = a.seqs or sorted(by_seq)
    jobs = []
    for s in seqs:
        mem = by_seq[s]
        gt = sorted(int(Path(n).stem) for n in mem if re.search(r'/processed_data/smpl/\d+\.npy$', n))
        keep = set(gt[::a.step])
        want = [n for n in mem if re.search(r'/colmap/workplace/(cameras\.txt|images\.txt|scale\.npy|\w+_transforms\.pkl)$', n)
                or (re.search(r'/processed_data/smpl/(\d+)\.npy$', n) and int(Path(n).stem) in keep)]
        for n in mem:
            m = re.search(r'/exo/(cam\d+)/images/(\d+)\.jpg$', n) or re.search(r'/processed_data/bbox/(cam\d+)/(\d+)\.npy$', n)
            if m and m.group(1) in cams and int(m.group(2)) in keep:
                want.append(n)
        jobs += [n for n in want if not (H4D/n).exists()]
    print(f'{act}: {len(seqs)} sequences, {len(jobs)} files to fetch', flush=True)
    chunks = [jobs[k::a.threads] for k in range(a.threads)]

    def fetch(chunk):
        z = open_zip(a.zip)
        for n in chunk:
            dst = H4D/n
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(dst.suffix + '.part')
            tmp.write_bytes(z.read(n))
            tmp.rename(dst)          # never leave a half-written file at the real path
        return len(chunk)
    t0 = time.time()
    with ThreadPoolExecutor(a.threads) as pool:
        n = sum(pool.map(fetch, chunks))
    print(f'fetched {n} files in {time.time() - t0:.0f}s', flush=True)
    import pickle
    import numpy as np
    for s in seqs:
        w = H4D/act/s/'colmap/workplace'
        if not (w/'scale.npy').exists() and (w/'aria_from_colmap_transforms.pkl').exists():
            np.save(w/'scale.npy', np.asarray(pickle.load(open(w/'aria_from_colmap_transforms.pkl', 'rb'))['aria01'], np.float64))


if __name__ == '__main__':
    main()
