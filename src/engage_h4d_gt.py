"""GT engagement relations on the local Harmony4D sequences (external/harmony4d) -> one json line per frame. CPU, .venv-pose.
  ../.venv-pose/bin/python engage_h4d_gt.py OUT.jsonl"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_relations import relations
from engage_paths import H4D
E = str(H4D)
out = open(sys.argv[1], 'w')
for s in sorted(glob.glob(E + '/03_grappling2/*')) + sorted(glob.glob(E + '/15_mma4/*')):
    fs = sorted(glob.glob(s + '/processed_data/smpl/*.npy'))
    D = [np.load(f, allow_pickle=True).item() for f in fs]
    floor = np.percentile([min(d[p]['vertices'][:, 2].min() for p in d) for d in D], 5)
    for f, d in zip(fs, D):
        a, b = sorted(d)
        sh = np.array([0, 0, floor])
        r = relations(d[a]['joints'] - sh, d[b]['joints'] - sh, d[a]['vertices'] - sh, d[b]['vertices'] - sh)
        r.update(seq=os.path.basename(s), frame=int(os.path.basename(f)[:-4]))
        out.write(json.dumps(r) + '\n')
    print(os.path.basename(s), len(fs), flush=True)
