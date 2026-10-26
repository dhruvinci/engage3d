"""Render GT relations on Harmony4D frames: mesh overlay per athlete, the threaded loop (yellow) and chain (magenta). CPU, .venv-pose.
  ../.venv-pose/bin/python engage_render.py REL.jsonl "B.torso through A.arm_loop|..." OUT.jpg"""
import json, os, random, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tangle_h4d import Sequence
from engage_relations import LEG_LOOP, ARM_LOOP, CHAINS
from engage_paths import H4D
E = str(H4D)
R = [json.loads(l) for l in open(sys.argv[1])]
keys, out = sys.argv[2].split('|'), sys.argv[3]
random.seed(3)
tiles = []
for key in keys:
    cand = [r for r in R if key in r['threads'] and 'grappling' in r['seq']] or [r for r in R if key in r['threads']]
    for r in random.sample(cand, min(3, len(cand))):
        seq = Sequence(f"{E}/{'03_grappling2' if 'grappling' in r['seq'] else '15_mma4'}/{r['seq']}")
        d = np.load(seq.root/'processed_data/smpl'/f"{r['frame']:05d}.npy", allow_pickle=True).item()
        a, b = sorted(d)
        cam = sorted(seq.poses)[0]
        img = cv2.imread(str(seq.image_path(cam, r['frame'])))
        ov = img.copy()
        for who, col in ((a, (60, 60, 230)), (b, (230, 120, 40))):
            uv, z = seq.project(d[who]['vertices'], cam)
            for x, y in uv[::3].astype(int):
                cv2.circle(ov, (x, y), 3, col, -1)
        img = cv2.addWeighted(img, 0.55, ov, 0.45, 0)
        other, _, owner_loop = key.split(' ')[0], None, key.split(' ')[-1]
        chain_who, chain = key.split(' ')[0].split('.')
        loop_who, loop = owner_loop.split('.')
        P = {'A': a, 'B': b}
        lj, _ = seq.project(d[P[loop_who]]['joints'][(LEG_LOOP if loop == 'leg_loop' else ARM_LOOP)], cam)
        cj, _ = seq.project(d[P[chain_who]]['joints'][CHAINS[chain]], cam)
        cv2.polylines(img, [lj.astype(np.int32)], True, (0, 255, 255), 6)
        cv2.polylines(img, [cj.astype(np.int32)], False, (255, 0, 255), 6)
        ys, xs = np.nonzero(ov.sum(2) != img.sum(2))
        h, w = img.shape[:2]
        allp = np.r_[lj, cj]; cx, cy = allp.mean(0).astype(int); half = int(max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1]))*0.9) + 150
        x1, y1, x2, y2 = max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half)
        t = cv2.resize(img[y1:y2, x1:x2], (420, 420))
        cv2.rectangle(t, (0, 0), (420, 46), (0, 0, 0), -1)
        cv2.putText(t, f"{r['seq']} #{r['frame']} {cam}", (6, 18), 0, 0.5, (255, 255, 255), 1)
        cv2.putText(t, f"{key} @ {r['threads'][key]}", (6, 38), 0, 0.45, (0, 255, 255), 1)
        tiles.append(t)
rows = [np.hstack(tiles[i:i + 3] + [np.zeros_like(tiles[0])]*(3 - len(tiles[i:i + 3]))) for i in range(0, len(tiles), 3)]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
