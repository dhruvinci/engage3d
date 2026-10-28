"""L1 stage 5: embrace / arm-hold episodes over time, predicted (SAM 3D Body + combined depth) vs truth (Harmony4D)."""
import json
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_relations import CANON, relations
from engage_temporal import load_tracks
from engage_vicos_timeline import segments, transitions, match_events, segmental_f1, min_duration

def held(A, B):
    thr = relations({k: A[i] for i, k in enumerate(CANON)}, {k: B[i] for i, k in enumerate(CANON)})['threads']
    s = {k for k, (_, m) in thr.items() if m >= 0.05}
    return any('torso through' in k and 'arm_loop' in k for k in s)

for seqs, fps_step in ((['002_hugging'], 2), (['007_ballroom2', '008_ballroom2', '009_ballroom2', '010_ballroom2'], 8)):
    P, G, _, key, tr = load_tracks('../benchmarks/engage_h4d/place_feats2', seqs, {'cam01', 'cam06', 'cam11', 'cam16'})
    tp = fp = fn = 0; T = []; S = []
    for (s, c), fr in sorted(tr.items()):
        frames = np.array(sorted(fr)); idx = [fr[f] for f in frames]
        yt = ['embrace' if held(G[i, 0], G[i, 1]) else 'apart_or_touch' for i in idx]
        yp = ['embrace' if held(P[i, 0], P[i, 1]) else 'apart_or_touch' for i in idx]
        for a, b in zip(yt, yp):
            tp += a == b == 'embrace'; fp += b == 'embrace' and a != b; fn += a == 'embrace' and a != b
        f30 = frames  # ground-truth frame numbers at 30 fps
        yp = min_duration(f30, yp, 0.5); st, sp = segments(f30, yt), segments(f30, yp)
        T.append(match_events(transitions(st), transitions(sp), 30, False)); S.append(segmental_f1(st, sp, 0.5))
        if s == '002_hugging' and c == 'cam01':
            print('  truth   :', [(round(a / 30, 1), round(b / 30, 1), l) for a, b, l in st])
            print('  predicted:', [(round(a / 30, 1), round(b / 30, 1), l) for a, b, l in sp])
    nt = sum(m['true'] for m in T); npred = sum(m['pred'] for m in T); hit = sum(m['recall'] * m['true'] for m in T)
    print(seqs[0][:12], f'frames {sum(len(f) for f in tr.values())}: embrace F1 per frame {2*tp/max(2*tp+fp+fn,1):.2f}; '
          f'start/end events within 1 s: recall {hit/max(nt,1):.2f} precision {hit/max(npred,1):.2f} ({nt} true, {npred} predicted); segment F1@0.5 {np.mean(S):.2f}')
