"""Demo stage 5 on real BJJ: from per-frame position predictions to a timeline, transitions, control and words
(CPU). Experimental; exploratory (not pre-registered). Reads `engage_vicos_positions.py`'s predictions (blocked
temporal cross-validation: every frame is predicted by a model that never saw its block).

Per ViCoS stream (one phone), frames are grouped into segments of one label (a gap of more than GAP_S without both
athletes' poses splits a segment). Measures against the expert labels:
  transitions   label changes, matched one-to-one within 0.5 / 1 / 2 s (timing only, and timing plus the new label)
  segments      segmental F1 at IoU 0.10 / 0.25 / 0.50 (Lea et al.)
  control       IBJJF-inspired: a dominant position held >= 3 s scores (mount 4, back 4, side control 3 as a guard
                pass, takedown 2 when the initiator then holds a top position >= 3 s), once until the opponent scores;
                points per athlete per stream, predicted vs expert
Predicted runs shorter than MIN_SEG_S are merged into a neighbour first (positions last seconds).
ViCoS roles: mount / side control / turtle number = athlete on top; back = athlete controlling the back; takedown =
initiator; guards are worded neutrally (the dataset does not define which athlete the guard number denotes).
Also writes a side-by-side narration (expert timeline vs predicted) for one example window.

  ../.venv-depth/bin/python engage_vicos_timeline.py --pred DIR/predictions.npz --out DIR [--stream 06 --start 0 --secs 90]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FPS, GAP_S, HOLD_S = 30.0, 0.5, 3.0
MIN_SEG_S = 1.5                                                # predicted positions shorter than this merge into a neighbour
POINTS = {'mount': 4, 'back': 4, 'side_control': 3}
TOP_POSITIONS = {'mount', 'side_control', 'back', 'turtle'}
WORDS = {'standing': 'both standing', '5050_guard': '50/50 guard', 'open_guard': 'open guard', 'closed_guard': 'closed guard',
         'half_guard': 'half guard', 'side_control': 'side control', 'mount': 'mount', 'back': 'back control',
         'turtle': 'turtle', 'takedown': 'takedown attempt'}


def split(label):
    if label in ('standing', '5050_guard'):
        return label, None
    return label[:-1], int(label[-1])


def segments(frames, labels):
    """[(start_frame, end_frame, label)] runs of one label; a gap > GAP_S splits a run."""
    out, s = [], 0
    for i in range(1, len(frames) + 1):
        if i == len(frames) or labels[i] != labels[s] or frames[i] - frames[i - 1] > GAP_S * FPS:
            out.append((int(frames[s]), int(frames[i - 1]), labels[s]))
            s = i
    return out


def min_duration(frames, labels, min_s=MIN_SEG_S):
    """Relabel predicted runs shorter than min_s with the previous run's label (or the next one's at the start)."""
    labels = list(labels)
    changed = True
    while changed:
        changed = False
        segs = segments(frames, labels)
        idx = {}
        k = 0
        for j, (s, e, lab) in enumerate(segs):
            n = int(np.sum((frames >= s) & (frames <= e)))
            idx[j] = (k, k + n); k += n
        for j, (s, e, lab) in enumerate(segs):
            if (e - s + 1) / FPS >= min_s or len(segs) == 1:
                continue
            new = segs[j - 1][2] if j else segs[j + 1][2]
            if new != lab:
                a, b = idx[j]
                labels[a:b] = [new] * (b - a)
                changed = True
                break
    return labels


def transitions(segs):
    """[(frame, new_label)] at each change of label between adjacent segments."""
    return [(b[0], b[2]) for a, b in zip(segs, segs[1:]) if a[2] != b[2]]


def match_events(true, pred, tol, need_label):
    used, tp = set(), 0
    for f, lab in true:
        best = None
        for j, (g, l2) in enumerate(pred):
            if j in used or abs(g - f) > tol or (need_label and l2 != lab):
                continue
            if best is None or abs(g - f) < abs(pred[best][0] - f):
                best = j
        if best is not None:
            used.add(best); tp += 1
    p = tp / max(len(pred), 1); r = tp / max(len(true), 1)
    return {'precision': p, 'recall': r, 'f1': 2 * p * r / max(p + r, 1e-9), 'true': len(true), 'pred': len(pred)}


def segmental_f1(true, pred, thr):
    used, tp = set(), 0
    for s, e, lab in pred:
        best, bi = 0.0, None
        for j, (s2, e2, l2) in enumerate(true):
            if j in used or l2 != lab:
                continue
            inter = max(0, min(e, e2) - max(s, s2) + 1); union = max(e, e2) - min(s, s2) + 1
            if inter / union > best:
                best, bi = inter / union, j
        if bi is not None and best >= thr:
            used.add(bi); tp += 1
    p = tp / max(len(pred), 1); r = tp / max(len(true), 1)
    return 2 * p * r / max(p + r, 1e-9)


def control(segs):
    """Scoring events [(frame, athlete, points, why)] and control seconds per athlete, IBJJF-inspired."""
    events, secs = [], defaultdict(float)
    scored = {1: set(), 2: set()}                                # an athlete scores a position once until the other scores
    for i, (s, e, lab) in enumerate(segs):
        base, who = split(lab)
        dur = (e - s + 1) / FPS
        if base in TOP_POSITIONS and who:
            secs[who] += dur
        if base in POINTS and who and dur >= HOLD_S and base not in scored[who]:
            events.append((s + int(HOLD_S * FPS), who, POINTS[base], WORDS[base]))
            scored[who].add(base); scored[3 - who].clear()
        if base == 'takedown' and who:
            for s2, e2, l2 in segs[i + 1:i + 3]:
                b2, w2 = split(l2)
                if w2 == who and b2 in TOP_POSITIONS and (e2 - s2 + 1) / FPS >= HOLD_S and s2 - e < 5 * FPS:
                    events.append((s2 + int(HOLD_S * FPS), who, 2, 'takedown'))
                    scored[3 - who].clear()
                    break
    return events, dict(secs)


def clock(frame, t0):
    s = round((frame - t0) / FPS, 1)
    return f'{int(s // 60)}:{s - 60 * int(s // 60):04.1f}'


def narrate(segs, t0):
    lines = []
    events, _ = control(segs)
    ev = {f: (w, p, why) for f, w, p, why in events}
    for s, e, lab in segs:
        base, who = split(lab)
        dur = (e - s + 1) / FPS
        if dur < 1.0:
            continue
        actor = f'athlete {who}' if who else 'both athletes'
        if base in ('mount', 'side_control', 'turtle'):
            text = f'athlete {who} on top in {WORDS[base]}' if base != 'turtle' else f'athlete {3 - who} turtles, athlete {who} on top'
        elif base == 'back':
            text = f'athlete {who} takes the back'
        elif base == 'takedown':
            text = f'athlete {who} shoots a takedown'
        elif who:
            text = f'{WORDS[base]} (athlete {who} as labelled)'
        else:
            text = WORDS[base]
        line = f'{clock(s, t0)} - {clock(e, t0)} ({dur:.1f} s): {text}'
        hits = [v for f, v in ev.items() if s <= f <= e + 1]
        for w, p, why in hits:
            line += f'  [athlete {w} +{p}: {why} held 3 s]'
        lines.append(line)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stream', default='06')
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--secs', type=float, default=90.0)
    a = ap.parse_args()
    z = np.load(a.pred)
    classes = [str(c) for c in z['classes']]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    per = defaultdict(list)
    for s, f, t, p in zip(z['stream'], z['frame'], z['true'], z['pred_smoothed']):
        per[str(s)].append((int(f), classes[t], classes[p]))
    rep = {'transitions': {}, 'segmental_f1': {}, 'control': {}}
    T, P, ST, SP, pts = defaultdict(list), defaultdict(list), [], [], []
    for s, rows in sorted(per.items()):
        rows.sort()
        f = np.array([r[0] for r in rows])
        st, sp = segments(f, [r[1] for r in rows]), segments(f, min_duration(f, [r[2] for r in rows]))
        ST.append(st); SP.append(sp)
        for tol in (0.5, 1.0, 2.0):
            for need in (False, True):
                k = f'{tol}s' + ('_label' if need else '')
                m = match_events(transitions(st), transitions(sp), tol * FPS, need)
                T[k].append(m)
        et, _ = control(st); ep, _ = control(sp)
        for w in (1, 2):
            pts.append((sum(p for _, a_, p, _ in et if a_ == w), sum(p for _, a_, p, _ in ep if a_ == w)))
    for k, ms in T.items():
        tp_t = sum(m['recall'] * m['true'] for m in ms); n_t = sum(m['true'] for m in ms); n_p = sum(m['pred'] for m in ms)
        p = tp_t / max(n_p, 1); r = tp_t / max(n_t, 1)
        rep['transitions'][k] = {'precision': p, 'recall': r, 'f1': 2 * p * r / max(p + r, 1e-9), 'expert_transitions': n_t,
                                 'predicted_transitions': n_p}
    for thr in (0.10, 0.25, 0.50):
        rep['segmental_f1'][f'{thr:.2f}'] = float(np.mean([segmental_f1(t, p, thr) for t, p in zip(ST, SP)]))
    pts = np.array(pts)
    rep['control'] = {'points_per_athlete_stream_expert_mean': float(pts[:, 0].mean()), 'points_pred_mean': float(pts[:, 1].mean()),
                      'points_abs_err_mean': float(np.abs(pts[:, 0] - pts[:, 1]).mean()),
                      'same_leader_share': float(np.mean([np.sign(pts[i, 0] - pts[i + 1, 0]) == np.sign(pts[i, 1] - pts[i + 1, 1])
                                                         for i in range(0, len(pts), 2)]))}
    json.dump(rep, open(out/'timeline.json', 'w'), indent=1)
    print(json.dumps(rep, indent=1))
    rows = sorted(per[a.stream])
    f0 = rows[0][0] + int(a.start * FPS)
    win = [r for r in rows if f0 <= r[0] < f0 + a.secs * FPS]
    f = np.array([r[0] for r in win])
    lines = ['# Narration example: ViCoS stream ' + a.stream + f' ({a.secs:.0f} s from {a.start:.0f} s)', '',
             '## From the expert labels', ''] + ['- ' + x for x in narrate(segments(f, [r[1] for r in win]), f0)] + \
            ['', '## From the 2D position model (blocked cross-validation)', ''] + \
            ['- ' + x for x in narrate(segments(f, min_duration(f, [r[2] for r in win])), f0)]
    (out/f'narration_{a.stream}.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
