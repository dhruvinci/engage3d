"""E23: engagement beyond contact, measured the same way on true and predicted bodies (CPU, .venv-depth). Experimental.
Spec: `docs/engagement/MEASUREMENT_SPEC.v1.md`. Pre-registration: `docs/engagement/results/measures_prereg.md` (a draft:
the prediction-against-truth comparison is not run until the definitions are signed off).

  measures  per clip and frame, on 'truth' and/or 'pred' bodies:
            floor (RANSAC plane through each body's lowest points), centre of mass (de Leva 1996 segment model), base of
            support and stability margin, supported by the partner (a geometric proxy) and top pressure, lever arms at
            contact, level and posture, head position, inside position of each arm and leg; per clip, contact episodes
            (E22's contact layer) and closing speed before each episode.
  check     truth only: the segment-model centre of mass against the uniform-density volume centre of the true mesh.
  score     C1-C9 of the pre-registration on the 'both' output (prediction against truth). Not run before sign-off.

  python engage_measures.py measures --clips NAME[,NAME|all] --which truth|pred|both --out DIR
  python engage_measures.py check --clips all --out DIR

Definitions signed off by Dhruva on 2026-10-30: MASS_TABLE 'mean', SUPPORTED_BY_PARTNER_M 0.05, lever-arm length only
(LEVER_JOINT), HEAD_INSIDE_M 0.15, support relative to the body's lowest point; frames (braced arm) left for later.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_relations import REGIONS, canon_from_pred, vertex_regions   # noqa: E402

DEMO = Path(__file__).resolve().parents[1]/'benchmarks/engage_h4d/demo_layers'
FACES = Path(__file__).resolve().parents[1]/'benchmarks/engage_h4d/depth_probe/faces.npz'
TRUTH_CLIPS = ['002_hugging__cam01_final', 'h4d_hugging_001', 'h4d_sword_009', 'medium_ballroom', 'h4d_karate_007',
               'hard_mma', 'h4d_mma5_009', 'hard_grappling',
               'mamma_wcs_sugarpush', 'mamma_wcs_rollinout', 'mamma_wcs_underarm', 'mamma_wcs_improv08']

# ---- definitions (spec v1); [decide] items are flagged
TAU, TAU_HAND = 0.03, 0.10          # contact distance; hands and forearms on predictions (E22's rule)
FLOOR_LOWEST, FLOOR_EVERY, FLOOR_INLIER = 40, 3, 0.02
SUPPORT_M = 0.03                    # a vertex within this of the body's own lowest vertex supports it ...
SUPPORT_REACH_M = 0.06              # ... when that lowest vertex is within this of the floor (signed off 2026-10-30, (c))
DEGENERATE_SPAN_M = 0.02
SUPPORTED_BY_PARTNER_M = 0.05       # signed off CoM outside its own base by more than this, while touching
HEAD_INSIDE_M = 0.15                # signed off
INSIDE_DEPTH_M = 0.40
CLOSING_WINDOW_S = 0.5
MASS_TABLE = 'mean'                 # signed off: 'mean' of de Leva's female and male tables, or 'male'
# de Leva (1996), J. Biomech. 29:1223-1230, adjusted Zatsiorsky-Seluyanov parameters (Table 4; values as tabulated in the
# BMC notebooks, data/BSP_ZdeLeva.txt): mass % (F, M) and CoM position % from the first endpoint (F, M).
# Endpoints are mapped to the canonical points: head -> 'head' (between the ears; de Leva's is vertex to cervicale,
# not available), trunk cervicale -> 'neck' and mid-hip -> 'pelvis', MET3 -> 'hand', ankle joint centre -> 'ankle',
# foot heel -> toe tip approximated by ankle -> toe at 40%.
DE_LEVA = {'trunk': ((42.57, 43.46), (49.64, 51.38), 'neck', 'pelvis'),
           'upper_arm': ((2.55, 2.71), (57.54, 57.72), 'shoulder', 'elbow'),
           'forearm': ((1.38, 1.62), (45.59, 45.74), 'elbow', 'wrist'),
           'hand': ((0.56, 0.61), (74.74, 79.00), 'wrist', 'hand'),
           'thigh': ((14.78, 14.16), (36.12, 40.95), 'hip', 'knee'),
           'shank': ((4.81, 4.33), (43.52, 43.95), 'knee', 'ankle'),
           'foot': ((1.29, 1.37), (40.0, 40.0), 'ankle', 'toe')}
HEAD_MASS = (6.68, 6.94)
# signed off: lever arm the joint a touched region turns about (direction is not measured in v1)
LEVER_JOINT = {'l_forearm': 'l_elbow', 'r_forearm': 'r_elbow', 'l_hand': 'l_elbow', 'r_hand': 'r_elbow',
               'l_upper_arm': 'l_shoulder', 'r_upper_arm': 'r_shoulder', 'l_shin': 'l_knee', 'r_shin': 'r_knee',
               'l_foot': 'l_knee', 'r_foot': 'r_knee', 'l_thigh': 'l_hip', 'r_thigh': 'r_hip', 'head': 'neck'}
ARM, LEG = {'l_arm': ('l_wrist', 'l_hand'), 'r_arm': ('r_wrist', 'r_hand')}, {'l_leg': ('l_knee', 'l_toe'), 'r_leg': ('r_knee', 'r_toe')}


def mass_table():
    pick = (lambda fm: (fm[0] + fm[1]) / 2) if MASS_TABLE == 'mean' else (lambda fm: fm[1])
    segs = [('head', pick(HEAD_MASS), None, 'head', 'head')]
    for name, (m, c, a, b) in DE_LEVA.items():
        if name == 'trunk':
            segs.append((name, pick(m), pick(c) / 100, a, b))
        else:
            for s in ('l', 'r'):
                segs.append((f'{s}_{name}', pick(m), pick(c) / 100, f'{s}_{a}', f'{s}_{b}'))
    total = sum(x[1] for x in segs)
    return [(n, m / total, c, a, b) for n, m, c, a, b in segs]


SEGMENTS = mass_table()


def com(C):
    """Centre of mass from the canonical points: sum of segment masses at their centres."""
    out = np.zeros(3)
    for _, m, c, a, b in SEGMENTS:
        out += m * (C[a] if c is None else C[a] + c * (C[b] - C[a]))
    return out


def volume_com(V, F):
    """Uniform-density centre of a closed mesh (signed tetrahedra from the origin)."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = np.einsum('ij,ij->i', a, np.cross(b, c)) / 6
    return (vol[:, None] * (a + b + c) / 4).sum(0) / vol.sum()


def fit_floor(bodies, rng):
    """Plane through each body's lowest vertices in every FLOOR_EVERY-th frame (RANSAC, then least squares).
    bodies: list over frames of [(V, C), (V, C)] in camera coordinates. 'Lowest' first uses the camera's down (+y),
    then the fitted normal, oriented so the bodies are above the floor. Returns (point, unit normal, axes e1, e2)."""
    def lowest(up):
        P = []
        for fr in bodies[::FLOOR_EVERY]:
            for V, _ in fr:
                h = V @ up
                P.append(V[np.argsort(h)[:FLOOR_LOWEST]])
        return np.concatenate(P)
    up = np.array([0.0, -1.0, 0.0])
    for _ in range(2):
        P = lowest(up)
        best, inl = 0, None
        for _ in range(500):
            s = P[rng.choice(len(P), 3, replace=False)]
            n = np.cross(s[1] - s[0], s[2] - s[0])
            if np.linalg.norm(n) < 1e-9:
                continue
            n /= np.linalg.norm(n)
            d = np.abs((P - s[0]) @ n)
            k = int((d < FLOOR_INLIER).sum())
            if k > best:
                best, inl = k, d < FLOOR_INLIER
        Q = P[inl]
        p0 = Q.mean(0)
        n = np.linalg.svd(Q - p0, full_matrices=False)[2][2]
        cen = np.mean([V.mean(0) for fr in bodies for V, _ in fr], 0)
        if (cen - p0) @ n < 0:
            n = -n
        up = n
    e1 = np.cross(n, [1.0, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    return p0, n, e1, np.cross(n, e1), {'points': int(len(P)), 'inliers': int(best)}


def signed_dist_polygon(q, poly):
    """Signed distance from 2D point q to a convex polygon (counter-clockwise vertices): positive inside."""
    d = np.inf
    inside = True
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        ab = b - a
        t = np.clip((q - a) @ ab / max(ab @ ab, 1e-12), 0, 1)
        d = min(d, np.linalg.norm(q - (a + t * ab)))
        if ab[0] * (q - a)[1] - ab[1] * (q - a)[0] < 0:
            inside = False
    return d if inside else -d


def base_margin(V, c, floor):
    """M2: support points (within SUPPORT_M of the body's lowest vertex, if that is within SUPPORT_REACH_M of the floor),
    and the stability margin of the CoM's floor projection (positive inside the base)."""
    p0, n, e1, e2 = floor[:4]
    h = (V - p0) @ n
    q = np.array([(c - p0) @ e1, (c - p0) @ e2])
    if h.min() > SUPPORT_REACH_M:                                   # lowest point too far above the floor: no support
        return np.nan, 0, True
    S = V[h <= h.min() + SUPPORT_M]                                 # relative to the body's own lowest point
    S2 = np.c_[(S - p0) @ e1, (S - p0) @ e2]
    span = np.linalg.norm(S2.max(0) - S2.min(0))
    if len(S2) >= 3 and span >= DEGENERATE_SPAN_M:
        try:
            hull = ConvexHull(S2)
            return float(signed_dist_polygon(q, S2[hull.vertices])), int(len(S)), False
        except Exception:
            pass
    return float(-np.min(np.linalg.norm(S2 - q, axis=1))), int(len(S)), False


def torso_frame(C):
    """Partner's torso frame: origin pelvis; y up the spine; x left -> right shoulder; z forward (= y x x)."""
    y = C['neck'] - C['pelvis']; y /= np.linalg.norm(y)
    x = C['r_shoulder'] - C['l_shoulder']; x -= (x @ y) * y; x /= np.linalg.norm(x)
    return C['pelvis'], x, y, np.cross(y, x)


def measures_pair(bodies, floor, hand_tau):
    """Every per-frame measure for one frame's two bodies [(V, C), (V, C)] (V: vertices, C: canonical points)."""
    p0, n = floor[0], floor[1]
    coms = [com(C) for _, C in bodies]
    trees = [cKDTree(V) for V, _ in bodies]
    regs = [vertex_regions(V, C) for V, C in bodies]
    gap = [trees[1 - k].query(bodies[k][0])[0] for k in range(2)]               # each vertex's distance to the partner
    touching = bool(min(g.min() for g in gap) < TAU)
    out = {'touching': touching, 'persons': []}
    for k in range(2):
        V, C = bodies[k]; Vo, Co = bodies[1 - k]
        margin, n_support, airborne = base_margin(V, coms[k], floor)
        h = float((coms[k] - p0) @ n)
        P = {'com': coms[k].tolist(), 'com_height_m': h, 'support_vertices': n_support, 'no_support': airborne,
             'margin_m': margin, 'stable': bool(margin > 0) if margin == margin else False}
        out_of_base = airborne or (margin == margin and margin < -SUPPORTED_BY_PARTNER_M)
        P['supported_by_partner'] = bool(touching and out_of_base)
        P['top_pressure'] = bool(P['supported_by_partner'] and h > (coms[1 - k] - p0) @ n)
        sp = C['neck'] - C['pelvis']
        P['trunk_deg'] = float(np.degrees(np.arccos(np.clip(sp @ n / np.linalg.norm(sp), -1, 1))))
        P['pelvis_height_m'] = float((C['pelvis'] - p0) @ n)
        # M4: lever arms on this body, where the partner touches it
        lev = {}
        for r, joint in LEVER_JOINT.items():
            ri = REGIONS.index(r)
            tau = hand_tau if (hand_tau and r in ('l_hand', 'r_hand', 'l_forearm', 'r_forearm')) else TAU
            m = (regs[k] == ri) & (gap[k] < tau)
            if m.any():
                arm = float(np.linalg.norm(V[m].mean(0) - C[joint]))
                if arm > lev.get(joint, (0, None))[0]:
                    lev[joint] = (arm, r)
        P['levers'] = {j: {'arm_m': a, 'region': r} for j, (a, r) in lev.items()}
        # M6: this person's head relative to the partner
        o, x, y, z = torso_frame(Co)
        hd = C['head']
        chest = (Co['l_shoulder'] + Co['r_shoulder']) / 2
        P['head_lateral_m'] = float((hd - o) @ x)
        P['head_in_front'] = bool((hd - chest) @ z > 0)
        P['head_inside'] = bool(abs(P['head_lateral_m']) < HEAD_INSIDE_M and P['head_in_front'])
        P['head_above_partner_head_m'] = float((hd - Co['head']) @ n)
        P['head_above_partner_pelvis_m'] = float((hd - Co['pelvis']) @ n)
        # M7: this person's arms and legs inside the partner's frame
        half_sh = np.linalg.norm(Co['r_shoulder'] - Co['l_shoulder']) / 2
        half_hip = np.linalg.norm(Co['r_hip'] - Co['l_hip']) / 2
        ins = {}
        for limbs, half in ((ARM, half_sh), (LEG, half_hip)):
            for name, pts in limbs.items():
                ins[name] = any(abs((C[p] - o) @ x) < half and 0 < (C[p] - o) @ z < INSIDE_DEPTH_M for p in pts)
        P['inside'] = ins
        out['persons'].append(P)
    out['level_diff_m'] = out['persons'][0]['com_height_m'] - out['persons'][1]['com_height_m']
    out['hips_above'] = int(np.sign(out['persons'][0]['pelvis_height_m'] - out['persons'][1]['pelvis_height_m']))
    out['pelvis_dist_m'] = float(np.linalg.norm(bodies[0][1]['pelvis'] - bodies[1][1]['pelvis']))
    return out


def load_bodies(name, which):
    """[(frame, [(V, C), (V, C)])] for a demo clip's truth or predicted bodies (camera coordinates, metres)."""
    from engage_clip import clip_of
    q = {l.split('\t')[0]: l.split('\t') for l in (DEMO/'queue.tsv').read_text().splitlines() if l and not l.startswith('#')}
    seq, cam = q[name][1], q[name][2]
    clip = clip_of(seq if not seq.startswith('..') else str((DEMO.parents[2]/'scripts'/seq).resolve()), cam if cam != '-' else None)
    T = json.load(open(DEMO/name/'track.json'))
    out = []
    if which == 'truth':
        for f in T['frames']:
            out.append((f, [(np.asarray(V, float), C) for V, C in clip.truth_bodies(f)]))
    else:
        B = np.load(DEMO/name/'body3d.npz')
        for k, f in enumerate(T['frames']):
            out.append((f, [(B['verts'][k, i].astype(np.float64) + B['t'][k, i], canon_from_pred(B['kp3d'][k, i].astype(np.float64) + B['t'][k, i]))
                            for i in range(2)]))
    return out, clip


def episodes(name, which, frames, fps, pelvis):
    """M8: E22's sustained-contact episodes, and the closing speed over CLOSING_WINDOW_S before each starts."""
    from engage_demo_layers import timeline
    C = json.load(open(DEMO/name/'contact.json'))
    if which == 'truth' and not C.get('truth'):
        return None
    ev = [(e['what'], e['start_s'], e['end_s']) for e in timeline(C, which)[0] if e['what'] == 'touching']
    t = (np.array(frames) - frames[0]) / fps
    closing = []
    for _, s, _e in ev:
        i0, i1 = np.searchsorted(t, s - CLOSING_WINDOW_S), np.searchsorted(t, s)
        if i1 > i0 and i0 >= 0 and t[i1] - t[i0] > 0:
            closing.append(float((pelvis[i0] - pelvis[i1]) / (t[i1] - t[i0])))
    share = float(np.mean([p['touching'] for p in C[which]]))
    return {'episodes': [list(e) for e in ev], 'n_episodes': len(ev), 'share_touching_frames': share,
            'closing_speed_mps': closing}


def cmd_measures(a):
    clips = TRUTH_CLIPS if a.clips == 'all' else a.clips.split(',')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    whiches = ['truth', 'pred'] if a.which == 'both' else [a.which]
    for name in clips:
        R = {'clip': name, 'definitions': {'tau_m': TAU, 'tau_hand_pred_m': TAU_HAND, 'support_m': SUPPORT_M, 'support_reach_m': SUPPORT_REACH_M,
                                           'supported_by_partner_m': SUPPORTED_BY_PARTNER_M, 'head_inside_m': HEAD_INSIDE_M,
                                           'mass_table': MASS_TABLE, 'segments': [s[:3] for s in SEGMENTS]}}
        for which in whiches:
            frames_bodies, clip = load_bodies(name, which)
            fl = fit_floor([b for _, b in frames_bodies], np.random.default_rng(0))
            per = [measures_pair(b, fl, TAU_HAND if which == 'pred' else None) for _, b in frames_bodies]
            frames = [f for f, _ in frames_bodies]
            R[which] = {'frames': frames, 'floor': {'point': fl[0].tolist(), 'normal': fl[1].tolist(), **fl[4]},
                        'per_frame': per,
                        'over_time': episodes(name, which, frames, clip.fps, np.array([p['pelvis_dist_m'] for p in per]))}
            print(name, which, len(per), 'frames; floor inliers', fl[4]['inliers'], '/', fl[4]['points'], flush=True)
        json.dump(R, open(out/f'{name}.json', 'w'), default=float)


def cmd_check(a):
    """Truth only: segment-model CoM against the mesh's volume centre; CoM height while standing; floor fit."""
    clips = TRUTH_CLIPS if a.clips == 'all' else a.clips.split(',')
    F = np.load(FACES)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = {}
    for name in clips:
        fb, clip = load_bodies(name, 'truth')
        faces = clip._truth()['faces'] if name.startswith('mamma') else F['smpl']
        fl = fit_floor([b for _, b in fb], np.random.default_rng(0))
        d, hgt = [], []
        for _, b in fb[::3]:
            for V, C in b:
                d.append(np.linalg.norm(com(C) - volume_com(V, faces)))
                hgt.append(float((com(C) - fl[0]) @ fl[1]))
        rep[name] = {'segment_vs_volume_com_cm': {'median': float(np.median(d) * 100), 'p90': float(np.percentile(d, 90) * 100)},
                     'com_height_m': {'median': float(np.median(hgt)), 'min': float(np.min(hgt)), 'max': float(np.max(hgt))},
                     'floor': fl[4]}
        print(name, json.dumps(rep[name]), flush=True)
    json.dump(rep, open(out/'check_truth.json', 'w'), indent=1)


ACTIVITY = {'002_hugging__cam01_final': 'hugging', 'h4d_hugging_001': 'hugging', 'h4d_sword_009': 'range (sword)',
            'medium_ballroom': 'ballroom', 'h4d_karate_007': 'karate', 'hard_mma': 'MMA', 'h4d_mma5_009': 'MMA',
            'hard_grappling': 'grappling'}
BLOCK_S, B = 1.0, 2000


def block_boot(stat, units, rng):
    """95% interval of stat(list of frame indices per clip) by resampling 1 s blocks within each clip."""
    vals = []
    for _ in range(B):
        vals.append(stat({c: np.concatenate([blk[i] for i in rng.integers(0, len(blk), len(blk))]) for c, blk in units.items()}))
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def cmd_score(a):
    """C1-C9 of the pre-registration: prediction against truth, pooled over the Harmony4D clips (MAMMA separately),
    per activity, with 95% intervals from 1 s blocks within each clip. Run only after the definitions are signed off."""
    src = Path(a.out)
    rng = np.random.default_rng(2611)
    rows = {}
    for name in TRUTH_CLIPS:
        R = json.load(open(src/f'{name}.json'))
        T, P = R['truth'], R['pred']
        assert T['frames'] == P['frames']
        tp = np.array([[f['persons'][k]['com'] for k in range(2)] for f in T['per_frame']])
        pp = np.array([[f['persons'][k]['com'] for k in range(2)] for f in P['per_frame']])
        swap = np.linalg.norm(pp - tp[:, ::-1], axis=-1).mean() < np.linalg.norm(pp - tp, axis=-1).mean()
        order = [1, 0] if swap else [0, 1]                       # the predicted person matched to each true person
        t = (np.array(T['frames']) - T['frames'][0]) / (30.0 if not name.startswith('mamma') else R.get('fps', 30.0))
        rows[name] = {'T': T, 'P': P, 'order': order, 't': t}
    def per_clip(name, fn):
        r = rows[name]; T, P, o = r['T']['per_frame'], r['P']['per_frame'], r['order']
        return np.array([fn(T[i], P[i], o) for i in range(len(T))], float)
    pt = lambda f, k: np.array(f['persons'][k]['com'])
    measures = {
        'C1_com_cm': (lambda T, P, o: np.mean([np.linalg.norm(pt(P, o[k]) - pt(T, k)) for k in range(2)]) * 100, 'median', 10.0),
        'C1_com_vector_cm': (lambda T, P, o: np.linalg.norm((pt(P, o[0]) - pt(P, o[1])) - (pt(T, 0) - pt(T, 1))) * 100, 'median', 8.0),
        'C2_level_cm': (lambda T, P, o: abs((P['persons'][o[0]]['com_height_m'] - P['persons'][o[1]]['com_height_m'])
                                            - (T['persons'][0]['com_height_m'] - T['persons'][1]['com_height_m'])) * 100, 'median', 6.0),
        'C3_stable_agree': (lambda T, P, o: np.mean([T['persons'][k]['stable'] == P['persons'][o[k]]['stable'] for k in range(2)]), 'mean', 0.80),
        'C5_hips_above_agree': (lambda T, P, o: (np.sign(T['persons'][0]['pelvis_height_m'] - T['persons'][1]['pelvis_height_m'])
                                                 == np.sign(P['persons'][o[0]]['pelvis_height_m'] - P['persons'][o[1]]['pelvis_height_m']))
                                if abs(T['persons'][0]['pelvis_height_m'] - T['persons'][1]['pelvis_height_m']) >= 0.10 else np.nan, 'mean', 0.80),
        'C6_head_inside_agree': (lambda T, P, o: np.mean([T['persons'][k]['head_inside'] == P['persons'][o[k]]['head_inside'] for k in range(2)])
                                 if T['touching'] else np.nan, 'mean', 0.75),
        'C7_inside_agree': (lambda T, P, o: np.mean([T['persons'][k]['inside'][l] == P['persons'][o[k]]['inside'][l] for k in range(2)
                                                     for l in ('l_arm', 'r_arm', 'l_leg', 'r_leg')]) if T['touching'] else np.nan, 'mean', 0.70),
        'C8_lever_cm': (lambda T, P, o: np.nanmedian([abs(T['persons'][k]['levers'][j]['arm_m'] - P['persons'][o[k]]['levers'][j]['arm_m']) * 100
                                                      for k in range(2) for j in T['persons'][k]['levers'] if j in P['persons'][o[k]]['levers']] or [np.nan]),
                        'median', 6.0)}
    out = {'definitions': json.load(open(src/f'{TRUTH_CLIPS[0]}.json'))['definitions'], 'pooled_harmony4d': {}, 'mamma': {}, 'per_activity': {}}
    for key, (fn, how, bar) in measures.items():
        vals = {n: per_clip(n, fn) for n in rows}
        units = {n: [np.where((rows[n]['t'] >= s) & (rows[n]['t'] < s + BLOCK_S))[0] for s in np.arange(0, rows[n]['t'][-1] + 1e-9, BLOCK_S)] for n in rows}
        units = {n: [u for u in v if len(u)] for n, v in units.items()}
        agg = (lambda x: np.nanmedian(x)) if how == 'median' else (lambda x: np.nanmean(x))
        for group, names in (('pooled_harmony4d', [n for n in rows if not n.startswith('mamma')]), ('mamma', [n for n in rows if n.startswith('mamma')])):
            x = np.concatenate([vals[n] for n in names])
            ci_ = block_boot(lambda sel: agg(np.concatenate([vals[n][sel[n]] for n in names])), {n: units[n] for n in names}, rng)
            v = float(agg(x))
            out[group][key] = {'value': v, 'ci': ci_, 'n_frames': int(np.isfinite(x).sum()),
                               'pass': bool(v <= bar if how == 'median' else v >= bar) if group == 'pooled_harmony4d' else None, 'bar': bar}
        for act in sorted(set(ACTIVITY.values())):
            names = [n for n, a_ in ACTIVITY.items() if a_ == act]
            x = np.concatenate([vals[n] for n in names])
            out['per_activity'].setdefault(act, {})[key] = {'value': float(agg(x)) if np.isfinite(x).any() else None,
                                                            'ci': block_boot(lambda sel: agg(np.concatenate([vals[n][sel[n]] for n in names])), {n: units[n] for n in names}, rng)
                                                            if np.isfinite(x).any() else None, 'n_frames': int(np.isfinite(x).sum())}
    # C4: supported by the partner, F1 on clips where it occurs in truth on >= 10% of frames
    tp_ = fp_ = fn_ = 0; used = []
    for n in rows:
        T, P, o = rows[n]['T']['per_frame'], rows[n]['P']['per_frame'], rows[n]['order']
        tr = np.array([[f['persons'][k]['supported_by_partner'] for k in range(2)] for f in T])
        pr = np.array([[P[i]['persons'][o[k]]['supported_by_partner'] for k in range(2)] for i in range(len(P))])
        if n.startswith('mamma') or tr.mean() < 0.10:
            continue
        used.append(n); tp_ += int((tr & pr).sum()); fp_ += int((~tr & pr).sum()); fn_ += int((tr & ~pr).sum())
    f1 = 2 * tp_ / max(2 * tp_ + fp_ + fn_, 1)
    out['pooled_harmony4d']['C4_supported_f1'] = {'value': f1 if used else None, 'clips': used, 'pass': (f1 >= 0.6) if used else None, 'bar': 0.6}
    # C9: closing speed over matched contact episodes (Harmony4D), starts within 1 s
    err = []
    for n in rows:
        if n.startswith('mamma'):
            continue
        To, Po = rows[n]['T']['over_time'], rows[n]['P']['over_time']
        if not To or not Po:
            continue
        for (_, s, _e), v in zip(To['episodes'], To['closing_speed_mps']):
            cand = [(abs(sp - s), w) for (_, sp, _ep), w in zip(Po['episodes'], Po['closing_speed_mps'])]
            if cand and min(cand)[0] <= 1.0:
                err.append(abs(min(cand)[1] - v))
    out['pooled_harmony4d']['C9_closing_mps'] = {'value': float(np.median(err)) if err else None, 'episodes': len(err),
                                                 'pass': (float(np.median(err)) <= 0.3) if err else None, 'bar': 0.3}
    json.dump(out, open(src/'score.json', 'w'), indent=1, default=float)
    print(json.dumps({k: {m: v.get('value') for m, v in g.items()} for k, g in out.items() if k in ('pooled_harmony4d', 'mamma')}, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['measures', 'check', 'score'])
    ap.add_argument('--clips', default='all')
    ap.add_argument('--which', default='truth', choices=['truth', 'pred', 'both'])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    {'measures': cmd_measures, 'check': cmd_check, 'score': cmd_score}[a.cmd](a)


if __name__ == '__main__':
    main()
