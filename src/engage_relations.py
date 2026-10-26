"""Engagement relations between two 3D bodies (CPU; .venv-pose or .venv-s3d: numpy + scipy). Experimental.

Implements layers 1-2 of docs/engagement/MEASUREMENT_SPEC.v0.md for any body model, through one canonical skeleton
of 19 named points (CANON) that both SMPL (Harmony4D ground truth) and MHR (SAM 3D Body's 70 keypoints) map onto,
so ground truth and predictions are measured with identical definitions:
  canon_from_smpl / canon_from_mhr   model output -> {name: xyz}
  vertex_regions()   16 body regions per mesh vertex: nearest bone of the canonical skeleton (same rule for any mesh;
                     the torso is split into chest / back by the side of the body it lies on)
  contact_map()      16 x 16 region pairs closer than tau, and the minimum mesh-to-mesh distance
  threads()          does a body chain of one athlete cross the surface spanned by a limb loop of the other
  margin()           how far inside the loop the crossing is (small = grazing / ambiguous)
  relations()        all of the above for one frame, both directions
Threading is the core of guards, mounts, back takes and leg entanglements (B's torso through A's leg loop is a guard
or mount, one of B's legs through it is half guard or ashi garami, B's head and an arm through it is a triangle), and
of loop balance: who holds more loops on whom.
"""
import numpy as np
from scipy.spatial import cKDTree

CANON = ['pelvis', 'neck', 'head', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle', 'l_toe', 'r_toe',
         'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hand', 'r_hand']
REGIONS = ['head', 'chest', 'back', 'hips', 'l_upper_arm', 'r_upper_arm', 'l_forearm', 'r_forearm', 'l_hand', 'r_hand',
           'l_thigh', 'r_thigh', 'l_shin', 'r_shin', 'l_foot', 'r_foot']
# (region, from, to): a vertex belongs to the region of its nearest bone ('torso' is split into chest / back)
BONES = [('head', 'neck', 'head'), ('torso', 'pelvis', 'neck'), ('torso', 'neck', 'l_shoulder'),
         ('torso', 'neck', 'r_shoulder'), ('hips', 'l_hip', 'r_hip')] + \
        [(f'{s}_{r}', f'{s}_{a}', f'{s}_{b}') for s in 'lr' for r, a, b in
         (('upper_arm', 'shoulder', 'elbow'), ('forearm', 'elbow', 'wrist'), ('hand', 'wrist', 'hand'),
          ('thigh', 'hip', 'knee'), ('shin', 'knee', 'ankle'), ('foot', 'ankle', 'toe'))]
LEG_LOOP = ['l_ankle', 'l_knee', 'l_hip', 'r_hip', 'r_knee', 'r_ankle']          # closed by the ankle-to-ankle chord
ARM_LOOP = ['l_wrist', 'l_elbow', 'l_shoulder', 'r_shoulder', 'r_elbow', 'r_wrist']  # closed by the wrist chord
CHAINS = {'torso': ['pelvis', 'neck'], 'head': ['neck', 'head'],
          'l_leg': ['l_hip', 'l_knee', 'l_ankle', 'l_toe'], 'r_leg': ['r_hip', 'r_knee', 'r_ankle', 'r_toe'],
          'l_arm': ['l_shoulder', 'l_elbow', 'l_wrist', 'l_hand'], 'r_arm': ['r_shoulder', 'r_elbow', 'r_wrist', 'r_hand']}
SEGMENT = {'l_leg': ['thigh', 'shin', 'foot'], 'r_leg': ['thigh', 'shin', 'foot'],
           'l_arm': ['upper_arm', 'forearm', 'hand'], 'r_arm': ['upper_arm', 'forearm', 'hand'],
           'torso': ['torso'], 'head': ['neck_to_head']}

SMPL_IDX = dict(neck=12, head=15, l_hip=1, r_hip=2, l_knee=4, r_knee=5, l_ankle=7, r_ankle=8, l_toe=10, r_toe=11,
                l_shoulder=16, r_shoulder=17, l_elbow=18, r_elbow=19, l_wrist=20, r_wrist=21, l_hand=22, r_hand=23)
# SAM 3D Body's keypoint order (sam_3d_body/metadata/mhr70.py, the first 70 of 308 MHR keypoints)
MHR70 = ["nose", "left-eye", "right-eye", "left-ear", "right-ear", "left-shoulder", "right-shoulder", "left-elbow",
         "right-elbow", "left-hip", "right-hip", "left-knee", "right-knee", "left-ankle", "right-ankle",
         "left-big-toe-tip", "left-small-toe-tip", "left-heel", "right-big-toe-tip", "right-small-toe-tip", "right-heel"] + \
        [f"right-{f}-{p}" for f in ('thumb', 'index', 'middle', 'ring', 'pinky')
         for p in ('tip', 'first-joint', 'second-joint', 'third-joint')] + ["right-wrist"] + \
        [f"left-{f}-{p}" for f in ('thumb', 'index', 'middle', 'ring', 'pinky')
         for p in ('tip', 'first-joint', 'second-joint', 'third-joint')] + \
        ["left-wrist", "left-olecranon", "right-olecranon", "left-cubital-fossa", "right-cubital-fossa",
         "left-acromion", "right-acromion", "neck"]
M = {n: i for i, n in enumerate(MHR70)}


def canon_from_smpl(J):
    J = np.asarray(J, float)
    C = {k: J[i] for k, i in SMPL_IDX.items()}
    C['pelvis'] = (C['l_hip'] + C['r_hip'])/2
    return C


def canon_from_mhr(K):
    """SAM 3D Body's 70 keypoints -> canonical points (hands at the middle-finger base, toes between the toe tips,
    head between the ears)."""
    K = np.asarray(K, float)
    C = {}
    for s, side in (('l', 'left'), ('r', 'right')):
        for c in ('hip', 'knee', 'ankle', 'shoulder', 'elbow', 'wrist'):
            C[f'{s}_{c}'] = K[M[f'{side}-{c}']]
        C[f'{s}_toe'] = (K[M[f'{side}-big-toe-tip']] + K[M[f'{side}-small-toe-tip']])/2
        C[f'{s}_hand'] = K[M[f'{side}-middle-third-joint']]
    C['pelvis'] = (C['l_hip'] + C['r_hip'])/2
    C['neck'] = K[M['neck']]
    C['head'] = (K[M['left-ear']] + K[M['right-ear']])/2
    return C


def canon_from_pred(K):
    """Predicted keypoints of either model, told apart by count: SMPL's 24 joints (PromptHMR) or MHR70 (SAM 3D Body)."""
    return canon_from_smpl(K) if len(K) == 24 else canon_from_mhr(K)


def _seg_dist(P, a, b):
    ab = b - a
    t = np.clip((P - a) @ ab/max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(P - (a + t[:, None]*ab), axis=1)


def vertex_regions(V, C):
    """[n] region index per vertex: nearest canonical bone; torso split by the side of the coronal plane."""
    V = np.asarray(V, float)
    D = np.stack([_seg_dist(V, C[a], C[b]) for _, a, b in BONES], 1)
    names = np.array([r for r, _, _ in BONES])[D.argmin(1)]
    left = C['l_shoulder'] - C['r_shoulder'] + C['l_hip'] - C['r_hip']
    up = C['neck'] - C['pelvis']
    fwd = np.cross(left, up)                      # person's left x up = forward
    front = (V - C['pelvis']) @ fwd > 0
    names = np.where(names == 'torso', np.where(front, 'chest', 'back'), names)
    lut = {r: i for i, r in enumerate(REGIONS)}
    return np.array([lut[n] for n in names])


def contact_map(va, ra, vb, rb, tau=0.02):
    """Region pairs (A region, B region) with a vertex pair closer than tau (m), and the minimum distance."""
    d, idx = cKDTree(vb).query(va, k=1)
    Mx = np.zeros((len(REGIONS), len(REGIONS)), bool)
    hit = d < tau
    Mx[ra[hit], rb[idx[hit]]] = True
    return Mx, float(d.min())


def _fan(loop):
    c = loop.mean(0)
    return [(c, loop[i], loop[(i + 1) % len(loop)]) for i in range(len(loop))]


def _seg_tri(p, q, tri, eps=1e-9):
    a, b, c = tri
    d = q - p
    e1, e2 = b - a, c - a
    h = np.cross(d, e2)
    det = e1 @ h
    if abs(det) < eps:
        return None
    f = 1/det
    s0 = p - a
    u = f*(s0 @ h)
    if u < 0 or u > 1:
        return None
    qv = np.cross(s0, e1)
    v = f*(d @ qv)
    if v < 0 or u + v > 1:
        return None
    s = f*(e2 @ qv)
    return s if 0 <= s <= 1 else None


def threads(loop, chain):
    """Crossings (segment index, s) of the polyline `chain` through the surface spanned by the closed polygon `loop`
    (fanned from its centroid); an odd count means the chain passes through the loop."""
    out = set()
    for i in range(len(chain) - 1):
        for t in _fan(loop):
            s = _seg_tri(chain[i], chain[i + 1], t)
            if s is not None:   # a crossing on an edge shared by two fan triangles is one crossing
                out.add((i, round(float(s), 4)))
    return sorted(out)


def margin(loop, chain, crossings):
    """Smallest distance (m) from a crossing point to the loop's boundary: a chain grazing the edge of a loop, or a
    loop flattened into a line (arms stretched out), crosses with a small margin and its threading is ambiguous."""
    best = np.inf
    n = len(loop)
    for i, s in crossings:
        x = chain[i] + s*(chain[i + 1] - chain[i])
        for k in range(n):
            best = min(best, float(_seg_dist(x[None], loop[k], loop[(k + 1) % n])[0]))
    return best


def relations(Ca, Cb, va=None, vb=None, tau=0.02):
    """One frame. Ca, Cb: canonical points of athletes A and B (same frame of reference, metres); va, vb: mesh
    vertices or None. Returns threading in both directions ({'B.torso through A.leg_loop': [segment, margin]}) and,
    with meshes, the minimum distance and the contact region pairs."""
    P = {'A': Ca, 'B': Cb}
    thr = {}
    for owner, other in (('A', 'B'), ('B', 'A')):
        for lname, lkeys in (('leg_loop', LEG_LOOP), ('arm_loop', ARM_LOOP)):
            loop = np.array([P[owner][k] for k in lkeys])
            for cname, ckeys in CHAINS.items():
                chain = np.array([P[other][k] for k in ckeys])
                x = threads(loop, chain)
                if len(x) % 2:
                    thr[f'{other}.{cname} through {owner}.{lname}'] = [SEGMENT[cname][x[0][0]],
                                                                       round(margin(loop, chain, x), 3)]
    r = {'threads': thr}
    if va is not None and vb is not None:
        ra, rb = vertex_regions(va, Ca), vertex_regions(vb, Cb)
        Mx, dmin = contact_map(np.asarray(va, float), ra, np.asarray(vb, float), rb, tau)
        r['min_dist'] = dmin
        r['contacts'] = [(REGIONS[i], REGIONS[j]) for i, j in zip(*np.nonzero(Mx))]
    return r
