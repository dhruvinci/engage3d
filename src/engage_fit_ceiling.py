"""Fitting ceiling: given perfect 2D, how much of SAM 3D Body's tangled-limb error can a skeleton fit remove?
(CPU, .venv-depth). Development sequences only; no test view. Experimental.

Both bodies in `place_feats2` are in the camera frame (centred on the camera), so perfect 2D for a joint is its true
viewing ray u = G / |G|; what a fit must still find is each joint's distance along its ray. Variants, per athlete and
view:
  none                SAM 3D Body as stored (with the combined depth)
  rays                every joint moved onto its true ray at its predicted distance (perfect 2D, predicted depth)
  rays_bones_hard     limb joints re-solved along the rays at the athlete's own bone lengths exactly (median of SAM 3D
                      Body's over the camera's track), the candidate nearer the prediction. Ill-conditioned: when a limb
                      lies across the image, a bone a few % too long moves the joint many cm along its ray
  rays_soft           the 8 limb joints' distances fitted by least squares: near the predicted distance (sigma 10 cm),
                      bone lengths near the athlete's own (sigma 2 cm); torso joints as `rays`
  rays_soft_time      + smooth over the camera's track (frame-to-frame distance change, sigma 5 cm); also with calib
  rays_soft_calib     rays_soft with calibrated bone lengths: SAM 3D Body's divided by the median predicted/true ratio
                      per bone measured on the other development sequences (leave one sequence out; no oracle)
  rays_soft_truebones rays_soft with the true bone lengths (oracle body shape, e.g. from a calibration capture)
Measure as E14 / E18: rigid alignment over CORE14; median error of the 8 limb joints inside a tangle and elsewhere.

  ../.venv-depth/bin/python engage_fit_ceiling.py --feats DIR --out DIR [--smoke] [--noise-px 0,5,10,20]
With --noise-px, the rays are perturbed to the given median 2D error (pixels on the 4K frames, focal length 1,766 px,
Gaussian in both image directions, every joint) and only `rays` and `rays_soft_calib` are scored: how exact the 2D
must be for the fit to pay.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engage_limb_head as L                                            # noqa: E402
from engage_limb_head import CORE14, JOINT8, kabsch, tangle_mask       # noqa: E402
from engage_temporal import DEV, load_tracks                           # noqa: E402

L.ALIGN = CORE14
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', 8)))
CHAINS = [(11, 13, 15), (12, 14, 16), (3, 5, 7), (4, 6, 8)]   # shoulder-elbow-wrist, hip-knee-ankle (left, right)
LIMB = [13, 15, 14, 16, 5, 7, 6, 8]                            # the joints a fit moves (elbows, wrists, knees, ankles)
SIGMA_D, SIGMA_L, SIGMA_T = 0.10, 0.02, 0.05
FOCAL_PX = 1766.0                                              # Harmony4D's fisheye focal length (1,752-1,789 px)


def perturb(U, median_px, rng):
    """Rays rotated by Gaussian angular noise whose median 2D error is median_px at FOCAL_PX."""
    if median_px <= 0:
        return U
    sigma = median_px / 1.1774 / FOCAL_PX                      # median of a 2D Gaussian's radius = 1.1774 sigma
    t1 = np.cross(U, np.array([0.0, 1.0, 0.0]))
    t1 /= np.linalg.norm(t1, axis=-1, keepdims=True)
    t2 = np.cross(U, t1)
    a, b = rng.normal(0, sigma, U.shape[:-1] + (1,)), rng.normal(0, sigma, U.shape[:-1] + (1,))
    V = U + a * t1 + b * t2
    return V / np.linalg.norm(V, axis=-1, keepdims=True)


def bone_lengths(X, tracks):
    """[V,4 chains,2 bones] per-track median bone lengths of points X [V,19,3]."""
    out = np.zeros((len(X), len(CHAINS), 2))
    for idx in tracks:
        for k, (a, b, c) in enumerate(CHAINS):
            out[idx, k, 0] = np.median(np.linalg.norm(X[idx, b] - X[idx, a], axis=-1))
            out[idx, k, 1] = np.median(np.linalg.norm(X[idx, c] - X[idx, b], axis=-1))
    return out


def ray_sphere(u, c, length):
    """Distances along unit rays u of the two points at `length` from c: [...,2] (the closest point twice on a miss)."""
    b = (u * c).sum(-1)
    s = np.sqrt(np.clip(b * b - (c * c).sum(-1) + length ** 2, 0, None))
    return np.clip(np.stack([b - s, b + s], -1), 1e-3, None)


def hard_fit(U, Pr, p, lengths):
    """Exact bone lengths along the rays, each joint the candidate nearer the prediction p."""
    Q = Pr.copy()
    for k, (a, b, c) in enumerate(CHAINS):
        for j, (par, ch) in enumerate(((a, b), (b, c))):
            d = ray_sphere(U[:, ch], Q[:, par], lengths[:, k, j])
            cand = d[..., None] * U[:, ch, None]                                  # [V,2,3]
            pick = np.linalg.norm(cand - p[:, ch, None], axis=-1).argmin(1)
            Q[:, ch] = cand[np.arange(len(Q)), pick]
    return Q


def soft_fit(U, Pr, lengths, prev=None, iters=200):
    """Least-squares distances of the limb joints along their rays (torso fixed): near the start, bones near `lengths`,
    and, with prev (index of the previous frame in the track, -1 at its start), smooth over time."""
    Ut = torch.tensor(U)
    base = torch.tensor(Pr)
    d0 = torch.tensor(np.ascontiguousarray(np.linalg.norm(Pr[:, LIMB], axis=-1)))
    d = d0.clone().contiguous().requires_grad_(True)
    Lt = torch.tensor(lengths)
    opt = torch.optim.LBFGS([d], max_iter=iters, line_search_fn='strong_wolfe')
    has_prev = None if prev is None else torch.tensor(prev >= 0)
    prev_i = None if prev is None else torch.tensor(np.maximum(prev, 0))

    def closure():
        opt.zero_grad()
        X = base.clone()
        X[:, LIMB] = d[..., None] * Ut[:, LIMB]
        loss = ((d - d0) ** 2).sum() / SIGMA_D ** 2
        for k, (a, b, c) in enumerate(CHAINS):
            loss = loss + ((torch.linalg.norm(X[:, b] - X[:, a], dim=-1) - Lt[:, k, 0]) ** 2).sum() / SIGMA_L ** 2
            loss = loss + ((torch.linalg.norm(X[:, c] - X[:, b], dim=-1) - Lt[:, k, 1]) ** 2).sum() / SIGMA_L ** 2
        if prev is not None:
            loss = loss + (((d - d[prev_i]) ** 2).sum(-1) * has_prev).sum() / SIGMA_T ** 2
        loss.backward()
        return loss
    opt.step(closure)
    Q = Pr.copy()
    Q[:, LIMB] = d.detach().numpy()[..., None] * U[:, LIMB]
    return Q


def summarise(E, t):
    e, tt = E[..., JOINT8], t[..., JOINT8]
    return {'tangle_joint8_median_cm': float(np.median(e[tt]) * 100), 'other_joint8_median_cm': float(np.median(e[~tt]) * 100)}


def boot(a, b, t, n=1000):
    rng, V, r = np.random.default_rng(1), len(t), []
    a, b, t = a[..., JOINT8], b[..., JOINT8], t[..., JOINT8]
    for _ in range(n):
        s = rng.integers(0, V, V)
        r.append(np.median(a[s][t[s]]) / np.median(b[s][t[s]]) - 1)
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--noise-px', default='')
    ap.add_argument('--seqs', default='', help='target sequences; bone-length calibration then comes from the development grappling')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    allcams = {f'cam{i:02d}' for i in range(1, 21)}
    seqs = a.seqs.split(',') if a.seqs else (DEV[:1] if a.smoke else DEV)
    P, G, _, key, tr = load_tracks(a.feats, seqs, allcams)
    ratio_ext = None
    if a.seqs:   # calibration learned on the development grappling, applied to other activities
        Pd, Gd, _, kd, trd = load_tracks(a.feats, DEV, allcams)
        td = [np.array([fr[f] for f in sorted(fr)]) for fr in trd.values()]
        ratio_ext = [np.median(bone_lengths(Pd[:, n], td) / bone_lengths(Gd[:, n], td), 0) for n in range(2)]
    tracks = [np.array([fr[f] for f in sorted(fr)]) for fr in tr.values()]
    prev = np.full(len(P), -1)
    for idx in tracks:
        prev[idx[1:]] = idx[:-1]
    tang = tangle_mask(G)
    if a.noise_px:
        seq = np.array([k[0] for k in key])
        res = {}
        for px in [float(x) for x in a.noise_px.split(',')]:
            En = {v: np.zeros(P.shape[:3]) for v in ('rays', 'rays_soft_calib')}
            rng = np.random.default_rng(0)
            for n in range(2):
                p, g = P[:, n], G[:, n]
                U = perturb(g / np.linalg.norm(g, axis=-1, keepdims=True), px, rng)
                Pr = np.linalg.norm(p, axis=-1, keepdims=True) * U
                Lp, Lg = bone_lengths(p, tracks), bone_lengths(g, tracks)
                Lc = Lp.copy()
                for s in np.unique(seq):
                    Lc[seq == s] = Lp[seq == s] / np.median(Lp[seq != s] / Lg[seq != s], 0) if (seq != s).any() else Lp[seq == s]
                for v, Q in (('rays', Pr), ('rays_soft_calib', soft_fit(U, Pr, Lc))):
                    En[v][:, n] = np.linalg.norm(kabsch(Q, g) - g, axis=-1)
            none = np.zeros(P.shape[:3])
            for n in range(2):
                none[:, n] = np.linalg.norm(kabsch(P[:, n], G[:, n]) - G[:, n], axis=-1)
            res[str(px)] = {v: summarise(e, tang) for v, e in En.items()}
            res[str(px)]['none'] = summarise(none, tang)
            print(px, json.dumps(res[str(px)]), flush=True)
        json.dump(res, open(out/'fit_noise.json', 'w'), indent=1)
        return
    E = {}
    for n in range(2):
        p, g = P[:, n], G[:, n]
        U = g / np.linalg.norm(g, axis=-1, keepdims=True)
        Pr = np.linalg.norm(p, axis=-1, keepdims=True) * U
        Lp, Lg = bone_lengths(p, tracks), bone_lengths(g, tracks)
        seq = np.array([k[0] for k in key])
        Lc = Lp.copy()
        for s in np.unique(seq):     # per-bone predicted/true ratio from the other sequences
            Lc[seq == s] = Lp[seq == s] / np.median(Lp[seq != s] / Lg[seq != s], 0) if (seq != s).any() else Lp[seq == s]
        if ratio_ext is not None:
            Lc = Lp / ratio_ext[n]
        V = {'none': p, 'rays': Pr, 'rays_bones_hard': hard_fit(U, Pr, p, Lp), 'rays_soft': soft_fit(U, Pr, Lp),
             'rays_soft_time': soft_fit(U, Pr, Lp, prev), 'rays_soft_calib': soft_fit(U, Pr, Lc),
             'rays_soft_calib_time': soft_fit(U, Pr, Lc, prev), 'rays_soft_truebones': soft_fit(U, Pr, Lg)}
        for name, Q in V.items():
            E.setdefault(name, np.zeros(P.shape[:3]))[:, n] = np.linalg.norm(kabsch(Q, g) - g, axis=-1)
        print(f'athlete {n} done', flush=True)
    r = {'views': int(len(P)), 'tangle_points': int(tang[..., JOINT8].sum()), 'sigmas_m': [SIGMA_D, SIGMA_L, SIGMA_T]}
    val = np.array([k[0] in ('025_grappling2', '035_grappling2') for k in key])
    for name, sel in (('all_dev', np.ones(len(P), bool)), ('val_025_035', val)):
        if not sel.any():
            continue
        r[name] = {v: summarise(e[sel], tang[sel]) for v, e in E.items()}
        for v in E:
            if v != 'none':
                r[name][f'{v}_vs_none_tangle_95ci'] = boot(E[v][sel], E['none'][sel], tang[sel])
    json.dump(r, open(out/'fit_ceiling.json', 'w'), indent=1)
    np.savez(out/'errors.npz', tang=tang, key=np.array(key, dtype=object), **E)
    print(json.dumps(r, indent=1), flush=True)


if __name__ == '__main__':
    main()
