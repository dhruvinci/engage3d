"""Demo clips for the engage3D README: where single-camera engagement measurement works (ballroom dancing) and where it
does not yet (grappling inside a tangle). Runs in .venv-depth (CPU).

Each frame: the photo with the true bodies, the photo with the measured bodies (SAM 3D Body + combined depth + the
learned placement, the E20 model saved with --save-model), the pair seen from above (truth grey, measured in colour),
and a caption with the distance between the two people and the gap between their bodies (measured vs true); for
grappling also how far the limb joints in contact are off (E11's measure, released model). Harmony4D frames (MIT).
The windows are picked by rule, to show the typical case: for ballroom, among windows with at least half the frames in
contact, the one whose median distance error is closest to E20's ballroom median (2.7 cm); for grappling, among the
quarter of windows with the most limb joints in a tangle, the one whose median limb error is closest to the held-out
median (5.4 cm). In the measured panel, red lines join each limb joint in contact to where it truly is.

  ../.venv-depth/bin/python engage_demo_video.py --model D/placement_e08.pt --ballroom-feats D/feats \
      --ballroom-s3d D/s3d --out ~/Code/engage3d/docs/assets
"""
import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

import engage_demo as DM
from engage_depth_probe import FACES, render, virtual_cam
from engage_limb_head import CORE14, JOINT8, tangle_mask
from engage_place_train import load_model
from engage_relations import CANON, canon_from_pred, canon_from_smpl

B = DM.B
COL = np.array([[0, 0, 0], [214, 120, 42], [52, 104, 235]], np.uint8)     # BGR: blue, orange
PANEL_H, FPS_MS, WINDOW = 400, 140, 44


def corrected(P, Y, Rv, m, t_blend, kp, verts):
    """engage_place_train.write's rigid move of one body from P to Y, applied in the camera frame."""
    ca, cb = P.mean(0), Y.mean(0)
    U, _, Vt = np.linalg.svd((P - ca).T @ (Y - cb))
    Rn = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    Rcam = Rv.T @ Rn @ Rv
    off = Rv.T @ (cb - Rn @ (ca + m) + m)
    t = Rcam @ t_blend + off
    return verts @ Rcam.T + t, kp @ Rcam.T + t


def limb_error(Pv, Gv):
    """Median rigid-aligned error (cm) of the 8 limb joints inside a tangle in one view, or None."""
    t = tangle_mask(Gv[None])[0]
    errs = []
    for n in range(2):
        A, Bv = Pv[n], Gv[n]
        a, b = A[CORE14].mean(0), Bv[CORE14].mean(0)
        U, _, Vt = np.linalg.svd((A[CORE14] - a).T @ (Bv[CORE14] - b))
        R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        e = np.linalg.norm((A - a) @ R.T + b - Bv, axis=1)
        errs += list(e[JOINT8 & t[n]])
    return float(np.median(errs) * 100) if errs else None


def gap(Va, Vb):
    return float(cKDTree(Va[::8]).query(Vb[::8])[0].min() * 100)


def frames_of(feats_file, s3d_file, predict):
    """Per frame of a feature file: measured bodies (camera frame) after the learned placement."""
    fe, z = np.load(feats_file), np.load(s3d_file)
    X = fe['X'][:, :171]
    Y = predict(fe['P'], X)
    out = {}
    pair = lambda A: np.linalg.norm(A[1, CANON.index('pelvis')] - A[0, CANON.index('pelvis')])
    for j, f in enumerate(fe['frame']):
        i = int(fe['i'][j])
        assert int(z['frames'][i]) == int(f), f'{feats_file}: feature row {j} is frame {f}, body file row {i} is {z["frames"][i]}'
        bodies = [corrected(fe['P'][j, n], Y[j, n], fe['R'][j], fe['m'][j], fe['t_blend'][j, n],
                            z['kp3d'][i, n].astype(float), z['verts'][i, n].astype(float)) for n in range(2)]
        G = fe['G'][j].astype(float)
        out[int(f)] = (bodies, fe['P'][j].astype(float), G, abs(pair(Y[j]) - pair(G)) * 100,
                       limb_error(fe['P'][j].astype(float), G))
    return out


def draw_frame(seq, cam, f, measured, Pv, Gv, title, show_limbs, scale):
    K, Dk, tocam = DM.cam_of(seq, cam)
    b = np.load(seq.root/'processed_data/bbox'/cam/f'{f:05d}.npy', allow_pickle=True).item()
    bx = np.array([np.asarray(b[w], float).ravel()[:4] for w in sorted(b)])
    Rv, Pc, (w, h) = virtual_cam(K, Dk, np.r_[bx[:, :2].min(0), bx[:, 2:].max(0)])
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, Dk, Rv, Pc, (w, h), cv2.CV_32FC1)
    img = cv2.remap(cv2.imread(str(seq.image_path(cam, f))), m1, m2, cv2.INTER_LINEAR)
    fc = np.load(FACES)
    gt = np.load(seq.root/'processed_data/smpl'/f'{f:05d}.npy', allow_pickle=True).item()
    truth = [(tocam(gt[k]['vertices']), canon_from_smpl(tocam(gt[k]['joints'][:24]))) for k in sorted(gt)]
    meas = [(V, canon_from_pred(J)) for V, J in measured]
    panels = []
    tmask = tangle_mask(Gv[None])[0] & JOINT8[None]
    proj = lambda X: ((X @ Rv.T)[:, :2] / (X @ Rv.T)[:, 2:] * Pc[[0, 1], [0, 1]] + Pc[:2, 2]).astype(int)
    for name, bodies, faces in (('truth', truth, fc['smpl']), ('measured from one camera', meas, fc['mhr'])):
        own = render([(V @ Rv.T, faces) for V, _ in bodies], Pc, w, h)[0]
        p = np.where(own[..., None] > 0, cv2.addWeighted(img, 0.35, COL[own], 0.65, 0), img)
        if name != 'truth':     # red lines: measured limb joint in contact -> its true position
            for n in range(2):
                for k in np.where(tmask[n])[0]:
                    a_, b_ = proj(np.array([meas[n][1][CANON[k]], truth[n][1][CANON[k]]]))
                    cv2.line(p, tuple(a_), tuple(b_), (30, 30, 230), 3, cv2.LINE_AA)
                    cv2.circle(p, tuple(b_), 4, (30, 30, 230), -1)
        cv2.rectangle(p, (0, 0), (w, 30), (251, 252, 252), -1)
        cv2.putText(p, name, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (11, 11, 11), 2, cv2.LINE_AA)
        panels.append(cv2.resize(p, (int(w * PANEL_H / h), PANEL_H), interpolation=cv2.INTER_AREA))
    # top-down: lateral (x) and depth (z) in the virtual camera, centred on the true pair
    td = np.full((PANEL_H, PANEL_H, 3), 251, np.uint8)
    mid = np.concatenate([V for V, _ in truth]).mean(0) @ Rv.T
    for bodies, cols in ((truth, [(160, 160, 160)] * 2), (meas, [(214, 120, 42), (52, 104, 235)])):
        for (V, _), c in zip(bodies, cols):
            q = (V[::25] @ Rv.T - mid)[:, [0, 2]] * scale + PANEL_H / 2
            for x, y in q.astype(int):
                if 0 <= x < PANEL_H and 0 <= y < PANEL_H:
                    cv2.circle(td, (x, PANEL_H - 1 - y), 1, c, -1)
    cv2.putText(td, 'seen from above (grey: truth)', (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (11, 11, 11), 1, cv2.LINE_AA)
    panels.append(td)
    strip = np.hstack(panels)
    dt = np.linalg.norm(truth[0][1]['pelvis'] - truth[1][1]['pelvis']) * 100
    dm = np.linalg.norm(meas[0][1]['pelvis'] - meas[1][1]['pelvis']) * 100
    le = limb_error(Pv, Gv)
    lines = [(title, 0.62, 2, (11, 11, 11)),
             (f'distance between them: {dm:.0f} cm measured, {dt:.0f} cm true   |   gap between bodies: '
              f'{gap(meas[0][0], meas[1][0]):.0f} cm measured, {gap(truth[0][0], truth[1][0]):.0f} cm true', 0.5, 1, (11, 11, 11)),
             ((f'limb joints in contact: {le:.1f} cm off (red lines)' if le is not None
               else 'limb joints in contact: none in this frame'), 0.5, 1,
              (30, 30, 230) if (le or 0) > 5 else (11, 11, 11))]
    bar = np.full((78, strip.shape[1], 3), 251, np.uint8)
    for i, (txt, sc, th, col) in enumerate(lines):
        cv2.putText(bar, txt, (8, 22 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)
    return np.vstack([bar, strip])


def windows(frames):
    keys = sorted(frames)
    return [keys[s:s + WINDOW] for s in range(0, max(1, len(keys) - WINDOW), 2)]


def pick_typical(frames, eligible, value, target):
    """Among eligible windows, the one whose median `value` is closest to `target`."""
    ws = [w for w in windows(frames) if eligible(w)]
    vals = [np.nanmedian([value(frames[k]) for k in w]) for w in ws]
    return ws[int(np.nanargmin(np.abs(np.array(vals) - target)))]


def write_gif(frames_bgr, path):
    """All frames padded to one canvas (the crops differ in width), so no frame leaves remnants of another."""
    H = max(f.shape[0] for f in frames_bgr)
    W = max(f.shape[1] for f in frames_bgr)
    canvas = []
    for f in frames_bgr:
        c = np.full((H, W, 3), 251, np.uint8)
        c[:f.shape[0], :f.shape[1]] = f
        canvas.append(c)
    ims = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).convert('P', palette=Image.ADAPTIVE, colors=128)
           for f in canvas]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=FPS_MS, loop=0, optimize=True, disposal=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--ballroom-feats', required=True)
    ap.add_argument('--ballroom-s3d', required=True)
    ap.add_argument('--grappling', default='043_grappling2,cam06', help='held-out sequence and camera')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    predict = load_model(a.model)
    out = Path(a.out)
    # where it works: ballroom (the frames where the two touch most, by the truth)
    seq_b = DM.Sequence(DM.seq_root('009_ballroom2'))
    fb = frames_of(Path(a.ballroom_feats)/'009_ballroom2__cam11.npz', Path(a.ballroom_s3d)/'009_ballroom2/cam11.npz', predict)
    in_contact = lambda k: bool(tangle_mask(fb[k][2][None])[0].any())
    win = pick_typical(fb, lambda w: np.mean([in_contact(k) for k in w]) >= 0.5, lambda v: v[3], 2.7)
    clip = [draw_frame(seq_b, 'cam11', f, fb[f][0], fb[f][1], fb[f][2], 'Ballroom dancing: works (a typical window)', True, 160)
            for f in win]
    write_gif(clip, out/'demo_ballroom.gif')
    print('ballroom frames', win[0], '-', win[-1], flush=True)
    # where it does not yet: grappling, the window with the most limb joints inside a tangle
    sname, cam = a.grappling.split(',')
    seq_g = DM.Sequence(DM.seq_root(sname))
    fg = frames_of(B/'place_feats2'/f'{sname}__{cam}.npz', B/'s3d_box'/sname/f'{cam}.npz', predict)
    tangled = lambda w: np.mean([tangle_mask(fg[k][2][None])[0][:, JOINT8].sum() for k in w])
    ws = windows(fg)
    cut = np.quantile([tangled(w) for w in ws], 0.75)
    win = pick_typical(fg, lambda w: tangled(w) >= cut, lambda v: v[4] if v[4] is not None else np.nan, 5.4)
    clip = [draw_frame(seq_g, cam, f, fg[f][0], fg[f][1], fg[f][2], 'Grappling, limbs tangled: not yet (a typical window)',
                       True, 160) for f in win]
    write_gif(clip, out/'demo_grappling.gif')
    print('grappling frames', win[0], '-', win[-1], flush=True)


if __name__ == '__main__':
    main()
