"""Render the engagement demo video from `engage_demo_layers.py`'s outputs (CPU, .venv-depth).

Six panels per frame, one per layer, on the clip's own time: 1 people and identities, 2 segmentation, 3 skeletons,
4 the 3D bodies drawn back onto the video through the clip's lens, 5 both bodies in one 3D space seen from a
slowly turning viewpoint with their contact points lit, 6 Astra's phase and narrative. Under them, the measured
timeline (touching and embrace; measured vs true where the clip has truth) with a playhead. Each panel's title carries
its layer's score against the truth, or without truth the pipeline's own health numbers.
Writes demo.mp4 (real time), demo.gif (half size) and four stills.
"""
import json

import cv2
import numpy as np
from PIL import Image

PW, PH = 640, 360                       # panel size
COL = [(60, 60, 230), (230, 140, 40)]   # A red, B blue (BGR)
LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def to_h264(path):
    """Re-encode OpenCV's MPEG-4 Part 2 output as H.264 (yuv420p, fast start) so browsers can play it; in place."""
    import subprocess
    tmp = path.with_suffix('.h264.mp4')
    r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(path), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-threads', '2', str(tmp)])
    if r.returncode == 0 and tmp.exists():
        tmp.replace(path)


def crop_box(boxes, W, H):
    """One fixed window (16:9 where the picture allows) around both people over the whole clip."""
    b = np.array(boxes).reshape(-1, 4)
    x1, y1 = np.nanmin(b[:, :2], 0); x2, y2 = np.nanmax(b[:, 2:], 0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = min(W, max(x2 - x1, (y2 - y1) * 16 / 9) * 1.1)
    h = min(H, max(w * 9 / 16, (y2 - y1) * 1.1))
    x0, y0 = int(np.clip(cx - w / 2, 0, W - w)), int(np.clip(cy - h / 2, 0, H - h))
    return x0, y0, int(w), int(h)


def title(img, text, sub=''):
    cv2.rectangle(img, (0, 0), (PW, 44 if sub else 26), (250, 250, 250), -1)
    cv2.putText(img, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    return img


def fit(img, cb):
    """The window, scaled to fit the panel; letterboxed in grey when its shape is not 16:9."""
    x0, y0, w, h = cb
    s = min(PW / w, PH / h)
    small = cv2.resize(img[y0:y0 + h, x0:x0 + w], (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    out = np.full((PH, PW, 3), 60, np.uint8)
    oy, ox = (PH - small.shape[0]) // 2, (PW - small.shape[1]) // 2
    out[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
    return out


def render_meshes(img, meshes, project):
    """Paint meshes (camera-frame vertices, faces, colour) onto img through the clip's lens, far to near."""
    tris, deps, cols = [], [], []
    for V, F, col in meshes:
        uv = project(V)
        t = uv[F]; z = V[F, 2].mean(1)
        tris.append(t); deps.append(z); cols += [col] * len(F)
    tris, deps = np.concatenate(tris), np.concatenate(deps)
    out = img.copy()
    pts = np.round(tris * 4).astype(np.int32)
    order = np.argsort(-deps)
    shade = np.zeros(len(order))
    for i in order:
        cv2.fillConvexPoly(out, pts[i], cols[i], lineType=cv2.LINE_8, shift=2)
    return cv2.addWeighted(img, 0.35, out, 0.65, 0)


def scene_view(Vs, contact_masks, angle, centre, scale):
    """Both bodies as points from a viewpoint turning about the vertical axis; contact points in yellow."""
    img = np.full((PH, PW, 3), 252, np.uint8)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])            # camera frame: y down, z forward
    for x in np.linspace(-1.5, 1.5, 7):                           # floor grid at the feet
        for (a_, b_) in (((x, 0, -1.5), (x, 0, 1.5)), ((-1.5, 0, x), (1.5, 0, x))):
            p = (np.array([a_, b_]) + [0, centre[3], 0]) @ R.T
            q = (p[:, [0, 1]] * scale + [PW / 2, PH * 0.55]).astype(int)
            cv2.line(img, tuple(q[0]), tuple(q[1]), (225, 225, 225), 1)
    pts, cols, dz = [], [], []
    for n, (V, cm) in enumerate(zip(Vs, contact_masks)):
        sub = slice(None, None, 6)
        P = (V[sub] - centre[:3]) @ R.T
        pts.append(P); dz.append(P[:, 2])
        cl = np.tile(np.array(COL[n]), (len(P), 1)); cl[cm[sub]] = (0, 210, 255)
        cols.append(cl)
    P, Cc, Z = np.concatenate(pts), np.concatenate(cols), np.concatenate(dz)
    q = (P[:, [0, 1]] * scale + [PW / 2, PH * 0.55]).astype(int)
    for i in np.argsort(-Z):
        x, y = q[i]
        if 0 <= x < PW and 0 <= y < PH:
            cv2.circle(img, (x, y), 2, tuple(int(v) for v in Cc[i]), -1)
    return img


def wrap(text, width):
    words, lines, cur = text.split(), [], ''
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = (cur + ' ' + w).strip()
    return lines + ([cur] if cur else [])


def render(a, out):
    from scipy.spatial import cKDTree
    from engage_demo_layers import COCO_LIMB, load_masks, timeline
    from engage_depth_probe import FACES
    clip = a.clip
    T = json.load(open(out/'track.json')); P = np.load(out/'pose.npz'); B = np.load(out/'body3d.npz')
    C = json.load(open(out/'contact.json')); X = json.load(open(out/'context.json')); S = json.load(open(out/'scorecard.json'))
    M = load_masks(out)
    det = json.load(open(out/'detect.json'))
    faces = np.load(FACES)['mhr']
    W, H = clip.size
    u = H / 2160                                                    # line and text scale: 1 on 4K frames
    cb = crop_box(T['boxes'], W, H)
    frames = T['frames']; t = (np.array(frames) - frames[0]) / clip.fps
    s1, s2, s3, s4, s5 = (S['L1_detection_tracking'], S['L2_segmentation'], S['L3_skeleton'], S['L4_3d'], S['L5_contact'])
    sam = T.get('method') == 'SAM 3.1 video'
    pm = 'presence' in P.files
    truth = bool(C.get('truth'))
    num = lambda v, fmt: 'n/a' if v is None else format(v, fmt)
    lens = 'estimated lens' if 'MoGe' in str(getattr(clip, 'meta', {}).get('lens_source', '')) else 'true lens'
    if truth:
        subs = [f'both found {s1["both_found_iou50"]:.0%} of frames; {s1["identity_switches"]} identity switches',
                f'mask overlap with truth {s2["mask_iou_median"]:.2f} (median)',
                f'joints {s3["visible_joints_px"]:.0f} px off when visible, {num(s3["hidden_joints_px"], ".0f")} px when hidden ({H}p)',
                f'limbs {s4["limb_joints_other_cm"]:.1f} cm, in contact {num(s4["limb_joints_in_contact_cm"], ".1f")} cm off',
                f'distance {s4["pair_distance_err_cm"]:.1f} cm off; '
                + (f'touching F1 {s5["touching_frame_f1"]:.2f}' if s5.get('touching_frame_f1') is not None
                   else f'no touch, measured none on {s5.get("touching_frame_agreement", 0):.0%} of frames')
                + (f', embrace F1 {s5["embrace_frame_f1"]:.2f}' if s5.get('embrace_frame_f1') is not None else '')]
    else:
        subs = [f'no ground truth; a person unmasked in {s1["held_person_frames"]:.0%} of frames',
                f'masks overlap {s2["mask_overlap_median"]:.0%} (median); area jump {s2["mask_area_jump_median"]:.0%}/frame',
                f'median keypoint score {s3["keypoint_score_median"]:.2f}' + (f'; {s3["visible_share"]:.0%} of joints visible' if s3.get('visible_share') is not None else ''),
                f'fit residual {s4["fit_residual_share_of_person_height"]:.1%} of body height; {s4["depth_m_median"]:.1f} m away',
                f'pelvises {s4["pair_pelvis_distance_m_median"]:.2f} m apart (median); touching {s5["touching_share"]:.0%} of frames']
    heads = list(zip(['1  people + identities (' + ('SAM 3.1 video tracking' if sam else 'RF-DETR + tracker') + ')',
                      '2  segmentation (' + ('SAM 3.1 masklets' if sam else 'BBoxMaskPose v2 SAM-pose2seg') + ')',
                      '3  skeletons (' + ('PMPose, mask-conditioned; hollow = hidden' if pm else 'ViTPose+') + ')',
                      f'4  3D bodies (SAM 3D Body, {lens})', '5  one 3D space + contact points'], subs)) + \
        [('6  context (Astra)', X['astra']['interaction_type'][:70])]
    ev_pred, _ = timeline(C, 'pred')
    ev_true = timeline(C, 'truth')[0] if truth else []
    phases = X['astra']['phases']
    # the 3D view follows the pair: centred on both bodies each frame (smoothed over about a second), zoomed only as far
    # as their own spread needs (170 px per metre unless wider)
    sub = [(B['verts'][:, n].astype(np.float64) + B['t'][:, n][:, None])[:, ::50] for n in range(2)]
    both = np.concatenate(sub, 1)                                   # [frames, points, 3]
    raw_c = both.mean(1)
    win = max(1, int(round(clip.fps / np.median(np.diff(frames)))))
    ker = np.ones(win) / win
    pad = np.pad(raw_c, ((win // 2, win - 1 - win // 2), (0, 0)), mode='edge')
    centres = np.stack([np.convolve(pad[:, d], ker, mode='valid') for d in range(3)], 1)
    feet = np.percentile(both[..., 1].ravel(), 99)
    reach = max(np.percentile(np.linalg.norm((both - centres[:, None])[..., [0, 2]], axis=-1), 99), 0.5)
    scale = float(min(170, 0.46 * PW / reach))
    rate = clip.fps / np.median(np.diff(frames))
    writer = cv2.VideoWriter(str(out/'demo.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), rate, (3 * PW, 2 * PH + 170))
    stills = set(np.linspace(0, len(frames) - 1, 6).round().astype(int)[1:5])
    gif = []
    for k, f in enumerate(frames):
        img = cv2.imread(str(clip.image_path(f)))
        p1 = img.copy()
        lw = lambda v: max(1, int(round(v * u)))
        for b in det['frames'][k]['boxes']:
            cv2.rectangle(p1, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (170, 170, 170), lw(4))
        for n in range(2):
            b = T['boxes'][k][n]
            cv2.rectangle(p1, tuple(map(int, b[:2])), tuple(map(int, b[2:])), COL[n], lw(10))
            cv2.putText(p1, 'AB'[n] + (' (held)' if T['held'][k][n] else ''), (int(b[0]) + lw(10), int(b[1]) + lw(70)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.4 * u, COL[n], lw(6), cv2.LINE_AA)
        p2 = img.copy()
        for n in range(2):
            p2[M[k][n]] = (0.45 * p2[M[k][n]] + 0.55 * np.array(COL[n])).astype(np.uint8)
        p3 = (img * 0.6).astype(np.uint8)
        for n in range(2):
            kp, sc = P['kp'][k, n], P['score'][k, n]
            for i, j in LIMBS:
                if sc[i] > 0.3 and sc[j] > 0.3:
                    cv2.line(p3, tuple(kp[i].astype(int)), tuple(kp[j].astype(int)), COL[n], lw(9), cv2.LINE_AA)
            vis = P['visibility'][k, n] if pm else np.ones(17)
            for i in COCO_LIMB:
                cv2.circle(p3, tuple(kp[i].astype(int)), lw(12), (255, 255, 255), -1 if vis[i] > 0.5 else lw(4))
        Vs = [B['verts'][k, n].astype(np.float64) + B['t'][k, n] for n in range(2)]
        p4 = render_meshes(img, [(Vs[n], faces, COL[n]) for n in range(2)], clip.project)
        cm = [cKDTree(Vs[1 - n]).query(Vs[n][::1], distance_upper_bound=0.03)[0] < 0.03 for n in range(2)]
        p5 = scene_view(Vs, cm, np.radians(-35 + 70 * k / max(len(frames) - 1, 1)), np.r_[centres[k], feet - centres[k][1]], scale)
        pc = C['pred'][k]
        cv2.putText(p5, f'gap {pc["gap_cm"]:.0f} cm' + (f' (true {C["truth"][k]["gap_cm"]:.0f})' if truth else ''), (10, PH - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        regions = ', '.join(' + '.join(p) for p in pc['contacts'][:3]) or 'no contact'
        cv2.putText(p5, regions[:80], (10, PH - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
        p6 = np.full((PH, PW, 3), 252, np.uint8)
        ph = min(phases, key=lambda p: 0 if p['start_s'] <= t[k] <= p['end_s'] else min(abs(t[k] - p['start_s']), abs(t[k] - p['end_s'])),
                 default=None)                                      # between phases: the nearest one
        y = 70
        if ph:
            cv2.putText(p6, f'{ph["label"].upper()}  ({ph["start_s"]:.1f}-{ph["end_s"]:.1f} s)', (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
            y += 30
            for line in wrap(ph['description'], 62)[:5]:
                cv2.putText(p6, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA); y += 22
            if ph['cites']:
                cv2.putText(p6, 'cites measured events: ' + ', '.join(ph['cites']), (12, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 90, 20), 1, cv2.LINE_AA); y += 26
        y = max(y + 10, 240)
        for line in wrap('Labels: ' + ', '.join(X['astra']['labels'][:8]), 66)[:3]:
            cv2.putText(p6, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA); y += 20
        panels = [fit(p1, cb), fit(p2, cb), fit(p3, cb), fit(p4, cb), p5, p6]
        panels = [title(p, h, s) for p, (h, s) in zip(panels, heads)]
        grid = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])
        # timeline: touching and embrace, measured vs true, with a playhead
        strip = np.full((170, 3 * PW, 3), 250, np.uint8)
        x0, x1 = 160, 3 * PW - 20
        tx = lambda s: int(x0 + (x1 - x0) * s / max(t[-1], 1e-6))
        rows = [('touching, measured', ev_pred, 'touching', (60, 160, 60)), ('touching, true', ev_true, 'touching', (170, 170, 170)),
                ('embrace, measured', ev_pred, 'embrace', (40, 120, 220)), ('embrace, true', ev_true, 'embrace', (170, 170, 170))]
        if not truth:
            rows = [r if 'true' not in r[0] else (r[0].replace('true', 'true: n/a'), [], r[2], r[3]) for r in rows]
        for i, (name, evs, what, col) in enumerate(rows):
            yy = 18 + 26 * i
            cv2.putText(strip, name, (8, yy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
            cv2.rectangle(strip, (x0, yy), (x1, yy + 16), (235, 235, 235), -1)
            for e in evs:
                if e['what'] == what:
                    cv2.rectangle(strip, (tx(e['start_s']), yy), (tx(e['end_s']), yy + 16), col, -1)
                elif what == 'touching' and e['what'] == 'brief touch':        # brief touches: a lighter tick
                    light = tuple(int(c + (255 - c) * 0.55) for c in col)
                    cv2.rectangle(strip, (tx(e['start_s']), yy), (max(tx(e['end_s']), tx(e['start_s']) + 3), yy + 16), light, -1)
        yy = 18 + 26 * 4
        cv2.putText(strip, 'Astra phases', (8, yy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
        for p in phases:
            cv2.rectangle(strip, (tx(p['start_s']), yy), (tx(p['end_s']), yy + 16), (215, 225, 240), -1)
            cv2.rectangle(strip, (tx(p['start_s']), yy), (tx(p['end_s']), yy + 16), (150, 150, 150), 1)
            cv2.putText(strip, p['label'][:14], (tx(p['start_s']) + 3, yy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.line(strip, (tx(t[k]), 10), (tx(t[k]), 160), (30, 30, 200), 2)
        cv2.putText(strip, f'{t[k]:.1f} s', (tx(t[k]) + 4, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 200), 1, cv2.LINE_AA)
        frame = np.vstack([grid, strip])
        writer.write(frame)
        if k % 2 == 0:
            gif.append(Image.fromarray(cv2.cvtColor(cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)))
        if k in stills:
            cv2.imwrite(str(out/f'still_{k:03d}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    writer.release()
    to_h264(out/'demo.mp4')
    pal = [g.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for g in gif]
    pal[0].save(out/'demo.gif', save_all=True, append_images=pal[1:], duration=int(2000 / rate), loop=0, optimize=True)
    print('wrote', out/'demo.mp4', out/'demo.gif')
