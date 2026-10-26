"""Demo figures for the engage3D experiment READMEs (runs in .venv-depth; GPU only for the depth-map panel).

  ../.venv-depth/bin/python engage_demo.py --out DIR [--no-depth]

Picks one held-out Harmony4D view (test sequences, cameras 01/06/11/16) where the two athletes touch, SAM 3D Body's
depth placement is far off, and the learned correction (placement v1) fixes it. Then writes:
  view_overlay.png      the photo, the true bodies, SAM 3D Body's, the learned correction's (crop, true lens)
  view_topdown.png      the same view from above (lateral x depth): truth vs SAM 3D Body / combined depth / learned
  view_depth.png        the photo and the MoGe-2 and Depth Pro depth maps of the crop (unless --no-depth)
  view_prompthmr.png    truth, SAM 3D Body and PromptHMR on the same view
  chart_*.png           result charts (values from each experiment's RESULTS.md)
Colours: the dataviz reference palette (blue / orange / aqua), grey for truth and baselines.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_bench_s3d import seq_root                                   # noqa: E402
from engage_depth_probe import FACES, render, virtual_cam                # noqa: E402
from engage_relations import canon_from_pred, canon_from_smpl            # noqa: E402
from tangle_h4d import Sequence                                          # noqa: E402

B = Path(__file__).resolve().parents[1]/'benchmarks/engage_h4d'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
GREY, INK, INK2, SURF = '#9a9993', '#0b0b0b', '#52514e', '#fcfcfb'
TEST = ['031_grappling2', '037_grappling2', '042_grappling2', '043_grappling2']


def plt_setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GREY, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True,
                         'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'axes.spines.top': False,
                         'axes.spines.right': False, 'text.color': INK, 'legend.frameon': False})
    return plt


def cam_of(seq, cam):
    Rc, tc, cid = seq.poses[cam]
    _, _, P = seq.cams[cid]
    K = np.array([[P[0], 0, P[2]], [0, P[1], P[3]], [0, 0, 1]])
    A4 = seq.world_to_colmap
    sc = np.cbrt(np.linalg.det(A4[:3, :3]))
    return K, np.array(P[4:8]).reshape(1, 4), lambda X: ((X @ A4[:3, :3].T + A4[:3, 3]) @ Rc.T + tc)/sc


_SEQ = {}


def bodies(sname, cam, f):
    """Camera-frame vertices and canonical points: truth, SAM 3D Body, combined depth, learned (None if missing)."""
    if sname not in _SEQ:
        _SEQ[sname] = Sequence(seq_root(sname))
    seq = _SEQ[sname]
    K, D, tocam = cam_of(seq, cam)
    gt = np.load(seq.root/'processed_data/smpl'/f'{f:05d}.npy', allow_pickle=True).item()
    out = {'truth': [(tocam(gt[k]['vertices']), canon_from_smpl(tocam(gt[k]['joints'][:24]))) for k in sorted(gt)]}
    z = np.load(B/'s3d_box'/sname/f'{cam}.npz')
    i = int(np.nonzero(z['frames'] == f)[0][0])
    sam = [(z['verts'][i, n].astype(float) + z['t'][i, n], canon_from_pred(z['kp3d'][i, n].astype(float) + z['t'][i, n]))
           for n in range(2)]
    out['SAM 3D Body'] = sam
    fe = np.load(B/'place_feats2'/f'{sname}__{cam}.npz')
    j = np.nonzero(fe['frame'] == f)[0]
    if len(j):
        t = fe['t_blend'][j[0]]
        out['combined depth'] = [(z['verts'][i, n].astype(float) + t[n], canon_from_pred(z['kp3d'][i, n].astype(float) + t[n]))
                                 for n in range(2)]
    mp = B/'place_v1_demo/model'/sname/f'{cam}.npz'
    if mp.exists():
        m = np.load(mp)
        k = np.nonzero(m['frames'] == f)[0]
        if len(k):
            out['learned'] = [(m['verts'][k[0], n].astype(float) + m['t'][k[0], n],
                               canon_from_pred(m['kp3d'][k[0], n].astype(float) + m['t'][k[0], n])) for n in range(2)]
    return seq, K, D, out


def dist_err(b, truth):
    d = lambda bb: np.linalg.norm(bb[0][1]['pelvis'] - bb[1][1]['pelvis'])
    return abs(d(b) - d(truth))


def pick_views():
    """Two held-out views where the athletes touch: (a) the largest SAM 3D Body -> combined depth gain, (b) the largest
    combined depth -> learned gain (combined still >= 4 cm off, learned <= 2 cm)."""
    a = b = None
    for p in sorted(glob.glob(str(B/'place_v1_demo/model/*/cam??.npz'))):
        sname, cam = p.split('/')[-2], Path(p).stem
        for f in np.load(p)['frames'][::3]:
            _, _, _, bb = bodies(sname, cam, int(f))
            if 'combined depth' not in bb or 'learned' not in bb:
                continue
            gap = np.min(np.linalg.norm(bb['truth'][0][0][::50, None] - bb['truth'][1][0][None, ::50], axis=-1))
            if gap >= 0.05:
                continue
            e = {k: dist_err(v, bb['truth']) for k, v in bb.items() if k != 'truth'}
            if e['learned'] < 0.03 and e['SAM 3D Body'] > 0.15:
                sa = e['SAM 3D Body'] - e['combined depth']
                if a is None or sa > a[0]:
                    a = (sa, sname, cam, int(f), e)
            if e['combined depth'] >= 0.04 and e['learned'] <= 0.02:
                sb = e['combined depth'] - e['learned']
                if b is None or sb > b[0]:
                    b = (sb, sname, cam, int(f), e)
    return a, b


def topdown(plt, out, bb, title):
    names = [k for k in ('SAM 3D Body', 'combined depth', 'learned') if k in bb]
    fig, axs = plt.subplots(1, len(names), figsize=(3.4*len(names), 3.6), sharex=True, sharey=True)
    mid = np.concatenate([v[0] for v in bb['truth']]).mean(0)
    for ax, name in zip(np.atleast_1d(axs), names):
        for n, col in ((0, BLUE), (1, ORANGE)):
            T = (bb['truth'][n][0][::12] - mid)*100
            P = (bb[name][n][0][::12] - mid)*100
            ax.scatter(T[:, 0], T[:, 2], s=3, c=GREY, alpha=0.35, linewidths=0, label='truth' if n == 0 else None)
            ax.scatter(P[:, 0], P[:, 2], s=3, c=col, alpha=0.6, linewidths=0, label=f'athlete {"AB"[n]}')
        ax.set_title(f'{name}\ndistance error {dist_err(bb[name], bb["truth"])*100:.1f} cm', fontsize=10)
        ax.set_xlabel('side to side (cm)')
        ax.set_aspect('equal')
    np.atleast_1d(axs)[0].set_ylabel('depth from the camera (cm)\n(camera below)')
    fig.legend(*np.atleast_1d(axs)[0].get_legend_handles_labels(), loc='lower center', ncol=3, markerscale=4, fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def overlay(out, seq, K, D, cam, f, bb, names, phmr=None):
    b = np.load(seq.root/'processed_data/bbox'/cam/f'{f:05d}.npy', allow_pickle=True).item()
    bx = np.array([np.asarray(b[w], float).ravel()[:4] for w in sorted(b)])
    Rv, Pv, (w, h) = virtual_cam(K, D, np.r_[bx[:, :2].min(0), bx[:, 2:].max(0)])
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, Rv, Pv, (w, h), cv2.CV_32FC1)
    img = cv2.remap(cv2.imread(str(seq.image_path(cam, f))), m1, m2, cv2.INTER_LINEAR)
    fc = np.load(FACES)
    col = np.array([[0, 0, 0], [214, 120, 42], [52, 104, 235]], np.uint8)   # BGR: blue, orange
    panels, labels = [img.copy()], ['photo']
    for name in names:
        if name == 'PromptHMR':
            meshes = [(V @ Rv.T, fc['smpl']) for V in phmr]
        else:
            meshes = [(v[0] @ Rv.T, fc['smpl'] if name == 'truth' else fc['mhr']) for v in bb[name]]
        own = render(meshes, Pv, w, h)[0]
        panels.append(np.where(own[..., None] > 0, cv2.addWeighted(img, 0.35, col[own], 0.65, 0), img))
        labels.append(name)
    for p, lab in zip(panels, labels):
        cv2.rectangle(p, (0, 0), (w, 34), (251, 252, 252), -1)
        cv2.putText(p, lab, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (11, 11, 11), 2, cv2.LINE_AA)
    strip = np.hstack(panels)
    scale = min(1.0, 1400/strip.shape[1])
    cv2.imwrite(str(out), cv2.resize(strip, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return img, Pv


def charts(plt, out):
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out/name, dpi=130)
        plt.close(fig)
    # E01 oracle ablation (RESULTS.md of E01, grappling)
    rows = ['nothing', 'side-to-side position', 'limbs + orientation', 'depth along the ray', 'position + orientation']
    vals = {'region pairs': [0.46, 0.44, 0.52, 0.58, 0.67], 'body lock': [0.80, 0.81, 0.87, 0.86, 0.93],
            'legs': [0.46, 0.43, 0.43, 0.58, 0.74]}
    fig, ax = plt.subplots(figsize=(7, 3.4))
    y = np.arange(len(rows))
    for k, (name, c) in enumerate(zip(vals, (BLUE, ORANGE, AQUA))):
        ax.barh(y + (k - 1)*0.26, vals[name], height=0.24, color=c, label=name)
    ax.set_yticks(y, rows)
    ax.set_xlim(0, 1)
    ax.set_xlabel('F1 (1 = perfect)')
    ax.set_title('Corrected with the truth: what fixes the relations?', fontsize=11)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8)
    save(fig, 'chart_oracle.png')
    # E03 depth-gap error by size of the true gap (MoGe-2 vs SAM 3D Body, grappling)
    bins = ['0-10', '10-25', '25-50', '> 50']
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(4)
    ax.bar(x - 0.2, [6.3, 8.4, 11.7, 21.6], 0.38, color=GREY, label='SAM 3D Body (size cue)')
    ax.bar(x + 0.2, [4.5, 5.3, 6.7, 12.0], 0.38, color=BLUE, label='MoGe-2 depth map')
    ax.set_xticks(x, [b + ' cm' for b in bins])
    ax.set_xlabel('true depth gap between the two athletes')
    ax.set_ylabel('median error (cm)')
    ax.set_title('Depth gap between two grapplers: error by gap size', fontsize=11)
    ax.legend(fontsize=8)
    save(fig, 'chart_depth_gap.png')
    # E05 ViCoS defining relations
    pos = ['mount', 'closed guard', 'half guard', 'back']
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(4)
    ax.bar(x - 0.2, [0.14, 0.20, 0.17, 0.39], 0.38, color=GREY, label='before')
    ax.bar(x + 0.2, [0.29, 0.33, 0.26, 0.64], 0.38, color=BLUE, label='with combined depth')
    ax.set_xticks(x, pos)
    ax.set_ylim(0, 1)
    ax.set_ylabel('share of frames with the\ndefining relation')
    ax.set_title('BJJ ground work (ViCoS): defining relation per position', fontsize=11)
    ax.legend(fontsize=8)
    save(fig, 'chart_vicos.png')
    # E07 learning curve
    fig, ax = plt.subplots(figsize=(6, 3.2))
    xs, ys = [6012, 10540, 24124], [6.91, 6.82, 6.71]
    ax.axhline(8.12, color=GREY, lw=1.5, ls='--')
    ax.text(25500, 8.02, 'combined depth (no learning)', ha='right', va='top', fontsize=8, color=INK2)
    ax.axhline(5.77, color=GREY, lw=1.5, ls=':')
    ax.text(25500, 5.87, 'floor: perfect whole-body move', ha='right', va='bottom', fontsize=8, color=INK2)
    ax.plot(xs, ys, color=BLUE, lw=2, marker='o', ms=8)
    for xx, yy, lab in zip(xs, ys, ('4 cameras', '8 cameras', '20 cameras')):
        ax.annotate(f'{lab}\n{yy:.2f}', (xx, yy), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=8)
    ax.set_ylim(5.4, 8.5)
    ax.set_xlabel('training views (same two athletes)')
    ax.set_ylabel('arrangement error (cm)')
    ax.set_title('More camera angles of the same people: small gains', fontsize=11)
    save(fig, 'chart_learning_curve.png')
    # E08 transfer to new people
    fig, ax = plt.subplots(figsize=(6, 3.0))
    x = np.arange(2)
    ax.bar(x - 0.2, [22, 24], 0.38, color=GREY, label='familiar people (held-out moments)')
    ax.bar(x + 0.2, [15, 19], 0.38, color=BLUE, label='new people (MMA)')
    ax.set_xticks(x, ['arrangement error', 'relative depth error'])
    ax.set_ylabel('reduction vs combined depth (%)')
    ax.set_title('The learned correction on people it never saw', fontsize=11)
    ax.legend(fontsize=8)
    save(fig, 'chart_new_people.png')
    # E09 variety vs quantity: dot plot (differences are small; no truncated bars)
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    models = ['A (one pair)', 'A + same pair', 'A + new pair']
    for row, (name, vals_) in enumerate((('MMA (third pair)', [7.73, 7.59, 7.54]),
                                         ('new pair, held out', [6.45, 6.32, 6.18]))):
        for v, c, lab in zip(vals_, (GREY, ORANGE, BLUE), models):
            ax.scatter(v, row, s=70, color=c, zorder=3, label=lab if row == 0 else None)
        ax.plot([min(vals_), max(vals_)], [row, row], color='#e6e5e0', lw=2, zorder=1)
    ax.set_yticks([0, 1], ['MMA (third pair)', 'new pair, held out'])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel('arrangement error (cm, lower is better)')
    ax.set_title('Variety vs quantity: new pair best, but by little', fontsize=11)
    ax.legend(loc='upper center', ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    save(fig, 'chart_variety.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-depth', action='store_true')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plt = plt_setup()
    charts(plt, out)
    va, vb = pick_views()
    if va is None or vb is None:
        sys.exit('no demo view found')
    for tag, (_, sname, cam, f, e) in (('a', va), ('b', vb)):
        print(f'view {tag} {sname}/{cam}/{f}: distance errors ' + ', '.join(f'{k} {v*100:.1f} cm' for k, v in e.items()))
        seq, K, D, bb = bodies(sname, cam, f)
        im, pv = overlay(out/f'view_{tag}_overlay.png', seq, K, D, cam, f, bb, ['truth', 'SAM 3D Body', 'learned'])
        topdown(plt, out/f'view_{tag}_topdown.png', bb, f'Held-out moment seen from above ({sname}, {cam}, frame {f})')
        if tag == 'a':
            img, Pv = im, pv
            pz = B/'phmr_box'/sname/f'{cam}.npz'
            if pz.exists():
                z = np.load(pz)
                k = np.nonzero(z['frames'] == f)[0]
                if len(k):
                    ph = [z['verts'][k[0], n].astype(float) + z['t'][k[0], n] for n in range(2)]
                    overlay(out/'view_prompthmr.png', seq, K, D, cam, f, bb, ['truth', 'SAM 3D Body', 'PromptHMR'], phmr=ph)
    if not a.no_depth:
        import torch
        from engage_depth_probe import load_model
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        maps = []
        for name in ('moge', 'depthpro'):
            run = load_model(name)
            maps.append(run(rgb, Pv[0, 0]))
            del run
            torch.cuda.empty_cache()
        asp = img.shape[1]/img.shape[0]
        fig, axs = plt.subplots(1, 3, figsize=(3*3.2*asp + 0.6, 3.6))
        axs[0].imshow(rgb)
        axs[0].set_title('photo', fontsize=10)
        for ax, dm, t in zip(axs[1:], maps, ('MoGe-2', 'Depth Pro')):
            lo, hi = np.nanpercentile(dm, [1, 60])
            ax.imshow(np.clip(dm, lo, hi), cmap='viridis_r')
            ax.set_title(t, fontsize=10)
        for ax in axs:
            ax.axis('off')
        fig.suptitle('Monocular depth maps (yellow = nearer)', fontsize=11)
        fig.tight_layout()
        fig.savefig(out/'view_depth.png', dpi=110)
        plt.close(fig)
    print('wrote', sorted(p.name for p in out.glob('*.png')))


if __name__ == '__main__':
    main()
