"""Make folder clips for the engagement demo (see engage_clip.py): frames at a fixed rate from any video, and for MAMMA
cameras their ground truth in the common truth.npz format. Experimental.

  video  (.venv-depth, CPU + network)  a window of an internet video (yt-dlp) or a local file, cut to frames
      ../.venv-depth/bin/python engage_clip_make.py video --name NAME --src URL_OR_PATH --start 12.5 --dur 10 [--fps 15]
  mamma  (.venv-phmr, CPU)       a MAMMA evaluation camera: frames from its video, SMPL-X truth posed into that camera
      ../.venv-phmr/bin/python engage_clip_make.py mamma --name NAME --seq mamma_eval_dance/SEQ --cam IOI_01 [--first 0 --count 300 --step 2]
  sheet  (.venv-depth, CPU)          a contact sheet of a video (4x4 stills) to pick a window by eye
      ../.venv-depth/bin/python engage_clip_make.py sheet --src URL_OR_PATH --out sheet.jpg

Clips land in benchmarks/engage_h4d/clips/NAME (frames/00000.jpg..., clip.json, truth.npz). Internet clips stay
internal research material; they are not published.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT/'benchmarks/engage_h4d/clips'
YTDLP = str(ROOT/'.venv/bin/yt-dlp')
MAMMA = Path.home()/'Code/shared/datasets/mamma/data'
SMPLX = Path.home()/'Code/shared/models/prompthmr-data/body_models'


def fetch(src, d, start=None, dur=None):
    """A local copy of the video (only the window, for a URL); returns its path and title."""
    if Path(src).exists():
        return Path(src), Path(src).name
    out = d/'source.mp4'
    cmd = [YTDLP, '-q', '--no-playlist', '-f', 'bv*[height<=1080][ext=mp4]/bv*[height<=1080]/b[height<=1080]',
           '--remux-video', 'mp4', '-o', str(out), '--print', 'after_move:title', '--no-simulate']
    if start is not None:
        cmd += ['--download-sections', f'*{start:.2f}-{start + dur + 1:.2f}', '--force-keyframes-at-cuts']
    r = subprocess.run(cmd + [src], capture_output=True, text=True)
    if r.returncode or not out.exists():
        sys.exit(f'yt-dlp failed: {r.stderr[-800:]}')
    return out, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''


def probe(path):
    r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate',
                        '-of', 'json', str(path)], capture_output=True, text=True)
    s = json.loads(r.stdout)['streams'][0]
    a, b = s['r_frame_rate'].split('/')
    return int(s['width']), int(s['height']), float(a) / float(b)


def cmd_video(a):
    d = CLIPS/a.name
    if (d/'clip.json').exists() and not a.force:
        print('exists:', d); return
    shutil.rmtree(d/'frames', ignore_errors=True); (d/'frames').mkdir(parents=True)
    src, title = fetch(a.src, d, a.start, a.dur)
    local = Path(a.src).exists()
    ss = a.start if local else min(a.start, 0.0)                    # a downloaded section starts at its window
    vf = f'fps={a.fps}' + (f",scale='if(gt(iw,ih),min({a.max_side},iw),-2)':'if(gt(iw,ih),-2,min({a.max_side},ih))'" if a.max_side else '')
    subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{ss:.3f}', '-i', str(src), '-t', f'{a.dur:.3f}', '-vf', vf, '-q:v', '2',
                    '-start_number', '0', str(d/'frames/%05d.jpg')], check=True)
    import cv2
    first = sorted((d/'frames').glob('*.jpg'))
    H, W = cv2.imread(str(first[0])).shape[:2]
    meta = {'name': a.name, 'fps': a.fps, 'size': [W, H], 'K': None, 'lens_source': None, 'truth': False,
            'source': {'kind': 'file' if local else 'url', 'src': a.src, 'title': title, 'start_s': a.start, 'dur_s': a.dur},
            'engagement': a.engagement, 'note': a.note}
    json.dump(meta, open(d/'clip.json', 'w'), indent=1)
    if not local and not a.keep_source:
        (d/'source.mp4').unlink(missing_ok=True)
    print(f'{a.name}: {len(first)} frames {W}x{H} at {a.fps} fps; "{title}"')


def cmd_sheet(a):
    import cv2
    tmp = Path(a.out).with_suffix('.src.mp4')
    if Path(a.src).exists():
        src = Path(a.src)
    elif tmp.exists():
        src = tmp
    else:
        r = subprocess.run([YTDLP, '-q', '--no-playlist', '-f', 'bv*[height<=360][ext=mp4]/bv*[height<=480]/b', '-o', str(tmp), a.src],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(r.stderr[-500:])
        src = tmp
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(src)], capture_output=True, text=True)
    dur = float(r.stdout.strip() or 0)
    t0, t1 = (a.window or [0, dur])
    tiles = []
    for t in np.linspace(t0, t1, a.grid * a.grid, endpoint=False):    # ffmpeg grabs: any codec the source uses
        r = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.2f}', '-i', str(src), '-frames:v', '1', '-vf', 'scale=480:270',
                            '-f', 'image2pipe', '-vcodec', 'png', '-'], capture_output=True)
        img = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR) if r.stdout else None
        img = np.zeros((270, 480, 3), np.uint8) if img is None else img
        cv2.putText(img, f'{t:.0f}s', (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)
    rows = [np.hstack(tiles[i:i + a.grid]) for i in range(0, len(tiles), a.grid)]
    cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f'{a.out}: {dur:.0f} s video, stills {t0:.0f}-{t1:.0f} s')
    if src == tmp and not a.keep_source:
        tmp.unlink()


def smplx_posed(G, n):
    """Vertices [T, V, 3] and SMPL-order joints [T, 24, 3] (world) of person n of a MAMMA global.npz, from its own
    template (v_template) and SMPL-X pose (flat hand mean as the file says)."""
    import smplx
    import torch
    V0 = torch.tensor(G['v_template'][n], dtype=torch.float32)
    m = smplx.SMPLX(str(SMPLX/'smplx'), gender='neutral', use_pca=False, flat_hand_mean=bool(G['flat_hand_mean']),
                    num_betas=10, v_template=V0)
    p = torch.tensor(G['pose_world'][:, n], dtype=torch.float32); tr = torch.tensor(G['pose_trans_world'][:, n], dtype=torch.float32)
    T = len(p)
    z = lambda k: torch.zeros(T, k)
    with torch.no_grad():
        o = m(global_orient=p[:, :3], body_pose=p[:, 3:66], jaw_pose=p[:, 66:69], leye_pose=p[:, 69:72], reye_pose=p[:, 72:75],
              left_hand_pose=p[:, 75:120], right_hand_pose=p[:, 120:165], transl=tr, betas=z(10), expression=z(10))
    J = o.joints.numpy()
    # SMPL's 24: the 22 body joints are shared; SMPL's hand joints ~ the middle-finger base (SMPL-X 28 left, 43 right)
    return o.vertices.numpy(), np.concatenate([J[:, :22], J[:, [28, 43]]], 1), m.faces.astype(np.int32)


def cmd_mamma(a):
    import cv2
    d = CLIPS/a.name
    sd = MAMMA/a.seq
    G = np.load(sd/'gt/global.npz', allow_pickle=False)
    Cm = np.load(sd/f'gt/{a.cam}.npz', allow_pickle=False)
    E = Cm['cam_ext'].astype(np.float64); K = Cm['cam_int'].astype(np.float64)
    W, H = int(Cm['cam_img_w']), int(Cm['cam_img_h'])
    n_gt = int(G['frames_len'])
    gt_frames = list(range(a.first, min(n_gt, a.first + a.count), a.step))
    shutil.rmtree(d/'frames', ignore_errors=True); (d/'frames').mkdir(parents=True)
    vid = next(p for p in (sd/f'videos/{a.cam}.mp4', sd/f'videos_crf16/{a.cam}.mp4', sd/f'videos_crf24/{a.cam}.mp4') if p.exists())
    cap = cv2.VideoCapture(str(vid))
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = {g: i for i, g in enumerate(gt_frames)}
    g = 0
    while True:
        ok, img = cap.read()
        if not ok or g > gt_frames[-1]:
            break
        if g in want:
            cv2.imwrite(str(d/'frames'/f'{want[g]:05d}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        g += 1
    verts, joints = [], []
    faces = None
    for n in range(int(G['people_len'])):
        V, J, faces = smplx_posed(G, n)
        verts.append(V[gt_frames]); joints.append(J[gt_frames])
    verts, joints = np.stack(verts, 1), np.stack(joints, 1)                    # [N, 2, V, 3], world
    R, t = E[:3, :3], E[:3, 3]
    cam = lambda X: X @ R.T + t
    vc, jc = cam(verts), cam(joints)
    np.savez_compressed(d/'truth.npz', frames=np.arange(len(gt_frames)), verts=vc.astype(np.float16), joints=jc.astype(np.float32),
                        faces=faces, gt_frames=np.array(gt_frames), in_image=Cm['is_body_in_img'][gt_frames])
    meta = {'name': a.name, 'fps': 30.0 / a.step, 'size': [W, H], 'K': K.tolist(), 'lens_source': f'MAMMA calibration ({a.cam}; radial distortion ignored)',
            'truth': True, 'source': {'kind': 'mamma', 'seq': a.seq, 'cam': a.cam, 'gt_frames': [gt_frames[0], gt_frames[-1], a.step],
                                      'video': vid.parent.name, 'video_frames': n_vid, 'gt_frames_len': n_gt, 'keywords': [str(x) for x in G['keywords']]},
            'engagement': a.engagement, 'note': a.note}
    json.dump(meta, open(d/'clip.json', 'w'), indent=1)
    z = vc[..., 2]
    print(f'{a.name}: {len(gt_frames)} frames {W}x{H}; video {n_vid} frames vs truth {n_gt}; body depth {np.median(z):.2f} m '
          f'(min {z.min():.2f}); people in image {Cm["is_body_in_img"][gt_frames].mean():.0%}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('video')
    v.add_argument('--name', required=True); v.add_argument('--src', required=True)
    v.add_argument('--start', type=float, default=0.0); v.add_argument('--dur', type=float, default=10.0)
    v.add_argument('--fps', type=float, default=15.0); v.add_argument('--max-side', type=int, default=1920)
    v.add_argument('--engagement', default=''); v.add_argument('--note', default='')
    v.add_argument('--keep-source', action='store_true'); v.add_argument('--force', action='store_true')
    m = sub.add_parser('mamma')
    m.add_argument('--name', required=True); m.add_argument('--seq', required=True); m.add_argument('--cam', default='IOI_01')
    m.add_argument('--first', type=int, default=0); m.add_argument('--count', type=int, default=300); m.add_argument('--step', type=int, default=2)
    m.add_argument('--engagement', default=''); m.add_argument('--note', default='')
    s = sub.add_parser('sheet')
    s.add_argument('--src', required=True); s.add_argument('--out', required=True)
    s.add_argument('--grid', type=int, default=4); s.add_argument('--window', type=float, nargs=2)
    s.add_argument('--keep-source', action='store_true')
    a = ap.parse_args()
    {'video': cmd_video, 'mamma': cmd_mamma, 'sheet': cmd_sheet}[a.cmd](a)


if __name__ == '__main__':
    main()
