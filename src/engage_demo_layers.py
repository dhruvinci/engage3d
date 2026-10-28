"""The engagement demo: peel six layers of understanding off one camera of a clip, without ground truth, then score
each layer against the truth where the clip has one. Experimental.

A clip is a Harmony4D sequence + camera (--seq NAME --cam camNN --frames first:last:step) or a folder of frames cut
from any video (--seq DIR holding clip.json; see engage_clip.py), e.g. a MAMMA camera with its truth or an internet
video without.

  lens     (.venv-depth)   folder clips without a known lens: MoGe-2's focal length, median over 5 frames, into clip.json
  detect   (.venv-rfdetr)  L1 people: RF-DETR (COCO, Medium) person boxes on every frame
  track    (.venv-depth)   L1 identities: the two people kept apart through contact. Hungarian matching on box IoU
                           with the previous frame plus clothing colour (HSV histogram of the torso area against each
                           person's appearance, learned while they stand apart), forward and backward from a frame where
                           they stand apart; a track with no match keeps its last box and is flagged held
  pose     (.venv-pose)    L3 skeletons: ViTPose+ 2D keypoints in each tracked box
  segment  (.venv-bmp)     L2 segmentation: BBoxMaskPose v2's SAM-pose2seg mask from each person's 3 most confident
                           keypoints; overlaps go to the higher-scoring mask
  body3d   (.venv-s3d)     L4 3D: SAM 3D Body per person from box + mask, its translation re-solved through the
                           clip's lens (Harmony4D's true fisheye, else the pinhole), so both bodies sit in one camera space
  place    (.venv-depth)   L4 placement: each body moved along its ray toward MoGe-2's relative depth of the pair
                           (the combined depth of depthmap_results.md)
  contact  (.venv-depth)   L5 contact: minimum gap, touching body-region pairs (16 regions, within TAU) and the
                           embrace (a torso threading the other's arm loop), for the prediction and for the truth
  context  (.venv)         L6 context: Astra (gpt-6-astra, logged-in Codex CLI) writes the narrative and labels from
                           keyframes marked with the two people plus the measured L1-L5 timeline, citing its events
  score    (.venv-depth)   every layer against the clip's truth; without truth, the pipeline's own health numbers
  render   (.venv-depth)   the demo video: six panels, one per layer, plus the measured timeline
  check    (.venv-depth)   quality gates on every layer's output; exit 1 (and GATE_FAILED.txt) when one fails

  ../.venv-<env>/bin/python engage_demo_layers.py <stage> --seq 002_hugging --cam cam01 --out DIR
  ../.venv-<env>/bin/python engage_demo_layers.py <stage> --seq ../benchmarks/engage_h4d/clips/NAME --out DIR
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engage_clip import SMPL_FOR_COCO, clip_of   # noqa: E402,F401

TAU = 0.03                     # contact distance between two bodies, metres
TAU_HAND = 0.10                # predictions only: a hand or forearm within this distance counts as touching
HAND_REGIONS = [8, 9, 6, 7]    # engage_relations.REGIONS indices: l_hand, r_hand, l_forearm, r_forearm
APPEARANCE_W = 1.0             # weight of the clothing-colour distance in tracking
ACTIVE = 0.008                 # a person moving at least this many box heights per frame counts as engaged
SALIENCE_POW = 0.5             # pair salience = motion (box heights per frame, camera motion removed) x height^this
MIN_EVENT_S = 0.5              # contact / embrace states shorter than this are merged into a neighbour
COCO_LIMB = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

WINDOW = None   # (first, last, step) in the clip's frame numbers, from --frames


def frames_of(clip):
    fs = clip.frames
    if WINDOW:
        a, b, s = WINDOW
        fs = [f for f in fs if a <= f <= b and (f - a) % s == 0]
    return fs


def seed_frames(T):
    """Clip indices where samtrack seeds each person: the first and last frames where the two stand apart; for a pair
    that never stands apart, the tracker's own start frame (its most separable engaged frame)."""
    if T.get('seed_frames'):                                        # set by the tracker (a pair hint): trust only that
        return list(T['seed_frames'])
    bx = np.array(T['boxes']); hd = np.array(T['held'])
    apart = [k for k in range(len(T['frames'])) if iou(bx[k, 0], bx[k, 1]) < 0.05 and not hd[k].any()]
    if apart:
        # the box tracker can drift onto an onlooker late in a clip (sumo: a ringside spectator after the wrestlers
        # merged into one box), so the late seed is the last apart frame where the tracked pair is still the frame's
        # most engaged pair (the tracker's own per-frame choice, stored as engaged_pair)
        E = T.get('engaged_pair')
        if E is None:                                               # tracks from before this check
            return sorted({apart[0], apart[-1]})
        def agrees(k):
            e = E[k]
            if e is None:
                return False
            e = np.array(e)
            return max(min(iou(bx[k, 0], e[0]), iou(bx[k, 1], e[1])), min(iou(bx[k, 0], e[1]), iou(bx[k, 1], e[0]))) >= 0.5
        late = [k for k in apart if agrees(k)]
        return sorted({apart[0], late[-1]}) if late else [apart[0]]
    return [T['frames'].index(T['bootstrap_frame'])] if T.get('bootstrap_frame') in T['frames'] else [0, len(T['frames']) - 1]


def iou(a, b):
    x1, y1 = np.maximum(a[:2], b[:2]); x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def pack(masks):
    return np.packbits(masks.astype(bool), axis=None), np.array(masks.shape)


def unpack(bits, shape):
    return np.unpackbits(bits)[:int(np.prod(shape))].reshape(shape).astype(bool)


# ---------------------------------------------------------------- L1: people and identities
def cmd_detect(a, out):
    import cv2
    import rfdetr
    clip = a.clip
    det = rfdetr.RFDETRMedium(pretrain_weights=str(Path.home()/'Code/shared/models/rfdetr/rf-detr-medium.pth'))
    rec, t0 = [], time.time()
    for f in frames_of(clip):
        img = cv2.cvtColor(cv2.imread(str(clip.image_path(f))), cv2.COLOR_BGR2RGB)
        d = det.predict(img, threshold=0.3)
        keep = d.class_id == 1
        rec.append({'frame': int(f), 'boxes': d.xyxy[keep].round(1).tolist(), 'scores': d.confidence[keep].round(3).tolist()})
    json.dump({'size': list(img.shape[1::-1]), 'frames': rec}, open(out/'detect.json', 'w'))
    print(f'detect: {len(rec)} frames, people per frame {np.mean([len(r["boxes"]) for r in rec]):.2f}, {time.time() - t0:.0f}s')


def appearance(img, b):
    """Normalised HSV histogram of a box's torso area (middle half across, 20-80% down)."""
    import cv2
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    crop = img[int(y1 + 0.2 * h):int(y1 + 0.8 * h), int(x1 + 0.25 * w):int(x2 - 0.25 * w)]
    if crop.size == 0:
        return np.ones((16, 8), np.float32) / 128
    hist = cv2.calcHist([cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return hist / max(hist.sum(), 1)


def bhatt(p, q):
    return float(np.sqrt(max(0.0, 1 - np.sum(np.sqrt(p * q)))))


def cmd_track(a, out):
    import cv2
    from scipy.optimize import linear_sum_assignment
    clip = a.clip
    D = json.load(open(out/'detect.json'))
    F = D['frames']
    W, H = D['size']
    # bootstrap: the engaged pair is the two people closest to each other for their size (similar scale), not the
    # largest; onlookers stand apart. Start at the first frame where that pair is close but still separate.
    def camera_shift():
        """Per frame step, the crowd's median image motion (pixels): with 4 or more people matched between frames, a
        moving camera moves everyone, so it is subtracted before judging who moves; with fewer, zero (no change)."""
        out = [np.zeros(2)]
        for kk in range(1, len(F)):
            a_, b_ = (np.array(F[kk - 1]['boxes']) if F[kk - 1]['boxes'] else np.zeros((0, 4))), \
                     (np.array(F[kk]['boxes']) if F[kk]['boxes'] else np.zeros((0, 4)))
            d = []
            for q in a_:
                if len(b_):
                    j = int(np.argmax([iou(q, x) for x in b_]))
                    if iou(q, b_[j]) >= 0.5:
                        d.append((b_[j, :2] + b_[j, 2:]) / 2 - (q[:2] + q[2:]) / 2)
            out.append(np.median(d, 0) if len(d) >= 4 else np.zeros(2))
        return out
    shift = camera_shift()

    def activity(k, box, span=30):
        """Mean centre displacement per frame (in box heights, camera motion removed) following box forward by best
        IoU for span frames."""
        cur, moves = np.asarray(box, float), []
        for kk in range(k + 1, min(len(F), k + span + 1)):
            b2 = np.array(F[kk]['boxes']) if F[kk]['boxes'] else np.zeros((0, 4))
            if not len(b2):
                break
            j = int(np.argmax([iou(cur, q) for q in b2]))
            if iou(cur, b2[j]) < 0.3:
                break
            moves.append(np.linalg.norm((b2[j, :2] + b2[j, 2:]) / 2 - (cur[:2] + cur[2:]) / 2 - shift[kk]) / (cur[3] - cur[1]))
            cur = b2[j]
        return float(np.mean(moves)) if len(moves) >= 5 else 0.0

    def pair(r, k):
        """The engaged pair: people who move and are in the foreground (salience = motion x sqrt(height)), close to
        each other and of similar size. Returns (boxes, normalised distance)."""
        b = np.array(r['boxes']) if r['boxes'] else np.zeros((0, 4))
        s = np.array(r['scores']) if r['scores'] else np.zeros(0)
        b = b[s >= 0.5]
        if len(b) < 2:
            return None, np.inf, 0.0
        h = b[:, 3] - b[:, 1]; c = (b[:, :2] + b[:, 2:]) / 2
        sal = np.array([activity(k, q) for q in b]) * h ** SALIENCE_POW
        best, score, dist = None, -1.0, np.inf
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                if max(h[i], h[j]) / max(min(h[i], h[j]), 1) > 2.0:
                    continue
                d = np.linalg.norm(c[i] - c[j]) / ((h[i] + h[j]) / 2)
                sc = (sal[i] + sal[j]) / (1 + d)
                if sc > score:
                    best, score, dist = b[[i, j]], sc, d
        return best, dist, max(score, 0.0)

    cand = [(k, *pair(r, k)) for k, r in enumerate(F)]
    # only frames where the best pair is strongly engaged count: a pair that holds each other all clip (ballroom) is
    # never apart, and the closest separate pair in some frame is then a pair of bystanders
    k_top = max(range(len(F)), key=lambda k: cand[k][3])
    top = cand[k_top][3]
    # ... and at the scale of the clip's most engaged pair: in a frame where one of that pair drops out of the
    # detections, the best remaining pair can be two onlookers (seated spectators at a tango, 40% of the dancers' height)
    top_h = float(np.mean(cand[k_top][1][:, 3] - cand[k_top][1][:, 1])) if cand[k_top][1] is not None else 0.0
    same = lambda b: float(np.mean(b[:, 3] - b[:, 1])) >= 0.6 * top_h
    starts = [k for k, b, d, sc in cand if b is not None and d < 1.5 and iou(b[0], b[1]) < 0.05 and sc >= 0.5 * top and same(b)]
    if not starts:   # a pair that never stands apart (close-embrace dance): the most separable strongly engaged frame
        near = [k for k, b, d, sc in cand if b is not None and sc >= 0.5 * top and iou(b[0], b[1]) < 0.5 and same(b)]
        starts = [min(near, key=lambda k: iou(*cand[k][1]))] if near else []
    k0 = starts[0] if starts else k_top
    first = cand[k0][1]
    hint = getattr(clip, 'meta', {}).get('pair_hint')
    if hint and hint.get('boxes'):   # a drawn prompt: {"frame": f, "boxes": [[x1, y1, x2, y2], ...]}, for when the
        k0 = [r['frame'] for r in F].index(hint['frame'])               # detector does not find the two people at all
        first = np.array(hint['boxes'], float)
        print(f'track: start from the two hint boxes at frame {hint["frame"]}')
    elif hint:   # a one-click prompt (clip.json): {"frame": f, "points": [[x, y], [x, y]]}, one point on each person
        k0 = [r['frame'] for r in F].index(hint['frame'])
        bb = np.array(F[k0]['boxes'])
        ctr = (bb[:, :2] + bb[:, 2:]) / 2
        pick = []
        for x, y in hint['points']:
            inside = (bb[:, 0] <= x) & (x <= bb[:, 2]) & (bb[:, 1] <= y) & (y <= bb[:, 3])
            d = np.linalg.norm(ctr - [x, y], axis=1) + np.where(inside, 0, 1e6)
            d[pick] = np.inf
            pick.append(int(np.argmin(d)))
        first = bb[pick]
        print(f'track: start from the pair hint at frame {hint["frame"]}')
    first = first[np.argsort(first[:, 0])]                         # A = left person at the start
    boxes = np.full((len(F), 2, 4), np.nan); held = np.zeros((len(F), 2), bool)
    boxes[k0] = first
    img0 = cv2.imread(str(clip.image_path(F[k0]['frame'])))
    model0 = [appearance(img0, b) for b in first]
    for order in (range(k0 + 1, len(F)), range(k0 - 1, -1, -1)):
        prev, model = first.copy(), [m.copy() for m in model0]
        for k in order:
            b = np.array(F[k]['boxes']) if F[k]['boxes'] else np.zeros((0, 4))
            cur = prev.copy(); h = np.ones(2, bool)
            if len(b):
                img = cv2.imread(str(clip.image_path(F[k]['frame'])))
                app = [appearance(img, q) for q in b]
                C = np.array([[(1 - iou(p, q)) + APPEARANCE_W * bhatt(m, ap) for q, ap in zip(b, app)]
                              for p, m in zip(prev, model)])
                r, c = linear_sum_assignment(C)
                for i, j in zip(r, c):
                    if C[i, j] < 0.7 + APPEARANCE_W * 0.5:
                        cur[i], h[i] = b[j], False
                        clear = all(iou(b[j], q) < 0.1 for jj, q in enumerate(b) if jj != j)
                        if clear:   # learn appearance only while the person stands apart
                            model[i] = 0.9 * model[i] + 0.1 * app[j]
            boxes[k], held[k] = cur, h
            prev = cur
    json.dump({'frames': [r['frame'] for r in F], 'boxes': boxes.round(1).tolist(), 'held': held.tolist(),
               'bootstrap_frame': F[k0]['frame'], 'salience_pow': SALIENCE_POW,
               'engaged_pair': [None if c_[1] is None else np.round(c_[1], 1).tolist() for c_ in cand],
               **({'seed_frames': [k0], 'pair_hint': hint} if hint else {})}, open(out/'track.json', 'w'))
    print(f'track: bootstrap at frame {F[k0]["frame"]}; held (no detection matched) {held.mean():.1%} of person-frames')


# ---------------------------------------------------------------- L1+L2 by video segmentation (alternative)
def cmd_samtrack(a, out):
    """SAM 3.1 video tracking: each person seeded with their mask where the two stand apart (first and last frames
    of --seed's run: box corners, interior points, a negative point on the partner), then propagated through the clip
    at 1280x720. Writes track.json (boxes from the masks; held = empty mask) and segment.npz (masks at full size)."""
    import shutil
    import tempfile
    import cv2
    from autolabel_sam31_video import add_points, build_predictor, close, interactive_mode, propagate, start
    from autolabel_densify import interior_points
    from autolabel_common import mask_box
    clip = a.clip
    src = Path(a.seed)
    T = json.load(open(src/'track.json')); M0 = load_masks(src)
    frames = T['frames']
    H0, W0 = M0[0].shape[1:]
    sc = 1280 / max(W0, H0)
    w, h = int(round(W0 * sc)), int(round(H0 * sc))
    out_masks = {}
    with tempfile.TemporaryDirectory(dir=str(out)) as wd:
        for j, f in enumerate(frames):
            cv2.imwrite(f'{wd}/{j:05d}.jpg', cv2.resize(cv2.imread(str(clip.image_path(f))), (w, h), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        predictor = interactive_mode(build_predictor())
        sid = start(predictor, wd)
        try:
            seeds = seed_frames(T)
            print('samtrack seeds at clip frames', seeds, flush=True)
            # a later seed must keep the first seed's identities: the box tracker can swap the two through a scramble
            # (wrestling takedown), so each later seed is relabelled if clothing colour matches better swapped
            app = {}
            for k in seeds:
                img = cv2.imread(str(clip.image_path(frames[k])))
                app[k] = [appearance(img, np.array(T['boxes'][k][n])) for n in range(2)]
            swap = {}
            for k in seeds[1:]:
                a0, ak = app[seeds[0]], app[k]
                keep = bhatt(a0[0], ak[0]) + bhatt(a0[1], ak[1]); cross = bhatt(a0[0], ak[1]) + bhatt(a0[1], ak[0])
                swap[k] = cross < keep - 0.05
                if swap[k]:
                    print(f'samtrack: seed at clip frame {k} relabelled (clothing colour: kept {keep:.2f}, swapped {cross:.2f})', flush=True)
            for k in seeds:
                small = [cv2.resize(M0[k][n].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool) for n in range(2)]
                if swap.get(k):
                    small = small[::-1]
                    T['boxes'][k] = T['boxes'][k][::-1]
                boxes_k = [np.array(T['boxes'][k][n]) * sc for n in range(2)]
                tiny = [small[n].sum() < 0.15 * max(1.0, (boxes_k[n][2] - boxes_k[n][0]) * (boxes_k[n][3] - boxes_k[n][1]))
                        for n in range(2)]
                if any(tiny):    # a person mostly behind the partner: seed BOTH from their tracked boxes, each with a
                    free = []    # positive point where the partner's box does not reach and a negative one on the partner's
                    for n in range(2):
                        x1, y1, x2, y2 = boxes_k[n]
                        ox1, _, ox2, _ = boxes_k[1 - n]
                        xs = [x for x in np.linspace(x1, x2, 9)[1:-1] if not (ox1 <= x <= ox2)] or list(np.linspace(x1, x2, 9)[1:-1])
                        free.append((float(np.median(xs)), float(y1 + 0.45 * (y2 - y1))))
                    for n in range(2):
                        x1, y1, x2, y2 = boxes_k[n]
                        (px, py), (qx, qy) = free[n], free[1 - n]
                        add_points(predictor, sid, k, n + 1, [[x1 / w, y1 / h], [x2 / w, y2 / h], [px / w, py / h], [qx / w, qy / h]],
                                   [2, 3, 1, 0])
                    print(f'samtrack: both people seeded from their boxes at clip frame {k} (a seed mask was unusable)', flush=True)
                    continue
                for n in range(2):
                    own = interior_points(small[n]); neg = interior_points(small[1 - n], 1)[:1] if not tiny[1 - n] else []
                    x1, y1, x2, y2 = mask_box(small[n])
                    add_points(predictor, sid, k, n + 1, [[x1 / w, y1 / h], [x2 / w, y2 / h]] + [[x / w, y / h] for x, y in own + neg],
                               [2, 3] + [1] * len(own) + [0] * len(neg))
            for j, masks in propagate(predictor, sid, direction='both', start_frame_idx=seeds[0]):
                out_masks[j] = masks
        finally:
            close(predictor, sid)
    boxes, held, bits = [], [], []
    for j in range(len(frames)):
        ms = []
        for n in range(2):
            m = out_masks.get(j, {}).get(n + 1)
            m = np.zeros((h, w), bool) if m is None else m.reshape(h, w)
            ms.append(cv2.resize(m.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool))
        both = ms[0] & ms[1]
        ms[0][both] = ms[1][both] = True                            # overlap kept: SAM gives each person their own mask
        bx = [mask_box(m) for m in ms]
        held.append([b is None for b in bx])
        boxes.append([b if b is not None else T['boxes'][j][n] for n, b in enumerate(bx)])
        bits.append(pack(np.stack(ms))[0])
    shutil.copy(src/'detect.json', out/'detect.json')
    json.dump({'frames': frames, 'boxes': boxes, 'held': held, 'bootstrap_frame': frames[0], 'method': 'SAM 3.1 video'},
              open(out/'track.json', 'w'))
    np.savez_compressed(out/'segment.npz', bits=np.stack(bits), shape=np.array([2, H0, W0]), scores=np.ones((len(frames), 2)))
    print(f'samtrack: {len(frames)} frames; empty person masks {np.mean(held):.1%}')


def cmd_pose_masked(a, out):
    """ViTPose+ with the partner greyed out: each person's own mask kept, the partner's other pixels set to grey,
    in a box around the person's mask (10% margin)."""
    from PIL import Image
    from transformers import AutoProcessor, VitPoseForPoseEstimation
    import torch
    from autolabel_pose2d import MODEL, estimate
    clip = a.clip
    T = json.load(open(out/'track.json')); M = load_masks(out)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = VitPoseForPoseEstimation.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
    kp = np.zeros((len(T['frames']), 2, 17, 2)); sc = np.zeros((len(T['frames']), 2, 17))
    for k, f in enumerate(T['frames']):
        arr = np.asarray(Image.open(clip.image_path(f)).convert('RGB'))
        for n in range(2):
            img = arr.copy()
            img[M[k][1 - n] & ~M[k][n]] = 128
            x1, y1, x2, y2 = T['boxes'][k][n]
            mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
            bx = [max(0, x1 - mx), max(0, y1 - my), (x2 - x1) + 2 * mx, (y2 - y1) + 2 * my]
            r = estimate(proc, model, [Image.fromarray(img)], [[bx]])[0]
            kp[k, n], sc[k, n] = r[0][0], r[1][0]
    np.savez_compressed(out/'pose.npz', kp=kp, score=sc)
    print(f'pose_masked: {len(kp)} frames, median keypoint confidence {np.median(sc):.2f}')


def cmd_pose_pmpose(a, out):
    """BBoxMaskPose v2's PMPose (mask-conditioned, with presence and visibility per keypoint) on each tracked person's
    box and mask. --variant PMPose-b / PMPose-l / PMPose-h."""
    import cv2
    sys.path.insert(0, str(Path.home()/'Code/kakashi-research/external/BBoxMaskPose'))
    from pmpose.api import PMPose
    clip = a.clip
    T = json.load(open(out/'track.json')); M = load_masks(out)
    model = PMPose(device='cuda', variant=a.variant, from_pretrained=True)
    n_f = len(T['frames'])
    kp = np.zeros((n_f, 2, 17, 2)); sc = np.zeros((n_f, 2, 17)); pres = np.zeros((n_f, 2, 17)); vis = np.zeros((n_f, 2, 17))
    for k, f in enumerate(T['frames']):
        img = cv2.imread(str(clip.image_path(f)))
        bx = np.array(T['boxes'][k], np.float32)
        try:
            K_, P_, V_, _ = model.predict(img, bx, masks=np.stack(filled(M[k], T['boxes'][k])).astype(np.uint8))
        except ValueError:                                          # a mask too small for PMPose's crop
            try:
                full = [np.zeros_like(M[k][0]) for _ in range(2)]
                K_, P_, V_, _ = model.predict(img, bx, masks=np.stack(filled(full, T['boxes'][k])).astype(np.uint8))
                print(f'pose_pmpose: frame {k}: box-shaped masks used')
            except ValueError:
                print(f'pose_pmpose: frame {k}: no skeleton (left at zero confidence)')
                continue
        K_, P_, V_ = K_[:, :17], P_.reshape(2, -1)[:, :17], V_.reshape(2, -1)[:, :17]   # COCO body (23 incl. feet)
        kp[k], sc[k] = K_[..., :2], K_[..., 2]
        pres[k], vis[k] = P_, V_
    np.savez_compressed(out/'pose.npz', kp=kp, score=sc, presence=pres, visibility=vis)
    print(f'pose_pmpose ({a.variant}): {n_f} frames, median keypoint score {np.median(sc):.2f}')


SAPIENS = {'0.3b': ('facebook/sapiens-pose-0.3b-torchscript', 'sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2'),
           '0.6b': ('facebook/sapiens-pose-0.6b-torchscript', 'sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2'),
           '1b': ('facebook/sapiens-pose-1b-torchscript', 'sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2')}
GOLIATH_TO_COCO = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]


def cmd_pose_sapiens(a, out):
    """Sapiens pose (Goliath, 308 keypoints; COCO body joints kept), top-down at 1024x768 on each tracked person's
    box (25% margin, 3:4), with the partner greyed out as in pose_masked. --variant 0.3b / 0.6b / 1b."""
    import cv2
    import torch
    from huggingface_hub import hf_hub_download
    clip = a.clip
    T = json.load(open(out/'track.json')); M = load_masks(out)
    repo, fname = SAPIENS[a.variant]
    model = torch.jit.load(hf_hub_download(repo, fname), map_location='cuda').eval()
    mean = torch.tensor([123.675, 116.28, 103.53], device='cuda').view(1, 3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], device='cuda').view(1, 3, 1, 1)
    IW, IH = 768, 1024
    n_f = len(T['frames'])
    kp = np.zeros((n_f, 2, 17, 2)); sc = np.zeros((n_f, 2, 17))
    for k, f in enumerate(T['frames']):
        rgb = cv2.cvtColor(cv2.imread(str(clip.image_path(f))), cv2.COLOR_BGR2RGB)
        for n in range(2):
            img = rgb.copy()
            img[M[k][1 - n] & ~M[k][n]] = 128
            x1, y1, x2, y2 = T['boxes'][k][n]
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 1.25, (y2 - y1) * 1.25
            if w / h > IW / IH:
                h = w * IH / IW
            else:
                w = h * IW / IH
            A = np.array([[IW / w, 0, IW / 2 - cx * IW / w], [0, IH / h, IH / 2 - cy * IH / h]])
            crop = cv2.warpAffine(img, A, (IW, IH), flags=cv2.INTER_LINEAR)
            x = torch.from_numpy(crop).permute(2, 0, 1)[None].float().cuda()
            with torch.inference_mode():
                hm = model((x - mean) / std)[0][GOLIATH_TO_COCO].float()          # [17, 256, 192]
            HH, WW = hm.shape[1:]
            flat = hm.reshape(17, -1)
            conf, idx = flat.max(1)
            ys, xs = (idx // WW).float(), (idx % WW).float()
            px = (xs.cpu().numpy() + 0.5) * IW / WW; py = (ys.cpu().numpy() + 0.5) * IH / HH
            kp[k, n] = np.c_[(px - A[0, 2]) / A[0, 0], (py - A[1, 2]) / A[1, 1]]
            sc[k, n] = conf.cpu().numpy()
        torch.cuda.empty_cache()
    np.savez_compressed(out/'pose.npz', kp=kp, score=sc)
    print(f'pose_sapiens ({a.variant}): {n_f} frames, median heatmap peak {np.median(sc):.2f}')


# ---------------------------------------------------------------- L3: skeletons
def cmd_pose(a, out):
    from PIL import Image
    from transformers import AutoProcessor, VitPoseForPoseEstimation
    import torch
    from autolabel_pose2d import MODEL, estimate
    clip = a.clip
    T = json.load(open(out/'track.json'))
    proc = AutoProcessor.from_pretrained(MODEL)
    model = VitPoseForPoseEstimation.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
    kp = np.zeros((len(T['frames']), 2, 17, 2)); sc = np.zeros((len(T['frames']), 2, 17))
    for k in (seed_frames(T) if a.seeds_only else range(len(T['frames']))):
        img = Image.open(clip.image_path(T['frames'][k])).convert('RGB')
        bx = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in T['boxes'][k]]
        kp[k], sc[k] = estimate(proc, model, [img], [bx])[0]
    np.savez_compressed(out/'pose.npz', kp=kp, score=sc)
    print(f'pose: {len(kp)} frames, median keypoint confidence {np.median(sc):.2f}')


# ---------------------------------------------------------------- L2: segmentation
def cmd_segment(a, out):
    import cv2
    sys.path.insert(0, str(Path.home()/'Code/kakashi-research/external/BBoxMaskPose'))
    import bboxmaskpose.sam2  # noqa: F401  (initialises SAM2's config module)
    from bboxmaskpose.sam2_utils import prepare_model
    from engage_vicos_masks import BMP, POSE2SEG
    clip = a.clip
    T = json.load(open(out/'track.json')); P = np.load(out/'pose.npz')
    model = prepare_model(str(BMP/'bboxmaskpose/sam2/configs/sam-pose2seg/sam-pose2seg_hiera_b+.yaml'), str(POSE2SEG))
    bits, scores = [], []
    todo = set(seed_frames(T)) if a.seeds_only else set(range(len(T['frames'])))
    W, H = clip.size
    for k, f in enumerate(T['frames']):
        if k not in todo:
            b, shape = pack(np.zeros((2, H, W), bool)); bits.append(b); scores.append([0.0, 0.0])
            continue
        img = cv2.cvtColor(cv2.imread(str(clip.image_path(f))), cv2.COLOR_BGR2RGB)
        model.set_image(img)
        ms, ss = [], []
        for n in range(2):
            top = np.argsort(-P['score'][k, n])[:3]
            m, s, _ = model.predict(point_coords=P['kp'][k, n, top], point_labels=np.ones(3), box=None, multimask_output=False)
            ms.append(m[0].astype(bool)); ss.append(float(s[0]))
        ms[1 - int(np.argmax(ss))][ms[0] & ms[1]] = False
        b, shape = pack(np.stack(ms)); bits.append(b); scores.append(ss)
    np.savez_compressed(out/'segment.npz', bits=np.stack(bits), shape=shape, scores=np.array(scores))
    print(f'segment: {len(bits)} frames')


def load_masks(out):
    z = np.load(out/'segment.npz')
    return [unpack(b, z['shape']) for b in z['bits']]


def filled(ms, boxes):
    """A frame's two masks, with an empty one (the video tracker lost the person) replaced by its tracked box."""
    ms = [m.copy() for m in ms]
    for n in range(2):
        if not ms[n].any():
            x1, y1, x2, y2 = np.round(boxes[n]).astype(int)
            ms[n][max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = True
    return ms


# ---------------------------------------------------------------- L4: 3D bodies in one camera space
def resolve_t(V, t_model, f_used, c, clip, nv=128):
    """Translation that puts V (root-relative) on the model's own pinhole projection, through the clip's lens
    (engage_bench_s3d.resolve_t, with the lens as the clip's projection)."""
    from scipy.optimize import least_squares
    Vs = V[np.linspace(0, len(V) - 1, nv).astype(int)].astype(np.float64)
    X = Vs + t_model
    uv_model = f_used * X[:, :2] / X[:, 2:3] + c

    def res(t):
        return (clip.project(Vs + t) - uv_model).ravel()
    s = least_squares(res, t_model * (clip.K[0, 0] / f_used), method='lm', max_nfev=400)
    return s.x, float(np.median(np.linalg.norm(res(s.x).reshape(-1, 2), axis=1)))


def cmd_body3d(a, out):
    import cv2
    import torch
    from autolabel_body3d import load_estimator
    clip = a.clip
    T = json.load(open(out/'track.json')); M = load_masks(out)
    W, H = clip.size
    c = np.array([W / 2, H / 2])
    est = load_estimator()
    rec = {k: [] for k in ('kp3d', 'verts', 't', 'resid_px')}
    t0 = time.time()
    for k, f in enumerate(T['frames']):
        img = cv2.cvtColor(cv2.imread(str(clip.image_path(f))), cv2.COLOR_BGR2RGB)
        per = []
        for n in range(2):
            box = np.asarray(T['boxes'][k][n], np.float32).reshape(1, 4)
            with torch.inference_mode():
                o = est.process_one_image(img, bboxes=box, masks=filled(M[k], T['boxes'][k])[n][None].astype(np.uint8),
                                          use_mask=True, inference_type='body')[0]
            est.batch = est.image_embeddings = est.output = None
            V = np.asarray(o['pred_vertices'], np.float64)
            ts, rp = resolve_t(V, np.asarray(o['pred_cam_t'], np.float64), float(np.asarray(o['focal_length']).ravel()[0]),
                               c, clip)
            per.append((np.asarray(o['pred_keypoints_3d'], np.float32), V.astype(np.float16), ts, rp))
        torch.cuda.empty_cache()
        for key, i in (('kp3d', 0), ('verts', 1), ('t', 2), ('resid_px', 3)):
            rec[key].append([p[i] for p in per])
    np.savez_compressed(out/'body3d.npz', **{k: np.asarray(v) for k, v in rec.items()})
    print(f'body3d: {len(rec["t"])} frames {time.time() - t0:.0f}s')


PLACE_W = 0.7     # weight of the depth map's relative depth (fitted 0.65-0.70 leave-one-sequence-out, depthmap_results.md)
PLACE_BAND = 5    # px kept from the partner's predicted silhouette when comparing the map with a body


def cmd_place(a, out):
    """L4 placement: each body moved along its camera ray toward MoGe-2's relative depth of the pair (the combined
    depth of depthmap_results.md, `pred_blend`): on a pinhole crop around both people (a virtual camera turned toward
    them for a fisheye clip), delta_n = median(map - rendered body depth) over the body's visible pixels away from the
    partner's edge; each body moves by PLACE_W x (delta_n - mean delta), so the pair's mean depth stays SAM 3D Body's.
    Rewrites body3d.npz's t (the original kept as t_raw, so the stage can run again)."""
    import cv2
    import torch
    from engage_depth_probe import FACES, render, virtual_cam
    from engage_relations import canon_from_pred
    from moge.model.v2 import MoGeModel
    clip = a.clip
    T = json.load(open(out/'track.json'))
    B = dict(np.load(out/'body3d.npz'))
    t_raw = B.get('t_raw', B['t']).astype(np.float64)
    Fm = np.load(FACES)['mhr']
    model = MoGeModel.from_pretrained('Ruicheng/moge-2-vitl-normal').cuda().eval()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * PLACE_BAND + 1, 2 * PLACE_BAND + 1))
    W, H = clip.size
    t_new, shifts = t_raw.copy(), np.zeros((len(T['frames']), 2))
    for k, f in enumerate(T['frames']):
        img = cv2.cvtColor(cv2.imread(str(clip.image_path(f))), cv2.COLOR_BGR2RGB)
        bx = np.array(T['boxes'][k], np.float64)
        box = np.r_[np.clip(bx[:, :2].min(0), 0, [W, H]), np.clip(bx[:, 2:].max(0), 0, [W, H])]
        if clip.D is not None:                                       # fisheye: a pinhole view turned toward the pair
            Rv, Pv, (w, h) = virtual_cam(clip.K, clip.D, box)
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(clip.K, clip.D, Rv, Pv, (w, h), cv2.CV_32FC1)
            crop = cv2.remap(img, m1, m2, cv2.INTER_LINEAR)
        else:                                                        # pinhole: the frame cropped and scaled
            c, half = (box[:2] + box[2:]) / 2, (box[2:] - box[:2]) / 2 * 1.15
            x0, y0 = np.maximum(0, c - half).astype(int); x1, y1 = np.minimum([W, H], c + half).astype(int)
            s = min(1.0, 768 / max(x1 - x0, y1 - y0))
            w, h = max(1, int((x1 - x0) * s)), max(1, int((y1 - y0) * s))
            crop = cv2.resize(img[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_AREA)
            Rv = np.eye(3)
            Pv = np.array([[clip.K[0, 0] * s, 0, (clip.K[0, 2] - x0) * s], [0, clip.K[1, 1] * s, (clip.K[1, 2] - y0) * s], [0, 0, 1]])
        x = torch.from_numpy(crop).cuda().permute(2, 0, 1).float() / 255
        with torch.no_grad():
            dm = model.infer(x, fov_x=float(np.degrees(2 * np.arctan(w / 2 / Pv[0, 0]))))['depth'].float().cpu().numpy()
        dm[~np.isfinite(dm)] = np.nan
        Vc = [B['verts'][k, n].astype(np.float64) + t_raw[k, n] for n in range(2)]
        own, dep, sil = render([(V @ Rv.T, Fm) for V in Vc], Pv, w, h)
        delta = []
        for n in range(2):
            pp = (own == n + 1) & np.isfinite(dm) & ~cv2.dilate(sil[1 - n].astype(np.uint8), kernel).astype(bool)
            delta.append(float(np.median(dm[pp] - dep[pp])) if pp.sum() >= 50 else None)
        if None in delta:
            continue                                                 # a body hidden: keep SAM 3D Body's depth
        for n in range(2):
            sh = PLACE_W * (delta[n] - np.mean(delta))
            pel = canon_from_pred(B['kp3d'][k, n].astype(np.float64) + t_raw[k, n])['pelvis']
            u = pel / np.linalg.norm(pel)
            t_new[k, n] = t_raw[k, n] + sh / (u @ Rv[2]) * u
            shifts[k, n] = sh
        torch.cuda.empty_cache()
    B.update(t=t_new.astype(B['t'].dtype), t_raw=t_raw.astype(B['t'].dtype), place_shift_m=shifts.astype(np.float32))
    np.savez_compressed(out/'body3d.npz', **B)
    moved = np.abs(shifts).max(1)
    json.dump({'weight': PLACE_W, 'moved_cm_median': float(np.median(moved) * 100), 'moved_cm_p90': float(np.percentile(moved, 90) * 100),
               'unchanged_share': float((moved == 0).mean())}, open(out/'place.json', 'w'), indent=1)
    print(f'place: {len(T["frames"])} frames; bodies moved {np.median(moved) * 100:.1f} cm (median), '
          f'{np.percentile(moved, 90) * 100:.1f} cm (p90); unchanged where a body was hidden: {(moved == 0).mean():.0%}')


# ---------------------------------------------------------------- L5: contact and interaction
def interaction(Ca, Cb, Va, Vb, hand_tau=None):
    """Contact between two bodies. With hand_tau (predictions only), a hand or forearm counts as touching within
    hand_tau rather than TAU: reconstructed hands are the least accurate part, so a held hand often comes out several
    cm from the partner (see demo_spectrum_notes.md). touching_strict keeps the plain TAU rule."""
    from scipy.spatial import cKDTree
    from engage_relations import REGIONS, relations, vertex_regions
    r = relations(Ca, Cb, Va, Vb, TAU)
    embrace = sorted(k for k, (_, m) in r['threads'].items() if 'torso through' in k and 'arm_loop' in k and m >= 0.05)
    contacts = {tuple(sorted(p)) for p in r['contacts']}
    strict = bool(r['min_dist'] < TAU)
    touching = strict
    if hand_tau and hand_tau > TAU:
        ra, rb = vertex_regions(Va, Ca), vertex_regions(Vb, Cb)
        d, idx = cKDTree(np.asarray(Vb, float)).query(np.asarray(Va, float), k=1)
        hand = np.isin(ra, HAND_REGIONS) | np.isin(rb[idx], HAND_REGIONS)
        hit = hand & (d < hand_tau)
        contacts |= {tuple(sorted((REGIONS[a], REGIONS[b]))) for a, b in zip(ra[hit], rb[idx[hit]])}
        touching = strict or bool(hit.any())
    return {'gap_cm': round(float(r['min_dist']) * 100, 1), 'touching': touching, 'touching_strict': strict,
            'contacts': sorted(contacts), 'embrace': embrace,
            'pelvis_cm': round(float(np.linalg.norm(Ca['pelvis'] - Cb['pelvis'])) * 100, 1)}


def cmd_contact(a, out):
    from engage_relations import canon_from_pred
    clip = a.clip
    T = json.load(open(out/'track.json')); B = np.load(out/'body3d.npz')
    pred, truth = [], []
    for k, f in enumerate(T['frames']):
        bodies = [(B['verts'][k, n].astype(np.float64) + B['t'][k, n], canon_from_pred(B['kp3d'][k, n].astype(np.float64) + B['t'][k, n]))
                  for n in range(2)]
        pred.append(interaction(bodies[0][1], bodies[1][1], bodies[0][0], bodies[1][0], hand_tau=TAU_HAND))
        if clip.has_truth:
            tb = clip.truth_bodies(f)
            truth.append(interaction(tb[0][1], tb[1][1], tb[0][0], tb[1][0]))
    json.dump({'frames': T['frames'], 'fps': clip.fps, 'tau_m': TAU, 'hand_tau_m': TAU_HAND, 'pred': pred, 'truth': truth or None},
              open(out/'contact.json', 'w'), default=list)
    tp = [p['touching'] for p in pred]
    print(f'contact: touching predicted {np.mean(tp):.0%}, embrace predicted {np.mean([bool(p["embrace"]) for p in pred]):.0%}'
          + (f'; true {np.mean([x["touching"] for x in truth]):.0%} / {np.mean([bool(x["embrace"]) for x in truth]):.0%}' if truth else ''))


def smooth(states, min_n):
    """Merge runs shorter than min_n frames into the previous run (the next at the start)."""
    s = list(states)
    changed = True
    while changed:
        changed = False
        runs, i = [], 0
        while i < len(s):
            j = i
            while j < len(s) and s[j] == s[i]:
                j += 1
            runs.append((i, j)); i = j
        for r, (i, j) in enumerate(runs):
            if j - i < min_n and len(runs) > 1:
                fill = s[runs[r - 1][0]] if r else s[runs[r + 1][0]]
                if fill != s[i]:
                    s[i:j] = [fill] * (j - i); changed = True; break
    return s


def events(t, flags, name):
    """[(name, start_s, end_s)] for runs of True; t = each frame's time in seconds."""
    out, i = [], 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j + 1 < len(flags) and flags[j + 1]:
                j += 1
            out.append((name, round(float(t[i]), 1), round(float(t[j]), 1)))
            i = j + 1
        else:
            i += 1
    return out


def timeline(C, which='pred'):
    """Measured events and a per-second summary, from the contact layer. Touching = sustained contact (runs of at least
    MIN_EVENT_S after smoothing); brief touch = a raw contact run the smoothing removed (a strike, a tap)."""
    frames = C['frames']; S = C[which]
    base = C.get('fps', 30.0)                                       # contact.json from before folder clips: Harmony4D
    t = (np.array(frames) - frames[0]) / base
    n = max(1, int(round(MIN_EVENT_S * base / np.median(np.diff(frames)))))
    raw = [bool(s['touching']) for s in S]
    touch = smooth(raw, n)
    emb = smooth([bool(s['embrace']) for s in S], n)
    brief = [r and not s_ for r, s_ in zip(raw, touch)]
    ev = events(t, touch, 'touching') + events(t, brief, 'brief touch') + events(t, emb, 'embrace')
    ev = [{'id': f'E{i + 1}', 'what': w, 'start_s': a_, 'end_s': b_} for i, (w, a_, b_) in enumerate(sorted(ev, key=lambda e: e[1]))]
    per_s = []
    for sec in range(int(t[-1]) + 1):
        idx = [i for i in range(len(frames)) if sec <= t[i] < sec + 1]
        if not idx:
            continue
        pairs = {}
        for i in idx:
            for p in S[i]['contacts']:                              # contacts exist only on frames with a raw touch
                pairs[' + '.join(p)] = pairs.get(' + '.join(p), 0) + 1
        per_s.append({'t_s': sec, 'gap_cm': float(np.median([S[i]['gap_cm'] for i in idx])),
                      'pelvis_distance_cm': float(np.median([S[i]['pelvis_cm'] for i in idx])),
                      'touching': bool(any(touch[i] for i in idx)),     # any frame inside a sustained-touch event
                      'brief_touch': bool(any(brief[i] for i in idx)),
                      'touch_frames': f'{sum(raw[i] for i in idx)}/{len(idx)}',
                      'embrace': bool(any(emb[i] for i in idx)),
                      'regions_in_contact': [k for k, v in sorted(pairs.items(), key=lambda x: -x[1])[:5]]})
    return ev, per_s


# ---------------------------------------------------------------- L6: context with Astra
SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['interaction_type', 'summary', 'phases', 'people', 'contact_description', 'labels', 'disagreements'],
    'properties': {
        'interaction_type': {'type': 'string'},
        'summary': {'type': 'string'},
        'phases': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
                   'required': ['start_s', 'end_s', 'label', 'description', 'cites'],
                   'properties': {'start_s': {'type': 'number'}, 'end_s': {'type': 'number'}, 'label': {'type': 'string'},
                                  'description': {'type': 'string'}, 'cites': {'type': 'array', 'items': {'type': 'string'}}}}},
        'people': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
                   'required': ['id', 'appearance', 'posture', 'role'],
                   'properties': {'id': {'type': 'string'}, 'appearance': {'type': 'string'}, 'posture': {'type': 'string'},
                                  'role': {'type': 'string'}}}},
        'contact_description': {'type': 'string'},
        'labels': {'type': 'array', 'items': {'type': 'string'}},
        'disagreements': {'type': 'array', 'items': {'type': 'string'}}}}

PROMPT = """You are labelling a short video of two people interacting, for a research demo of how far a computer-vision
pipeline can understand human engagement. Images: {n} keyframes in time order; person A is tinted red with an A label,
person B tinted blue with a B label; each image shows its time in seconds.

The pipeline measured, from the same single camera (3D bodies, contact between them):
EVENTS (cite these ids): {events}
PER SECOND: {per_second}
("touching" is sustained contact of at least 0.5 s; a "brief touch" is a shorter contact, such as a strike or a tap;
touch_frames counts the frames of that second with any contact, and regions_in_contact lists the body regions touching
in those frames; gap_cm is the median smallest distance between the two bodies.)

Write: the interaction type; a 2-4 sentence summary; the phases in time order (start and end in seconds, a short
label such as approach / greeting / embrace / release, a description, and the measured event ids each phase relies
on); each person's appearance, posture and role (who initiates, who receives); which body parts touch; a list of short
labels useful for annotating this clip; and any place where what you see disagrees with the measurements. Use the
measurements for timing and contact; use the images for what the people are doing. Do not invent times that are not
in the measurements or visible in the images."""


def cmd_context(a, out):
    import cv2
    from autolabel_astra import call_astra, check_usage
    clip = a.clip
    T = json.load(open(out/'track.json')); C = json.load(open(out/'contact.json')); P = np.load(out/'pose.npz')
    M = load_masks(out)
    ev, per_s = timeline(C)
    keys = np.linspace(0, len(T['frames']) - 1, 6).round().astype(int)
    kdir = out/'astra_keyframes'; kdir.mkdir(exist_ok=True)
    images = []
    for k in keys:
        f = T['frames'][k]
        img = cv2.imread(str(clip.image_path(f)))
        u = img.shape[0] / 2160                                     # text scale: 1 on 4K frames
        for n, col in ((0, (40, 40, 220)), (1, (220, 120, 40))):
            img[M[k][n]] = (0.55 * img[M[k][n]] + 0.45 * np.array(col)).astype(np.uint8)
            ys, xs = np.nonzero(M[k][n])
            if len(ys):                                             # label above the person's own mask, not the box
                top = ys <= np.percentile(ys, 3)
                x1, y1 = int(np.median(xs[top])), int(ys.min())
            else:
                x1, y1 = np.array(T['boxes'][k][n][:2]).astype(int)
            cv2.putText(img, 'AB'[n], (max(0, x1 - int(20 * u)), max(int(40 * u), y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.8, 2.0 * u), col, max(2, int(5 * u)), cv2.LINE_AA)
        bx = np.array(T['boxes'][k]); x1, y1 = bx[:, :2].min(0); x2, y2 = bx[:, 2:].max(0)
        pad = 0.25 * max(x2 - x1, y2 - y1)
        crop = img[int(max(0, y1 - pad)):int(min(img.shape[0], y2 + pad)), int(max(0, x1 - pad)):int(min(img.shape[1], x2 + pad))]
        crop = cv2.resize(crop, (int(900 * crop.shape[1] / crop.shape[0]), 900))
        cv2.putText(crop, f't = {(f - T["frames"][0]) / clip.fps:.1f} s', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
        p = kdir/f'key_{k:03d}.jpg'; cv2.imwrite(str(p), crop, [cv2.IMWRITE_JPEG_QUALITY, 90]); images.append(p)
    usage = check_usage()
    prompt = PROMPT.format(n=len(images), events=json.dumps(ev), per_second=json.dumps(per_s))
    res, receipt = call_astra(prompt, images, SCHEMA)
    json.dump({'astra': res, 'events': ev, 'per_second': per_s, 'receipt': receipt, 'usage_before': usage, 'prompt_version': 2},
              open(out/'context.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


# ---------------------------------------------------------------- scoring against the truth
def cmd_score(a, out):
    clip = a.clip
    if not clip.has_truth:
        return score_health(a, out)
    import engage_limb_head as L
    from engage_limb_head import CORE14, JOINT8, kabsch, tangle_mask
    from engage_relations import CANON, canon_from_pred
    L.ALIGN = CORE14
    T = json.load(open(out/'track.json')); P = np.load(out/'pose.npz'); B = np.load(out/'body3d.npz')
    C = json.load(open(out/'contact.json')); M = load_masks(out)
    frames = T['frames']; boxes = np.array(T['boxes'])
    gtb = np.array([clip.truth_boxes(f) for f in frames])
    k0 = frames.index(T['bootstrap_frame'])
    perm = [0, 1] if iou(boxes[k0, 0], gtb[k0, 0]) + iou(boxes[k0, 1], gtb[k0, 1]) >= iou(boxes[k0, 0], gtb[k0, 1]) + iou(boxes[k0, 1], gtb[k0, 0]) else [1, 0]
    gtb0 = gtb.copy()
    gtb = gtb[:, perm]
    ious = np.array([[iou(boxes[k, n], gtb[k, n]) for n in range(2)] for k in range(len(frames))])
    swapped = np.array([iou(boxes[k, 0], gtb[k, 1]) + iou(boxes[k, 1], gtb[k, 0]) > ious[k].sum() for k in range(len(frames))])
    # per-frame assignment of tracks to true people (for scoring layers 2-4 on their own merit)
    pf = [([0, 1] if iou(boxes[k, 0], gtb0[k, 0]) + iou(boxes[k, 1], gtb0[k, 1]) >= iou(boxes[k, 0], gtb0[k, 1]) + iou(boxes[k, 1], gtb0[k, 0])
           else [1, 0]) for k in range(len(frames))]
    switches = int(sum(pf[k] != pf[k - 1] for k in range(1, len(pf))))
    W, H = clip.size
    rep = {'frames': len(frames), 'truth': True, 'image_size': [W, H],
           'L1_detection_tracking': {'identity_switches': switches, 'both_found_iou50': float((ious >= 0.5).all(1).mean()),
           'person_found_iou50': float((ious >= 0.5).mean()), 'identity_swapped_frames': float(swapped.mean()),
           'held_person_frames': float(np.array(T['held']).mean()), 'median_box_iou': float(np.median(ious))}}
    # L2 masks vs the truth's ownership (rendered at half resolution)
    mi, vis = [], []
    for k, f in enumerate(frames):
        lab = clip.truth_labels(f, (W // 2, H // 2))
        for n in range(2):
            g = lab == pf[k][n] + 1
            pm = M[k][n][::2, ::2][:H // 2, :W // 2]
            mi.append((g & pm).sum() / max((g | pm).sum(), 1))
        tr = clip.truth_2d(f)[pf[k]]
        for n in range(2):
            for j in COCO_LIMB:
                u, v = (tr[n, j] / 2).astype(int)
                if not (0 <= u < W // 2 and 0 <= v < H // 2):
                    continue                                        # joint outside the picture: not scored
                vis.append((np.linalg.norm(P['kp'][k, n, j] - tr[n, j]), lab[v, u] == pf[k][n] + 1))
    vis = np.array(vis)
    rep['L2_segmentation'] = {'mask_iou_median': float(np.median(mi)), 'mask_iou_mean': float(np.mean(mi))}
    rep['L3_skeleton'] = {'joint_error_px_median': float(np.median(vis[:, 0])),
                          'visible_joints_px': float(np.median(vis[vis[:, 1] == 1, 0])),
                          'hidden_joints_px': float(np.median(vis[vis[:, 1] == 0, 0])) if (vis[:, 1] == 0).any() else None,
                          'hidden_share': float((vis[:, 1] == 0).mean())}
    # L4 3D: pose (limb joints, rigid aligned) and placement (pelvis, pair distance) in camera space
    Gc, Pcn = [], []
    for k, f in enumerate(frames):
        tb = clip.truth_bodies(f)
        Gc.append([[tb[pf[k][n]][1][q] for q in CANON] for n in range(2)])
        Pcn.append([[canon_from_pred(B['kp3d'][k, n].astype(np.float64) + B['t'][k, n])[q] for q in CANON] for n in range(2)])
    Gc, Pcn = np.array(Gc), np.array(Pcn)
    e = np.stack([np.linalg.norm(kabsch(Pcn[:, n], Gc[:, n]) - Gc[:, n], axis=-1) for n in range(2)], 1)
    tang = tangle_mask(Gc)
    pel = CANON.index('pelvis')
    rep['L4_3d'] = {'limb_joints_in_contact_cm': float(np.median(e[..., JOINT8][tang[..., JOINT8]]) * 100) if tang[..., JOINT8].any() else None,
                    'limb_joints_other_cm': float(np.median(e[..., JOINT8][~tang[..., JOINT8]]) * 100),
                    'pelvis_position_err_cm': float(np.median(np.linalg.norm(Pcn[:, :, pel] - Gc[:, :, pel], axis=-1)) * 100),
                    'pair_distance_err_cm': float(np.median(np.abs(np.linalg.norm(Pcn[:, 1, pel] - Pcn[:, 0, pel], axis=-1)
                                                                   - np.linalg.norm(Gc[:, 1, pel] - Gc[:, 0, pel], axis=-1))) * 100)}
    # L5 contact vs the truth's (person order mapped)
    pr, tr_ = C['pred'], C['truth']
    tp = np.array([p['touching'] for p in pr]); tt = np.array([x['touching'] for x in tr_])
    ep = np.array([bool(p['embrace']) for p in pr]); et = np.array([bool(x['embrace']) for x in tr_])
    f1 = lambda p, t: float(2 * (p & t).sum() / max(p.sum() + t.sum(), 1)) if (p.any() or t.any()) else None
    rp = []
    for p, x in zip(pr, tr_):
        Ps = {tuple(q) for q in p['contacts']}; Ts = {tuple(q) for q in x['contacts']}
        if Ps or Ts:
            rp.append(2 * len(Ps & Ts) / max(len(Ps) + len(Ts), 1))
    ev_p, _ = timeline(C, 'pred'); ev_t, _ = timeline(C, 'truth')
    ts = np.array([p.get('touching_strict', p['touching']) for p in pr])
    rep['L5_contact'] = {'touching_frame_f1': f1(tp, tt), 'touching_frame_agreement': float((tp == tt).mean()),
                         'strict_3cm_touching_f1': f1(ts, tt), 'strict_3cm_touching_agreement': float((ts == tt).mean()),
                         'touching_share_pred_true': [float(tp.mean()), float(tt.mean())],
                         'embrace_frame_f1': f1(ep, et), 'region_pairs_f1_mean': float(np.mean(rp)) if rp else None,
                         'gap_err_cm_median': float(np.median([abs(p['gap_cm'] - x['gap_cm']) for p, x in zip(pr, tr_)])),
                         'events_pred': ev_p, 'events_true': ev_t}
    json.dump(rep, open(out/'scorecard.json', 'w'), indent=1, default=float)
    print(json.dumps(rep, indent=1, default=float))


def score_health(a, out):
    """No truth: the pipeline's own health numbers per layer (what the demo can say about itself)."""
    clip = a.clip
    T = json.load(open(out/'track.json')); P = np.load(out/'pose.npz'); B = np.load(out/'body3d.npz')
    C = json.load(open(out/'contact.json')); M = load_masks(out)
    W, H = clip.size
    boxes = np.array(T['boxes'])
    merged = [iou(b[0], b[1]) > 0.7 for b in boxes]
    area = np.array([[m.sum() for m in ms] for ms in M], float)
    jump = np.abs(np.diff(np.log(np.maximum(area, 1)), axis=0))
    overlap = [(ms[0] & ms[1]).sum() / max(min(ms[0].sum(), ms[1].sum()), 1) for ms in M]
    pm = 'visibility' in P.files
    hgt = boxes[..., 3] - boxes[..., 1]
    pel = []
    from engage_relations import canon_from_pred
    for k in range(len(T['frames'])):
        cs = [canon_from_pred(B['kp3d'][k, n].astype(np.float64) + B['t'][k, n]) for n in range(2)]
        pel.append(np.linalg.norm(cs[0]['pelvis'] - cs[1]['pelvis']))
    pr = C['pred']; ev_p, _ = timeline(C, 'pred')
    rep = {'frames': len(T['frames']), 'truth': False, 'image_size': [W, H], 'lens': clip.meta.get('lens_source'),
           'L1_detection_tracking': {'held_person_frames': float(np.array(T['held']).mean()), 'merged_box_frames': float(np.mean(merged)),
                                     'people_detected_per_frame': float(np.mean([len(r['boxes']) for r in json.load(open(out/'detect.json'))['frames']]))},
           'L2_segmentation': {'mask_area_jump_median': float(np.median(jump)), 'mask_area_jump_p95': float(np.percentile(jump, 95)),
                               'mask_overlap_median': float(np.median(overlap))},
           'L3_skeleton': {'keypoint_score_median': float(np.median(P['score'])),
                           'visible_share': float((P['visibility'] > 0.5).mean()) if pm else None},
           'L4_3d': {'fit_residual_px_median': float(np.median(B['resid_px'])),
                     'fit_residual_share_of_person_height': float(np.median(B['resid_px'] / np.maximum(hgt, 1))),
                     'depth_m_median': float(np.median(B['t'][..., 2])), 'pair_pelvis_distance_m_median': float(np.median(pel))},
           'L5_contact': {'touching_share': float(np.mean([p['touching'] for p in pr])),
                          'embrace_share': float(np.mean([bool(p['embrace']) for p in pr])), 'events_pred': ev_p}}
    json.dump(rep, open(out/'scorecard.json', 'w'), indent=1, default=float)
    print(json.dumps(rep, indent=1, default=float))


# ---------------------------------------------------------------- lens for clips without one
def cmd_lens(a, out):
    """MoGe-2's focal length (median over 5 frames spread through the clip), principal point at the centre."""
    import cv2
    import torch
    from moge.model.v2 import MoGeModel
    clip = a.clip
    if clip.K is not None and not a.force:
        print('lens: clip already has K from', clip.meta.get('lens_source')); return
    model = MoGeModel.from_pretrained('Ruicheng/moge-2-vitl-normal').cuda().eval()
    W, H = clip.size
    fs = []
    for f in [clip.frames[i] for i in np.linspace(0, len(clip.frames) - 1, 5).round().astype(int)]:
        img = cv2.cvtColor(cv2.imread(str(clip.image_path(f))), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img).cuda().permute(2, 0, 1).float() / 255
        with torch.no_grad():
            Kn = model.infer(x)['intrinsics'].float().cpu().numpy().reshape(3, 3)
        fs.append(float(Kn[0, 0] * W))
    f = float(np.median(fs))
    clip.meta.update(K=[[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], lens_source='MoGe-2 (median of 5 frames)',
                     lens_fov_x_deg=float(np.degrees(2 * np.arctan(W / 2 / f))), lens_focals=[round(v, 1) for v in fs])
    tmp = clip.root/'clip.json.tmp'
    json.dump(clip.meta, open(tmp, 'w'), indent=1)
    tmp.replace(clip.root/'clip.json')                              # never leave a half-written clip.json
    print(f'lens: focal {f:.0f} px (horizontal field of view {clip.meta["lens_fov_x_deg"]:.0f} deg); per frame {clip.meta["lens_focals"]}')


# ---------------------------------------------------------------- quality gates
def cmd_check(a, out):
    """Every layer must have produced a usable result before the queue moves on. Writes GATE.json; on any failure
    also GATE_FAILED.txt and exit status 1."""
    need = ['detect.json', 'track.json', 'segment.npz', 'pose.npz', 'body3d.npz', 'contact.json', 'context.json',
            'scorecard.json', 'demo.mp4', 'demo.gif']
    G = {}
    missing = [n for n in need if not (out/n).exists()]
    G['files'] = (not missing, f'missing {missing}' if missing else 'all outputs present')
    if not missing:
        T = json.load(open(out/'track.json')); P = np.load(out/'pose.npz'); B = np.load(out/'body3d.npz')
        M = load_masks(out); X = json.load(open(out/'context.json')); S = json.load(open(out/'scorecard.json'))
        boxes = np.array(T['boxes'])
        held = float(np.array(T['held']).mean())
        G['L1 both people tracked'] = (held <= 0.15, f'a person has no mask in {held:.0%} of person-frames (limit 15%)')
        same = float(np.mean([(ms[0] & ms[1]).sum() / max(min(ms[0].sum(), ms[1].sum()), 1) > 0.6 for ms in M]))
        G['L1/L2 two distinct people'] = (same <= 0.15, f'both tracks on one body in {same:.0%} of frames (limit 15%)')
        ar = np.array([[m.sum() for m in ms] for ms in M], float)
        both = float(np.mean(ar.min(1) >= 0.10 * np.maximum(ar.max(1), 1)))
        G['L2 both people masked'] = (both >= 0.80, f'the smaller mask is at least 10% of the larger in {both:.0%} of frames '
                                                     f'(limit 80%; one mask holding both people fails this)')
        if S.get('truth', True) and 'both_found_iou50' in S['L1_detection_tracking']:
            bf = S['L1_detection_tracking']['both_found_iou50']
            G['L1 on the true pair'] = (bf >= 0.6, f'both true people found (IoU 0.5) in {bf:.0%} of frames (limit 60%)')
        ks = float(np.median(P['score']))
        G['L3 skeletons'] = (ks >= 0.3, f'median keypoint score {ks:.2f} (limit 0.30)')
        hgt = np.maximum(boxes[..., 3] - boxes[..., 1], 1)
        rr = float(np.median(B['resid_px'] / hgt))
        dz = float(np.median(B['t'][..., 2]))
        G['L4 3D fit'] = (rr <= 0.05 and 0.5 <= dz <= 40, f'fit residual {rr:.1%} of person height (limit 5%), depth {dz:.1f} m')
        from engage_relations import canon_from_pred
        pel = np.median([np.linalg.norm(canon_from_pred(B['kp3d'][k, 0].astype(np.float64) + B['t'][k, 0])['pelvis']
                                        - canon_from_pred(B['kp3d'][k, 1].astype(np.float64) + B['t'][k, 1])['pelvis'])
                         for k in range(len(T['frames']))])
        G['L4/L5 one space'] = (pel <= 4.0, f'median distance between the two pelvises {pel:.2f} m (limit 4 m)')
        ast = X.get('astra') or {}
        ok6 = bool(ast.get('interaction_type')) and len(ast.get('phases', [])) >= 1
        G['L6 context'] = (ok6, f'Astra: "{ast.get("interaction_type", "")}", {len(ast.get("phases", []))} phases')
    passed = all(v[0] for v in G.values())
    json.dump({'passed': passed, 'gates': {k: {'ok': bool(v[0]), 'detail': v[1]} for k, v in G.items()}},
              open(out/'GATE.json', 'w'), indent=1)
    for k, (ok, d) in G.items():
        print(f'{"PASS" if ok else "FAIL"}  {k}: {d}')
    if not passed:
        open(out/'GATE_FAILED.txt', 'w').write('\n'.join(f'{k}: {d}' for k, (ok, d) in G.items() if not ok) + '\n')
        sys.exit(1)
    (out/'GATE_FAILED.txt').unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['lens', 'detect', 'track', 'samtrack', 'pose', 'pose_masked', 'pose_pmpose', 'pose_sapiens', 'segment', 'place',
                                      'body3d', 'contact', 'context', 'score', 'render', 'check'])
    ap.add_argument('--seed', default='', help='samtrack: the run whose first/last-frame masks seed the tracker')
    ap.add_argument('--variant', default='PMPose-b', help='pose_pmpose: PMPose-b / PMPose-l / PMPose-h')
    ap.add_argument('--seeds-only', action='store_true', help='pose / segment: only the frames samtrack seeds from')
    ap.add_argument('--force', action='store_true', help='lens: re-estimate even if the clip has a lens')
    ap.add_argument('--seq', default='002_hugging', help='Harmony4D sequence name, or a folder clip directory')
    ap.add_argument('--cam', default='cam01')
    ap.add_argument('--out', required=True)
    ap.add_argument('--frames', default='', help='window in the clip\'s frame numbers: first:last:step')
    a = ap.parse_args()
    global WINDOW
    if a.frames:
        WINDOW = tuple(int(x) for x in a.frames.split(':'))
    a.clip = clip_of(a.seq, a.cam)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.stage == 'render':
        from engage_demo_render import render
        return render(a, out)
    globals()[f'cmd_{a.stage}'](a, out)


if __name__ == '__main__':
    main()
