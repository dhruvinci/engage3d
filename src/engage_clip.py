"""Clip sources for the engagement demo (`engage_demo_layers.py`): one interface over a Harmony4D camera and over a
plain folder of frames cut from any video, with its lens and, when it exists, ground truth in one common format.

A folder clip (`--seq` names its directory):
  clip.json        {"name", "fps", "size": [W, H], "K": 3x3 pinhole (from the `lens` stage or the dataset),
                    "lens_source", "source": {...}, "truth": true/false}
  frames/00000.jpg ...  numbered 0..N-1 at `fps`
  truth.npz        optional: frames [N], verts [N, 2, V, 3] float16 camera-frame metres, faces [F, 3],
                   joints [N, 2, 24, 3] float32 in SMPL's joint order (camera frame)

Every clip offers: frames, image_path(f), size, fps, project(X camera frame -> pixels), K and D (D None = pinhole),
and with truth: truth_bodies(f), truth_2d(f), truth_boxes(f), truth_labels(f, size).
"""
import json
from pathlib import Path

import cv2
import numpy as np

SMPL_FOR_COCO = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}


def label_map(meshes, project, size, scale):
    """Z-buffer ownership map at `size` (w, h): 0 background, n for the n-th mesh (1-based), nearer body wins.
    meshes = [(V camera frame, F)]; project maps camera-frame points to full-resolution pixels."""
    tris = []
    for n, (V, F) in enumerate(meshes, start=1):
        uv = project(V) * scale
        z = V[:, 2]
        ok = (z[F] > 0).all(1)
        for tri, dep in zip(uv[F][ok], z[F].mean(1)[ok]):
            tris.append((dep, n, tri))
    tris.sort(key=lambda x: -x[0])
    out = np.zeros((size[1], size[0]), np.uint8)
    for _, n, tri in tris:
        cv2.fillConvexPoly(out, np.round(tri * 4).astype(np.int32), n, lineType=cv2.LINE_8, shift=2)
    return out


class H4DClip:
    """One camera of a Harmony4D sequence (4K fisheye), truth from its SMPL fits."""

    def __init__(self, name, cam):
        from engage_paths import H4D
        from tangle_h4d import Sequence
        self.seq = Sequence(next(H4D.glob(f'*/{name}')))
        self.cam, self.name, self.fps, self.has_truth = cam, name, 30.0, True
        R, t, cid = self.seq.poses[cam]
        W, H, P = self.seq.cams[cid]
        self.size = (int(W), int(H))
        self.K = np.array([[P[0], 0, P[2]], [0, P[1], P[3]], [0, 0, 1]], float)
        self.D = np.array(P[4:8], float).reshape(1, 4)
        self.frames = [f for f in self.seq.frames if self.seq.image_path(cam, f).exists()]

    def image_path(self, f):
        return self.seq.image_path(self.cam, f)

    def project(self, X):
        uv, _ = cv2.fisheye.projectPoints(np.asarray(X, np.float64).reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.D)
        return uv.reshape(-1, 2)

    def _smpl(self, f):
        return np.load(self.seq.root/'processed_data/smpl'/f'{f:05d}.npy', allow_pickle=True).item()

    def truth_bodies(self, f):
        import engage_demo as DM
        from engage_relations import canon_from_smpl
        _, _, tocam = DM.cam_of(self.seq, self.cam)
        gt = self._smpl(f)
        return [(tocam(np.asarray(gt[k]['vertices'], float)), canon_from_smpl(tocam(np.asarray(gt[k]['joints'][:24], float))))
                for k in sorted(gt)]

    def truth_2d(self, f):
        from tangle_h4d import PEOPLE
        d = self._smpl(f)
        out = np.zeros((2, 17, 2))
        for n, person in enumerate(PEOPLE):
            uv, _ = self.seq.project(np.asarray(d[person]['joints'][:24], float), self.cam)
            for c, s in SMPL_FOR_COCO.items():
                out[n, c] = uv[s]
        return out

    def truth_boxes(self, f):
        d = np.load(self.seq.root/'processed_data/bbox'/self.cam/f'{f:05d}.npy', allow_pickle=True).item()
        return np.array([np.asarray(v, float) for _, v in sorted(d.items())])

    def truth_labels(self, f, size):
        return self.seq.gt(self.cam, f, size=size)


class FolderClip:
    """Frames cut from any video, a pinhole lens, optional truth.npz."""

    def __init__(self, root):
        self.root = Path(root)
        self.meta = json.load(open(self.root/'clip.json'))
        self.name, self.fps = self.meta['name'], float(self.meta['fps'])
        self.size = tuple(self.meta['size'])
        self.frames = sorted(int(p.stem) for p in (self.root/'frames').glob('*.jpg'))
        K = self.meta.get('K')
        self.K = np.array(K, float) if K is not None else None
        self.D = None
        self.has_truth = bool(self.meta.get('truth')) and (self.root/'truth.npz').exists()
        self._T = None

    def image_path(self, f):
        return self.root/'frames'/f'{f:05d}.jpg'

    def project(self, X):
        X = np.asarray(X, np.float64).reshape(-1, 3)
        z = np.maximum(X[:, 2:3], 1e-6)
        return X[:, :2] / z * self.K[[0, 1], [0, 1]] + self.K[:2, 2]

    def _truth(self):
        if self._T is None:
            z = np.load(self.root/'truth.npz')
            self._T = {k: z[k] for k in z.files}
            self._T['index'] = {int(f): i for i, f in enumerate(self._T['frames'])}
        return self._T

    def truth_bodies(self, f):
        from engage_relations import canon_from_smpl
        T = self._truth(); i = T['index'][f]
        return [(T['verts'][i, n].astype(np.float64), canon_from_smpl(T['joints'][i, n].astype(np.float64))) for n in range(2)]

    def truth_2d(self, f):
        T = self._truth(); i = T['index'][f]
        out = np.zeros((2, 17, 2))
        for n in range(2):
            uv = self.project(T['joints'][i, n])
            for c, s in SMPL_FOR_COCO.items():
                out[n, c] = uv[s]
        return out

    def truth_boxes(self, f):
        T = self._truth(); i = T['index'][f]
        W, H = self.size
        out = []
        for n in range(2):
            uv = self.project(T['verts'][i, n])
            x1, y1 = np.clip(uv.min(0), 0, [W, H]); x2, y2 = np.clip(uv.max(0), 0, [W, H])
            out.append([x1, y1, x2, y2])
        return np.array(out)

    def truth_labels(self, f, size):
        T = self._truth(); i = T['index'][f]
        return label_map([(T['verts'][i, n].astype(np.float64), T['faces']) for n in range(2)], self.project, size,
                         np.array([size[0] / self.size[0], size[1] / self.size[1]]))


def clip_of(seq, cam):
    """A folder clip if `seq` is a directory holding clip.json, else a Harmony4D sequence name."""
    p = Path(seq)
    if (p/'clip.json').exists():
        return FolderClip(p)
    return H4DClip(seq, cam)
