# Pre-registration: relative depth from a monocular depth map — written 2026-10-24, while the first run was in progress and before any result was scored

## Why
Every single-camera fix so far used cues whose depth error grows with the distance to the camera: how big each body
looks (SAM 3D Body: relative depth off 14.5 cm, about 3.5% of the 4.2 m median distance), 2D keypoints (too noisy to
measure a 3.5% size change), and the floor (too shallow from 1.2-1.5 m cameras). Cues that compare the two people
where they meet in the picture should have an error that grows with the gap between them instead: which one covers
the other, the depth step at their shared edge, and contact. A depth-map model trained on millions of images (MoGe-2,
Depth Pro) reads these cues from the whole picture. One smoke frame (025_grappling2/cam06, view 20): true gap 35 cm,
MoGe-2 28 cm, Depth Pro 17 cm, SAM 3D Body 1.5 cm (and its render puts the far athlete over the near one's back).

## What runs
`scripts/engage_depth_probe.py` (.venv-depth, GPU) on every 8th view of `benchmarks/engage_h4d/s3d_box` (same frames,
ground-truth boxes, true lens), MoGe-2 (`Ruicheng/moge-2-vitl-normal`) and Depth Pro (`apple/DepthPro`), each given
the true focal length. Each predicted body is moved along its camera ray only (orientation and limbs unchanged), so the
ceiling is the oracle's "depth along each camera ray" row (`jointfit_results.md`). Variants: `gt_rel` (true visibility
masks: the map's ceiling), `pred_rel` (no masks: the predicted render minus a 5 px band around the other body; the
real method), `pred_abs` (as `pred_rel` but trusting the map's metric depth). Scored with `engage_bench_oracle.py`;
the `pred` variant is the baseline on the same subset.

## Criteria (grappling; MMA reported separately)
| | Measure | Pass if (`pred_rel`) |
|---|---|---|
| D0 | Relative depth of the two athletes' visible surfaces, median error, map vs truth on the same pixels | below SAM 3D Body's on the same pixels, and <= 7 cm |
| D1 | Relative pelvis depth error (oracle "rel depth") | <= 7 cm (baseline 14.5) |
| P1 | Distance between the two bodies, median error | <= 6 cm |
| P2 | Region-pair F1 | >= 0.57 |
| P5 | Leg through leg loop F1 | >= 0.56 |
| P4 | No harm: body lock F1 | >= 0.75 |
P3 (loop balance >= 70%) is not tested: moving bodies along the ray cannot reach it (oracle ceiling 53% matched).

**Decision rule.** D1 + P1 + P2 + P5 pass with `pred_rel`: one camera can place the two people from a depth map; next
come orientation (rigid fit to the map, the oracle's best row), real masks (SAM 3.1) and ViCoS ground work. Only `gt_rel`
passes: the depth map carries the information, and the masks are the bottleneck. Neither passes: depth maps do not
fix this at 4 m; next is visibility order (render-and-compare against masks) and training a two-person placement
model on Harmony4D (held-out sequences).

## Caveats
Harmony4D boxes are ground truth. Frames are resampled from fisheye to pinhole for the depth models. The `gt_rel`
masks come from the true bodies, so they are an upper bound. Every 8th view of the benchmark (about 900 views), so
the confidence is lower than the full benchmark's.
