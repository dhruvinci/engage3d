# Relative depth from a monocular depth map — results (pre-registration: `depthmap_prereg.md`)

## MoGe-2 — 2026-10-24
`benchmarks/engage_h4d/depth_probe/moge/` (912 views = every 8th view of `s3d_box`; 724 grappling, 188 MMA; 18 min on
the RTX 3060, peak 2.4 GB). Each variant's scores are in `<variant>.score.log` and `<variant>/score/`. The direct
measurement comes from `scripts/engage_depth_probe_score.py moge/views.jsonl`.

### Direct measurement: depth gap between the two athletes' visible surfaces (grappling, 724 views)
| | median error | p75 | nearer athlete right (true gap >= 10 cm) |
|---|---|---|---|
| SAM 3D Body | 14.0 cm | 23.9 | 98% |
| MoGe-2 | **8.4 cm** | 14.7 | 99% |

The error grows with the true gap, as expected of a cue that compares the two people where they meet. MoGe-2 gives
4.5 / 5.3 / 6.7 / 12.0 cm for true gaps of 0-10 / 10-25 / 25-50 / > 50 cm; SAM 3D Body gives 6.3 / 8.4 / 11.7 / 21.6.
D0 fails: better than SAM 3D Body, but above 7 cm.

### Pre-registered: moving each body along its ray to agree with the map (grappling)
| | distance err | rel depth | region pairs | in contact | body lock | legs | arms | balance (as scored) |
|---|---|---|---|---|---|---|---|---|
| baseline on this subset | 10.3 cm | 14.3 | 0.46 | 0.89 | 0.76 | 0.41 | 0.27 | 45% |
| `gt_rel` (true visibility masks) | 5.1 | 8.0 | 0.50 | 0.96 | 0.74 | 0.43 | 0.25 | 50% |
| `pred_rel` (no masks; the method) | **5.2** | 8.4 | 0.49 | 0.96 | **0.68** | 0.47 | 0.26 | 50% |
| `pred_abs` (map's metric depth) | 7.5 | 8.5 | 0.47 | 0.96 | 0.64 | 0.48 | 0.27 | 49% |
| oracle: depth along the ray from truth | 1.5 | 0.25 | 0.58 | 0.96 | 0.87 | 0.57 | 0.32 | 52% |

**Verdict (`pred_rel`): fails.** P1 passes (5.2 <= 6 cm). D1 (8.4 > 7), P2 (0.49 < 0.57) and P5 (0.47 < 0.56) fail,
and P4 fails (body lock 0.68 < 0.75). `gt_rel` fails the same way, so under the decision rule a depth map alone does
not fix this at 4 m. The masks are not the bottleneck.

### Exploratory (after the pre-registered result): combining the map with SAM 3D Body
The two errors oppose. The map shrinks the gap (fit: 0.84 x truth) and SAM 3D Body stretches it (1.20 x), so the
errors correlate at -0.26. A weighted mean was tested with the weight fitted leave-one-sequence-out (0.65-0.70 toward
the map in every fold; `scripts/engage_depth_blend.py`, weights in `moge/blend_weights.json`).
- Visible-surface gap, held out by sequence (95% intervals: bootstrap over sequences): SAM 3D Body 14.0 cm
  [11.4, 16.4], map 8.4 [6.9, 9.0], map / fitted slope 8.0 [7.1, 8.7], **combined 6.0 [5.5, 6.4]**.
- Bodies moved to the combined gap:

| grappling | distance err | rel depth | region pairs | in contact | body lock | legs | arms | balance |
|---|---|---|---|---|---|---|---|---|
| `gt_blend` (true masks) | 4.0 cm | 5.8 | 0.53 | 0.96 | 0.81 | 0.49 | 0.31 | 52% |
| `pred_blend` (no masks) | **3.9** | **5.5** | **0.53** | 0.96 | 0.78 | **0.52** | 0.30 | 51% |
| for comparison: 2 cameras 60-120 deg apart, late fusion (full benchmark) | 3.3 | | 0.58 | 0.96 | 0.88 | 0.51 | 0.31 | |

MMA (188 views, `pred_blend`): distance error 9.7 -> 5.8 cm, region pairs 0.22 -> 0.33.
This is the best single-camera placement so far, close to a second camera on distance and legs. It still misses
P2 (0.53), P5 (0.52 vs 0.56) and the depth target only narrowly (5.5 cm). It is exploratory: the combination was
chosen after seeing the pre-registered result. It needs a fresh pre-registered test on views not used here (the other
7/8 of the benchmark), and on ViCoS ground work.

### Other checks
- A temporal median over 3 samples (about 1 s) made the map worse (8.4 -> 10.6 cm): the athletes move between samples
  half a second apart. Smoothing needs consecutive frames to test.
- The map's metric scale is 0.95 x truth (median), so trusting its absolute depth (`pred_abs`) is worse than using
  only the relative gap.

## Depth Pro — 2026-10-24
`benchmarks/engage_h4d/depth_probe/depthpro/` (same 912 views; 36 min, peak 4.0 GB). Grappling, 724 views:
- Visible-surface gap: median error 8.8 cm (MoGe-2 8.4, SAM 3D Body 14.0); 5.8 / 7.0 / 6.1 / 12.2 cm by true gap
  0-10 / 10-25 / 25-50 / > 50 cm. It shrinks gaps less than MoGe-2 does (0.91 x) and its errors oppose SAM 3D Body's
  less (-0.21).
- Pre-registered `pred_rel`: distance 5.6 cm, rel depth 8.5, region pairs 0.46, body lock 0.67, legs 0.37. **Fails**
  like MoGe-2's, and is worse on the relations. `gt_rel`: 5.4 / 8.6 / 0.46 / 0.69 / 0.40.
- The two models' errors correlate at 0.60. Averaging both maps gives 7.3 cm. Map average + SAM 3D Body (0.35 / 0.35 /
  0.3) gives 5.3 cm, against 5.8 for MoGe-2 + SAM 3D Body (in-sample, exploratory). Not worth twice the GPU time now.
- Combined with SAM 3D Body (exploratory, weight leave-one-sequence-out, first sample): `pred_blend` distance 4.6 cm,
  rel depth 6.0, region pairs 0.52, body lock 0.80, legs 0.52. Close to MoGe-2's combination (3.9 / 5.5 / 0.53 / 0.78
  / 0.52).
Conclusion: two independent depth models show the same pattern. MoGe-2 is better and twice as fast, so it is the one
carried forward.

## Fresh-view test of the combined depth — PASSES (pre-registration: `depthblend_prereg.md`)
`benchmarks/engage_h4d/depth_probe_o4/moge/` (every 8th view from view 4: 696 grappling + 188 MMA views, none shared
with the first sample). Weight fixed at 0.70 in advance. No masks. Grappling:

| | distance err | rel depth | region pairs | in contact | body lock | legs | arms | balance (as scored) |
|---|---|---|---|---|---|---|---|---|
| SAM 3D Body (same views) | 10.0 cm | 14.7 | 0.45 | 0.88 | 0.80 | 0.42 | 0.16 | 49% |
| map alone (`pred_rel`) | 4.4 | 8.1 | 0.49 | 0.96 | 0.71 | 0.47 | 0.17 | 54% |
| **combined (`pred_blend`)** | **3.9** | **6.0** | **0.52** | 0.96 | 0.78 | **0.56** | 0.22 | 59% |
| combined, true masks (`gt_blend`) | 4.0 | 5.9 | 0.52 | 0.96 | 0.81 | 0.49 | 0.21 | 59% |

C1 6.0 <= 7 cm **pass**; C2 3.9 <= 6 cm **pass**; C3 0.52 >= 0.49 **pass**; C4 0.56 >= 0.48 **pass**; C5 0.78 >= 0.77
**pass**; C6 visible-surface gap 6.0 cm [5.4, 6.4] <= 7 **pass** (SAM 3D Body 14.3 [11.1, 16.5], map 7.9 [7.1, 8.5]).
Against the original targets, legs reaches P5 (0.56) and region pairs stay short of P2 (0.52 vs 0.57). MMA: distance
error 9.6 -> 4.9 cm, region pairs 0.26 -> 0.37.

**Decision (per the rule):** the combined depth becomes the single-camera placement step. Next comes orientation from
the depth map (`scripts/engage_depth_fit.py`, developed on the first sample's maps in `depth_probe_o0maps/`).

**What this means.** From one ordinary camera, a monocular depth model and the per-person body model err in opposite
directions about how far apart two people are in depth. Combined, they place two grapplers about as well as two cameras
did with simple averaging (3.9 vs 3.3 cm distance error; legs 0.56 vs 0.51; but body lock 0.78 vs 0.88).
**Caveats:** ground-truth boxes; Harmony4D standing grappling (the weight came from other views of the same
sequences); no ground work yet (ViCoS has no 3D truth, so the test there is the pre-registered H1-H6 and duplicates).

## Development checks after the fresh-view pass (first sample only; exploratory)
- **What is left after the combined depth** (oracle on top of `pred_blend`, `pred_blend/oracle/`; two new oracle
  variants: `fix_rot` = the true rotation only, about the predicted pelvis; `fix_center` = the centroid of the 19
  canonical points moved onto the truth). Grappling:

| on top of the combined depth | distance | rel depth | region pairs | body lock | legs | arms |
|---|---|---|---|---|---|---|
| as is | 3.9 cm | 5.5 | 0.53 | 0.78 | 0.52 | 0.30 |
| + true rotation only | 3.9 | 5.5 | 0.54 | 0.83 | 0.49 | 0.33 |
| + true pelvis position | 0 | 0 | 0.55 | 0.88 | 0.56 | 0.30 |
| + true body centre | 2.4 | 2.8 | 0.61 | 0.92 | 0.52 | 0.40 |
| + true position and rotation | 2.0 | 2.1 | 0.67 | 0.97 | 0.68 | 0.43 |
  Rotation alone adds little. Placing the body's centre (where its mass is) beats placing the pelvis exactly, and legs
  need position and rotation together.
- **Torso facing direction from the map** (plane fit to the visible chest/back): worse than SAM 3D Body's own
  (MMA, 351 torso patches: 21.0 deg vs 15.7 deg median error; the map better in 43%). Rotating the bodies to the map's
  surface by ICP (`engage_depth_fit.py`, one MMA camera) made the torso orientation worse (16.9 -> 18.2 deg). The map
  places the two people; it does not orient them.
- **SAM 3D Body's torso orientation error** (grappling, n = 2,798 bodies): median 10.2 deg. It is persistent over time
  (correlation 0.75 between samples 4 frames apart), and averaging over 0.3-0.8 s does not help (10.3 / 11.1 deg).
- **Where to measure the map** (`engage_depth_place.py`, in-sample weight sweeps; grappling relative pelvis depth at
  the best weight): all visible pixels 5.4 cm (reproduces the probe); only within 15 cm of the other athlete 6.7 cm;
  interior only (eroded 6 / 15 px) 5.7 / 5.8 cm. The map blurs depth across edges, so whole-body medians are best.
- **Averaging the map's correction over time** (`depth_probe_dense/`: MoGe-2 on every view, i.e. every 4th frame, of
  4 development sequences, 025 / 028 / 033 / 036, 1,068 views; `engage_depth_place.py --smooth K`, median of each
  athlete's shift over +-K views). Grappling relative pelvis depth at the best weight: no averaging 6.0 cm (weight 0.7;
  these sequences are easier, SAM 3D Body alone 9.8 cm), +-1 view 6.2, +-2 6.4, +-4 6.6, +-8 7.3. Averaging does not
  help. The remaining error changes as fast as the athletes move, so there is no slow-varying noise to remove.
- **The joint two-body fit started from the combined depth** (`engage_jointfit.py` arms full / no_pen on
  `pred_blend`; ViTPose keypoints subset to these views; `depth_fit_dev/`). Grappling: distance 3.9 -> 5.9 / 7.5 cm,
  rel depth 5.5 -> 8.7 / 11.1, region pairs 0.53 -> 0.50 / 0.43, legs 0.52 -> 0.33 / 0.27, body lock 0.78 -> 0.75 /
  0.64. Lateral improves (5.1 -> 4.0 cm), but fitting 2D keypoints drags depth wrong (median move 8 cm): at 4 m the
  keypoints cannot see depth, and the fit trades away the depth the map fixed.

**Where single-camera fixes stand (2026-10-24).** Combined depth: confirmed twice (Harmony4D fresh views; ViCoS
ground work). Every cheap fix tried on top of it is worse or flat: map orientation, ICP, edge-zone / interior pixels,
temporal averaging, the 2D + no-overlap joint fit. The oracle ceiling on top of it (true body centre + rotation:
region pairs 0.67, body lock 0.97, legs 0.68) needs information that no hand-built cue here supplies. The next step
is a learned two-person placement model, trained on Harmony4D with sequences held out.
