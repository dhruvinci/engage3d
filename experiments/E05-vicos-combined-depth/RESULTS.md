# Combined depth on ViCoS ground work — results (pre-registration: `vicos_depth_prereg.md`) — 2026-10-24

`benchmarks/engage_vicos/depth_s3d_clicks/` (main: SAM 3D Body with keypoint clicks) and `depth_s3d/` (secondary:
plain SAM 3D Body). 1,426 frames, MoGe-2 with SAM 3D Body's default camera, weight 0.70 fixed in advance, no masks
(`scripts/engage_vicos_depth.py`). Scores: `depth_<base>.<blend|rel>.score.log` (`engage_vicos_score.py`). Camera
check: the bodies rendered with that camera land 2-7 px from ViCoS's 2D keypoints on the smoke frames.

## Main (clicks bodies): PASSES
| | Measure | Clicks baseline | Combined depth | Pass if |
|---|---|---|---|---|
| V1 | mount: O.torso -> D.leg_loop | 0.14 | **0.29** | >= +0.05 |
| V1 | closed guard: O.torso -> D.leg_loop | 0.20 | **0.33** | >= +0.05 |
| V1 | half guard: O.leg -> D.leg_loop | 0.17 | **0.26** | >= +0.05 |
| V2 | controls, any leg-loop threading (standing / takedown / turtle / side control) | 0.00 / 0.20 / 0.14 / 0.14 | 0.00 / 0.20 / 0.12 / 0.09 | all <= 0.20 |
| V3 | duplicates (mount / closed / half / side control) | 7 / 3 / 3 / 3% | 9 / 1 / 4 / 2% | all <= 10% |
| V4 | H1-H6 passed | 2/6 (H4, H5) | 2/6 (H5, H6) | reported |
V1 (3 of 3 up by >= 0.05, none down), V2 and V3 pass. Back (torso through either leg loop) rises 0.39 -> 0.64 and
passes H6. H4 is lost: 50/50 mutual leg threading falls 0.12 -> 0.09, below half guard's 0.13. Arm-loop threading in
standing rises 0.12 -> 0.20 (not a criterion; worth watching as a false-positive sign).

**Decision (per the rule):** the combined depth carries over to ground work in direction. It is now the default
placement. The defining relations are still found in only about a third of mount and closed-guard frames, against
the definitions' >= 0.60. So most of the gap remains, and 3D ground truth for ground work is still the open need
(capture).

## Secondary and weight-1 variants
| | mount | closed | half (leg) | back | 50/50 mutual | duplicates mount/closed/half/side |
|---|---|---|---|---|---|---|
| plain SAM 3D Body | 0.22 | 0.23 | 0.23 | 0.49 | 0.06 | 24-30% |
| plain + combined depth | 0.29 | 0.34 | 0.23 | 0.67 | 0.10 | 21 / 17 / 23 / 17% |
| clicks + map only (weight 1) | 0.29 | 0.36 | 0.26 | 0.53 | 0.24 | 4 / 3 / 5 / 3% |
The plain bodies improve too, but their duplicates (one athlete rebuilt twice) remain. The clicks and the combined
depth fix different errors, and together they are the best ground-work pipeline so far.

## Caveats
No 3D truth: these are rates of the defining relation per expert-labelled position, not accuracies. The intrinsics
are SAM 3D Body's default and unknown in truth. Frames are limited to those where ViCoS's detector found both
athletes. The intervals (bootstrap over streams) are optimistic because streams of one session share moments.
