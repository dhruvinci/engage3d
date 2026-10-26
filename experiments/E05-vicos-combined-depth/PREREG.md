# Pre-registration: combined depth on ViCoS ground work — written 2026-10-24, before the run

## Why
The combined depth (0.7 x MoGe-2 depth map + 0.3 x SAM 3D Body for the gap between the athletes) passed a
pre-registered test on Harmony4D standing grappling (`depthblend_prereg.md`: distance 10.0 -> 3.9 cm, legs 0.42 ->
0.56). Ground work is where BJJ positions live and Harmony4D has none. ViCoS has no 3D truth, but its expert position
labels say which relation must hold. Mount and closed guard both mean the top athlete's torso is inside the bottom
athlete's leg loop; half guard means a leg is inside it.

## What runs (fixed now)
- Bodies: `benchmarks/engage_vicos/s3d_clicks/vicos_s3d.npz` (SAM 3D Body with keypoint clicks: duplicate athletes
  3-7%, so moving them is meaningful). Secondary: `benchmarks/engage_vicos/s3d/` (no clicks, 24-30% duplicates).
- Same 1,426 frames. Camera: SAM 3D Body's own default (focal sqrt(W^2 + H^2), principal point at the image centre),
  given to MoGe-2 as its field of view.
- Method as on Harmony4D, with no masks. Each athlete's depth offset is the median of (map - predicted render) over its
  rendered pixels minus a band around the other body. The band is 5 px scaled by image size, at least 2 px. Each body
  moves along its ray by 0.70 x (offset - mean offset). Script: `scripts/engage_vicos_depth.py`.
- Scored with `scripts/engage_vicos_score.py` (the same pre-registered H1-H6 and duplicate measure).

## Criteria (main: clicks bodies)
| | Measure | Clicks baseline | Pass if |
|---|---|---|---|
| V1 | Defining relations: mount O.torso->D.leg_loop; closed guard O.torso->D.leg_loop; half guard O.leg->D.leg_loop | 0.14 / 0.20 / 0.17 | at least 2 of 3 rise by >= 0.05, and none falls by > 0.02 |
| V2 | Controls (H5): standing / takedown / turtle / side control, any leg-loop threading | 0.00 / 0.20 / 0.14 / 0.14 | all still <= 0.20 |
| V3 | Duplicates in mount / closed / half guard / side control | 3-7% | all <= 10% |
| V4 | H1-H6 passed (reported) | 2/6 | reported |

**Decision rule.** V1-V3 pass: the combined depth carries over to ground work in direction (without 3D truth, that is
the strongest claim ViCoS allows). It becomes the default placement, and 3D ground truth for ground work (capture)
stays the open need. V1 fails: the Harmony4D gain does not carry over to phone crops of ground work at these settings.
The suspects then are unknown intrinsics, low resolution and lying bodies seen from the side, and are checked before
anything else is built on it.

## Caveats
Intrinsics are unknown (SAM 3D Body's default), and the crops are small (231-585 px tall). The 2D poses and boxes come
from ViCoS's own detector, so only frames with both athletes are used. The expected rates are definitions, not
measurements (`vicos_prereg.md`).
