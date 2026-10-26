# Pre-registration: E12, ground work without 3D truth, using ViCoS's synchronised phones — written 2026-10-26, before any body or relation was computed on these frames

## Why
There is no 3D truth for ground work, so E05 could only report how often the defining relation of each expert-labelled
position appears. ViCoS was filmed by 2-3 phones at once in each of 6 sessions. Two phones looking at the same moment
see the same bodies. So the agreement between them is a reliability measure that needs no truth: if the relations we
measure from one phone are about the bodies, two phones of one moment should agree more than two phones of different
moments that carry the same position label. Also tested: whether the learned placement correction (E06, trained on
standing Harmony4D) carries over to ground work, and whether MapAnything can turn the synchronised phones into usable
multi-view 3D.

## Data (`scripts/engage_vicos_sync.py select`; selected before writing this, nothing else computed)
- Sessions and phones: {00,01,02}, {03,04,05}, {06,07,08}, {09,10}, {12,13}, {14,15}. Stream 11 is left out: it runs
  at an offset from 12/13 (labels agree only 84-85%).
- Moments: both athletes' 2D poses present in every phone of the session, the same label in every phone, at least 30
  frames apart, spread over the positions; up to 100 per session.
- Result: 492 moments, 1,223 images, 239 of them seen by three phones. Positions: turtle 85, open guard 68, back 67,
  takedown 49, 50/50 47, closed guard 44, standing 39, side control 38, mount 31, half guard 24.

## Pipelines compared (all with SAM 3D Body's default camera; no intrinsics are known)
- `clicks`: SAM 3D Body with keypoint clicks from ViCoS's 2D poses (E05's main bodies).
- `blend`: `clicks` + the combined depth (MoGe-2 map, weight 0.7; E05's default).
- `place`: `blend` + E06's learned placement correction (v1 weights, unchanged). Features are built as in
  `engage_place_features.py`, with ViCoS's own 2D poses in place of ViTPose and a pinhole camera instead of the fisheye.

## Measures
- **Relations:** threading with margin >= 5 cm, as E05.
  - **Identity-free (primary):** chain type (torso, head, leg, arm) through loop type (leg loop, arm loop), either
    athlete. The athlete numbering is consistent across phones in the labels, but not verified for the 2D poses.
  - **Identity-aware (secondary):** with athlete identity (`pose1` / `pose2`).
- **Agreement:** the Dice overlap of two images' relation sets.
  - **Synchronised:** two phones of the same moment.
  - **Baseline:** the same two phones, the second at another moment of the same session and label, at least 300
    frames away when possible.
  - Pairs with no relation in either image are counted separately.
  - Intervals: 1,000 bootstrap resamples of moments.
- **Defining relations per position,** as E05 V1 and V2 (`engage_vicos_score.py`), on this sample.

## Criteria
| | Comparison | Passes if |
|---|---|---|
| A1 | `blend`: synchronised vs baseline agreement (identity-free) | difference >= 0.05, interval above 0 |
| A2 | synchronised agreement: `blend` vs `clicks` | reported: the more reliable pipeline is the one with higher agreement (>= 0.03 counts as a difference) |
| B1 | `place` vs `blend`: defining relations (mount O.torso -> D.leg loop; closed guard the same; half guard O.leg -> D.leg loop) | up by >= 0.05 in at least 2 of 3, none down by more than 0.05; controls (standing, takedown, turtle, side control: any leg-loop threading) all <= 0.20; synchronised agreement not lower than `blend` by more than 0.03 |
| C1 | MapAnything feasibility (Apache-licensed checkpoint), 60 three-phone moments (20 from each three-phone session, fixed seed) | runs on the 6 GB GPU; in >= 60% of moments, (i) the distance between the two athletes, measured from each phone's points, varies by < 15 cm across the three phones, and (ii) each athlete's 3D centre varies by < 25 cm across them |

Athlete points for C1: MapAnything's 3D points at ViCoS's 2D keypoints of that athlete (confidence >= 0.5), median.
Reported without a criterion: how well MapAnything's distance between the athletes correlates with `blend`'s (Spearman).

**Reading.**
- **A1 passes:** single-phone relations carry information about the moment, not just the label. Cross-phone agreement
  becomes the no-truth reliability score for ground work, and A2 / B1 rank the pipelines with it.
- **A1 fails:** what one phone measures is mostly label prior or noise; single-view ground-work relations are not yet
  instance measurements.
- **B1 passes:** the learned placement transfers from standing to ground work and becomes the ground-work default.
- **C1 passes:** MapAnything can give pseudo-3D for ground work. A follow-up builds a ground-work benchmark from it.

Compute: GPU about 30-45 minutes (bodies, depth maps, MapAnything), run after E11 finished, never alongside another GPU
job; CPU minutes. Disk: depth maps about 0.6 GB; a new environment for MapAnything with its weights, about 8-10 GB.
