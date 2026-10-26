# Pre-registration: joint two-person fitting — written 2026-10-22, before running

## Why this experiment (agreed with the user)
It decides the biggest fork in the plan: can engagement between two people be measured from **one ordinary camera**
with today's models plus a cheap fix, or does it need **multi-camera capture or a model we train**?
- Works -> the first usable single-camera engagement measures; the user's multi-camera capture becomes validation, not
  a prerequisite.
- Fails -> stop tuning toward single-camera; capture or two-person model training is the critical path, and the
  capture project is planned knowing it is necessary.
Scope: any human engagement, not BJJ only (user, 2026-10-22). So the test scores contact surfaces and the distance
between the bodies as well as loops. Harmony4D (standing grappling + MMA, multi-view 3D truth) and ViCoS (real BJJ
ground work, expert position labels) are both used, because a standing-only answer would not cover ground work.

## What changes
Baseline: SAM 3D Body per person from a box (Harmony4D: ground-truth boxes, translation re-solved through the true lens;
ViCoS: boxes from its 2D poses, default focal length). Today's scores: `h4d_s3d_results.md`, `vicos_results.md`.

**Arm 1, joint rigid fit** (`scripts/engage_jointfit.py`, CPU). Per view, each body may rotate (about its pelvis) and
translate as a whole; limbs are not re-bent. Minimised together:
1. 2D evidence: each body's keypoints projected through the camera (Harmony4D: true fisheye; ViCoS: the model's own
   pinhole) against independent 2D keypoints of that person (Harmony4D: ViTPose+ base from the ground-truth box;
   ViCoS: the dataset's own 2D poses), weighted by keypoint confidence, robust loss;
2. no interpenetration: vertices of one body inside the other's limb capsules are penalised, both ways;
3. a prior keeping each body near the model's estimate (depth looser than lateral).
Controls: the same fit without (2) (2D only) and without (1) (no-overlap only).
**Arm 2** (after Arm 1, aimed at ViCoS duplication): SAM 3D Body's own click refinement fed the same 2D keypoints,
then Arm 1. Reported separately.

## Success criteria (Harmony4D, 4 cameras x every 2nd frame; grappling and MMA reported separately and pooled)
| | Measure | Baseline | Pass if |
|---|---|---|---|
| P1 | Distance between the two bodies, median error | 10.2 cm | <= 6 cm |
| P2 | Which surfaces touch: region-pair F1 (best of 2 / 5 / 10 cm thresholds) | 0.47 | >= 0.57 |
| P3 | Loop balance sign agreement (truth not tied) | 46% | >= 70% |
| P4 | No harm: body lock F1 / in-contact F1 | 0.80 / 0.85-0.92 | not below 0.75 / 0.85 |
| P5 | Leg through leg loop F1 | 0.46 | >= 0.56 |

**Decision rule.** P1 + P2 + P3 pass: single-camera measurement is viable with this fix. P1 + P2 pass, P3 fails:
placement and contact are fixable cheaply; loops need limb-level refinement (Arm 2 / re-bending limbs). P1 and P2 fail:
the cheap fix does not work; capture or training is the critical path.
**ViCoS:** re-test the pre-registered H1-H6 (`vicos_prereg.md`, same thresholds) on the fitted bodies, and the share of
frames where both predicted bodies are the same person (24-30% in mount / guard / side control; Arm 2 target <= 10%).

## Caveats that travel with the result
Harmony4D boxes are ground truth (detection solved by oracle). ViTPose runs from the box with no masks, so in contact
it can take the partner's limbs. ViCoS 2D poses are the dataset's own detections (the same ones that gave the boxes),
and only frames where both athletes were detected are used.
