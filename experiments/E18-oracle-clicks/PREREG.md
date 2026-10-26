# Pre-registration: E18, do true 2D keypoints fix tangled limbs in SAM 3D Body? — written 2026-10-26, before any run on the test views

## Why
E11 showed that fine-tuning SAM 3D Body's decoder lowers tangled-limb error only modestly (8% against the released
model). Two routes remain for the cloud budget: a bigger fine-tune, or better input. SAM 3D Body accepts 2D keypoint
"clicks" (E05 showed they help on ViCoS). This measures the ceiling of the input route: if the model is given the
true 2D positions of the joints, how much of the tangled-limb error goes away? And how much of that does the current 2D
model (ViTPose+) already deliver?

## Data
The E11 test views, ground-truth boxes: held-out sequences 031 / 037 / 042 / 043 on cameras 01/06/11/16 (1,068 views)
and MMA 016 (1,484 views; new people).

## Method (`scripts/engage_h4d_clicks.py`; checked on development views only)
- Per athlete crop, SAM 3D Body runs once (`original`). Its own click loop then reuses that image embedding: the
  model's prompt sampler clicks the body keypoint whose current 2D estimate is furthest from the given 2D pose (as
  in E05).
- Variants:
  - `vitpose2`: ViTPose+ keypoints (the pipeline's 2D model today), 2 clicks, the model's trained maximum;
  - `oracle2`: the true joints projected through the true fisheye camera (12 body joints; face points get no click),
    2 clicks;
  - `oracle4`: the same with 4 clicks (beyond the model's training; exploratory, no criterion).
- Check before the run (development views): confident ViTPose joints lie a median 21 px from the projected true
  joints of the same athlete and 150 px from the other's, so the projection and the athlete order are right.
- Measure as E11: the 19 canonical points aligned rigidly onto the truth over the 14 points both body models define
  alike; median error of the 8 limb joints, for tangle points and the rest. Intervals: 1,000 bootstrap resamples of
  views.

## Criteria (median error of tangle points among the 8 limb joints)
| | Comparison | Passes if |
|---|---|---|
| O1 | held-out: `oracle2` vs `original` | >= 25% lower (released 5.40 cm -> <= 4.05), 95% interval below 0 |
| O2 | held-out: share of `oracle2`'s gain that `vitpose2` reaches | reported |
| O3 | held-out, the other limb joints: `oracle2` vs `original` | at most 2% higher |
| O4 | new people (MMA): `oracle2` vs `original` | >= 15% lower |

**Reading.**
- **O1 passes and `vitpose2` reaches less than half of the gain:** better 2D keypoints inside tangles are the lever.
  E16 (RF-DETR keypoints fine-tuned on Harmony4D contact, feeding clicks) comes before any cloud fine-tune.
- **O1 passes and `vitpose2` reaches half or more:** clicks with today's 2D model already deliver; they join the
  pipeline now, and E16 targets the rest.
- **O1 fails:** even true 2D does not fix tangled limbs in 3D; what remains is depth ambiguity that 2D input cannot
  settle. Next: multi-view supervision, or the cloud fine-tune with more people.

Caveat: the true 2D includes hidden joints, which a detector only predicts if trained to; the ceiling is an upper
bound on what any 2D model could give.

Compute: GPU about 60-90 minutes (one backbone pass per crop; the click loops reuse it), after E12 finished, alone on
the GPU; no new disk.
