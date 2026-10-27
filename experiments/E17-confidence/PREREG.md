# Pre-registration: E17, can the pipeline flag the frames where its measurement is wrong? — written 2026-10-27, before the full run

## Why
E20/E21 mapped where single-camera engagement measurement works (ballroom, distance at range) and where it does not
yet (limbs hidden at contact). A product can use today's pipeline if it knows, frame by frame, when to trust it. This
tests whether truth-free signals flag the bad frames.

## Disclosure (read first)
While checking the code, a smoke run (2 training epochs, 3,000 development views) scored the test sets below on
cameras 01/06/11/16 by mistake, before this was written. Seen: held-out grappling distance flag AUROC 0.77 (trusted half
2.4 cm vs 3.6 cm overall), limbs 0.79; MMA 0.71 / 0.74; hugging 0.76; ballroom 0.78; sword 0.60. The thresholds below
were drafted before that run and are unchanged. To keep a clean test, the **confirmatory sets are views the smoke run
never scored**: held-out grappling on cameras 03/08/13/18, and MMA on its other 16 cameras. The camera-01/06/11/16 sets
and the other activities are reported as secondary.

## Method (`scripts/engage_confidence.py`)
- **Placement:** E08's settings (width 512, 60 epochs, 3 seeds, mirroring, `--drop-overlap`), trained on the 12
  development grappling2 sequences x 20 cameras. Out-of-fold predictions for the flag's training come from 4-fold
  cross-validation by sequence; a full model scores the test sets.
- **Targets (truth used only to train and score the flag):**
  - the distance error between the two people after the learned placement, "bad" above 5 cm;
  - the error of the limb joints in contact (E11's measure, released model), "bad" above 6 cm.
- **Signals (no truth):**
  - how much the 3 ensemble members disagree (pair distance and all points);
  - the predicted closest approach and distance of the two bodies;
  - how far the correction moved them;
  - ViTPose+ confidence and 2D residuals of the limb joints;
  - the depth map's evidence (gap, valid flag, visible pixels);
  - the 7 overlap (mutual occlusion) features.
- **Flag:** logistic regression per target (L2), trained on the out-of-fold development predictions.
- **Measures:**
  - AUROC: how well the flag ranks bad frames above good ones;
  - risk at half coverage: the median error of the half of frames the flag trusts most, against all frames.

## Criteria (confirmatory sets)
| | Measure | Passes if |
|---|---|---|
| C1 | held-out grappling, cameras 03/08/13/18: distance flag AUROC | >= 0.70 |
| C2 | the same views: median distance error of the trusted half vs all | >= 30% lower |
| C3 | MMA (new people), other 16 cameras: distance flag AUROC | >= 0.65 |
| C4 | held-out grappling, cameras 03/08/13/18: limb flag AUROC | >= 0.65 |

**Reading.** C1-C4 pass: every measurement can ship with a trust flag, and the trusted half is reliable enough to use
now, in grappling too. If C1 passes and C3 fails, the flag is tuned to one pair and needs more people. If C4 fails,
limb errors are not predictable from these signals, and hidden limbs need direct work (E14, E16).

Compute: CPU only, about 1-1.5 hours (5 placement trainings); no GPU, no new disk.
