# Pre-registration: E11 pilot, fine-tune SAM 3D Body's decoder on two-person contact — written 2026-10-26, before any run on the test views

## Why
E10 showed SAM 3D Body's final pose token holds no extra information about limbs inside tangles: a head on it does no
better than a head on the predicted points. If tangled limbs can be fixed, the decoder has to change: it is the part
that reads the image for each person. This pilot fine-tunes the decoder locally, on the 6 GB GPU, before any cloud run.

## Data (views of `benchmarks/engage_h4d/place_feats2` that carry SAM 3D Body tokens: the same views as E10)
- **Training:** 5,000 views (10,000 athlete crops) from 10 of the 12 development grappling2 sequences, drawn with a fixed
  seed; views with a tangle point are 3x as likely to be drawn.
- **Validation (monitoring only, selects nothing):** 500 views from the other 2 development sequences (025, 035).
- **Held-out test:** the E06 views (sequences 031 / 037 / 042 / 043, cameras 01/06/11/16; 1,068 views).
- **New people:** MMA 016 (1,484 views), never trained on.

## Method (`scripts/engage_s3d_finetune.py`; settings fixed here)
- Ground-truth person boxes, as in every earlier experiment. The ViT-H backbone and the MHR head are frozen; only the
  body decoder trains (39.4M parameters). The backbone's output is cached once per crop; feeding it back reproduces the
  model's keypoints (checked to 0.001 mm).
- **Loss per athlete:** the 19 canonical points (a fixed linear map of the 70 keypoints), plus a learned fixed offset
  per point in the body's own frame (SMPL and MHR define some points differently), aligned rigidly onto the truth over
  the 14 points both models define alike (pelvis, neck, hips, knees, ankles, shoulders, elbows, wrists). The mean error
  over those 14, with limb points inside a tangle weighted 3x, plus 0.1 x the mean distance of the 70 keypoints from the
  original model's.
- AdamW (decoder lr 1e-5, offsets 1e-3, weight decay 0.01), one-cycle schedule, 3 epochs, batch 8, gradient clip 1.
- **Measure:** align each body rigidly onto its truth over the same 14 points; error of the 8 limb joints (elbows,
  wrists, knees, ankles). Tangle points: as E10 (within 20 cm of a bone of the other athlete's true skeleton).
- **Baselines** on the same views:
  - `original`: SAM 3D Body as released;
  - `original + points-only head`: E10's head (`engage_limb_head.py --align core14 --measure joint8`, trained on all
    12 development sequences), the cheap fix to beat.

## Criteria (median error of tangle points among the 8 limb joints)
| | Comparison | Passes if |
|---|---|---|
| F1 | held-out: fine-tuned vs original + points-only head | fine-tuned >= 15% lower, and its 95% bootstrap interval against `original` is below 0 |
| F2 | held-out, the other limb joints: fine-tuned vs original + points-only head | at most 2% higher |
| F3 | new people (MMA): fine-tuned vs original + points-only head | >= 5% lower |
| F4 | pipeline check: the evaluation's `original` errors vs errors from E10's stored points for the same views | median difference < 1 mm (otherwise the run is void) |

**Reading.**
- F1-F3 pass: fine-tuning the decoder fixes limbs in contact and transfers to new people. A full run (more data,
  cloud) is justified.
- F1 passes, F3 fails: it learned this pair, not contact. More people are needed in training (Harmony4D's second pair,
  Hi4D / CHI3D) before a full run.
- F1 fails, but fine-tuned is clearly below `original`: the pilot is too small, or the decoder alone is too weak. Next:
  a larger run, or adapters in the backbone, in the cloud.
- F1 fails and fine-tuned is about `original`: the frozen backbone features may not carry who owns which pixel in a
  tangle. Next: change the input (masks or keypoint prompts) rather than the decoder.

Compute: GPU about 3-4 hours (cache about 1 h, training about 1.5 h, evaluation about 40 min), one GPU job at a time;
the cache (about 22 GB) is deleted after the run. E12 starts only after E11 has finished.
