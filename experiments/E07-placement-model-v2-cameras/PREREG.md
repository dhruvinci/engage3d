# Pre-registration: placement model v2 (more cameras, optional pose tokens) — written 2026-10-25 00:05, before any new view was processed

## Why
Placement model v1 (`placemodel_results.md`) was trained on 4 of Harmony4D's 20 cameras (6,012 non-test views) and
passed its held-out test (distance 3.95 -> 3.17 cm, relative depth 5.82 -> 4.30, region pairs 0.55 -> 0.60). The
test-split zips hold the other 16 cameras of the same sequences (`~/Code/shared/README.md`). The user asked
(2026-10-24) for the best training set possible, using as many camera angles as available.

## Data (overnight queue `benchmarks/engage_h4d/night/queue.sh`)
- Training: the 12 development grappling sequences x up to 16 extra cameras (every 2nd ground-truth frame, as the
  benchmark), in waves of 4 / 8 / 4 cameras, plus the MMA sequence x 16 extra cameras (every 4th frame).
  Outputs: `s3d_box_xcam/`, `depth_maps_xcam/`.
- Extended test: the 4 held-out sequences (031 / 037 / 042 / 043) x 4 extra cameras (cam03 / 08 / 13 / 18). These are
  kept in their own folders (`s3d_box_xtest/`, `depth_maps_xtest/`) and never trained on.
- Pose tokens: SAM 3D Body's final pose token per athlete (1,024 numbers, `--save-tokens`) for every view. The
  original 4 cameras are re-run for their tokens (`s3d_box_tok/`).

## Choosing v2 (development only; the held-out sequences are not looked at)
4-fold cross-validation by sequence over the 12 development sequences, as for v1. Evaluated on the development views
of the original 4 cameras (the same views as v1's CV) and on all cameras. Candidates:
- training cameras: original 4 / + wave 1 / + waves 1-2 / all;
- tokens: none / 32 components (no mirroring with tokens; a no-mirror run without tokens is the control);
- settings: v1's (60 epochs, 3 seeds, dropout 0.2, weight decay 1e-3), plus at most two capacity variants if the data
  grows past 4x.
v2 = the candidate with the lowest median configuration error on the original-4-camera CV views. Ties within 0.1 cm
go to the simpler candidate (no tokens, fewer settings changes).

## Test (once, after v2 is fixed)
v2 is trained on every non-test view and evaluated:
- **Primary:** the same 1,068 views as v1's test (`place_test/`: held-out sequences, cameras 01 / 06 / 11 / 16).
- **Extended:** the held-out sequences x 8 cameras (the primary views + cam03 / 08 / 13 / 18), for the families that
  had few true cases (legs: 8 views; body lock: 68 views). v1 is also run on the extended views, for comparison.

| | Measure | v1 on the primary test | Pass if (v2) |
|---|---|---|---|
| X1 | Distance between the bodies, median error (primary) | 3.17 cm | < 3.17 cm |
| X2 | Relative pelvis depth error (primary) | 4.30 cm | < 4.30 cm |
| X3 | Region-pair F1 (primary) | 0.60 | >= 0.59 |
| X4 | Body lock and legs F1 (extended, with true-case counts) | v1 on the same views | v2 >= v1 on both |

**Decision rule.** X1-X4 pass: v2 replaces v1. X1 or X2 fails: more cameras of the same 12 sequences do not improve
placement, which points to sequence and athlete diversity (the Harmony4D training split) rather than viewpoints.
Either way, the learning curve (CV error against training cameras) is reported.
