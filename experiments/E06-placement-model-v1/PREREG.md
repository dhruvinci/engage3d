# Pre-registration: learned two-person placement model, held-out sequences — written 2026-10-24 22:45, before the test run

## What it is
A small network (`scripts/engage_place_train.py`; MLP 285 -> 256 -> 256 -> 12, an ensemble of 3, about 140k
parameters each) that corrects each body's centre and rotation after the confirmed combined depth
(`depthblend_prereg.md`). Inputs (`scripts/engage_place_features.py`):
- both bodies' 19 canonical points in the crop's virtual camera frame;
- per body region, how the MoGe-2 depth map disagrees with the body's shape;
- the 2D disagreement of ViTPose+ keypoints with the projected bodies, and their confidences;
- the map-vs-SAM 3D Body gap.

Loss: the pair's configuration against the truth's. Augmentation: athletes swapped, view mirrored.

## Development (done; the test sequences were never used)
4-fold cross-validation by sequence over the 12 development grappling sequences (MMA always in training); 4,528
out-of-fold views (`benchmarks/engage_h4d/place_cv/`). Combined depth -> model: distance error 3.81 -> 3.12 cm, relative
depth 5.76 -> 4.60, relative lateral 5.00 -> 3.61, region pairs 0.53 -> 0.57, in contact 0.96 -> 0.97, body lock 0.79
-> 0.82, legs 0.57 -> 0.56, arms 0.23 -> 0.27, balance (as scored) 56% -> 61%. All four folds improved the
configuration error. Settings were fixed before the folds' relation scores were seen (epochs 60, 3 seeds, dropout 0.2,
weight decay 1e-3) and are not changed for the test.

## Test (fixed now)
- Sequences: grappling 031, 037, 042, 043 (all their views: 1,068 across 4 cameras), held out since the model was
  designed. Caveat: every 8th view of them was in the earlier depth samples, and the 0.7 combination weight was fitted
  on one of those samples. The learned model never saw them.
- Final model: trained once on every non-test view (12 development grappling sequences + MMA), with the settings
  above. `engage_place_train.py --test`.
- Scored with `engage_bench_oracle.py` (`pred` rows of `blend` and `model`, same views).

## Criteria (model vs combined depth on the test sequences)
| | Measure | CV change | Pass if |
|---|---|---|---|
| T1 | Distance between the bodies, median error | -0.69 cm | model <= blend - 0.4 cm |
| T2 | Relative pelvis depth error | -1.16 cm | model <= blend - 0.6 cm |
| T3 | Region-pair F1 | +0.04 | model >= blend + 0.02 |
| T4 | Body lock F1 (no harm) | +0.03 | model >= blend |
| T5 | Legs F1 (no harm) | -0.01 | model >= blend - 0.03 |
Also reported: the original targets (P1 <= 6 cm, P2 >= 0.57, P4 >= 0.75, P5 >= 0.56), arms, balance, and MMA is
not tested (it was in training).

**Decision rule.** T1-T5 pass: the learned model becomes the single-camera placement step (combined depth, then the
model). Next: ground work (ViCoS; the model never saw lying bodies, so it is applied only after its own pre-registered
check) and more training data. T1 or T2 fails: the cross-validation gain does not hold on new sequences; keep the
combined depth and report the model as not generalising.
