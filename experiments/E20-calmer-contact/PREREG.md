# Pre-registration: E20, does single-camera engagement measurement already work where contact is calmer? (hugging, ballroom dancing) — written 2026-10-26, before any of these views was processed

## Why
Grappling is the hardest case of two-person engagement on purpose: constant, shifting, interlocked contact. Every
experiment so far ran on it (and on MMA). The broader claim is about human engagement in general: who touches whom,
with which body surfaces, in what hold. Hugging and ballroom dancing are meaningful engagement with sustained but
calmer contact: arms around the torso, a dance hold, hands joined. If the same pipeline already measures them well,
that is a usable result now, and it locates the hard part precisely in grappling's tangles.

## Data (Harmony4D test split, fetched with `engage_h4d_fetch.py`; cameras 01/06/11/16; ground-truth boxes)
- **Hugging:** sequence 002_hugging, every 2nd ground-truth frame (about 600 views).
- **Ballroom dancing:** sequences 007-010_ballroom2, every 8th ground-truth frame (about 1,120 views).
- Different people from the grappling pair; activities never seen in training.

## Pipeline (unchanged; the E08 recipe)
- SAM 3D Body per person box (`engage_bench_s3d.py`), ViTPose+ 2D (`engage_pose2d_h4d.py`), MoGe-2 maps
  (`engage_depth_probe.py`), placement features (`engage_place_features.py`).
- Learned placement: v2 settings trained only on the 12 development grappling2 sequences x 20 cameras (the E08 model;
  MMA, the held-out grappling sequences and every other sequence excluded), applied with `engage_place_train.py
  --test --test-seqs <hugging, ballroom>`.
- Three stages scored: SAM 3D Body alone, + combined depth, + learned placement.

## Measures
- From `engage_bench_oracle.py` (`pred` rows), per activity: distance error between the two people, relative depth,
  configuration error, region-pair F1 (which body surfaces touch), threading F1 by family (torso through an arm loop
  is an embrace or a hold).
- Limbs in contact, as E11: rigid-aligned error of the 8 limb joints for points within 20 cm of the other person
  (from the feature files).
- Reference values (not re-run): grappling held-out (E06/E11) and MMA (E08).

## Criteria (full pipeline = SAM 3D Body + combined depth + learned placement)
| | Measure | Passes if, in **both** hugging and ballroom |
|---|---|---|
| W1 | distance error between the two people (median) | <= 4.0 cm |
| W2 | region-pair F1 (which body surfaces touch) | >= 0.70 (grappling held-out 0.60, MMA 0.46) |
| W3 | limb joints in contact (released SAM 3D Body, E11's measure) | <= 4.0 cm (grappling held-out 5.40, MMA 6.27) |
| W4 | the learned placement (trained on grappling) vs combined depth: configuration error | lower in both (it transfers) |

**Reading.**
- **W1-W3 pass:** single-camera engagement measurement already works for calmer human contact. This becomes the
  showcase (with demo figures), and grappling is the documented hard case.
- **W1 passes, W2 or W3 fail:** placement works, but surfaces and limbs in contact are hard even when calm; the
  limb work (E11, E18) matters beyond grappling.
- **W4** says whether the correction learned on grappling helps elsewhere.

Compute: GPU about 1-1.5 hours (bodies, 2D, maps), after E18 finished, one job at a time; CPU about 30 minutes; disk
about 3 GB (fetched frames, maps).
