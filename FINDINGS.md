# Findings log

One line per finding, newest last. Each links to the experiment that supports it.

## 2026-10-22
- Single-camera 3D bodies (SAM 3D Body) reproduce contact (F1 0.85-0.92) and body locks (0.80) on standing
  grappling, but not which surfaces touch (0.42-0.47) or loop balance (46%). A shuffled-partner control collapses
  every relation to <= 0.05, so the signal comes from real relative placement. ([E00](experiments/E00-benchmark-baseline/))
- On BJJ ground work the relations point the right way but are found in only about a quarter of mount and guard
  frames; 24-30% of those frames reconstruct the same athlete twice. ([E00](experiments/E00-benchmark-baseline/))
- With the truth, fixing each body's depth along the camera ray alone passes the distance, surface and leg criteria.
  The error that breaks the relations is relative depth (14.5 cm, against 5.0 cm side to side), not limbs.
  ([E01](experiments/E01-joint-rigid-fit/))
- A joint rigid fit to 2D keypoints cannot recover that depth; a whole-image model (PromptHMR) places better but
  halves body-lock recall. ([E01](experiments/E01-joint-rigid-fit/), [E02](experiments/E02-prompthmr/))
- A second camera at about 90 degrees cuts the distance error 10.2 -> 3.3 cm, even with naive fusion.
  ([E01](experiments/E01-joint-rigid-fit/))

## 2026-10-24
- A monocular depth map (MoGe-2) measures the depth gap between two people better than the body model (8.4 vs
  14.0 cm). Its error grows with the gap, not with the distance to the camera. ([E03](experiments/E03-depth-map/))
- The map shrinks gaps (0.84x) and the body model stretches them (1.20x). Blending them passes a pre-registered test on
  fresh views: distance error 10.0 -> 3.9 cm. ([E03](experiments/E03-depth-map/),
  [E04](experiments/E04-combined-depth-fresh-test/))
- The same combined depth raises the defining relations of mount, closed guard, half guard and back control on BJJ
  ground work. ([E05](experiments/E05-vicos-combined-depth/))
- Depth maps do not give orientation; edge-only or interior-only pixels, temporal averaging and 2D fitting all make
  it worse. ([E03](experiments/E03-depth-map/))
- A small learned correction (move + turn each body) on held-out sequences: distance 3.95 -> 3.17 cm, region pairs
  0.55 -> 0.60. ([E06](experiments/E06-placement-model-v1/))

## 2026-10-25
- 4x more camera angles of the same athletes: small, consistent gains, but a pre-registered criterion fails by
  0.04 cm. The learning curve flattens. ([E07](experiments/E07-placement-model-v2-cameras/))
- All Harmony4D grappling is two pairs of athletes. Trained on one pair, the correction keeps about 2/3 of its gain
  on a different pair (MMA). ([E08](experiments/E08-new-people/))

## 2026-10-26
- A new pair's data beats the same amount of the same pair's data on every measure, but below the pre-registered
  thresholds. On a third pair variety and quantity are equal: two pairs are too few to measure variety. The
  correction is near the limit of a whole-body move; what remains is limb error inside tangles.
  ([E09](experiments/E09-athlete-variety/))
- SAM 3D Body's own final pose token cannot fix limbs inside tangles: a head reading it does no better than one on the
  predicted points alone (5.90 vs 5.78 cm on tangle limb points). Outside tangles the points-only head halves the limb
  error (5.0 -> 2.9 cm). The remaining error sits in the arms where the bodies touch: the decoder itself has to change.
  ([E10](experiments/E10-limb-head/))
- Fine-tuning SAM 3D Body's decoder on Harmony4D contact (local pilot, backbone frozen) lowers tangled-limb error 8-9%
  against the released model, held-out and on new people, but only 2% beyond the cheap points-only head on the
  held-out pair (5% on new people), and it saturated after one epoch. ([E11](experiments/E11-decoder-finetune/))

- On ViCoS ground work, two phones filming the same moment agree on the measured relations about twice as much as the
  position label alone explains (combined depth: 0.28 vs 0.14), so cross-phone agreement is a truth-free reliability
  score. The combined depth doubles the moment-specific agreement over SAM 3D Body alone; the learned placement
  (trained standing) does not carry over to the ground. ([E12](experiments/E12-vicos-synced-phones/))
