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
  (trained standing) does not carry over to the ground, and MapAnything cannot register ViCoS's tight crops into one
  3D frame. ([E12](experiments/E12-vicos-synced-phones/))
- Given the true 2D joints as clicks, SAM 3D Body's tangled-limb error falls 15% (held-out pair) to 25% (new people);
  given today's 2D keypoints (ViTPose+) it rises 19-25%, because inside a tangle the 2D model takes the partner's
  limbs. ([E18](experiments/E18-oracle-clicks/))
- ViTPose+ fine-tuned on Harmony4D contact is 21-23% more accurate on limb joints inside tangles (held-out and new
  people) and halves the harm of its clicks into SAM 3D Body (+25% -> +14%), but the clicks still hurt: they land
  exactly where the 2D model is least reliable. ([E19](experiments/E19-contact-2d-keypoints/))
- Calmer engagement: ballroom dancing is measured well from one camera (the two dancers' distance within 2.7 cm,
  touching F1 0.96, limbs in contact 3.8 cm), but hugging is not easier than grappling (arms hidden behind the
  partner: limbs 6.6 cm), though the embrace itself is detected (F1 0.89). The hard part is hidden limbs at contact.
  ([E20](experiments/E20-calmer-contact/))
- Engagement at range (sword fighting, 2.4 m apart): the distance between the two people is right to about 4% (8.4 cm),
  the same relative accuracy as in close contact, but the pre-registered absolute bars fail; the placement learned on
  grappling hurts depth at range. ([E21](experiments/E21-engagement-at-range/))
- The pipeline can flag its own bad frames without any truth: a trust flag built mostly on the placement ensemble's
  disagreement ranks a bad frame above a good one 4 times in 5 (AUROC 0.795 held-out grappling, 0.76 new people, 0.77-0.80
  ballroom and hugging; limbs 0.79). Trusting half the frames cuts the distance error by 28% on the held-out pair (bar:
  30%) and 39-51% where errors are larger (MMA, hugging); at range it does not work (0.58).
  ([E17](experiments/E17-confidence/))
- Time does not fill in hidden limbs: a head reading 5-31 frames is no better than one reading a single frame (4.81 vs
  4.81 cm in tangles), because SAM 3D Body's errors on a hidden limb persist across neighbouring frames. Hiding frames
  shows time alone bridges about one frame (0.07 s); at 0.5 s the error doubles. The per-frame head itself is the best
  limb correction so far (5.39 -> 4.81 cm, exploratory). ([E14](experiments/E14-temporal-limbs/))
- **Caveat (found 2026-10-27):** SAM 3D Body was trained on Harmony4D (all 24 subjects, arXiv 2602.15989), so its
  absolute numbers on Harmony4D in E10-E21, including the MMA "new people", are likely optimistic. Comparisons within
  each experiment (corrections, prompts, fits against the same model on the same views) are unaffected.
