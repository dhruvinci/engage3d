# E10 results: SAM 3D Body's pose token does not fix limbs inside tangles (pre-registration: `limbhead_prereg.md`)

Run 2026-10-26 14:05-14:34 (CPU), after the pre-registration was committed and pushed (engage3d `fb20060`, 14:04).
Code: `scripts/engage_limb_head.py`. Outputs: `benchmarks/engage_h4d/limb_head/{cv,test}.json`.

## Criteria (held-out test: 1,068 views; median rigid-aligned error of tangle limb points, 6,072 points)
| | Comparison | Result | Verdict |
|---|---|---|---|
| L1 | `token` vs `ponly` | 5.90 vs 5.78 cm: +2.1% (95% interval +0.4% to +3.8%) | **fail** (the token makes it slightly worse) |
| L2 | `token` vs `none` | 5.90 vs 6.19 cm: -4.7% (-6.9% to -2.2%) | **fail** (needed -15%) |
| L3 | other limb points, `token` vs `none` | 2.87 vs 5.00 cm: -43% | pass |
| L4 | `shuffled` vs `ponly` | 5.81 vs 5.78 cm: +0.5% | pass (no leak) |
| N1 | MMA (new people, 7,068 tangle points), `token` vs `ponly` | 6.78 vs 6.73 cm: +0.7% (-0.9% to +2.7%) | **fail** |

## All variants (median cm; rigid-aligned per body)
| | tangle limb points | other limb points | all 19 points |
|---|---|---|---|
| held-out, none | 6.19 | 5.00 | 5.04 |
| held-out, points-only head | **5.78** | 2.88 | 2.82 |
| held-out, points + token | 5.90 | **2.87** | 2.82 |
| held-out, points + shuffled token | 5.81 | 2.91 | 2.84 |
| MMA, none | 7.10 | 5.35 | 5.30 |
| MMA, points-only head | 6.73 | **3.61** | 3.36 |
| MMA, points + token | 6.78 | 3.70 | 3.43 |
| development CV (22,640 views), none / points-only / token / shuffled | 6.25 / 5.55 / 5.53 / 5.61 | 5.28 / 3.05 / 3.00 / 3.10 | 5.27 / 2.98 / 2.92 / 3.02 |

Per limb, tangle points, held-out (none -> points-only -> token, cm): knees 5.88 -> 3.26 -> 2.93 (left), 5.42 -> 4.59
-> 4.32 (right); toes 8.59 -> 4.17 -> 3.82, 9.49 -> 4.74 -> 5.26; elbows 4.98 -> 5.59 -> 5.54, 4.77 -> 3.59 -> 3.75;
wrists 6.35 -> 5.67 -> 6.09, 5.77 -> 5.65 -> 5.59; hands 7.45 -> 6.98 -> 7.59, 7.08 -> 7.40 -> 7.13. The token's
per-limb changes go both ways; arms in tangles barely move with any head.

## Reading (as pre-registered: "L1 fails and L2 fails")
- **Neither the predicted points nor SAM 3D Body's own final pose token can correct limbs inside tangles.** The token
  adds nothing over the points (+2.1%, interval above zero), on the held-out pair and on new people.
- **Outside tangles the limb error is mostly fixed and learnable** (exploratory, not a criterion): a head on the
  points alone takes the other limb points from 5.00 to 2.88 cm, and all points from 5.04 to 2.82 cm. Part of this is
  the fixed difference between how the two body models define points (hands, toes), part a consistent pose bias.
  Inside tangles the same head moves the error only from 6.19 to 5.78 cm.
- So the remaining error is concentrated where the two bodies touch, in the arms above all, and the information to fix
  it is not in the decoder's final token. **E11 (fine-tuning the decoder on two-person contact) is needed**, and its
  target should be how the decoder reads the contested pixels.
- Not measured here: whether the points-only head's gain outside tangles improves the relation measures (they are
  computed on meshes, not on these 19 points). Worth a check before adding it to the pipeline.
