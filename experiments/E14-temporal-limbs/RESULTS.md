# E14 results: time does not fill in hidden limbs; neighbouring frames carry the same errors (pre-registration: `temporal_prereg.md`)

Run 2026-10-27 13:28-13:50 on the CPU, after the pre-registration and its amendment were pushed (engage3d `c6ac9f3`,
`637eb8b`). Code: `scripts/engage_temporal.py`. Outputs: `benchmarks/engage_h4d/temporal/` (`temporal.json`,
`run.log`). Training: 22,640 development views (12 sequences x 20 cameras, every 2nd frame), 2 seeds per head.

## Criteria (median error of the 8 limb joints inside a tangle, rigid-aligned over 14 points)
| | Comparison | Result | Verdict |
|---|---|---|---|
| T1 | held-out (1,068 views): window 15 vs window 1 | 4.81 vs 4.81 cm: +0.1% (95% interval -1.9% to +2.3%); needed -10% | **fail** |
| T2 | held-out: where the gain lands | no gain to place (tangle -0.1%, other limb joints -4.3%: window 15 is slightly worse outside tangles) | **fail** (met as written, but empty) |
| T3 | held-out, 940 views with frames on both sides: centre +-3 frames (about 0.5 s) hidden vs window 1 | 10.57 vs 4.80 cm: +120% (interval +110% to +129%); allowed +10% | **fail** |
| T4 | MMA, new people (1,484 views): window 15 vs window 1 | 6.08 vs 6.01 cm: +1.2% (interval -0.6% to +2.9%); needed -5% | **fail** |

## All numbers (cm)
| | held-out, tangle | held-out, other limb joints | MMA, tangle | MMA, other |
|---|---|---|---|---|
| SAM 3D Body (no head) | 5.39 | 4.24 | 6.25 | 4.61 |
| window 1 (per-frame head) | **4.81** | **2.46** | **6.01** | **3.13** |
| window 5 (+-2 frames, 0.27 s) | 4.78 | 2.43 | 6.04 | 3.19 |
| window 15 (+-7, 0.93 s) | 4.81 | 2.56 | 6.08 | 3.31 |
| window 31 (+-15, 2 s) | 4.85 | 2.67 | 6.16 | 3.46 |

Masked gap: the 31-frame head that never sees the centre frame, vs window 1 (all views):
| hidden | held-out, tangle | vs window 1 | plain interpolation | MMA, tangle | vs window 1 |
|---|---|---|---|---|---|
| centre only (0.07 s) | 5.25 | +9% | 5.55 | 6.66 | +11% |
| +-1 frame (0.2 s) | 6.99 | +45% | 7.39 | 9.86 | +64% |
| +-3 frames (0.47 s) | 11.14 | +132% | 11.22 | 15.07 | +151% |
| +-7 frames (1 s) | 14.81 | +208% | 14.70 | 19.05 | +217% |

## Reading (pre-registered: "T1 fails")
- **Time adds nothing.** Windows of 5, 15 and 31 frames land within +-1% of the single frame inside tangles, on the
  held-out pair and on new people. Longer windows are slightly worse outside tangles (more inputs, no new
  information).
- **Why: the errors persist over time.** Hiding the centre frame and simply averaging its two neighbours (1/15 s
  away) is almost as good as SAM 3D Body on the frame itself (5.55 vs 5.39 cm). When a limb is hidden, SAM 3D Body is
  wrong in the same way across neighbouring frames, so averaging or reading them cannot correct it. Grappling contacts
  last longer than any window tested.
- **Time alone bridges about one frame.** With one frame hidden (0.07 s) the head stays within 9-11% of seeing it; at
  0.2 s the error is 45-64% higher; at half a second it doubles. Grappling moves too fast for gaps to be filled from
  neighbours. That is the yardstick for video models (GVHMR, WHAM, PromptHMR-Video) on this protocol.
- **Exploratory, not pre-registered: the per-frame head is the best limb correction so far.** Trained on all 22,640
  development views and given the partner's points, window 1 brings tangled limbs from 5.39 to 4.81 cm (-11%; E11's
  decoder fine-tune: 4.97, E10's head: 5.07) and the other limb joints from 4.24 to 2.46 cm. On new people the
  tangle gain shrinks to -4% (6.25 -> 6.01).
- **Next:** hidden limbs need new evidence, not more of the same view: image features from a model trained on contact
  (E16: RF-DETR keypoints and masks as SAM 3D Body prompts), or a second camera.
