# Pre-registration: E10, can SAM 3D Body's own pose token fix limbs inside tangles? — written 2026-10-26, before any run on the test views

## Why
The learned placement correction is near its limit ([E07](placemodel_v2_results.md), [E09](diversity_results.md)):
moving and turning each body perfectly would still leave about 5.8 cm of configuration error, and what remains is
limb error inside tangles. Before fine-tuning SAM 3D Body's decoder (E11, the expensive step), this scopes the cheap
one: does the model's own final pose token already carry the information to correct the limbs, so that a small head
reading it can fix them?

Development baselines (computed before writing this, on development views only; no test view was looked at): after
aligning each predicted body rigidly onto the truth, limb points are 5.5 cm off (median); limb points inside a tangle
(29% of them) 6.25 cm; the other limb points 5.3 cm.

## Data
`engage_place_features.py` outputs that carry tokens (`benchmarks/engage_h4d/place_feats2`, 26,260 views): the 19
canonical points per athlete after the combined depth (`P`), the truth's points (`G`), and SAM 3D Body's final pose
token per athlete (`tok`, 1,024 values).
- **Development (training):** the 12 grappling2 development sequences, all cameras: 22,640 views.
- **Held-out test:** sequences 031 / 037 / 042 / 043, cameras 01/06/11/16: 1,068 views (the E06 test views).
- **New people:** MMA sequence 016 (1,484 views with tokens; a different pair, never trained on).

## Method (`scripts/engage_limb_head.py`, settings fixed here)
- Per athlete, the head sees its own 19 points (centred on its centroid), the other athlete's points in the same frame
  and, for `token`, its pose token reduced to 64 principal components fitted on the training rows only.
- It outputs a correction per point. Target: the truth rigidly aligned onto the prediction (Kabsch over the 19 points),
  so the head learns the pose, not the placement.
- MLP, 2 hidden layers of 512, GELU, dropout 0.2, AdamW (lr 1e-3, weight decay 1e-3, cosine), 60 epochs, batch 512;
  the last layer starts at zero (an untrained head changes nothing); 3 seeds averaged.
- Variants:
  - `none`: no correction;
  - `ponly`: points only. This also learns any fixed difference between how the two body models define points (hands,
    toes), so it is the fair baseline for the token;
  - `token`: points + token;
  - `shuffled`: points + the token of a random other view (control).
- **Measure:** align each corrected body rigidly onto its truth (rotation and translation, no scaling) and take each
  point's error. **Tangle points:** limb points (elbows, wrists, hands, knees, ankles, toes) whose true position is
  within 20 cm of a bone of the other athlete's true skeleton.
- Uncertainty: 1,000 bootstrap resamples of views for the ratio of medians.
- Development 4-fold cross-validation by sequence is run and reported; it selects nothing.

## Criteria (held-out test, median error of tangle limb points)
| | Comparison | Passes if |
|---|---|---|
| L1 | `token` vs `ponly` (the token carries limb information) | `token` >= 10% lower, and the 95% bootstrap interval of the change is below 0 |
| L2 | `token` vs `none` (a real correction) | `token` >= 15% lower |
| L3 | other limb points, `token` vs `none` (no harm elsewhere) | `token` at most 2% higher |
| L4 | `shuffled` vs `ponly` (control) | within ±3% |
| N1 | new people (MMA), `token` vs `ponly` | `token` >= 5% lower (secondary) |

**Reading.**
- L1-L4 pass: the frozen token already holds much of the limb correction; a light head is a cheap fix and joins the
  pipeline. E11 is then measured against this head.
- L2 passes, L1 fails: the gain is the fixed point-definition difference, not information in the token. Limbs in tangles
  have to be fixed where the image is read: E11 (decoder fine-tune) is needed.
- L1 fails and L2 fails: neither the points nor the token can correct the limbs; E11 is needed, and its target should
  be the decoder's attention to the contested pixels.
- L4 fails: the setup leaks (a shuffled token should not help); the run is void and the bug is reported.
- N1 says whether any gain transfers to new people; it does not change the decision above.

Compute: CPU only (`TORCH_THREADS=8`, `MemoryMax=8G`), under an hour; no GPU.
