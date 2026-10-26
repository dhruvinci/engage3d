# E11 pilot results: fine-tuning SAM 3D Body's decoder helps inside tangles, but only modestly (pre-registration: `s3dft_prereg.md`)

Run 2026-10-26 16:24-20:17 on the laptop GPU, after the pre-registration was pushed (engage3d `517a882`, 16:23).
Code: `scripts/engage_s3d_finetune.py`; baseline: `scripts/engage_limb_head.py --align core14 --measure joint8`.
Outputs: `benchmarks/engage_h4d/s3d_ft/` (decoder weights, offsets, logs, per-crop errors) and
`benchmarks/engage_h4d/limb_head_e11/`.

## Run
- Cache: 11,000 crops (10,000 training from 10 development sequences, 1,000 validation from 025 / 035), 75 minutes.
- Training: 3 epochs, 3,750 steps of 8 crops, 97 minutes; peak GPU memory under 5 GB.
- Validation, tangle limb joints (median cm): 4.81 at the start, 3.91 after epoch 1, 3.81 after epoch 2, 3.87 after
  epoch 3. Almost all the gain came in the first epoch.

## Criteria (median error of tangle points among the 8 limb joints, rigid-aligned over 14 points)
| | Comparison | Result | Verdict |
|---|---|---|---|
| F1 | held-out: fine-tuned vs original + points-only head | 4.97 vs 5.07 cm: -2.0% (needed -15%); against `original` 5.40 cm: -8.0%, 95% interval -10.5% to -5.2% | **fail** |
| F2 | held-out, other limb joints: fine-tuned vs original + points-only head | 2.50 vs 2.47 cm: +1.2% (allowed +2%) | pass |
| F3 | new people (MMA): fine-tuned vs original + points-only head | 5.72 vs 6.04 cm: -5.3% (needed -5%); against `original` 6.27 cm: -8.8%, interval -10.5% to -7.0% | pass |
| F4 | pipeline check: evaluation's `original` vs errors from E10's stored points | median difference 0.14 mm (held-out), 0.16 mm (MMA) | pass |

F4 note: the 99th percentile of the difference is 1.7-2.0 mm; a few crops differ by up to 13.5 mm (held-out) and 52.6 mm
(MMA), from bfloat16 differences when crops are batched differently. The medians are unaffected.

## All numbers (median cm)
| | tangle limb joints | other limb joints | tangle points |
|---|---|---|---|
| held-out, original | 5.40 | 4.24 | 3,760 |
| held-out, original + points-only head | 5.07 | 2.47 | |
| held-out, **fine-tuned decoder** | **4.97** | 2.50 | |
| MMA, original | 6.27 | 4.62 | 4,208 |
| MMA, original + points-only head | 6.04 | 3.14 | |
| MMA, **fine-tuned decoder** | **5.72** | 3.14 | |

## Reading (as pre-registered: "F1 fails, but fine-tuned is clearly below `original`")
- **Fine-tuning the decoder does move limbs inside tangles**, and the gain holds on new people: 8-9% lower error than
  the released model on both test sets, with intervals well below zero. That is more than E10's head on the frozen token
  could do (+2% worse than the points-only head).
- **But it is small:** only 2% better than the cheap points-only head on the held-out pair (5% on new people), far from
  the 15% bar. Outside tangles, both remove the same fixed error (4.2 -> 2.5 cm), mostly the different point definitions
  of the two body models and a consistent pose bias.
- **More of the same will not fix it:** validation stopped improving after the first epoch. Per the pre-registration,
  the next step is a larger run or adapters in the backbone (cloud). Before spending on that, a cheaper test fits first
  (not pre-registered here): give SAM 3D Body the true 2D keypoints of the tangled limbs as prompts ("clicks", which
  already helped on ViCoS in E05). If true clicks fix the limbs, the practical route is a 2D keypoint model that is good
  inside tangles (RF-DETR's keypoint model, fine-tuned on Harmony4D contact; see the revised E16), not a bigger
  fine-tune of SAM 3D Body.
- The fine-tuned decoder is kept locally (`s3d_ft/decoder.pt`, 158 MB, plus the 19 learned offsets); it is not
  published (SAM 3D Body's licence terms for derived weights are unchecked). The 21.6 GB embedding cache is deleted, as
  pre-registered.
