# E19 results: a 2D model fine-tuned on contact is 21% more accurate inside tangles, but its clicks still make SAM 3D Body worse (pre-registration: `vitposeft_prereg.md`)

Run 2026-10-27 00:24-06:04 (crop cache on the CPU while E18 ran; training and evaluation on the GPU after E20's GPU
stages), after the pre-registration was pushed (engage3d `d7f0ad7`). Code: `scripts/engage_vitpose_finetune.py`,
`scripts/engage_h4d_clicks.py --pose-file`. Outputs: `benchmarks/engage_h4d/vitpose_ft/` (model, `predict.json`,
`clicks_heldout/clicks.json`; the 2.5 GB crop cache is deleted).

## Training
17,000 crops (16,000 training from 8,000 views of 10 development sequences, 1,000 validation from 025 / 035); 3 epochs,
1,998 steps, 56 minutes. Validation, limb joints inside tangles (median, pixels in the 192 x 256 crop): 6.16 at the start,
4.57 / 4.26 / 4.25 after epochs 1-3 (-31%).

## Criteria (held-out views)
| | Comparison | Result | Verdict |
|---|---|---|---|
| V1 | 2D, limb joints inside tangles: fine-tuned vs released ViTPose+ (median pixel error, 4K frames) | 19.2 vs 24.5 px: -21% (needed -30%) | **fail** |
| V2 | 3D tangled-limb error with the fine-tuned clicks | 6.14 cm vs 5.39 without clicks: +14% (interval +11% to +17%); needed below 5.39 and <= 5.00 | **fail** |
| V3 | 3D, other limb joints with the fine-tuned clicks vs without | 4.44 vs 4.24 cm: +5% (allowed +2%) | **fail** |

## All numbers
| | 2D inside tangles (px) | 2D other (px) | 3D inside tangles (cm) | 3D other (cm) |
|---|---|---|---|---|
| no clicks (E18) | | | 5.39 | 4.24 |
| released ViTPose+ (E18 `vitpose2`) | 24.5 | 14.7 | 6.72 | 4.62 |
| **fine-tuned ViTPose+** | **19.2** | **9.4** | 6.14 | 4.44 |
| true 2D (E18 `oracle2`) | 0 | 0 | 4.60 | 4.00 |
New people (MMA), 2D only: 24.3 -> 18.7 px inside tangles (-23%), 16.7 -> 10.4 px elsewhere (-38%).

## Reading (pre-registered: "V1 fails")
- **Training on one pair's contact makes the 2D model better everywhere** (-21% inside tangles, -36% elsewhere on the
  held-out pair; -23% / -38% on new people), and it **halves the harm its clicks do** (+25% -> +14%). But clicks from
  it still make tangled limbs worse than no clicks.
- Why clicks are unforgiving: SAM 3D Body clicks the joint whose 2D estimate disagrees most with the given 2D pose,
  and inside a tangle that is usually the joint where the 2D model is wrong. A 2D model must be nearly exact there
  (true 2D gives -15%) before clicks pay off; 19 px median error on 4K frames is not enough.
- Next on this route: a stronger 2D model trained on more people (E16, RF-DETR keypoints; Harmony4D's second pair,
  Hi4D / CHI3D), or using 2D as a soft, confidence-weighted constraint instead of hard clicks.
