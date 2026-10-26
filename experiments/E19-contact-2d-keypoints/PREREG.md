# Pre-registration: E19, a 2D keypoint model trained on contact, feeding SAM 3D Body's clicks — written 2026-10-27, before any training or test run

## Why
E18's held-out half (1,068 views): with the true 2D joints as clicks, SAM 3D Body's tangled-limb error falls 15%
(5.39 -> 4.60 cm); with today's 2D model (ViTPose+) as clicks it rises 25% (5.39 -> 6.72 cm), because inside a tangle
ViTPose takes the partner's limbs and the clicks pull the body the wrong way. A 2D model that is right inside tangles
could turn that harm into most of the true-click gain. Harmony4D gives exact 2D joints in contact, including hidden
ones, to train on.

## Data
- **Training:** 8,000 views (16,000 athlete crops) from the same 10 development grappling2 sequences as E11
  (contact-weighted draw, fixed seed). Targets: the 12 true body joints projected through the true fisheye camera.
- **Validation (monitoring only):** 500 views from 025 / 035.
- **Test:** the E18 held-out views (031 / 037 / 042 / 043, cameras 01/06/11/16; 1,068 views). MMA only if time remains,
  reported without a criterion.

## Method (`scripts/engage_vitpose_finetune.py`; settings fixed here)
- ViTPose+ base (`usyd-community/vitpose-plus-base`, COCO head), ViTPose's own top-down crop (256 x 192); Gaussian
  heatmap targets (sigma 2, 64 x 48) for the 12 body joints; limb joints inside a tangle weighted 3x; face points
  untrained.
- AdamW (lr 5e-5, weight decay 0.01), one-cycle schedule, 3 epochs, batch 24, bfloat16; nothing selected on validation.
- Test inference through the pipeline's own `estimate()` path for both the released and the fine-tuned model.
- 3D: `engage_h4d_clicks.py --pose-file` with the fine-tuned keypoints, 2 clicks (`vitposeft2`), measured exactly as
  E18 (8 limb joints, rigid alignment over 14 points). Compared with E18's `original`, `vitpose2` and `oracle2` on the
  same views.

## Criteria (held-out views)
| | Comparison | Passes if |
|---|---|---|
| V1 | 2D: median pixel error of the 8 limb joints inside tangles, fine-tuned vs released ViTPose | >= 30% lower |
| V2 | 3D: tangled-limb error with fine-tuned clicks (`vitposeft2`) | below `original` (5.39 cm) and at least half of `oracle2`'s gain (<= 5.00 cm) |
| V3 | 3D: other limb joints with `vitposeft2` vs `original` | at most 2% higher |

**Reading.**
- **V1-V3 pass:** a contact-trained 2D model makes clicks useful inside tangles. It becomes a pipeline stage, and E16
  (a modern keypoint model, RF-DETR, trained the same way) has a baseline to beat.
- **V1 passes, V2 fails:** better 2D alone is not enough at 2 clicks; the click budget or depth ambiguity limits it.
- **V1 fails:** ViTPose+ base cannot learn contact from one pair; a stronger model or more people is needed.

Compute: CPU about 25 minutes (crop cache), GPU about 20 minutes (training) and about 70 minutes (2D and 3D
evaluation); after E20, alone on the GPU; disk about 2.5 GB (crop cache, deleted after).
