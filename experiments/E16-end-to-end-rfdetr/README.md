# E16 — End to end with no ground-truth boxes: RF-DETR keypoints and person masks trained on contact

**Objective.** Every result so far started from ground-truth person boxes, and limbs hidden at contact remain the
bottleneck. How much does real detection cost, how far can a person mask (a prompt SAM 3D Body accepts) take hidden
limbs, and do keypoints from a detector trained on contact finally make clicks pay?

**Setup.** RF-DETR's keypoint and segmentation models fine-tuned on Harmony4D pair crops (E19's training views). A
stock detector finds the pair on the full frame; SAM 3D Body runs from true vs detected boxes, with and without
masks, with and without clicks. Held-out grappling (criteria) and MMA (reported). Criteria fixed in advance:
[`PREREG.md`](PREREG.md).

**Stage 0 (exploratory, development data only, run before the training):** with perfect inputs, a true mask helps
hidden limbs by only 3%, true 2D clicks by 15%, and 2D plus a soft fit with a calibrated skeleton by 46%, provided the
2D error inside tangles is 10 px or less. That target is why this experiment trains a keypoint model on contact:
[`STAGE0.md`](STAGE0.md), data in `data/stage0_*.json`.

**Results** (held-out grappling, tangled limb joints, paired; [`RESULTS.md`](RESULTS.md))
| | Comparison | Result | |
|---|---|---|---|
| R1 | athletes found end to end | 92.7% (bar 95%) | **fail** |
| R2 | 2D inside tangles: RF-DETR (detected) vs ViTPose+ (true boxes) | 25.2 vs 19.2 px | **fail** |
| R3 | true mask vs none (the mask ceiling) | -0.4% | **fail** |
| R4 | detected box + mask vs true box (no ground truth at all) | +1.8% (interval spans 0) | **fail** |
| R5 | detected box vs true box (cost of detection) | +0.6% (bar: at most +5%) | **pass** |
| R6 | RF-DETR keypoints as clicks | +24% worse | **fail** |

New people (MMA, no pass marks): masks -6%, detected boxes -4%, clicks +18%; 77% of athletes found.

**Verdict.** R5 passes; R1-R4 and R6 fail.
**Learned.** Real detection costs nothing where it finds the athletes: the ground-truth-box results carry over. But it
finds 93% of them (77% on new people). A person mask does not tell SAM 3D Body where a hidden limb is, and RF-DETR's
contact-trained keypoints (25 px inside tangles) are far from the 10 px that stage 0 showed 2D must reach to help.
Hidden limbs need decoder training at scale or a second view.
**Disclosure.** A first launch on 2026-10-27 was stopped 14 minutes into training, before any evaluation (log kept).
