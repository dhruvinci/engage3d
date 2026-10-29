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

**Status.** Pre-registered 2026-10-27; running; results follow in `RESULTS.md`.
