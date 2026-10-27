# E16 — End to end with no ground-truth boxes: RF-DETR keypoints and person masks trained on contact

**Objective.** Every result so far started from ground-truth person boxes, and limbs hidden at contact remain the
bottleneck. How much does real detection cost, how far can a person mask (a prompt SAM 3D Body accepts) take hidden
limbs, and do keypoints from a detector trained on contact finally make clicks pay?

**Setup.** RF-DETR's keypoint and segmentation models fine-tuned on Harmony4D pair crops (E19's training views). A
stock detector finds the pair on the full frame; SAM 3D Body runs from true vs detected boxes, with and without
masks, with and without clicks. Held-out grappling (criteria) and MMA (reported). Criteria fixed in advance:
[`PREREG.md`](PREREG.md).

**Status.** Pre-registered 2026-10-27; results follow in `RESULTS.md`.
