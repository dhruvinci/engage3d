# E18 — Do true 2D keypoints fix tangled limbs in SAM 3D Body?

**Objective.** The ceiling of the input route: if SAM 3D Body is given the true 2D joints as clicks, how much of the
tangled-limb error goes away, and how much of that does today's 2D model (ViTPose+) already deliver? It decides
between a better 2D keypoint model (RF-DETR, E16) and a bigger fine-tune.

**Setup.** E11's test views; the model's own click loop with ViTPose+ keypoints or the true joints projected through
the true fisheye camera. Criteria fixed in advance: [`PREREG.md`](PREREG.md).

**Status.** Pre-registered 2026-10-26; results follow in `RESULTS.md`.
