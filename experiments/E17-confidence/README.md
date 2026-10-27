# E17 — Can the pipeline flag the frames where its measurement is wrong?

**Objective.** Ship every measurement with a trust flag: do truth-free signals (the ensemble's disagreement, how close
and overlapped the two bodies are, 2D confidence, the depth map's evidence) predict the frames with large errors?

**Setup.** A logistic flag per error type, trained on out-of-fold predictions of the development sequences; tested on
held-out grappling and MMA views never scored before, plus the other activities. Criteria fixed in advance, with a
disclosure: [`PREREG.md`](PREREG.md).

**Status.** Pre-registered 2026-10-27; results follow in `RESULTS.md`.
