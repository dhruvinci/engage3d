# E00 — Benchmark and single-camera baseline

**Objective.** Define two-person engagement as 3D measurements, and see how well a single-camera 3D body model
(SAM 3D Body) reproduces them.

**Setup.**
- Measures: contact between 16 body regions; distance between the bodies; **loop threading** (a limb or torso
  passing through the other person's arm or leg loop).
- Truth: Harmony4D multi-view 3D (5,596 views, standing grappling) and ViCoS expert BJJ position labels (1,426
  frames).

![Loops (yellow) and the chains threading them (magenta), from ground truth](../../docs/assets/truth_threading_examples.jpg)

**Results** (Harmony4D, one camera)
| | F1 [95% CI] |
|---|---|
| in contact | 0.85-0.92 |
| body lock (torso through arm loop) | **0.80** [0.64, 0.86] |
| leg through leg loop | 0.46 [0.22, 0.62] |
| hand fighting (arm through arm loop) | 0.22 [0.14, 0.31] |
| which surfaces touch | 0.42-0.47 |
- Distance between the athletes: 10.2 cm off.
- ViCoS: 2 of 6 hypotheses pass; the same athlete is rebuilt twice in 24-30% of ground positions.

**Verdict.** Baseline. Contact and body locks are measurable from one camera; fine relations are not.
**Learned.** The limit is relative placement between two interlocked bodies.
**Next.** Find which error breaks the relations ([E01](../E01-joint-rigid-fit/)).

Files: `RESULTS-harmony4d.md`, `PREREG-vicos.md`, `RESULTS-vicos.md`.
