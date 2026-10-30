# E23 — Engagement beyond contact: can one camera measure balance, level, leverage and position?

**Objective.** Contact, threading and distance were measured before. Combat sports also care about the centre of
gravity, base and balance, weight on the partner, leverage, level, head position and inside position. Can these be
measured from one camera?

**Setup.** Eight measures defined in [`docs/MEASUREMENT_SPEC.v1.md`](../../docs/MEASUREMENT_SPEC.v1.md) (signed off
2026-10-30), computed with the same code on the true bodies and on the single-camera pipeline's bodies (E22's 12 clips,
no ground-truth boxes). Criteria fixed in advance: [`PREREG.md`](PREREG.md).

**Results** (8 Harmony4D clips)
| | Measure | Result | |
|---|---|---|---|
| C1 | centre of mass; the vector between the two | 8.6 cm; 6.5 cm | **pass** |
| C2 | level difference | 1.7 cm | **pass** |
| C3 | stable over own base | 62.8% agree (bar 80%) | **fail** |
| C4 | supported by the partner (proxy) | F1 0.48 (bar 0.6) | **fail** |
| C5 | whose hips are higher | 100% | **pass** |
| C6 | head inside | 75.4% (bar 75%) | **pass**, narrowly |
| C7 | inside wrists and knees | 88.4% | **pass** |
| C8 | lever arm at contact | 3.5 cm | **pass** |
| C9 | closing speed | 0.15 m/s (6 episodes) | **pass** |

**Verdict.** 7 of 9 pass. Centre of mass, level, hips, head and limb inside position, lever arms and closing speed are
measurable from one camera; balance (stable, supported by the partner) is not yet, because it depends on where the
centre of mass falls relative to the feet to within a few centimetres.
**Learned.** The relational measures hold best: they depend on the two bodies relative to each other more than on
absolute depth (ballroom's centre of mass is 14.8 cm off, the vector between the dancers 5.7 cm).

Files: `PREREG.md` (committed before any prediction was scored), `RESULTS.md`, `data/score.json`. Code:
`src/engage_measures.py`.
