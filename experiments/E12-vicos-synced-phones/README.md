# E12 — Ground work without 3D truth: do synchronised phones agree?

**Objective.** No 3D truth exists for ground work. ViCoS was filmed by 2-3 phones at once, so agreement between phones
at the same moment is a reliability measure that needs none. Also: does the learned placement carry over to ground
work, and can MapAnything turn the phones into usable 3D?

**Setup.** 492 synchronised moments (1,223 images) over all 10 positions. Pipelines: SAM 3D Body with clicks, plus the
combined depth, plus the learned placement. Criteria fixed in advance: [`PREREG.md`](PREREG.md).

**Status.** Pre-registered 2026-10-26; results follow in `RESULTS.md`.
