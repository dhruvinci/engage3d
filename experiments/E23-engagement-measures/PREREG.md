# E23 pre-registration: engagement beyond contact from one camera — written 2026-10-30, before any prediction is scored

**Question.** Can the measures of `MEASUREMENT_SPEC.v1.md` be measured from one camera, and where does each work or
fail? The measures are centre of mass, base and stability, weight on the partner, leverage, level and posture, head
position, inside position, and engagement over time.

**Definitions.** As in the spec, signed off by Dhruva on 2026-10-30:
- de Leva's (1996) segment masses, the mean of the female and male tables;
- supported by the partner = CoM more than 5 cm outside one's own base while touching;
- lever-arm length only;
- head inside = within 15 cm of the partner's centreline and in front of the chest;
- no "frames" measure yet;
- support = vertices within 3 cm of the body's own lowest vertex, if that vertex is within 6 cm of the floor.

Code: `scripts/engage_measures.py` (`measures`, then `score`).

**What was seen before writing this** (truth only; no prediction has been scored against truth):
- the measures on the true bodies of all 12 clips (per-clip values: `measures_signoff_notes.md`);
- a check of the segment-model CoM against the true mesh's volume centre (3.1-4.4 cm apart);
- the floor fit (the true bodies sit up to +-3 cm off it on Harmony4D). This changed the support rule, before sign-off,
  from "within 3 cm of the floor" to "relative to the body's lowest point".
- With that rule, standing people are stable in 70-94% of frames where both feet bear weight and mostly unstable on
  one foot, as in stepping.

**Data.**
- The 12 E22 clips with truth, end to end with no ground-truth boxes:
  - Harmony4D: two hugging pairs, sword, ballroom, karate, two MMA, grappling;
  - MAMMA: four West Coast Swing clips.
- Predicted bodies: E22's bodies after the combined depth, as already computed (nothing is re-run).
- Truth: Harmony4D SMPL fits and MAMMA SMPL-X fits, in the same camera.
- Every measure is computed on truth and prediction with the same code; only the inputs differ. The one exception is
  E22's contact rule (hands and forearms at 10 cm on predictions only).
- The floor is fitted separately on each.
- Person correspondence: per clip, the assignment with the smaller mean CoM distance.
- The draft's secondary set (E06's 1,068 held-out views) is dropped. It holds 19 points per body and no vertices, so
  base, support, leverage and contact cannot be computed there.

**Uncertainty.** 95% intervals by bootstrap over 1 s blocks within each clip (2,000 resamples), pooled.

**Criteria** (single camera against truth; pooled over the 8 Harmony4D clips unless stated):
| # | Measure | Pass if |
|---|---|---|
| C1 | CoM position (3D) | median error <= 10 cm, and the A-B CoM vector <= 8 cm |
| C2 | Level difference (CoM heights) | median error <= 6 cm |
| C3 | Stable or not (inside own base) | frame agreement >= 80% |
| C4 | Supported by the partner (proxy) | F1 >= 0.6 on clips where it occurs in truth on >= 10% of frames |
| C5 | Hips above | agreement >= 80% on frames where the true hip heights differ by >= 10 cm |
| C6 | Head inside | agreement >= 75% on frames in contact |
| C7 | Inside wrists / knees | agreement >= 70% on frames in contact |
| C8 | Lever arm at contact | median error <= 6 cm on frames where truth and prediction share the touched limb |
| C9 | Closing speed | median error <= 0.3 m/s over contact episodes (Harmony4D) |

Every measure is also reported per activity (grappling, MMA, karate, hugging, ballroom, range) with its interval. MAMMA
is reported separately, without pass marks: a different rig, with a depth error of 16-33 cm in E22.

**Decision rule.** A measure that passes is "measurable from one camera" for the activities where it passes. A measure
that fails is reported as such, with where it fails. No threshold is changed after the run.

**Known risks.**
- CoM inherits the pelvis placement error (E22: 6-20 cm).
- Weight on the partner is a geometric proxy, not a force.
- Leverage has no direction in v1.
- Few clips (12), each short (10 s).
- SAM 3D Body was trained on Harmony4D, so absolute errors are likely optimistic.
