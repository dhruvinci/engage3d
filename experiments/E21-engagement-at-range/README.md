# E21 — Engagement at range: does single-camera measurement work for sword fighting?

**Objective.** Much of human engagement happens at range: facing off, managing distance. Does the pipeline measure the
distance between two people, who is nearer, and whether they touch, in Harmony4D's sword fighting?

**Setup.** Harmony4D `05_sword2` test sequences (1,120 views; new people, never trained on), the unchanged pipeline.
Criteria fixed in advance.

**Results** (medians; the fencers stand 2.36 m apart)
| | distance error | as a share of the distance | relative depth |
|---|---|---|---|
| SAM 3D Body | 11.5 cm | 5.2% | 15.9 cm |
| + combined depth | 8.5 cm | 3.7% | **8.4 cm** |
| + learned placement | **8.4 cm** | **3.7%** | 10.7 cm |

**Verdict.** **Fail** on S1-S3 (absolute bars set for close contact; "touching" is ill-defined here, perfect bodies
score 0.50); S4 passes.
**Learned.** At range the distance is right to about 4%, the same relative accuracy as in close contact; absolute bars
were the wrong yardstick. The placement learned on grappling pulls far-apart people together and hurts depth: at range
use the combined depth alone.
**Next.** Relative error as the measure for engagement at range; a placement model trained with non-contact data.

Files: `PREREG.md` (committed before the run), `RESULTS.md`.
