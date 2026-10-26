# E04 — The combined depth on fresh views (pre-registered)

**Objective.** E03's combination was found after seeing the data. Test it with everything fixed in advance, on views
never used.

**Setup.** 696 grappling + 188 MMA views, none shared with E03. The weight is fixed at 0.70. Six criteria.

![A held-out moment from the camera: SAM 3D Body looks right...](../../docs/assets/view_a_overlay.png)
![...but from above it puts one athlete 55 cm too far; the combined depth fixes it](../../docs/assets/view_a_topdown.png)

**Results**
| | SAM 3D Body | combined depth |
|---|---|---|
| distance error | 10.0 cm | **3.9** |
| relative depth | 14.7 cm | **6.0** |
| region pairs | 0.45 | **0.52** |
| legs | 0.42 | **0.56** |
| body lock | 0.80 | 0.78 |

**Verdict.** **Pass (6/6).** The first single-camera fix to pass; close to two cameras (3.3 cm).
**Next.** BJJ ground work ([E05](../E05-vicos-combined-depth/)); learn the rest ([E06](../E06-placement-model-v1/)).

Files: `PREREG.md`; results in `../E03-depth-map/RESULTS.md`.
