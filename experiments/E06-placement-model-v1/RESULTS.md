# Learned two-person placement model — results (pre-registration: `placemodel_prereg.md`) — 2026-10-24

Model: `scripts/engage_place_train.py` (MLP, ensemble of 3, corrects each body's centre and rotation after the
combined depth). Features: `scripts/engage_place_features.py`. Inputs: canonical points, per-region map-vs-body depth
disagreement, ViTPose 2D residuals and confidences. Trained on CPU in a few minutes. Depth maps: MoGe-2 on every
benchmark view (`benchmarks/engage_h4d/depth_maps_all/`, 7,080 views, 87 min on the RTX 3060).

## Cross-validation (development: 12 grappling sequences, 4 folds by sequence, 4,528 out-of-fold views)
`benchmarks/engage_h4d/place_cv/`
| | distance | rel depth | rel lateral | region pairs | in contact | body lock | legs | arms | balance (as scored) |
|---|---|---|---|---|---|---|---|---|---|
| combined depth | 3.81 cm | 5.76 | 5.00 | 0.53 | 0.96 | 0.79 | 0.57 | 0.23 | 56% |
| + model | **3.12** | **4.60** | **3.61** | **0.57** | 0.97 | **0.82** | 0.56 | **0.27** | **61%** |

## Held-out test (grappling 031 / 037 / 042 / 043, 1,068 views, never used in development): PASSES
`benchmarks/engage_h4d/place_test/`. Final model trained once on every non-test view.
| | distance | rel depth | rel lateral | region pairs | in contact | body lock | legs | arms | balance (as scored) |
|---|---|---|---|---|---|---|---|---|---|
| SAM 3D Body (start of the day) | 11.22 cm | 15.99 | 4.93 | 0.46 | 0.92 | 0.71 | 0.26 | 0.41 | 41% |
| combined depth | 3.95 | 5.82 | 5.30 | 0.55 | 0.97 | 0.59 | 0.11 | 0.42 | 49% |
| + model | **3.17** | **4.30** | **2.93** | **0.60** | **0.98** | **0.76** | **0.40** | **0.55** | **56%** |

T1 3.17 <= 3.55 cm **pass**; T2 4.30 <= 5.22 cm **pass**; T3 0.60 >= 0.57 **pass**; T4 0.76 >= 0.59 **pass**; T5 0.40 >=
0.08 **pass**. Original targets: P1 (<= 6 cm) pass, P2 (>= 0.57) pass, P4 body lock (>= 0.75) pass, P5 legs (>= 0.56)
fail.

**Caveat on T4 / T5.** The test sequences hold few true cases. Only 8 views have a confident true leg through a leg
loop, and 68 a body lock, all in 037. Their F1 values are noisy, and the large swings between rows (legs 0.26 / 0.11 /
0.40) are mostly that noise. For those two families the 12-sequence cross-validation is the better estimate: body lock
+0.03, legs -0.01. Distance, depth, region pairs, contact, arms and balance rest on all 1,068 views.

**Decision (per the rule):** the learned model becomes the single-camera placement step: SAM 3D Body -> combined
depth -> learned correction. From one camera on held-out sequences, the distance between the two athletes is off by a
median 3.2 cm (from 11.2), relative depth by 4.3 cm (from 16.0), and the touching surfaces score 0.60. Two cameras with
simple averaging gave 3.3 cm and 0.58 on the full benchmark.

## Caveats
- Ground-truth boxes; standing grappling only (Harmony4D has no ground work).
- 12 training sequences, all from one capture setting.
- Every 8th view of the test sequences was in earlier depth samples, and the 0.7 weight was fitted on one of them.
- The model has never seen lying bodies, so applying it to ViCoS needs its own pre-registered check.
