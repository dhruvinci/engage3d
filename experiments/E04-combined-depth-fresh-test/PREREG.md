# Pre-registration: depth map + SAM 3D Body combined depth, fresh views — written 2026-10-24, before the run

## Why
In `depthmap_results.md` the pre-registered fix (depth map alone) failed. An exploratory combination did best:
gap = 0.7 x map + 0.3 x SAM 3D Body, because the map shrinks the gap between the athletes (0.84 x) and SAM 3D Body
stretches it (1.20 x). It was chosen after seeing the data, on every 8th view starting at view 0. This test fixes
everything in advance and runs on views that run never touched.

## What runs (fixed now)
- Views: every 8th view of `benchmarks/engage_h4d/s3d_box` starting at view 4 (`--offset 4`), so none overlap the
  first sample. Grappling is the test; MMA is reported.
- Model: MoGe-2 (`Ruicheng/moge-2-vitl-normal`), true focal length, same crop and render code
  (`engage_depth_probe.py --offset 4 --save-maps`).
- Method: `pred_blend` with **weight 0.70** (fitted on the first sample's grappling views; not refitted). No masks:
  the predicted render minus a 5 px band around the other body. `engage_depth_blend.py DIR --weight 0.70`.
- Scoring: `engage_bench_oracle.py` (the `pred` row of each variant). The `pred` variant on the same views is the
  baseline.

## Criteria (grappling, `pred_blend` vs the baseline on the same views)
| | Measure | Exploratory value | Pass if |
|---|---|---|---|
| C1 | Relative pelvis depth error, median | 5.5 cm (baseline 14.3) | <= 7 cm |
| C2 | Distance between the bodies, median error | 3.9 cm | <= 6 cm |
| C3 | Region-pair F1 | 0.53 (+0.07) | >= baseline + 0.04 |
| C4 | Leg through leg loop F1 | 0.52 (+0.11) | >= baseline + 0.06 |
| C5 | Body lock F1 (no harm) | 0.78 (+0.02) | >= baseline - 0.03 |
| C6 | Visible-surface gap error, median (`engage_depth_probe_score.py`, blended as 0.7 map + 0.3 SAM 3D Body) | 6.0 cm | <= 7 cm |
The original targets (region pairs >= 0.57, legs >= 0.56) are reported but not expected to pass: they need
orientation, not only depth (oracle).

**Decision rule.** C1-C5 pass: the combined depth becomes the single-camera placement step, and the next work is
orientation from the depth map. Any of C1, C2 or C4 fails: the exploratory gain was partly chance or specific to the
first sample. Report it as such, and keep SAM 3D Body placement as the baseline.

## Caveats
Ground-truth boxes; Harmony4D standing grappling only; about 900 views. The weight comes from the same benchmark
(other views of the same sequences), so a test on another dataset (ViCoS ground work, no 3D truth) is still needed.
