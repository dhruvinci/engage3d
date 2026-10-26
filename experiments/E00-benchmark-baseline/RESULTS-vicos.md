# ViCoS result: predicted 3D relations vs expert BJJ position labels — 2026-10-22

Pre-registration: `vicos_prereg.md` (written before the run). Runner `scripts/engage_vicos_s3d.py`, scorer
`scripts/engage_vicos_score.py`, outputs `benchmarks/engage_vicos/s3d/` (1,426 frames, 69-80 per class, both athletes'
2D poses present; SAM 3D Body per athlete from a box around its 2D keypoints, default focal length).

## Pre-registered hypotheses: 2 of 6 pass
| | Hypothesis | Result |
|---|---|---|
| FAIL | H1 mount: O.torso through D.leg_loop >= 0.60; side control <= 0.20 | mount 0.22 [0.19, 0.23]; side control 0.03 |
| FAIL | H2 closed guard: O.torso through D.leg_loop >= 0.60 | 0.23 [0.18, 0.32] |
| FAIL | H3 half guard: O.leg through D.leg_loop >= 0.40, above closed guard | 0.23 [0.19, 0.31] vs 0.11 |
| FAIL | H4 50/50 has the most mutual leg threading | half guard 0.11 > 50/50 0.06 |
| PASS | H5 standing / takedown / turtle / side control: any leg-loop threading <= 0.20 | 0.00 / 0.20 / 0.14 / 0.10 |
| PASS | H6 back: a torso through the other's leg loop >= 0.40 | 0.49 [0.48, 0.51] |

The direction is right everywhere (mount 0.22 vs side control 0.03; closed guard 0.23 vs open guard 0.00; back 0.49;
standing 0.00), but the defining relation is found in only about a quarter of mount and guard frames.

## Diagnosis (post hoc, exploratory)
1. **Identity duplication.** In 24-30% of mount, closed guard, half guard and side control frames the two predicted
   bodies are the same body (median corresponding-point distance < 15 cm; standing 1%, back 6%, open guard 2%). The
   bottom athlete's box contains most of the top athlete, so SAM 3D Body reconstructs the more visible athlete twice:
   movement-research's "separation, not detection" failure, now in 3D. Leaving duplicates out raises the rates only
   modestly: mount 0.22 -> 0.28, closed guard 0.23 -> 0.30, half guard (leg) 0.23 -> 0.31.
2. **Definition.** Threading the other athlete's whole body axis (ankle - knee - hip - pelvis - neck - head) instead of
   pelvis - neck: closed guard 0.30 -> 0.43, mount 0.27 -> 0.34, standing and side control stay <= 0.05. Counting
   crossings of any margin adds ~0.08. Neither reaches the expected rates.
3. What remains is monocular 3D error on interlocked ground bodies (and possibly definition), which cannot be separated
   without 3D truth for ground work. Harmony4D has none (all grappling sequences are standing).

## ViCoS is synchronized multi-view (discovered here)
The 16 streams are 6 sessions filmed by 2-3 phones: {00,01,02}, {03,04,05}, {06,07,08}, {09,10}, {11,12,13}, {14,15}.
At the same frame number the position labels of a session's streams agree 98-100% (11 vs 12/13: 84-85%, likely an
offset). Images are per-frame crops around the athletes with no crop offsets recorded, so triangulation would first
need each crop registered to the static gym background. A cross-camera consistency check of predicted relations needs
no offsets.

## Caveats
Coverage: only frames where the 2D detector found both athletes (the hardest entanglements are missing). Intrinsics
unknown (default focal, principal point at the crop centre). Intervals from a bootstrap over streams; streams of one
session share moments, so they are optimistic.
