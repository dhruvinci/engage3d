# PromptHMR as the single-camera model — results (pre-registration: `phmr_prereg.md`)

## Harmony4D — 2026-10-22 23:17
`benchmarks/engage_h4d/phmr_box/` (7,080 views, no out-of-memory skips, re-solve residual median 1-6 px per camera);
scores `phmr_box/score/` (per activity in subdirectories). Same views, boxes and lens handling as SAM 3D Body.

Grappling (pre-registered criteria):

| | Measure | SAM 3D Body | PromptHMR | pass if |
|---|---|---|---|---|
| P1 | distance between the bodies, median error | 10.2 cm | 6.1 | <= 6 |
| P2 | region-pair F1 (best of 2/5/10 cm) | 0.47 | 0.42 | >= 0.57 |
| P3 | loop balance sign, matched | 41% | 20% | >= 70% |
| P4 | body lock F1 / in-contact F1 | 0.80 / 0.85-0.92 | 0.50 / 0.85-0.91 | >= 0.75 / 0.85 |
| P5 | leg through leg loop F1 | 0.46 | 0.30 | >= 0.56 |
| | pelvis position error | 10.8 cm | 8.5 | |
| | body orientation + limbs (root-relative) | 7.5 cm | 6.7 | |
| | cross-camera agreement of threading | 0.25 | 0.10 | |

MMA (1 sequence, athletes 4.6-5.8 m from the cameras): PromptHMR is worse there — pelvis 30.5 vs 9.3 cm, distance
12.9 vs 9.1 cm; in-contact F1 at 2 cm 0.86 vs 0.76.

**Verdict: fails (P1 misses by 0.1 cm; P2-P5 fail).** Per the decision rule, the rule of `jointfit_prereg.md`
stands: capture or training is the critical path.
What it shows: PromptHMR places the two athletes better (distance error -40%, pelvis -21%) and its poses are better
overall, but it resolves wrapped limbs much worse. Body lock precision is the same (0.74 vs 0.78) while recall halves
(0.38 vs 0.81); it predicts a confident loop in 16% of grappling views against 30% in truth (SAM 3D Body 25%). A
plausible cause (not tested): it sees each athlete at about 200 px inside one 896 px whole-image encoding, where
SAM 3D Body sees each athlete in its own high-resolution crop. Its placement also degrades with distance (MMA).

## ViCoS (real BJJ ground work) — 2026-10-22 23:48
`benchmarks/engage_vicos/phmr/` (same 1,426 frames and boxes as SAM 3D Body; default intrinsics); scores
`phmr/score/`. The ViCoS scorer now reports the duplicate rate itself (`duplicate`: corresponding canonical points a
median < 15 cm apart; it reproduces the earlier SAM 3D Body figures exactly).

| Defining relation rate | SAM 3D Body | PromptHMR |
|---|---|---|
| mount: bottom torso through top leg loop (H1 >= 0.60) | 0.22 | 0.12 |
| closed guard: top torso through guard player's leg loop (H2 >= 0.60) | 0.23 | 0.14 |
| half guard: top leg through guard player's leg loop (H3 >= 0.40) | 0.23 | 0.13 |
| back: a torso through the other's leg loop (H6 >= 0.40) | 0.49 | 0.37 |
| H5 (standing / takedown / turtle / side control low) | pass | pass |
| hypotheses passed | 2 / 6 | 1 / 6 |

Both predicted bodies are the same athlete (same place and, checked separately, the same pose: root-relative joints a
median < 10 cm apart, so these are true duplicates, not two stacked bodies):

| | standing | open guard | half guard | closed guard | side control | mount | back | turtle |
|---|---|---|---|---|---|---|---|---|
| SAM 3D Body | 0.01 | 0.03 | 0.28 | 0.22 | 0.24 | 0.23 | 0.05 | 0.03 |
| PromptHMR | 0.03 | 0.03 | 0.34 | 0.44 | 0.40 | 0.43 | 0.24 | 0.02 |

**Verdict: fails; worse than SAM 3D Body on ground work.** With both athletes decoded against one whole-image
encoding, PromptHMR reconstructs the same athlete twice in 40-44% of mount, closed guard and side control frames. On
interlocked ground bodies the first failure is telling the two people apart in 3D, before placement or limbs; a
model that sees the whole scene at low resolution makes it worse, not better.

## Exploratory (not pre-registered): SAM 3D Body bodies at PromptHMR's positions — Harmony4D
`scripts/engage_bench_hybrid.py` -> `benchmarks/engage_h4d/hybrid_{s3dpose_phmrplace,centroid}/`. Grappling:

| | distance err | region pairs | in contact (2/5/10 cm) | body lock | legs | balance (matched) |
|---|---|---|---|---|---|---|
| SAM 3D Body | 10.2 cm | 0.47 | 0.85 / 0.88 / 0.92 | 0.80 | 0.46 | 41% |
| PromptHMR | 6.1 | 0.42 | 0.85 / 0.89 / 0.91 | 0.50 | 0.30 | 20% |
| SAM 3D Body at PromptHMR's pelvis | 6.1 | 0.44 | **0.92 / 0.94 / 0.95** | 0.65 | 0.39 | 34% |
| SAM 3D Body at PromptHMR's mesh centroid | 9.7 | 0.46 | 0.85 / 0.90 / 0.93 | 0.67 | 0.37 | 31% |
The second anchor checks that the first result is not a hip-joint definition artefact. Transplanting PromptHMR's
positions gives the best "are they touching" detection so far but loses loops in both forms: a better average
position does not fix the relative arrangement of two bodies, and it breaks the arrangement SAM 3D Body's two bodies
did get right.

## Diagnostic: oracle ablation on PromptHMR (pre-registered as a diagnostic) — 2026-10-22 23:52
`benchmarks/engage_h4d/phmr_box/oracle/`. Grappling, body lock / legs / region pairs:

| Corrected with truth | SAM 3D Body | PromptHMR |
|---|---|---|
| nothing | 0.80 / 0.46 / 0.46 | 0.50 / 0.30 / 0.39 |
| depth along the line of sight | 0.86 / 0.58 / 0.58 | 0.60 / 0.42 / 0.45 |
| limbs + orientation, position as predicted | 0.87 / 0.43 / 0.52 | **0.72** / 0.46 / **0.58** |
| whole-body position + orientation, model's own limbs | **0.93 / 0.74 / 0.67** | 0.83 / 0.65 / 0.55 |
| relative depth error / lateral error | 14.5 / 5.0 cm | 9.5 / 4.1 cm |

The two models fail in opposite places: PromptHMR's main loss is its limbs (correcting them helps most), SAM 3D Body's
is depth. With position and orientation corrected, SAM 3D Body's own limbs are the better pair (0.93 / 0.74 vs 0.83 /
0.65). So the best single-camera ingredients available are SAM 3D Body's limbs plus a depth / orientation source that
neither model provides.
