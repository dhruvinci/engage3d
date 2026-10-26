# Joint two-person fit — results (pre-registration: `jointfit_prereg.md`)

## ViCoS (real BJJ ground work), Arm 1 rigid joint fit — 2026-10-22 17:33
Fits: `benchmarks/engage_vicos/fit_{full,no_pen,no_2d}/` (2D targets = the dataset's own poses). Same H1-H6 thresholds.

| Relation rate (defining relation) | baseline | full | 2D only | no-overlap only |
|---|---|---|---|---|
| mount: bottom torso through top leg loop | 0.22 | 0.11 | 0.13 | 0.11 |
| closed guard: top torso through guard player's leg loop | 0.23 | 0.26 | 0.19 | 0.19 |
| half guard: top leg through guard player's leg loop | 0.23 | 0.22 | 0.26 | 0.21 |
| back: a torso through the other's leg loop | 0.49 | 0.43 | 0.38 | 0.42 |
| H5 (standing / takedown / turtle / side control, low is right) | pass | pass | fail (takedown 0.24) | pass |
| Both predicted bodies on the same athlete (mount / guards / side control) | 26% | 8% | 0% | 20% |

**Arm 1 fails on ViCoS.** Moving and rotating whole bodies does not raise the defining relations (mount falls).
The drop in "same athlete twice" is mostly cosmetic: the 2D term moves the duplicated body onto the right athlete's
location, but it keeps the other athlete's pose, which a rigid fit cannot change (the caveat in the pre-registration).
This points at Arm 2 (limb-level correction: SAM 3D Body's own click refinement with the 2D keypoints).

Process note: the first ViCoS run was killed by its 4 GB memory cap (cgroup OOM, no effect on the rest of the
machine): indexing an `np.load` NpzFile decompresses the whole array on every access. Fixed in the fit and both
scorers (`dict(np.load(...))`), which also explains the 35-minute first Harmony4D scoring.

## Harmony4D (standing grappling + MMA, multi-view 3D truth), Arm 1 rigid joint fit — 2026-10-22 18:43
Fits: `benchmarks/engage_h4d/fit_{full,no_pen,no_2d}/` (2D targets = ViTPose+ base from the ground-truth box, no
masks); scores: `<dir>/score/` (per activity in subdirectories). Baseline re-scored with MMA added:
`s3d_box/score/` (the grappling-only numbers are unchanged; the old grappling-only score is in
`s3d_box_score_grappling_only/`). 4 cameras, every 4th frame: grappling 5,596 views / 16 sequences, MMA 1,484 views /
1 sequence.

Grappling (the pre-registered baselines):

| | Measure | baseline | full fit | 2D only | no-overlap only | pass if |
|---|---|---|---|---|---|---|
| P1 | distance between the bodies, median error | 10.2 cm | 10.1 | 10.1 | 10.4 | <= 6 |
| P2 | region-pair F1 (best of 2/5/10 cm) | 0.47 | 0.43 | 0.40 | 0.45 | >= 0.57 |
| P3 | loop balance sign (truth not tied) | 46% | 43% | 45% | 42% | >= 70% |
| P4 | body lock F1 / in-contact F1 (2/5/10 cm) | 0.80 / 0.85-0.92 | 0.74 / 0.87-0.93 | 0.63 / 0.87-0.93 | 0.76 / 0.84-0.92 | >= 0.75 / 0.85 |
| P5 | leg through leg loop F1 | 0.46 | 0.36 | 0.25 | 0.39 | >= 0.56 |
| | body orientation + limbs (root-relative error) | 7.5 cm | 6.7 | 6.6 | 7.7 | |
| | cross-camera agreement of threading | 0.25 | 0.23 | 0.17 | 0.25 | |

MMA (1 sequence, few loops; no truth body locks): distance error 9.1 -> 8.0 cm, region pairs 0.30 -> 0.26, balance
sign 20% -> 22% (120 views), in-contact F1 at 2 cm 0.76 -> 0.82. Pooled: distance 10.0 -> 9.6 cm, region pairs
0.45 -> 0.42, balance 44% -> 42%, body lock 0.77 -> 0.71, legs 0.43 -> 0.33.

**Decision rule: P1 and P2 fail -> the cheap fix does not work; capture or training is the critical path.** P3 and P5
fail too, and P4 fails on body locks (0.74 < 0.75).
The fit did move the bodies (median 4.9 cm, 90th percentile 13.5 cm; cost down to a third), so the prior did not
pin them. The informative part is the split: the 2D term makes each body's own orientation better (7.5 -> 6.7 cm)
yet makes the relations worse (2D only: body lock 0.80 -> 0.63, legs 0.46 -> 0.25, cross-camera agreement 0.25 ->
0.17). Matching one image better does not place the two bodies better relative to each other, and in contact the
2D model (no masks) can pull a body toward the partner's limbs. The no-overlap term alone is roughly neutral: it moves
bodies only 0-3 cm.

## Which error breaks the relations: oracle ablation on the baseline — 2026-10-22 19:05
`scripts/engage_bench_oracle.py` -> `benchmarks/engage_h4d/s3d_box_oracle/`. Each predicted body is corrected with
the truth in one respect at a time and re-scored with the benchmark's own definitions (not pre-registered: a
diagnostic, run after Arm 1 failed). Grappling, 5,596 views:

| Corrected with truth | distance err | relative depth err | relative lateral err | region pairs | body lock | legs | arms | balance (as scored / matched) |
|---|---|---|---|---|---|---|---|---|
| nothing (baseline) | 10.2 cm | 14.5 cm | 5.0 cm | 0.46 | 0.80 | 0.46 | 0.22 | 46% / 41% |
| depth along each camera ray | 1.5 | 0.2 | 4.8 | **0.58** | 0.86 | **0.58** | 0.30 | 56% / 53% |
| lateral (across the ray) | 9.6 | 14.3 | 1.3 | 0.44 | 0.81 | 0.43 | 0.20 | 42% / 37% |
| pelvis position (both) | 0 | 0 | 0 | 0.56 | 0.87 | 0.54 | 0.24 | 51% / 48% |
| whole-body position + orientation, limbs as predicted | 2.1 | 2.3 | 3.0 | **0.67** | **0.93** | **0.74** | 0.42 | 65% / 64% |
| limbs and orientation, position as predicted | 10.2 | 14.5 | 5.0 | 0.52 | 0.87 | 0.43 | 0.26 | 50% / 43% |
| truth (ceiling of each measure) | 0 | 0 | 0 | 0.85 | 1.00 | 1.00 | 1.00 | 81% / 100% |

MMA (1 sequence) shows the same order: depth fix 9.1 -> 1.9 cm, region pairs 0.27 -> 0.44; rigid fix region pairs
0.63.

What it says:
1. **The error that breaks the relations is where each body sits along the camera's line of sight.** The two
   athletes' relative depth is off by 14.5 cm (median), three times the side-to-side error (5.0 cm). Fixing depth
   alone passes P1, P2 and P5; fixing side-to-side does nothing.
2. **The predicted limbs are good enough for most measures on standing grappling.** With each body placed and turned
   correctly as a whole (no limb changes), body lock 0.93, legs 0.74, region pairs 0.67, balance 64% (matched).
   Correct limbs at the predicted position help little (legs 0.43). So a rigid fit *could* pass P1, P2, P4, P5; Arm 1
   failed because its evidence (one 2D image) cannot see depth, not because rigid moves are too weak.
3. Arm-through-arm-loop stays hard even with the rigid fix (0.42): arms are where limb errors matter.
4. This differs from ViCoS ground work, where a quarter of frames put both predicted bodies on one athlete; that is a
   whole-person error, which these numbers do not cover.

**Measurement flaw found (mine):** loop balance as scored counts every predicted crossing but only confident
(>= 5 cm) true ones, so perfect bodies score 81%, not 100%. Counted the same way on both sides ("matched"), the
baseline is 41% and the truth 100%. The Arm 1 verdict does not change; later pre-registrations use the matched form.

### Where the depth error comes from (quick checks, 6 grappling sequences, ad hoc scripts not kept)
- **Not a floor-fixable error from these cameras:** they sit at 1.2-1.5 m, 3.4-5.8 m from the athletes, so the line
  of sight to the pelvis is 3-8 degrees below horizontal. A 14 cm slide in depth moves the feet 1-2 cm up or down,
  less than the model's own foot-position noise. (Better conditioned for ground work filmed from higher up; still
  weak at 1.5 m.)
- **Not body size:** the model makes every athlete about 5.7% larger (interquartile 4.8-6.7%, i.e. very steady),
  but the depth error is unrelated to it (correlation 0.01). Rescaling each body to its true size: relative depth
  error 19.5 -> 16.1 cm on the checked subset.
- **Not frame-to-frame jitter:** the depth error drifts slowly (correlation 0.84 between samples 0.13 s apart), so a
  temporal median over 1.2 s only takes the relative error 16.6 -> 15.8 cm. The two athletes' errors are slightly
  opposed (correlation -0.21), which widens the gap between them.
So a single camera needs a learned depth cue (a depth model, or a model trained on two people), or a second viewpoint.

## What a second camera buys: late fusion of per-camera predictions — 2026-10-22 20:08
`scripts/engage_bench_multiview.py` -> `benchmarks/engage_h4d/s3d_box_multiview/`. No ground truth used in the
correction: each athlete's pelvis is the weighted fusion of the cameras' estimates (each trusted 5 cm across its line
of sight, 15 cm along it; these two numbers were measured on this same data), optionally with the body turned to the
cameras' mean whole-body orientation; scored in every camera of every camera subset. Balance in the matched form.
Grappling:

| Cameras | distance err | region pairs | in contact | body lock | legs | arms | balance (matched) |
|---|---|---|---|---|---|---|---|
| 1 (baseline) | 10.2 cm | 0.46 | 0.88 | 0.80 | 0.46 | 0.22 | 41% |
| 2, lines of sight 60-120 deg apart | 3.3 | 0.58 | 0.96 | 0.88 | 0.51 | 0.31 | 53% |
| 2, < 60 deg apart | 5.9 | 0.55 | 0.94 | 0.89 | 0.56 | 0.28 | 50% |
| 3 | 3.5 | 0.58 | 0.96 | 0.88 | 0.51 | 0.31 | 54% |
| 4 | 3.2 | 0.59 | 0.97 | 0.88 | 0.54 | 0.32 | 53% |
| oracle: true pelvis position (for comparison) | 0 | 0.56 | 0.95 | 0.87 | 0.54 | 0.24 | 48% |
| oracle: true whole-body position + orientation | 2.1 | 0.67 | 0.98 | 0.93 | 0.74 | 0.42 | 64% |
(rows 2-4 with orientation fusion; position-only fusion is 0.01-0.04 lower on region pairs and loops. Pairs more than
120 deg apart are few in grappling (1,134 views, clustered in particular moments) and score oddly low; not
interpreted.)

What it says:
1. **A second camera at a real angle fixes placement and contact:** distance error 10.2 -> 3.3 cm (P1 passes),
   region pairs 0.46 -> 0.58 (P2 passes), body lock 0.80 -> 0.88. Extra cameras beyond two add little in this form.
2. **It does not fix the loops:** legs 0.46 -> 0.51, balance 41% -> 53%. Late fusion reaches the "true pelvis"
   oracle, but loops need each body's whole-body orientation right too (oracle: legs 0.74), and averaging the
   cameras' orientation guesses does not get there.
3. For the capture project: two cameras roughly at right angles (or side + overhead) is the minimum geometry, and the
   multi-camera pipeline has to reconstruct jointly from all views (e.g. triangulated 2D keypoints), not average each
   camera's 3D guess. Tested next.

## ViCoS, Arm 2: SAM 3D Body's own keypoint clicks, then Arm 1 — 2026-10-23 00:14
`scripts/engage_vicos_s3d_clicks.py` -> `benchmarks/engage_vicos/s3d_clicks/` (+ rigid fit: `clicks_fit_full/`). Per
athlete, two click iterations from the athlete's own 2D pose (the model's eval-mode sampler picks the worst body
keypoint each time; all clicks were real keypoints, mostly hips). The repo's training-time click loop passes an
argument (`full_output`) its decoder no longer accepts; the script drops it with a wrapper, nothing vendored changed.

| | baseline | Arm 1 (rigid fit) | Arm 2 (clicks) | Arm 2 + Arm 1 |
|---|---|---|---|---|
| mount: bottom torso through top leg loop (H1 >= 0.60) | 0.22 | 0.11 | 0.14 | 0.14 |
| closed guard: top torso through guard player's leg loop (H2 >= 0.60) | 0.23 | 0.26 | 0.20 | 0.22 |
| half guard: top leg through guard player's leg loop (H3 >= 0.40) | 0.23 | 0.22 | 0.17 | 0.16 |
| back: a torso through the other's leg loop (H6 >= 0.40) | 0.49 | 0.43 | 0.39 | 0.38 |
| hypotheses passed | 2/6 (H5, H6) | | 2/6 (H4, H5) | 1/6 (H5) |
| both bodies the same athlete: mount / closed / half / side control | 25 / 24 / 30 / 26% | 8% (pooled) | **7 / 3 / 3 / 3%** | **2 / 0 / 0 / 1%** |

**Arm 2 meets its own target (duplicates <= 10%) and still fails the relations.** Clicking each athlete's own
keypoints separates the two people in 3D, but the separated bodies are not arranged so that one is inside the other's
legs: the defining relations fall slightly rather than rise. So duplication was not what was hiding the ground
relations; with it gone, what remains is the two bodies' relative placement and limb arrangement, the same error the
Harmony4D oracle points at (relative depth first), plus a definition that has never been checked against 3D truth
of real ground positions (Harmony4D has none).

## Overall, 2026-10-23
Every single-camera fix tried fails its pre-written criteria: the rigid joint fit (Arm 1), SAM 3D Body clicks
(Arm 2), PromptHMR (`phmr_results.md`), and PromptHMR placement under SAM 3D Body bodies (exploratory). Per the
decision rule, capture or training is the critical path. What the day established for that path:
1. **What is missing is relative depth** (and whole-body orientation) between the two athletes, not limbs:
   SAM 3D Body's own limbs with true position + orientation reach body lock 0.93, legs 0.74, region pairs 0.67.
2. **A second camera roughly at right angles** fixes placement and contact (distance 10.2 -> 3.3 cm, region pairs
   0.58, body lock 0.88), but averaging per-camera 3D guesses does not fix loops; joint multi-view reconstruction
   (triangulated keypoints / fitting to all views) is the untested next step, and the available Harmony4D cameras
   can test it before any capture.
3. **On ground work, separating the two people in 3D is solvable** with each athlete's 2D keypoints as clicks
   (duplicates 25% -> 3%), so detection / 2D keypoint quality per athlete is a prerequisite worth keeping.
4. **The ground definitions need 3D truth of ground positions**, which no dataset here has: the capture should
   include ground work with at least two views at an angle (or side + overhead).

**Not ruled out (added 2026-10-24).** "Every single-camera fix failed" covers the fixes above only; it does not show that
one camera cannot give relative depth. Ruled out so far: body size, temporal smoothing, the floor at these camera heights,
2D keypoint fitting with a no-overlap rule, and whole-image reasoning (PromptHMR). Untested single-camera cues:
(a) occlusion order (where one athlete covers the other, that part is in front; whether a limb passes through a loop is
largely decided by which part is in front at each image crossing, so this is the cue the loop measures need);
(b) a monocular depth-map model (Depth Anything V2 / Depth Pro / MoGe; none installed); (c) camera motion parallax
(not testable on Harmony4D: its cameras are fixed); (d) a model trained on two-person 3D data, which is still
single-camera at use time. For scale: the relative depth error (14.5 cm) is about 3.5% of the median camera-to-athlete
distance (4.2 m).
