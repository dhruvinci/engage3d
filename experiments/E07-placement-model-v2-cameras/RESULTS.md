# Placement model v2 — overnight log and results (pre-registration: `placemodel_v2_prereg.md`)

Running notes, newest last. All numbers are 4-fold cross-validation by sequence over the 12 development grappling
sequences, evaluated on the original-camera views (cam01/06/11/16, 4,528 views: the same views as v1's CV), unless
stated otherwise. The held-out sequences are untouched until the pre-registered test.

## Processing (GPU queue `benchmarks/engage_h4d/night/queue.sh`; log `night/NIGHT.log`; GPU log `night/gpu.log`)
- Wave 1 (cam03/08/13/18 x 12 development sequences): 4,528 new views. 23:57 -> 01:21. Peak GPU 3.3 GB of 6.
- Extended test (cam03/08/13/18 x 4 held-out sequences): 1,068 views, in `s3d_box_xtest/` / `depth_maps_xtest/`.
- Tokens for the original cameras (`s3d_box_tok/`): re-running SAM 3D Body reproduces the earlier bodies to 1e-6 m.

## Ceiling of a rigid correction (for scale)
On the 4,528 views: combined depth gives configuration error 8.12 cm, relative depth 5.76, relative lateral 5.03.
Moving and rotating each predicted body perfectly onto the truth (limbs as predicted) gives 5.77 / 2.25 / 3.06. That
is the floor for this model. v1 closed 51% of the configuration gap; relative depth is where most room is left.

## Learning curve and variants
| Training | Train views | Config err | Rel depth | Rel lateral |
|---|---|---|---|---|
| v1: original 4 cameras (width 256, 60 epochs, 3 seeds) | 6,012 | 6.91 cm | 4.59 | 3.59 |
| + wave 1 (8 cameras) | 10,540 | 6.84 | 4.44 | 3.56 |
| + wave 1, width 512 | 10,540 | 6.82 | 4.30 | 3.54 |
| + wave 1, 120 epochs | 10,540 | 6.85 | 4.39 | 3.56 |

### Pose tokens (SAM 3D Body's final pose token per athlete, 32 principal components fitted per training fold)
Same data (original cameras + wave 1), width 512. A learned token cannot be mirrored, so the token run is compared with
a no-mirror control:
| | Config err | Rel depth | Rel lateral |
|---|---|---|---|
| mirroring on, no tokens | 6.82 cm | 4.30 | 3.54 |
| mirroring off, no tokens (control) | 6.89 | 4.57 | 3.60 |
| mirroring off, tokens | 6.84 | 4.39 | 3.63 |
The tokens carry information (control -> tokens: -0.05 cm config, -0.18 cm relative depth), about as much as
mirroring adds. Using both would need SAM 3D Body tokens of mirrored images (a second pass); not tonight. To be
re-checked at full data, where augmentation matters less.

### Overlap (front-to-back) features: no gain
Seven extra features from the pixels where both predicted bodies overlap (which body the depth map says is visible
there, by how much, and how often the prediction's front-to-back order disagrees; `engage_place_features.py
--overlap`). Original cameras + wave 1, width 512: config err 6.82 -> 6.82 cm, rel depth 4.30 -> 4.39, rel lateral
3.54 -> 3.59. Not used further (`--drop-overlap`). The front-to-back evidence seems to be carried already by the
per-region depth disagreement.

### Process note (my error)
Between 03:30 and 09:00, three CV runs ran at once, each with 8 torch threads, next to the GPU jobs. On 16 cores
(load average 30) each run crawled: one fold in 5.5 h instead of minutes. They were stopped and are re-run one at a
time with a thread cap (`TORCH_THREADS`). The GPU queue was unaffected; the delay was CPU-side only.

## Learning curve at full data (13:11; the machine was suspended 10:17-12:00 by a power cut, no data lost)
| Training (width 512 unless noted) | Train views | Config err | Rel depth | Rel lateral | Gap to floor closed |
|---|---|---|---|---|---|
| v1: original 4 cameras (width 256) | 6,012 | 6.91 cm | 4.59 | 3.59 | 51% |
| + wave 1 (8 cameras) | 10,540 | 6.82 | 4.30 | 3.54 | 55% |
| + waves 1-3 (all 20 cameras) | 24,124 | **6.71** | **4.21** | **3.45** | **60%** |
| all 20 cameras + tokens (no mirroring) | 24,124 | 6.75 | 4.16 | 3.50 | 58% |
More viewpoints keep helping, with diminishing returns: 4x the data closes another 9% of the gap. **v2 (by the
pre-registered rule: lowest configuration error; simpler on ties) = all cameras, width 512, 60 epochs, 3 seeds,
mirroring, no tokens.** Final training also includes MMA's 16 extra cameras (every 4th frame).

## Held-out test (pre-registered; 15:16) — v2 does NOT pass (X2 fails by 0.04 cm)
v2 trained once on every non-test view (27,100: 12 development sequences x 20 cameras + MMA x 20). v1 re-trained
with its settings reproduces its earlier test exactly (3.175 / 4.302 / 0.601). Outputs: `place_v2_test/`,
`place_v1_ext/` (scores in `score_{primary,extended}_{model,blend}/`).

| Primary test: held-out sequences, cameras 01/06/11/16, 1,068 views | v1 | v2 |
|---|---|---|
| X1 distance error (pass if v2 < v1) | 3.175 cm | **3.115** (pass) |
| X2 relative depth (pass if v2 < v1) | **4.302** | 4.345 (**fail**) |
| X3 region pairs (pass if >= 0.59) | 0.601 | **0.609** (pass) |
| body lock / legs / arms | 0.760 / 0.400 / 0.548 | 0.811 / 0.333 / 0.516 |

| Extended test: the same sequences x 8 cameras (+ cam03/08/13/18), 2,136 views | v1 | v2 |
|---|---|---|
| distance error | 3.13 cm | **2.94** |
| relative depth | 4.30 | **4.08** |
| relative lateral | 3.07 | **2.97** |
| region pairs | 0.606 | **0.613** |
| X4 body lock (136 views with a confident true case) | 0.739 | **0.793** (pass) |
| X4 legs (16 views with a confident true case) | 0.277 | 0.278 (pass, a tie) |
| arms | **0.539** | 0.517 |
| configuration error | 6.67 | **6.50** |
(Combined depth alone on the extended views: distance 3.88, relative depth 5.35, region pairs 0.55, body lock 0.57.)

**Verdict (per the rule):** X2 fails on the primary test (4.345 vs 4.302 cm), so v2 does not replace v1. The
pre-registered reading applies: more viewpoints of the same 12 sequences do not decisively improve placement. The
evidence is consistent and small: cross-validation (6.91 -> 6.71 cm configuration error), the extended test (twice the
views: distance -0.19 cm, depth -0.22, body lock +0.05), and the primary test (distance -0.06, depth +0.04). The
learning curve flattens: 4x the data closed 9% more of the gap to the rigid floor. **The next lever is new sequences
and athletes (Harmony4D's training split: `train/03_grappling2` 17 GB, `train/02_grappling` 44 GB), not new
viewpoints.** The pose tokens and the overlap features gave no robust gain at this data size.
