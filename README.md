# engage3D — measuring how two people engage, in 3D, from one camera

Research code and pre-registered experiments on measuring **two-person engagement** (where two people touch, which
body surfaces meet, whose limbs wrap whom) as 3D quantities, from ordinary single-camera video. Test domains:
standing grappling and MMA (Harmony4D, multi-view 3D ground truth) and Brazilian jiu-jitsu ground work (ViCoS, expert
position labels). The relations generalise to any close human interaction.

Every experiment is **pre-registered**: its pass criteria are committed before it runs (see the commit history), and
its results are reported against them, failures included. One folder per experiment: [`experiments/`](experiments/).
Findings log: [`FINDINGS.md`](FINDINGS.md). Plan: [`docs/RESEARCH_QUEUE.md`](docs/RESEARCH_QUEUE.md).

## Where it stands (October 2026)
Single-camera pipeline: **SAM 3D Body** (one body per person) -> **combined depth** (a monocular depth map from MoGe-2
blended with SAM 3D Body's own estimate of the depth gap between the two people) -> **learned placement correction**
(a small network that moves and turns each body).

| Harmony4D, held-out sequences (1,068 views) | SAM 3D Body | + combined depth | + learned correction |
|---|---|---|---|
| distance between the two athletes, median error | 11.2 cm | 3.95 cm | **3.17 cm** |
| relative depth between them, median error | 16.0 cm | 5.8 cm | **4.3 cm** |
| which body surfaces touch (region-pair F1) | 0.46 | 0.55 | **0.60** |
| hand fighting (arm through arm loop, F1) | 0.41 | 0.42 | **0.55** |
| loop balance (sign agreement) | 41% | 49% | **56%** |

- **New people:** trained on one pair of athletes, the correction keeps about two-thirds of its gain on a different
  pair doing MMA (distance 5.07 -> 4.30 cm) ([E08](experiments/E08-new-people/)).
- **Ground work** (no 3D truth): the combined depth raises the defining relations of mount (0.14 -> 0.29), closed
  guard (0.20 -> 0.33) and back control (0.39 -> 0.64) without false positives in control positions
  ([E05](experiments/E05-vicos-combined-depth/)).
- **Limits:**
  - ground-truth person boxes were used;
  - all Harmony4D grappling is two pairs of athletes, too few to measure how athlete variety helps
    ([E09](experiments/E09-athlete-variety/));
  - leg entanglements and loop balance are not yet reliable, and what remains is limb error inside tangles.

## Layout
- `src/`: the code (flat modules; each script states its environment and usage in its docstring). Paths to data and
  models: `src/engage_paths.py` (environment variables `ENGAGE3D_*`).
- `experiments/E<nn>-<name>/`: `PREREG.md` (committed before the run) and `RESULTS.md`.
- `docs/`: measurement spec, capture design, research queue.
- `env/`: pinned package lists of the five environments (`venv-<name>.txt`; Python 3.11-3.12, CUDA 12/13).
- `tools/sync_from_workspace.py`: how this repo is exported from the private research workspace, with a privacy check.

## Data and models (not redistributed here)
- **Harmony4D** (Khirodkar et al., NeurIPS 2024; dataset card: MIT): `Jyun-Ting/Harmony4D` on Hugging Face.
  `src/engage_h4d_fetch.py` pulls selected cameras and frames straight out of the online zips.
- **ViCoS BJJ positions** (Hudovernik & Skocaj, MMSports'22; CC BY-NC-SA 4.0).
- **Models:**
  - SAM 3D Body and SAM 3 / 3.1 (Meta, SAM License);
  - MoGe-2 (`Ruicheng/moge-2-vitl-normal`) and Depth Pro (`apple/DepthPro`);
  - ViTPose+ (`usyd-community/vitpose-plus-base`);
  - PromptHMR (non-commercial research); SMPL / SMPL-X (their own licences).
- Published artifacts (models, benchmarks) live on Hugging Face, one repo per experiment, gathered in the engage3D
  collection (links in each experiment's README as they are released).

## Licence
Code: MIT ([`LICENSE`](LICENSE)). Documents and results: CC BY 4.0. Third-party data and models keep their own
licences.

## Citation
See [`CITATION.cff`](CITATION.cff).
