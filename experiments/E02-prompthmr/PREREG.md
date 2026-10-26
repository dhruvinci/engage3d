# Pre-registration: PromptHMR as the single-camera model — written 2026-10-22 20:25, before the full run

## Why
The rigid joint fit failed (`jointfit_results.md`), and the oracle ablation showed what breaks the relations: each
body's depth along the line of sight (relative depth error 14.5 cm vs 5.0 cm lateral), not the limbs. It is not body
size, jitter or floor-fixable. movement-research E13 (15 frames x 5 runs) found that PromptHMR, which encodes the whole
image once and decodes both people against it, placed two people better than SAM 3D Body, which reconstructs each
from its own crop. If a whole-image model already fixes relative depth, the single-camera path is open without capture
or training; if not, the decision rule of `jointfit_prereg.md` stands.

## What runs
`scripts/engage_bench_phmr.py` (Harmony4D) and `scripts/engage_vicos_phmr.py` (ViCoS): the same views, the same
ground-truth boxes / keypoint boxes, the same true-lens re-solve of translation as the SAM 3D Body baseline; both
athletes in one call, cross-person attention on. Not trained on Harmony4D (repo configs and README: BEDLAM 1/2, H36M,
3DPW, MPII, BEHAVE, CHI3D, Hi4D, COCO). Scored by the unchanged scorers (`engage_bench_score.py`,
`engage_vicos_score.py`), which now accept SMPL joints.
Already seen before writing this: a 60-view smoke test on one 15-frame sequence (distance error 5.1 vs 11.4 cm,
region pairs 0.55 vs 0.46, arm loops 0.11 vs 0.50 on 20 views).

## Criteria (Harmony4D grappling, same as P1-P5, balance in the matched form)
| | Measure | SAM 3D Body | Pass if |
|---|---|---|---|
| P1 | distance between the bodies, median error | 10.2 cm | <= 6 cm |
| P2 | region-pair F1 (best of 2/5/10 cm) | 0.47 | >= 0.57 |
| P3 | loop balance sign, matched (truth not tied) | 41% | >= 70% |
| P4 | body lock F1 / in-contact F1 | 0.80 / 0.85-0.92 | not below 0.75 / 0.85 |
| P5 | leg through leg loop F1 | 0.46 | >= 0.56 |
**Decision.** P1 + P2 pass: whole-image models fix single-camera placement and contact; the work moves to loops
(orientation / limbs) on that model. P1 + P2 + P3 + P5 pass: single-camera engagement measurement is viable with this
model. P1 or P2 fails: the decision rule of `jointfit_prereg.md` stands (capture or training is the critical path).
**ViCoS:** H1-H6 of `vicos_prereg.md` (same thresholds) and the duplicate-body rate (SAM 3D Body 24-30% on mount /
guards / side control; pass if <= 10%).
**Diagnostic, not a criterion:** the oracle ablation on PromptHMR (is depth still the dominant error?).
