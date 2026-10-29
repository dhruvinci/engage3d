# E16 stage 0: ceilings for hidden limbs — masks, clicks and fitting, with perfect inputs (exploratory, development data only)

Run 2026-10-27, before E16's training, to decide where its effort goes. No test view was used. Measure as E14 / E18:
median error of the 8 limb joints inside a tangle, rigid-aligned over 14 points.

## Prompts into SAM 3D Body (GPU; 500 views of 025 / 035, E19's validation views; `scripts/engage_s3d_ceiling.py`)
| Given to SAM 3D Body (true box) | Tangled limbs | vs nothing | 95% interval |
|---|---|---|---|
| nothing | 4.80 cm | | |
| the true visible mask | 4.64 cm | -3% | -4.5% to -0.8% |
| the true 2D joints as 2 clicks | 4.07 cm | -15% | -17% to -12% |
| both | 4.01 cm | -17% | -19% to -14% |

## Fitting the limbs to perfect 2D (CPU; all 22,640 views of the 12 development sequences; `scripts/engage_fit_ceiling.py`)
Both bodies are in the camera frame, so perfect 2D is each true joint's viewing ray; a fit must find each joint's
distance along its ray.
| Variant | Tangled limbs | Change | Other limb joints |
|---|---|---|---|
| SAM 3D Body | 5.46 cm | | 4.53 cm |
| every joint on its true ray at its predicted distance | 3.91 cm | -28% | 3.33 cm |
| exact bone lengths (SAM 3D Body's) along the rays | 6.78 cm | +24% | 7.54 cm |
| soft fit (distance sigma 10 cm, bones sigma 2 cm), SAM 3D Body's bone lengths | 5.23 cm | -4% | 5.89 cm |
| + smooth over time (sigma 5 cm per frame) | 4.91 cm | -10% | 5.50 cm |
| **soft fit, calibrated bone lengths** (per-bone predicted/true ratio from the other sequences) | **2.95 cm** | **-46%** | 2.64 cm |
| soft fit, true bone lengths (oracle) | 2.89 cm | -47% | 2.61 cm |
On 025 / 035 alone: calibrated -37%, true bones -40%.

SAM 3D Body's limbs are mis-proportioned against the truth (upper arms and thighs 9-14% long, forearms 6-9% short;
partly the two body models' joint definitions), and a limb lying across the image turns a small length error into a
large error along the ray. With a calibrated skeleton the fit removes almost half the tangled-limb error.

## Does the bone-length calibration transfer to new people? (same soft fit, calibration learned on the 12 grappling sequences)
| Target | Rays only | Soft fit, calibrated | Soft fit, true bones |
|---|---|---|---|
| Development grappling (leave one sequence out: **the same two athletes**) | -28% | -46% | -47% |
| MMA, new people (4,460 views) | -27% | **-43%** | -47% |
| Hugging + ballroom, new people (1,736 views) | -9% | **-14%** | -40% |
The calibration carries across similar athletes in fitted sportswear (grappling to MMA), not to the hugging and ballroom
people (other body types, loose clothes, a skirt). SAM 3D Body's limb bias is partly person-specific, so the fitting
stage should measure each person's own skeleton (for example from a few seconds in view of two phones) where body type
or clothing differs. Smoothing over time hurt on ballroom (+60%), whose frames are 8 apart.

## How exact must the 2D be? (rays perturbed to a median 2D error on the 4K frames, every joint)
| Median 2D error | Rays only | Soft fit, calibrated skeleton |
|---|---|---|
| 0 px | -28% | -46% |
| 2.5 px | -28% | -44% |
| 5 px | -25% | -38% |
| 10 px | -16% | -20% |
| 15 px | -3% | 0% |
| 20 px | +11% | +21% |

## Reading
- Masks do not fix hidden limbs (-3%); clicks are capped near -15%; **2D plus a soft fit with a calibrated skeleton
  can remove about 45%**: the strongest route found.
- The fit pays only with exact 2D: about 5-7 px for -30% or better, 10 px for -20%, break-even near 15 px. Today's
  best inside tangles is 19 px (E19's fine-tuned ViTPose+), which is why every 2D route so far made limbs worse.
- So E16's keypoint model has a quantitative target: 2D error inside tangles, hidden joints included, of 10 px or less
  (stretch: 5-7 px) on the 4K frames.
- Caveats: uniform noise on every joint (real errors concentrate on hidden joints, so uncertainty-weighted fitting may
  tolerate more); development data of one pair; the bone-length calibration must be checked on new people.
