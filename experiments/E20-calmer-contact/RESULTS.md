# E20 results: ballroom dancing is measured well; hugging is not easier than grappling (pre-registration: `easy_prereg.md`)

Run 2026-10-27 02:27-06:17, after E18 and after the pre-registration was pushed (engage3d `b517146`).
Outputs: `benchmarks/engage_h4d/easy/` (`s3d/`, `maps/`, `place/`, `score_*`). GPU stages one at a time (bodies 34 min,
2D 2 min, depth maps 48 min).

**Process notes (recipe fixes, no change to the method; all in `easy/run.log`):** (1) the run script pointed the feature
step at `maps/maps`; the maps were in `maps/moge/maps`, so features and placement were re-run; (2) E08's recipe builds
features with `--overlap` and trains with `--drop-overlap`, which the first re-run missed (the feature files did not
match); (3) the trainer's default is now 120 epochs, E08 used 60: the reported model uses 60 (the 120-epoch run, kept as
`place_120ep/`, gives nearly the same numbers, ballroom 2.74 / hugging 4.73 cm).

## Data
1,736 views: hugging 002_hugging (604 views), ballroom 007-010_ballroom2 (1,132 views); cameras 01/06/11/16; true
joints project inside their own person boxes in 100% of sampled views. The hugging sequence also contains greetings
(a handshake).

## Criteria (full pipeline: SAM 3D Body + combined depth + learned placement)
| | Measure | Ballroom | Hugging | Verdict (both must pass) |
|---|---|---|---|---|
| W1 | distance error between the two people, median (<= 4.0 cm) | **2.74** | 4.63 | **fail** (ballroom passes) |
| W2 | region-pair F1 (>= 0.70) | 0.52 | 0.19 | **fail** |
| W3 | limb joints in contact, released model (<= 4.0 cm) | **3.78** | 6.59 | **fail** (ballroom passes) |
| W4 | learned placement vs combined depth, configuration error | 6.66 -> 6.18 | 10.29 -> 8.21 | **pass** |

## All stages (`engage_bench_oracle.py`, `pred` rows; medians in cm, F1 otherwise)
| | distance | relative depth | region pairs | in contact | embrace (torso through arm loop) |
|---|---|---|---|---|---|
| ballroom, SAM 3D Body | 4.13 | 7.80 | 0.38 | 0.89 | - |
| ballroom, + combined depth | 3.07 | 5.65 | 0.46 | 0.95 | - |
| ballroom, **+ learned placement** | **2.74** | **5.19** | **0.52** | **0.96** | - |
| hugging, SAM 3D Body | 8.11 | 9.88 | 0.15 | 0.88 | **0.89** |
| hugging, + combined depth | 4.80 | 7.09 | 0.19 | 0.85 | 0.73 |
| hugging, + learned placement | 4.63 | 7.16 | 0.19 | 0.89 | 0.74 |
| perfect bodies, under the scorer's thresholds | 0 | 0 | 0.86 / 0.79 | 0.98 / 0.99 | 1.00 |
References (not re-run): grappling held-out 11.2 -> 3.17 cm distance, region pairs 0.60, limbs in contact 5.40 cm
(E06, E11); MMA 4.30 cm, region pairs 0.46, limbs 6.25 cm (E08, E11).

## Reading
- **Ballroom dancing is a working case now:** the two dancers are placed within 2.7 cm of each other's true distance,
  whether they touch is right 96% of the time (F1), and limbs in contact are within 3.8 cm, against 5.4 cm in
  grappling. The correction learned on grappling transfers (W4).
- **Hugging is not easier than grappling for this pipeline:** the arms wrap behind the partner's back, out of the
  camera's view (limbs in contact 6.6 cm), and the two chests at one depth make the depth gap hard (relative depth
  7.2 cm). But the defining relation is found: SAM 3D Body alone detects the embrace with F1 0.89. Both depth
  corrections lower that to 0.74, because moving whole bodies to fix the gap also moves the arm loops.
- **"Which surfaces touch" stays weak everywhere** (0.52 ballroom, 0.19 hugging), and even perfect bodies only reach
  0.79-0.86 under the scorer's 2 cm / 5 cm thresholds, so W2's 0.70 bar was close to the measure's own ceiling.
- **The hard part is hidden limbs at contact, not grappling as such:** wherever limbs touch out of view (hugs,
  tangles), errors are 5.4-6.6 cm; where the contact is visible (a dance hold), 3.8 cm.
