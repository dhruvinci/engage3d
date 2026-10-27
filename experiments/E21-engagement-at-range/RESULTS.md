# E21 results: at range the distance is right to about 4%, but every pre-registered bar fails (pre-registration: `sword_prereg.md`)

Run 2026-10-27 06:06-07:47, after E19 and after the pre-registration was pushed (engage3d `c0f4085`), with E20's
corrected recipe (maps in `maps/moge/maps`, `--overlap` features, `--drop-overlap`, 60 epochs). Outputs:
`benchmarks/engage_h4d/sword/`. 1,120 views of 009 / 010_sword2, cameras 01/06/11/16.

## Criteria (full pipeline)
| | Measure | Result | Verdict |
|---|---|---|---|
| S1 | distance error between the two people, median (<= 5.0 cm) | 8.36 cm | **fail** |
| S2 | relative depth error, median (<= 6.0 cm) | 10.72 cm (combined depth alone: 8.40) | **fail** |
| S3 | `in contact` F1 (>= 0.90) | 0.31; perfect bodies reach only 0.50 under the scorer's thresholds | **fail** (the bar was above the measure's ceiling) |
| S4 | learned placement vs combined depth, configuration error | 9.22 -> 8.70 cm | pass |

## All stages (medians)
| | distance error | relative to the true distance | relative depth | in contact (F1) |
|---|---|---|---|---|
| SAM 3D Body | 11.48 cm | 5.2% | 15.86 cm | 0.25 |
| + combined depth | 8.45 cm | 3.7% | **8.40 cm** | 0.21 |
| + learned placement | **8.36 cm** | **3.7%** | 10.72 cm | 0.31 |
| perfect bodies | 0 | 0 | 0 | 0.50 |
The two fencers stand a median 2.36 m apart (hugging and ballroom: 0.65 m, where the full pipeline's error is 4.4% of
the distance). The relative-error column was not pre-registered (exploratory).

## Reading
- **At range, the distance between two people is measured to within about 4% of the distance, the same relative
  accuracy as in close contact,** but 4% of 2.4 m is 8-9 cm, which fails the absolute bars set for close contact.
  For engagement at range, a relative bar would have been the right one; that is noted for the next test, and the
  result is not re-scored here.
- **The learned placement hurts depth at range** (8.4 -> 10.7 cm): trained on grappling, where the two people are
  close, it pulls far-apart people towards a contact configuration. It helps the overall configuration slightly (S4).
  At range the combined depth alone is the better choice.
- **"Touching or not" is ill-defined for sword fighting** with this scorer: bodies rarely touch (the swords do), and
  even perfect bodies score 0.50.
