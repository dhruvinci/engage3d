# Pre-registration: E21, engagement at range: does single-camera measurement work for sword fighting? — written 2026-10-27, before any of these views was processed

## Why
E20 tests calm contact (hugging, ballroom). Much of human engagement happens at range: two people facing off,
managing distance, closing and opening it (fencing, sword fighting, striking, also everyday conversation). There the
quantities that matter are the distance between the two people, who is closer to the camera, and whether they touch at
all. Harmony4D's sword fighting gives exact 3D truth for it.

## Data
Harmony4D test split `05_sword2` (sequences 009 and 010), cameras 01/06/11/16, every 8th ground-truth frame (about
1,120 views), fetched with `engage_h4d_fetch.py`; ground-truth boxes. New people, an activity never trained on.

## Pipeline
Identical to E20 (SAM 3D Body, ViTPose+, MoGe-2, combined depth, E08's placement model: 12 development grappling2
sequences, v2 settings, 60 epochs, `--drop-overlap`).

## Measures (`engage_bench_oracle.py`, `pred` rows)
- distance error between the two people;
- relative depth error (who is nearer the camera, and by how much);
- the `in contact` F1 (are they touching), the guard against false contact.

## Criteria (full pipeline: SAM 3D Body + combined depth + learned placement)
| | Measure | Passes if |
|---|---|---|
| S1 | distance error between the two people (median) | <= 5.0 cm |
| S2 | relative depth error (median) | <= 6.0 cm |
| S3 | `in contact` F1 | >= 0.90 |
| S4 | the learned placement vs combined depth: configuration error | lower (it transfers to engagement at range) |

**Reading.** S1-S3 pass: single-camera measurement of engagement at range works, and joins hugging / ballroom (E20)
in the map of where the pipeline already works. A fail locates the next problem (distance, depth or false contact).

Compute: GPU about 1 hour (bodies, 2D, maps), after E19 finished, alone on the GPU; CPU about 45 minutes; disk about
2 GB.
