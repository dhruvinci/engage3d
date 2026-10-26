# Pre-registration: does the placement model transfer to new people? — written 2026-10-25, before the run

## Why
All 16 Harmony4D grappling sequences show the same two athletes (identical body-shape parameters across sequences,
checked 2026-10-25). The placement model's held-out test therefore measured new moments of familiar people. The MMA
sequence (016_mma4) has a different pair (different shape parameters) doing a different activity (striking and
clinching, 4.6-5.8 m from the cameras). This test measures new people and new activity together; the two cannot be
separated here.

## What runs
v2's settings (width 512, 60 epochs, 3 seeds, mirroring, no tokens, `--drop-overlap`), trained on the 12 development
grappling sequences x 20 cameras only (no MMA; the 4 grappling test sequences excluded). It is evaluated on every MMA
view (20 cameras: 1,484 views at every 2nd frame on cam01/06/11/16, 2,968 at every 4th frame on the other 16) against
the combined depth on the same views. Relations are scored with `engage_bench_oracle.py` (mma4 section).

## Reference
The improvement on familiar people (v2, extended grappling test): configuration error 8.29 -> 6.50 cm (-22%),
relative depth 5.35 -> 4.08 (-24%), distance error 3.88 -> 2.94 (-24%).

## Criteria
| | Measure (model vs combined depth, MMA) | Transfers if | Does not transfer if |
|---|---|---|---|
| N1 | configuration error, relative change | <= -11% (half the familiar-people gain) | > -5%, or worse |
| N2 | relative depth error, relative change | <= -12% | > -5%, or worse |
| N3 | distance error and region pairs | no worse than the combined depth | worse on either |
Between those bands: "partial transfer".

**Decision.** Transfers: the method generalises across people, and data variety is needed mainly for coverage
(ground work, positions). Partial or none: the model has partly learned these two athletes, and varied athletes are
needed before it can be trusted on anyone else. That shapes how many people the capture must include.
