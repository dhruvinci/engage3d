# Harmony4D result: monocular 3D (SAM 3D Body) vs ground-truth engagement relations — 2026-10-22

Runner `scripts/engage_bench_s3d.py` (ground-truth boxes, one athlete at a time, translation re-solved through the
true fisheye lens), scorer `scripts/engage_bench_score.py`, fusion `scripts/engage_bench_fuse.py`. Outputs
`benchmarks/engage_h4d/s3d_box/` (+ `score/summary.txt`, `score/records.jsonl`). 5,596 views = 1,399 frames x 4
cameras, 16 standing-grappling sequences. 95% intervals: cluster bootstrap over sequences.

**Placement (medians):** pelvis 10.8 cm, body pose relative to pelvis 7.5 cm, distance between the athletes 10.2 cm.

| Relation (family) | views with truth | F1 one camera [95% CI] | shuffled pairing | 5-frame vote | 4-camera vote |
|---|---|---|---|---|---|
| torso through arm loop (body lock) | 860 | **0.80** [0.64, 0.86] | 0.05 | 0.83 | **0.89** |
| leg through leg loop | 152 | 0.46 [0.22, 0.62] | 0.03 | 0.54 | 0.61 |
| arm through arm loop (hand fighting) | 748 | 0.22 [0.14, 0.31] | 0.02 | 0.23 | 0.26 |
| leg through arm loop | 48 | 0.36 | 0.05 | 0.41 | 0.48 |
| head through arm loop | 56 | 0.16 | 0.00 | 0.05 | 0.19 |

- **In contact** (bodies within 2 cm, true in 75% of views): F1 0.85 at a 2 cm threshold, 0.92 at 10 cm.
  **Which body parts touch** (16 x 16 region pairs): F1 0.42-0.47.
- **Loop balance** (headline): predicted sign agrees with truth in 46% of views where truth is not tied (52% with a
  4-camera vote). Not usable yet: it is dominated by arm-loop relations, which are mostly grazing and poorly predicted.
- **Cross-camera agreement** of predicted threading (no ground truth needed): Jaccard 0.25 (truth 1.00). A free
  warning sign that single-view relations are unstable.
- **Shuffled pairing** (each athlete's predicted body kept, the partner taken from >= 1 s away) collapses every family
  to <= 0.05: the signal that exists comes from relative 3D placement, not from pose priors.

**Reading.** From one camera, coarse engagement is measurable: contact and body locks well, leg-between-legs
moderately. Fine engagement is not: which parts touch, hand fighting, and loop balance. The limiting quantity is
relative placement between two interlocked bodies (10 cm median error between athletes, against loop geometry of a
few cm). Naive voting over cameras helps the coarse relations (0.80 -> 0.89, 0.46 -> 0.61) but not loop balance.
