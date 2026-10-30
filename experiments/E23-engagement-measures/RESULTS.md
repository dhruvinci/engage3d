# E23 results: engagement beyond contact, from one camera

**Status.**
- Pre-registered ([PREREG.md](PREREG.md), engage3d 03a1367, pushed before any prediction was scored). Definitions
  signed off by Dhruva on 2026-10-30 (`docs/MEASUREMENT_SPEC.v1.md`).
- Run on CPU in about 25 minutes (`scripts/engage_measures.py measures --which both`, then `score`). It used E22's
  existing bodies; nothing was re-run.
- Every number below is in `data/score.json`.
- Intervals: 95%, from 2,000 resamples of 1 s blocks within each clip. C4 (an F1 pooled over clips) and C9 (6 episodes)
  are point values.

## Verdict (8 Harmony4D clips, single camera against truth)
| | Measure | Result | Bar | |
|---|---|---|---|---|
| C1 | centre of mass, 3D | **8.6 cm** [7.5, 9.7] | <= 10 cm | **pass** |
| C1 | the A-B centre-of-mass vector | **6.5 cm** [5.9, 7.3] | <= 8 cm | **pass** |
| C2 | level difference (CoM heights) | **1.7 cm** [1.5, 1.9] | <= 6 cm | **pass** |
| C3 | stable or not (CoM inside own base) | 62.8% agree [59.9, 65.9] | >= 80% | **fail** |
| C4 | supported by the partner (proxy) | F1 0.48 (4 clips where it occurs) | >= 0.6 | **fail** |
| C5 | hips above (true hip heights >= 10 cm apart) | **100%** agree (148 frames) | >= 80% | **pass** |
| C6 | head inside, frames in contact | **75.4%** agree [69.4, 80.6] | >= 75% | **pass** (narrowly; the interval reaches below the bar) |
| C7 | inside wrists, hands, knees, feet, frames in contact | **88.4%** agree [85.9, 90.6] | >= 70% | **pass** |
| C8 | lever arm at contact | **3.5 cm** [3.1, 3.8] | <= 6 cm | **pass** |
| C9 | closing speed before a contact episode | **0.15 m/s** (6 episodes) | <= 0.3 m/s | **pass** |

**Seven of nine pass.** The two that fail are the balance measures (stable, supported by the partner).

## Per activity (point values; intervals in `data/score.json`)
| activity | CoM (cm) | A-B vector (cm) | level (cm) | stable agree | head inside agree | inside agree | lever (cm) |
|---|---|---|---|---|---|---|---|
| hugging | 3.1 | 3.5 | 1.7 | 66% | 86% | 96% | 2.6 |
| karate | 7.4 | 6.6 | 0.8 | 65% | 75% | 95% | 6.3 |
| grappling | 10.2 | 7.9 | 1.9 | 64% | 77% | 92% | 3.7 |
| MMA | 10.5 | 9.3 | 2.3 | 58% | 75% | 87% | 4.7 |
| ballroom | 14.8 | 5.7 | 0.8 | 66% | 64% | 80% | 3.8 |
| sword (range) | 15.4 | 9.3 | 3.0 | 60% | (no contact) | (no contact) | (no contact) |

MAMMA (West Coast Swing, 4 clips; reported separately, without pass marks):
- CoM 13.4 cm, A-B vector 26.2 cm;
- level 3.1 cm, stable 55%, hips above 99%, head inside 83%, inside 88%, lever 2.7 cm.
- The A-B vector inherits E22's depth error on dance seen with one dancer in front of the other (16-33 cm).

## Reading
- **Measurable from one camera, for the activities tested:**
  - centre of mass and the vector between the two people's centres;
  - who is lower and by how much (level within 2 cm);
  - whose hips are higher;
  - head and limb inside position;
  - lever arms at contact;
  - closing speed.
  The relational measures (level, hips, inside position) hold best. They are what combat sports care about, and they
  depend on the two bodies relative to each other more than on absolute depth.
- **Not yet measurable:** whether a person is balanced over their own base, and whether they lean on the partner.
  - Both depend on where the centre of mass falls relative to the feet, to within a few centimetres.
  - A CoM 8.6 cm off, plus feet that rarely meet the floor exactly in the reconstruction, flips the verdict in a third
    of frames. The truth check before the run showed how sensitive the measure is: on the true bodies, standing people
    are stable in 70-94% of frames with both feet down and mostly unstable on one foot.
- **Absolute position matters less than the relation:** ballroom's CoM is 14.8 cm off, but the vector between the two
  dancers is 5.7 cm.

## Known limits
- 12 clips of 10 s; 6 contact episodes for C9.
- Weight on the partner is a geometric proxy, not a force; leverage has no direction.
- SAM 3D Body was trained on Harmony4D, so absolute errors are likely optimistic.
- The person correspondence between prediction and truth is chosen per clip by CoM distance.

## Reproduce
```
python scripts/engage_measures.py measures --clips all --which both --out benchmarks/engage_h4d/measures_e23
python scripts/engage_measures.py score --out benchmarks/engage_h4d/measures_e23
```
