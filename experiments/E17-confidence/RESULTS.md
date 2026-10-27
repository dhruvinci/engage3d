# E17 results: a per-frame trust flag finds the bad frames (AUROC 0.76-0.80 in grappling and MMA), but trusting half the frames cuts the distance error by 28%, just short of the 30% bar (pre-registration: `confidence_prereg.md`)

Run 2026-10-27 12:45-13:28 on the CPU, after the pre-registration was pushed (engage3d `6615e69`). Code:
`scripts/engage_confidence.py`. Outputs: `benchmarks/engage_h4d/confidence/` (`confidence.json`, `run.log`).
The flags were trained on out-of-fold predictions for the 22,640 development views (4 folds by sequence). There, 27%
of views have a distance error above 5 cm.

## Criteria (confirmatory sets: views the smoke run never scored)
| | Measure | Result | Verdict |
|---|---|---|---|
| C1 | held-out grappling, cameras 03/08/13/18 (1,068 views): distance flag AUROC | 0.795 | **pass** (needed >= 0.70) |
| C2 | the same views: median distance error of the trusted half vs all | 1.88 vs 2.60 cm: -28% | **fail** (needed -30%) |
| C3 | MMA (new people), its other 16 cameras (2,976 views): distance flag AUROC | 0.763 | **pass** (needed >= 0.65) |
| C4 | held-out grappling, cameras 03/08/13/18 (692 views with a tangle): limb flag AUROC | 0.792 | **pass** (needed >= 0.65) |

## All sets
Distance between the two people (bad = more than 5 cm off):
| Set | views | bad share | AUROC | median, all | trusted half | flagged half |
|---|---|---|---|---|---|---|
| **held-out grappling, cams 03/08/13/18** | 1,068 | 28% | **0.795** | 2.60 cm | **1.88 cm** | 4.14 cm |
| **MMA, other 16 cams** | 2,976 | 44% | **0.763** | 4.15 cm | **2.55 cm** | 8.33 cm |
| held-out grappling, cams 01/06/11/16 | 1,068 | 30% | 0.777 | 3.16 cm | 2.23 cm | 4.83 cm |
| MMA, cams 01/06/11/16 | 1,484 | 45% | 0.746 | 4.44 cm | 3.00 cm | 7.75 cm |
| hugging | 604 | 49% | 0.795 | 4.77 cm | 2.35 cm | 17.27 cm |
| ballroom | 1,132 | 31% | 0.772 | 2.72 cm | 1.87 cm | 4.48 cm |
| sword fighting (2.4 m apart) | 1,120 | 68% | 0.584 | 8.01 cm | 7.36 cm | 8.71 cm |

Limbs in contact (views with a tangle; bad = the median limb-joint error above 6 cm):
| Set | views | bad share | AUROC | median, all | trusted half |
|---|---|---|---|---|---|
| **held-out grappling, cams 03/08/13/18** | 692 | 43% | **0.792** | 5.58 cm | 4.65 cm |
| **MMA, other 16 cams** | 1,552 | 46% | 0.771 | 5.80 cm | 4.99 cm |
| held-out grappling, cams 01/06/11/16 | 692 | 40% | 0.790 | 5.51 cm | 4.70 cm |
| MMA, cams 01/06/11/16 | 776 | 54% | 0.760 | 6.23 cm | 5.28 cm |
| hugging | 264 | 56% | 0.691 | 6.39 cm | 5.70 cm |
| ballroom | 1,100 | 12% | 0.788 | 3.79 cm | 3.52 cm |

The disclosed smoke run (2 training epochs, 3,000 development views) had seen the camera-01/06/11/16 sets: distance AUROC
0.77 held-out, 0.71 MMA. The full run lands close (0.777, 0.746), and the confirmatory sets agree with them.

## Single signals (distance flag, AUROC on the same sets)
| Set | ensemble disagreement alone | closest approach (closer = riskier) alone | the flag |
|---|---|---|---|
| held-out, cams 03/08/13/18 | 0.715 | 0.429 | 0.795 |
| MMA, other 16 cams | 0.750 | 0.369 | 0.763 |
| hugging | 0.737 | 0.601 | 0.795 |
| sword fighting | 0.576 | 0.477 | 0.584 |

## Reading (pre-registered: C1, C3 and C4 pass; C2 misses by 2 points)
- **The pipeline knows when it is wrong.** Without any truth, the flag ranks a bad frame above a good one about 4
  times in 5 in grappling (0.795), and nearly as often on new people (MMA 0.76) and in ballroom and hugging (0.77-0.80).
  The frames it flags are 2-7 times worse than those it trusts (MMA: 8.3 vs 2.6 cm; hugging: 17.3 vs 2.4 cm).
- **Trusting half the frames is not quite enough on grappling's held-out pair** (-28% against a -30% bar). The gain is
  larger where there are more bad frames: -39% on MMA's other cameras, -51% in hugging, -31% in ballroom. In held-out
  grappling the error is already small (2.6 cm), which leaves less to remove.
- **Most of the signal is the placement ensemble's disagreement** (0.72-0.75 alone); 2D confidence, depth evidence and
  overlap add the rest. How close the two bodies are does not predict distance error in grappling (0.37-0.43: if
  anything, apart is riskier). It does in hugging (0.60).
- **Limb errors are predictable too** (0.77-0.79 on grappling and MMA): the flag finds the frames where hidden limbs
  are wrong. Trusting half cuts the limb error by 14-17%, less than for distance.
- **At range the flag does not work** (sword fighting 0.58). Most frames there are above the 5 cm bar, which E21
  showed is the wrong yardstick at 2.4 m.
- **For the product:** every measurement can ship with this flag now. Show the trusted frames, and mark or smooth the
  flagged ones. At range, use relative error instead.
