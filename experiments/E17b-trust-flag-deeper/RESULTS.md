# E17b results: making the trust flag usable

**Status.**
- Pre-registered ([PREREG.md](PREREG.md), engage3d a97955b, pushed before any run).
- Run on CPU (`scripts/engage_trust_b.py`; outputs `benchmarks/engage_h4d/trust_b/`: `trust_b.json`, `views.npz`,
  `run.log`). About 1.3 hours; $0.
- Every number below is in `data/trust_b.json`.
- **Disclosure.** After the pre-registration was pushed, a smoke run tested the code. It used 2 training epochs and
  3,000 development views, and it also scored the test sets; its numbers were seen. Nothing was changed after it: no
  threshold, rule or code path. Its verdicts were the same as the full run's (C1 and C5 pass, C2-C4 fail).
- **Re-run check.** The re-run reproduces E17 exactly: confirmatory distance-flag AUROC 0.7946 and 0.7632, equal to
  E17's.

## Verdict
| | Part | Set | Result | Bar | |
|---|---|---|---|---|---|
| C1 | error bars, 90% | held-out grappling, cameras 03/08/13/18 | covers **90.5%** [87.1, 93.8] | 85-95% | **pass** |
| C2 | error bars, 90% | MMA, new people, other 16 cameras | covers **74.9%** [72.0, 77.9] | >= 85% | **fail** |
| C3 | repair flagged frames | held-out grappling | flagged frames **worse**: 4.14 -> 5.17 cm (-25%) | >= 20% better | **fail** |
| C4 | trust-weighted clip summary | both confirmatory sets, 32 series | 4.86 vs 4.42 cm for the plain mean (difference -0.44 [-1.57, 0.58]) | lower, interval above 0 | **fail** |
| C5 | best camera of three | held-out grappling | **1.82 cm** vs 3.37 cm for a random camera: **-46%** [-40, -51] | >= 20% lower | **pass** |

**Learned.**
- **The flag's best use is choosing a camera.** With three phones, taking each moment from the camera the flag trusts
  most halves the distance error:
  - held-out grappling: 3.37 -> 1.82 cm;
  - new people (MMA): 5.38 -> 2.93 cm, also -46%;
  - hugging: 10.03 -> 3.40 cm (-66%);
  - ballroom: 3.61 -> 1.80 cm (-50%).
  - A perfect choice among the three would reach 1.03 cm, so the flag gets about two-thirds of the possible gain.
- **Error bars hold for the training pair, not yet for new people.** Normalised by the flag's signals, the 90% bars
  cover 90.5% on the held-out pair and are narrower than fixed-width bars at the same coverage (median half-width 7.2
  against 9.5 cm). On new people they cover 75%: the bars are too narrow there and need calibrating per person (or on
  more people).
- **Repairing a flagged frame from its neighbours makes it worse.** It matches E14 (errors persist across frames):
  - the true distance changes within 0.4 s;
  - half or more of the frames are flagged, so the trusted neighbours are few and far apart.
- **Discounting flagged frames does not improve a clip's average distance.** Keeping only unflagged frames is much
  worse (10.8 cm): the flag's signals include the predicted distance and closeness, so the trusted frames are not a
  random sample of the clip.
- **The flag does not work at range.** On sword fighting, AUROC is 0.584 for absolute error and 0.506 for relative
  error above 4%, and the 90% bars cover 60%.

## Per set
| set | AUROC | 90% bars: coverage, median half-width | 80% bars: coverage | best of 3 cameras: picked / random / best (cm) | repair: flagged frames before -> after (cm) |
|---|---|---|---|---|---|
| held-out grappling, cams 03/08/13/18 (confirmatory) | 0.795 | 0.905, 7.2 cm | 0.801 | 1.82 / 3.37 / 1.03 | 4.14 -> 5.17 |
| MMA, other 16 cams (confirmatory) | 0.763 | 0.749, 8.7 cm | 0.658 | 2.93 / 5.38 / 1.51 | 5.64 -> 11.74 |
| held-out grappling, cams 01/06/11/16 | 0.777 | 0.882, 7.4 cm | 0.770 | 2.56 / 3.55 / 1.40 | 4.67 -> 5.61 |
| MMA, cams 01/06/11/16 | 0.746 | 0.798, 9.3 cm | 0.674 | 3.57 / 5.68 / 1.95 | 5.30 -> 6.83 |
| hugging | 0.795 | 0.859, 8.9 cm | 0.745 | 3.40 / 10.03 / 1.52 | 5.98 -> 9.44 |
| ballroom | 0.772 | 0.915, 9.1 cm | 0.834 | 1.80 / 3.61 / 0.98 | 3.20 -> 4.32 |
| sword fighting | 0.584 | 0.598, 9.5 cm | 0.438 | 6.93 / 8.83 / 3.02 | 8.56 -> 15.15 |

Rules fitted on the development predictions only: risk threshold 0.210 (the development median); conformal factors 2.45
(90%) and 1.80 (80%); fixed half-widths 9.5 cm (90%) and 6.3 cm (80%).

**Exploratory (no bar).**
- B5: a small MLP flag is worse than the logistic one on every set (held-out 0.758 vs 0.795; MMA 0.667 vs 0.763).
- B6: for relative error above 4%, the flag ranks views at AUROC 0.76-0.77 in grappling and MMA, but at chance on
  sword fighting (0.506).
- B7: the limb flag's 90% bars cover 89.0% (held-out) and 87.2% (MMA).

## Not run, and why
As pre-registered:
- the ViCoS phones and a blind human review: they need GPU passes and Dhruva's time;
- E16's detection signals: development views need a detection pass;
- flags for touching and for who is on top;
- mirror consistency.

## Reproduce
```
TORCH_THREADS=6 ../.venv-depth/bin/python engage_trust_b.py --feats ../benchmarks/engage_h4d/place_feats2 \
    --out ../benchmarks/engage_h4d/trust_b
```
