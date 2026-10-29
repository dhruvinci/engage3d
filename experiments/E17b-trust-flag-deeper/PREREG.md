# Pre-registration: E17b, making the trust flag usable: error bars, repair, clip summaries, the best camera — written 2026-10-29, before any run

## Why
E17 showed that, with no ground truth, the pipeline finds its own bad frames (distance flag AUROC 0.795 on the held-out
pair, 0.763 on new people). A flag alone is not yet something a user can act on. This experiment tests four uses of
it, each against a bar fixed here:
- error bars with a tested coverage;
- repairing flagged frames from trusted neighbours;
- clip summaries that discount flagged frames;
- choosing the best camera when several phones film the same moment.

## What has been seen before writing this
- E17's summary numbers (`experiments/E17-confidence/RESULTS.md`). E17 saved nothing per frame, so none of the
  analyses below has been run, on any set.
- The feature files' layout: every 4th frame (7.5 per second) per sequence and camera, with frame numbers.

## Data and flag (E17's, unchanged)
- Features: E17's (`place_feats2`). Placement: E08's settings (width 512, 60 epochs, 3 seeds, mirroring,
  `--drop-overlap`).
- Development: the 12 development grappling2 sequences x 20 cameras (22,640 views). Out-of-fold predictions come from
  4-fold cross-validation by sequence. A full model scores the test sets.
- Flags: E17's logistic flags on the same truth-free signals (distance, bad above 5 cm; limbs in contact, bad above
  6 cm).
- **Test sets:** E17's.
  - Confirmatory: held-out grappling (031 / 037 / 042 / 043) on cameras 03/08/13/18 (1,068 views), and MMA (new
    people) on its other 16 cameras.
  - Secondary: the camera-01/06/11/16 sets, hugging, ballroom, sword fighting.
- **Re-run check:** the same code, settings and seeds re-create E17, this time saving every view. The confirmatory
  AUROCs must match E17's to within 0.01. A larger gap is reported, and the re-run's numbers are used.
- **Every rule below is fitted on the development sequences' out-of-fold predictions only.** Nothing is tuned on a
  test set.
- Script: `scripts/engage_trust_b.py`. Measure throughout: the error of the distance between the two people
  (pelvis to pelvis) after the learned placement, as in E17.

## Parts and criteria
**B1, error bars with a coverage guarantee (normalised split conformal).**
- Scale: a ridge regression of log(error + 0.5 cm) on the standardised signals, fitted on the development
  predictions. sigma = exp(its prediction).
- Score: error / sigma on the development predictions. q = the ceil((n + 1)(1 - a))-th smallest score.
- Interval: predicted distance +- q x sigma, for a = 0.1 (90%) and 0.2 (80%).
- Also reported: a fixed-width interval (the development 90th-percentile error) for comparison of widths.

| | Set | Passes if |
|---|---|---|
| C1 | held-out grappling, cameras 03/08/13/18 | the 90% interval covers 85-95% of views |
| C2 | MMA, new people, other 16 cameras | the 90% interval covers at least 85% of views |

**B2, repairing flagged frames.**
- Threshold: the median development risk (E17's trusted-half boundary, fixed on development). A view above it is
  flagged.
- Per sequence and camera, in frame order, a flagged view's distance is replaced by the (1 - risk)-weighted mean of the
  unflagged views within 3 samples (0.4 s) either side. With no unflagged neighbour it is left as it was.

| | Set | Passes if |
|---|---|---|
| C3 | held-out grappling, cameras 03/08/13/18 | the median error of flagged views falls by at least 20%, the 95% interval of the fall (bootstrap over sequence-camera series, 2,000) excludes 0, and the median error over all views does not rise |

**B3, clip summaries.**
- Per sequence and camera: the mean distance between the two people over its views.
- Estimators:
  - all views;
  - unflagged views only;
  - weighted by (1 - risk).
- Error: the estimate minus the true mean over all the series' views, absolute.

| | Set | Passes if |
|---|---|---|
| C4 | the two confirmatory sets pooled (32 series) | the weighted mean's error is lower than the all-views mean's (paired bootstrap over series; the interval of the difference excludes 0) |

**B4, the best camera of three.**
- For each test frame, 3 of the set's cameras are drawn at random, 20 times per frame (seeded). The camera with the
  lowest risk is picked.
- Compared with:
  - a random pick: the mean error of the 3;
  - the oracle: the best of the 3.
- The true distance is the same in every camera, so the errors are comparable.

| | Set | Passes if |
|---|---|---|
| C5 | held-out grappling, cameras 03/08/13/18 | the picked camera's median error is at least 20% below a random pick's (95% interval by bootstrap over frames excludes 0) |

**Exploratory (no bar):**
- B5: a stronger flag, a small MLP (one hidden layer of 32) on the same signals, against the logistic flag (AUROC,
  every set).
- B6: range. On sword fighting, the distance flag's AUROC for relative error above 4% (E21's level), and B1's
  coverage.
- B7: B1-B4 for the limb flag where they apply (coverage of the limb error; best camera by limb risk).

## Reading
- C1 and C2 pass: every distance ships with an error bar whose coverage was tested on new people.
- C1 passes and C2 fails: the bars hold for the training pair only and need re-calibration per person.
- C3: ship the repair rule (or not).
- C4: ship trust-weighted summaries (or not).
- C5: recommend two or three phones with selection by the flag.
- Each part stands alone. A failed part is reported as such.

## Not in this run, and why
- **ViCoS synced phones** (does the flag track disagreement between phones?): the flag's signals need ViTPose+,
  depth-map and overlap features on the ViCoS frames, a GPU pass not yet run.
- **Blind human review of flagged against trusted frames on public footage:** needs Dhruva's time and a GPU pass on
  the footage.
- **E16's detection signals:** E16 detects only on test views. Training a flag on them needs a detection pass on the
  22,640 development views (GPU).
- **Flags for touching and for who is on top:** their targets are not in the placement features. **Mirror
  consistency:** needs a mirrored SAM 3D Body pass (GPU).

Compute: CPU only, about 1.5 hours (E17's five placement trainings) plus minutes for the analyses; no new disk beyond
the per-view file (a few MB).
