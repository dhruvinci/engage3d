# E16 results: end to end with no ground-truth boxes (RF-DETR keypoints and person masks, then SAM 3D Body)

**Status.**
- Pre-registered ([PREREG.md](PREREG.md), engage3d c62da37, before any run on the test views).
- Run overnight 2026-10-29/30 (`benchmarks/engage_h4d/rfdetr/run_e16.sh`):
  - keypoint training 21:06-01:10;
  - segmentation training 01:10-03:38;
  - detection 03:38-04:02;
  - SAM 3D Body 04:02-06:06.
- Outputs: `benchmarks/engage_h4d/rfdetr/eval/` (`detect.json`, `s3d.json`); every number below is in `data/`.
- **Disclosure.** A first launch on 2026-10-27 was stopped 14 minutes into keypoint training, before any evaluation.
  No test view was scored; its log is kept (`run_aborted_2026-10-27.log`). The settings are unchanged.
- **Sanity check passes:** `gtbox` gives 5.39 cm on the held-out views, E18's number.
- Measure: the median error of the 8 limb joints inside a tangle, after aligning each body rigidly over 14 points.
  Comparisons are paired on the athletes every arm scored (1,980 of 2,136 held-out). Intervals: 1,000 bootstrap
  resamples of views, as the relative change.

## Verdict (held-out grappling: 031 / 037 / 042 / 043, cameras 01/06/11/16; 1,068 views)
| | Comparison | Result | Bar | |
|---|---|---|---|---|
| R1 | athletes found end to end (stock pair finder, then the fine-tuned segmentation model, IoU >= 0.5) | **92.7%** (both athletes 89.8%; the pair box 100%) | >= 95% | **fail** |
| R2 | 2D inside tangles: fine-tuned RF-DETR (detected boxes) vs fine-tuned ViTPose+ (true boxes) | **25.2 px vs 19.2 px**: 31% worse | >= 20% lower | **fail** |
| R3 | mask ceiling: true mask vs none, true box | 5.34 vs 5.36 cm: **-0.4%** [-2.0, +1.3] | >= 10% lower, interval below 0 | **fail** |
| R4 | end to end: detected box + its mask vs true box | 5.45 vs 5.36 cm: **+1.8%** [-0.5, +3.7] | lower, interval below 0 | **fail** |
| R5 | cost of detection: detected box vs true box | 5.39 vs 5.36 cm: **+0.6%** [-1.3, +2.4] | at most 5% higher | **pass** |
| R6 | clicks from the fine-tuned RF-DETR keypoints vs none | 6.78 vs 5.45 cm: **+24%** [+21, +29] | lower, interval below 0 | **fail** |

## New people (MMA 016, cameras 01/06/11/16; 742 views; reported without pass marks)
| Comparison | Result |
|---|---|
| athletes found | 76.9% (both 71.2%; the pair box 100%) |
| 2D inside tangles | RF-DETR 19.1 px vs ViTPose+ 18.0 px (true boxes) |
| true mask vs none | 5.68 vs 6.07 cm: **-6.4%** [-9.8, -3.2] |
| detected box vs true box | 5.80 vs 6.07 cm: **-4.3%** [-7.0, -1.0] |
| detected box + mask vs true box | 5.82 vs 6.07 cm: **-4.1%** [-7.4, -0.5] |
| clicks vs none | 6.89 vs 5.82 cm: +18% [+12, +23] |

## Reading (as pre-registered)
- **R3 fails: a person mask does not tell SAM 3D Body where a hidden limb is.** With the true mask, tangled limbs move
  by -0.4% on the held-out pair, as stage 0 predicted (-3%, `STAGE0.md`). On new people a mask helps a little (-6%).
  What remains for hidden limbs is decoder training at scale or a second camera (in a multi-camera simulation on Harmony4D,
  `src/engage_multiview_sim.py`, a second phone halves placement error and cuts tangled limbs by 11%).
- **R5 passes: real detection costs nothing where it finds the athletes.** Detected boxes match true boxes within 1% on
  the held-out pair and are 4% better on new people. The earlier ground-truth-box results carry over to real use for
  the athletes that are found.
- **R1 fails: detection recall is the bottleneck of its own.** The stock pair finder always finds the pair, but the
  fine-tuned segmentation model finds 92.7% of athletes on the held-out pair and 76.9% on new people. Which frames it misses is not analysed
  here.
- **R4 fails: a pipeline with no ground truth matches, but does not beat, the ground-truth-box numbers** on the held-out
  pair (+1.8%, interval spans 0). On new people it is 4% better.
- **R2 and R6 fail: RF-DETR's keypoints, trained on contact, are not exact enough for clicks.** Inside tangles they
  are 25 px off against ViTPose+'s 19 px, and as clicks they make limbs 24% worse. Stage 0 set the bar for 2D to pay at
  10 px or less; every 2D model tested is two to three times off it.

## Reproduce
```
nohup bash benchmarks/engage_h4d/rfdetr/run_e16.sh > benchmarks/engage_h4d/rfdetr/run.log 2>&1 &   # about 9 h on 6 GB
```
Scripts: `engage_rfdetr_data.py`, `engage_rfdetr_train.py`, `engage_rfdetr_eval.py`, `engage_rfdetr_s3d.py`.
