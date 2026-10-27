# Pre-registration: E16, end to end with no ground-truth boxes: RF-DETR keypoints and person masks trained on contact, feeding SAM 3D Body — written 2026-10-27, before any run on the test views

## Why
Every number so far started from ground-truth person boxes. The bottleneck is limbs hidden at contact (5.39 cm on
the held-out pair), and E14 showed time cannot fix them: SAM 3D Body is wrong in the same way across neighbouring
frames. New image evidence is needed. Two routes into SAM 3D Body remain:
- **2D keypoints as clicks.** True 2D helps (-15%, E18), but ViTPose+ trained on contact still hurts (+14%, E19).
- **A person mask as a prompt.** SAM 3D Body accepts a mask (it enters after the image backbone) and has never been
  tested here.
RF-DETR detects people, keypoints and masks with a model that reasons over the whole pair, so it may not merge
overlapping athletes. This experiment measures four things: how much real detection costs, how far a true mask can
take the limbs (the ceiling of the mask route), whether a contact-trained mask model gets there with no ground
truth, and whether RF-DETR's keypoints make clicks pay.

## Data
- **Training:** exactly E19's views. 8,000 views from the 10 development grappling2 sequences (contact-weighted, fixed
  seed) and 500 validation views from 025 / 035, read from E19's `targets.npz`.
  - Each view becomes one square crop around the pair: the union of both true boxes, side 1.3 x its longer edge,
    640 x 640. In training the crop is jittered (side x 0.85-1.2, centre +-8% of the side) to cover the detector's
    slack.
  - Labels per athlete:
    - the true box;
    - the 12 body joints projected through the true fisheye camera, marked hidden where the partner covers them
      (14% of joints in training, 17% in validation), with face points unlabelled as in E19;
    - the visible mask: both SMPL bodies painted far-to-near, so the nearer one owns a pixel.
  - Script: `scripts/engage_rfdetr_data.py`.
- **Test:** E18's held-out views (031 / 037 / 042 / 043, cameras 01/06/11/16; 1,068 views). New people: MMA 016,
  cameras 01/06/11/16. 2D is scored on all 1,484 MMA views; 3D on every 2nd view (742) to fit the night.
- **What was seen before writing this:** 200 development crops for speed and memory probes (1 epoch), a chain test on
  6 validation views (025 / 035), and the stock detector on one development frame. No test view has been scored.

## Method (settings fixed here; `benchmarks/engage_h4d/rfdetr/run_e16.sh`)
1. **Keypoint model:** RF-DETR keypoint preview (XLarge, Apache-2.0) at 576. 30 person queries instead of 100 (each
   image holds two people; halves memory and doubles speed). Batch 2 x 8 accumulation, lr 1e-4, 3 epochs, no
   multi-scale, EMA. Code: `scripts/engage_rfdetr_train.py`.
2. **Segmentation model:** RF-DETR Seg Medium at 432, 30 queries, batch 4 x 4, lr 1e-4, 3 epochs, no multi-scale.
3. **Checkpoint:** RF-DETR's own pick on the validation sequences, `checkpoint_best_total.pth`. With 3 epochs the
   keypoint model's pick is its final EMA weights.
4. **End-to-end detection (`scripts/engage_rfdetr_eval.py`):**
   - Stage A: the stock COCO RF-DETR Medium on the full 4K frame. Its pair box is the most confident person plus every
     person box touching it (grown by half its size), merged.
   - Stage B: the crop around that box goes to the fine-tuned keypoint and segmentation models, and to the released
     keypoint model; the two most confident people of each are kept.
   - Truth is used only to score: detections are matched one-to-one to the athletes at box IoU >= 0.5.
   - For clicks, each segmentation detection takes the keypoints of the keypoint detection that overlaps it most (no
     truth).
5. **SAM 3D Body arms (`scripts/engage_rfdetr_s3d.py`):**

   | Arm | Box | Mask | Clicks |
   |---|---|---|---|
   | `gtbox` | true | none | none (E18's `original`) |
   | `gtbox_truemask` | true | the true visible mask | none |
   | `detbox` | the segmentation model's | none | none |
   | `detbox_mask` | the segmentation model's | its mask | none |
   | `detbox_mask_kp2` | the segmentation model's | its mask | 2, from the fine-tuned RF-DETR keypoints (E18's loop) |

   Arms with the same box share one backbone pass; a check shows that matches a fresh pass to 0.0003 mm.
6. **Measure:** as E18. The 19 canonical points are aligned rigidly onto the truth over 14 points; the measure is the
   median error of the 8 limb joints inside a tangle. Comparisons are paired on athletes that every arm scored; 95%
   intervals come from 1,000 bootstrap resamples of views.
7. **2D:** the median error of the 8 limb joints inside tangles, in 4K pixels. Compared with E19's ViTPose+ (fine-tuned:
   19.2 px; released: 24.5 px), which used true boxes, on the same athletes.
8. **Sanity check:** `gtbox` must reproduce E18's 5.39 cm on the held-out views.

## Criteria (held-out views)
| | Comparison | Passes if |
|---|---|---|
| R1 | detection, end to end: athletes found (stage A + fine-tuned segmentation, IoU >= 0.5) | >= 95% |
| R2 | 2D, limb joints inside tangles: fine-tuned RF-DETR (detected boxes) vs fine-tuned ViTPose+ (true boxes) | >= 20% lower |
| R3 | mask ceiling: `gtbox_truemask` vs `gtbox`, tangle limb joints | >= 10% lower, interval below 0 |
| R4 | end to end: `detbox_mask` vs `gtbox`, tangle limb joints | lower, interval below 0 |
| R5 | cost of detection: `detbox` vs `gtbox`, tangle limb joints | at most 5% higher |
| R6 | clicks: `detbox_mask_kp2` vs `detbox_mask`, tangle limb joints | lower, interval below 0 |
MMA (new people) is reported with the same comparisons, without pass marks.

**Reading.**
- **R3 passes:** a person mask is a working route for hidden limbs. Where R4 lands says how much of that a contact-trained
  model delivers without any truth.
- **R4 passes:** a pipeline with no ground truth beats the ground-truth-box numbers reported so far, and becomes the
  pipeline.
- **R3 fails:** masks tell SAM 3D Body whose limb is whose but not where a hidden limb is. What remains is decoder
  training at scale (the E11 cloud run) or a second camera.
- **R1 and R5:** whether earlier results carry over to real use, or detection is a bottleneck of its own.
- **R6:** whether RF-DETR's keypoints, trained on contact, are exact enough for clicks (E19's were not).

Compute: GPU about 10 hours overnight (training about 7.5 h, detection about 0.5 h, SAM 3D Body about 2.4 h); disk
about 1 GB of crops plus about 1 GB of checkpoints. Weights in `~/Code/shared/models/rfdetr` (registry row added).
