# Pre-registration: E14, can time fill in limbs hidden at a contact? — written 2026-10-27, before any run on the test views

## Why
Limbs hidden at the point of contact are the remaining bottleneck (E20: 6.6 cm in hugging, 5.4 cm in grappling,
against 3.8 cm when the contact is visible). A limb hidden now was usually visible a moment before or after. This tests
whether a small model reading a window of frames corrects hidden limbs better than one reading a single frame, and how
long a gap time alone can bridge (the masked-gap protocol, also a yardstick for video models later).

## Data
`benchmarks/engage_h4d/place_feats2`, every 2nd ground-truth frame per camera (about 15 per second).
- **Training:** the 12 development grappling2 sequences x 20 cameras.
- **Held-out test:** sequences 031 / 037 / 042 / 043, cameras 01/06/11/16 (the E06 views).
- **New people:** MMA 016, cameras 01/06/11/16.
- The code was checked on development sequences only.

## Method (`scripts/engage_temporal.py`; settings fixed here)
- **Input, per athlete and frame:** a window of 1, 5, 15 or 31 frames of the same camera. Each frame holds the athlete's
  own 19 points and the other's (camera frame, centred on the athlete's centre-frame centroid), the ViTPose+ confidence
  of 17 keypoints, and a "present" flag.
- **Output:** a correction of the centre frame's own pose. Target: the truth aligned rigidly onto the prediction over
  the 14 points both body models define alike, as in E11.
- **Model:** MLP, 2 x 512, GELU, dropout 0.2, AdamW (lr 1e-3, weight decay 1e-3, cosine), 40 epochs, batch 512, 2 seeds
  averaged. Window 1 is the per-frame head.
- **Masked gap:** a window-15 head is trained with the centre frame and up to 7 frames around it hidden at random. At
  test, the centre +-g frames are hidden, g = 0, 1, 3, 7 (up to about 1 s).
- **Measure:** E11's (median error of the 8 limb joints inside a tangle, and of the others, rigid-aligned over 14
  points). Intervals: 1,000 bootstrap resamples of views.

## Criteria (primary window fixed in advance: 15 frames, about 1 s)
| | Comparison | Passes if |
|---|---|---|
| T1 | held-out: window 15 vs window 1, tangle limb joints | >= 10% lower, 95% interval below 0 |
| T2 | held-out: where the gain lands | relative gain on tangle limb joints >= relative gain on the other limb joints |
| T3 | held-out: masked window-15 head with the centre +-3 frames hidden (about 0.5 s) vs window 1 | at most 10% higher (time alone carries the limbs across half a second) |
| T4 | MMA (new people): window 15 vs window 1, tangle limb joints | >= 5% lower |

**Reading.**
- **T1 passes:** time fixes part of the hidden-limb error. A temporal stage joins the pipeline, and video models (GVHMR,
  WHAM, PromptHMR-Video) can be scored with the same masked-gap protocol.
- **T1 fails:** the per-frame points carry too little over time; hidden limbs need image evidence (E16, the decoder).
- **T3** says how long a contact can hide a limb before time stops helping.

Compute: CPU only, about 1.5-2.5 hours, after E17 finishes (never alongside another training).

## Amendment (2026-10-27, 13:40, before any run on the test views)
A code review before the run found that the masked-gap test, as written above, could not measure what T3 asks. These
fixes come only from reading the design and a smoke run on one development sequence; no test view has been scored.
1. **The masked head never sees the frame it predicts.** As written, its correction was added to the hidden frame's
   own per-frame prediction, so it was not "time alone". It now starts from the mean pose of the nearest visible frame
   on each side and learns a correction to that. The plain interpolation (no learning) is reported too.
2. **31 frames, not 15.** A 15-frame window (+-7) hiding +-7 frames leaves nothing visible, so the masked head reads 31
   frames (+-15). At the longest gap, 8 frames stay visible on each side.
3. **g = 0 hides the centre frame only**, as in training ("the centre +-g frames are hidden").
4. **T3 is scored on views with a visible frame on both sides of the +-3 gap.** Held-out grappling sequences are short
   (24-163 frames per camera at 15 per second), so near a sequence's ends one side is missing and the test becomes
   extrapolation. All views are reported as secondary.

T3 now reads: held-out, views with both sides visible, the masked 31-frame head with the centre +-3 frames hidden
(7 frames, about 0.5 s) vs window 1 on the same views: at most 10% higher. Thresholds and T1, T2 and T4 are unchanged.
