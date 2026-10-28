# E22: layered engagement from one camera: results (2026-10-27/28)

**Not pre-registered.** This is a demo and an engineering run, not a test: the pipeline was fixed and improved while
it ran, and one contact threshold was chosen after looking at the results (marked below). Numbers have no confidence
intervals (each clip is 10 s, 112-151 frames).

## What it does
One camera's video of two people is peeled into six layers over time; no ground truth goes in. Where a dataset has
multi-camera ground truth, it is used only to score each layer.

| Layer | Tool |
|---|---|
| 1 People + identities | RF-DETR person boxes; the engaged pair chosen by motion, size and closeness; SAM 3.1 video tracking seeded where the two stand apart |
| 2 Segmentation | SAM 3.1 masks, one per person |
| 3 2D skeletons | PMPose (BBoxMaskPose v2, mask-conditioned; visibility per joint) |
| 4 3D bodies in one space | SAM 3D Body per person from box and mask, translation re-solved through the clip's lens, then placed by the combined depth of E03/E04 (MoGe-2's relative depth of the pair blended 0.7 with SAM 3D Body's) |
| 5 Contact | per frame: smallest gap, touching body regions (16), embrace (a torso threading the partner's arm loop); over time: sustained touches (at least 0.5 s) and brief touches (shorter: strikes, taps) |
| 6 Context | GPT-6 Astra (a vision-language model) writes the interaction type, phases, labels and disagreements from 6 keyframes and the measured timeline, citing the measured events |

Any video works: a clip is a folder of frames with a lens. For datasets the lens is the calibration; for ordinary
video it is estimated by MoGe-2 (median focal length over 5 frames). A queue runs one clip at a time, one GPU stage at a
time, and a clip only counts when every layer passes a quality gate: both people tracked (a person unmasked in at most
15% of frames), two distinct people, both people masked (the smaller mask at least 10% of the larger in 80% of frames),
skeleton confidence, 3D fit residual, the two bodies in one plausible space, and a narration.

## Data
- **Scored against ground truth (12 clips):** 8 Harmony4D clips (Khirodkar et al., NeurIPS 2024; MIT per the dataset
  card): hugging `002_hugging` cam01 and `001_hugging` cam06 (train split), sword `009_sword2` cam01, ballroom
  `009_ballroom2` cam01, karate `007_karate` cam06 (train split), MMA `009_mma5` cam01 and `016_mma4` cam01, grappling
  `043_grappling2` cam01; and 4 MAMMA evaluation clips (West Coast Swing: sugar push, roll in/out, underarm turn,
  improvised; camera IOI_01; ground truth SMPL-X fits posed into the camera; Max Planck Institute for Intelligent
  Systems, mamma.is.tue.mpg.de; numbers only, MAMMA media is not redistributed).
- **No ground truth (17 clips):** ordinary single-camera videos (handheld and fixed, broadcast and phone, 480p-1080p):
  greetings, caregiving, acroyoga, stretching, partner dance, fencing, push-hands, sumo, judo, wrestling, MMA,
  BJJ, pad work. They are not redistributed. All 17 passed every layer gate; the pipeline's health numbers are
  reported for them, not accuracy.

## Results against ground truth
Touching uses the adopted rule (a hand or forearm counts within 10 cm, a body within 3 cm); strict 3 cm in brackets.

| clip | both found | identity switches | mask IoU | joints visible / hidden (px) | limbs (cm) | distance between the two, error (cm) | touching F1 | frame agreement |
|---|---|---|---|---|---|---|---|---|
| hugging (H4D 002) | 1.00 | 0 | 0.82 | 20 / 45 | 2.9 | 2.9 | 0.97 (0.96) | 0.97 (0.97) |
| hugging, second pair (H4D 001) | 0.98 | 0 | 0.80 | 18 / 27 | 3.2 | 1.5 | 1.00 (0.99) | 1.00 (0.99) |
| sword at range (H4D 009) | 0.99 | 0 | 0.69 | 15 / 18 | 4.0 | 4.3 | no true contact | 1.00 (1.00) |
| ballroom (H4D 009) | 0.87 | 0 | 0.74 | 7 / 16 | 3.4 | 3.1 | 0.96 (0.84) | 0.92 (0.74) |
| West Coast Swing, sugar push (MAMMA) | 0.99 | 0 | 0.90 | 13 / 19 | 4.7 | 32.7 | 0.92 (0.91) | 0.89 (0.87) |
| West Coast Swing, roll in/out (MAMMA) | 0.93 | 0 | 0.90 | 13 / 21 | 4.8 | 18.8 | 0.86 (0.76) | 0.79 (0.68) |
| West Coast Swing, underarm turn (MAMMA) | 0.91 | 0 | 0.89 | 15 / 20 | 5.1 | 18.9 | 0.84 (0.74) | 0.76 (0.65) |
| West Coast Swing, improvised (MAMMA) | 0.99 | 0 | 0.89 | 12 / 17 | 5.2 | 15.6 | 0.91 (0.81) | 0.84 (0.69) |
| karate (H4D 007) | 0.99 | 0 | 0.73 | 18 / 14 | 3.7 | 4.5 | 0.59 (0.53) | 0.93 (0.94) |
| MMA at range (H4D 009_mma5) | 0.99 | 0 | 0.75 | 12 / 17 | 4.8 | 4.6 | 0.75 (0.76) | 0.89 (0.90) |
| MMA clinch (H4D 016) | 1.00 | 0 | 0.71 | 15 / 16 | 4.5 | 5.0 | 0.92 (0.79) | 0.90 (0.77) |
| grappling (H4D 043) | 0.99 | 0 | 0.78 | 14 / 21 | 4.2 | 5.1 | 0.98 (0.98) | 0.98 (0.97) |

Joints in pixels on the clip's own frames (Harmony4D 2160p, MAMMA 1504p).

**What it shows.**
- Tracking holds across the spectrum: 0 identity switches on all 12 clips with truth.
- Difficulty depends on the layer. Hugging is easy for contact (F1 0.97-1.00) but its hidden joints are the worst
  (27-45 px). Strikes are hard for contact (F1 0.59-0.75: brief and often missed by a few cm). Partner dance seen by
  MAMMA's camera is hard for depth (the distance between the dancers 16-33 cm off: one dancer is often in front of the
  other, and SAM 3D Body pulls both body sizes toward an average, so their depths converge; the depth map has too few
  visible pixels of the hidden dancer to correct it). Harmony4D's distance error stays 1.5-5.1 cm.
- Brief touches: on karate the measured brief touches match the true ones 6 of 6, each within 0.2 s (measured
  2.9 / 3.7 / 4.0 / 4.3-4.5 / 7.1 / 7.6 s; true 2.9 / 3.7 / 4.1 / 4.4-4.5 / 6.9-7.1 / 7.7 s).

## Placement step (combined depth) in the live pipeline
The combined depth (E03, tested fresh in E04) was added as a stage after SAM 3D Body; weight 0.7 as fitted in E03, not
tuned here. Before -> after on the five truth clips that had run (same tracks, masks, skeletons and bodies):

| clip | distance error (cm) | pelvis error (cm) | touching F1 | region pairs F1 |
|---|---|---|---|---|
| hugging (H4D 002) | 4.5 -> 2.9 | 6.5 -> 5.6 | 0.86 -> 0.96 | 0.58 -> 0.73 |
| MMA clinch | 9.8 -> 5.0 | 9.3 -> 7.6 | 0.70 -> 0.79 | 0.18 -> 0.27 |
| grappling | 13.9 -> 5.1 | 12.5 -> 10.0 | 0.79 -> 0.98 | 0.30 -> 0.60 |
| ballroom | 7.4 -> 3.1 | 11.3 -> 13.4 | 0.54 -> 0.84 | 0.16 -> 0.37 |
| West Coast Swing (MAMMA) | 33.6 -> 32.7 | 20.6 -> 18.0 | 0.89 -> 0.91 | 0.14 -> 0.19 |

Better on every clip for distance and contact (strict 3 cm touching). The Harmony4D clips come from the rig the weight
was fitted on; MAMMA, the only other rig, barely moves.

## Contact threshold for hands (exploratory: chosen after seeing the data)
Reconstructed hands are the least accurate part: a held hand often comes out several cm from the partner.
- On the 11 truth clips available at the time, the body distance (1-4 cm) made no difference; the hand / forearm
  distance did. Mean frame agreement by hand distance: 3 cm 0.844, 5 cm 0.867, 8 cm 0.893, **10 cm 0.908**, 12 cm
  0.918, 15 cm 0.929, 20 cm 0.936, 25 cm 0.935. The two low-contact clips (sword, karate) lose as it loosens: 0.970 at
  3-5 cm, 0.963 at 8-10 cm, 0.947 at 15 cm, 0.930 at 20 cm. Leave-one-clip-out on a 3-15 cm grid picks 15 cm in all
  11 folds.
- **Adopted: 3 cm for bodies, 10 cm for a hand or forearm (predictions only; truth stays at 3 cm)**, the loosest that
  keeps the low-contact clips within a point of the strict rule.
- Fresh check (`009_mma5`, not in the sweep; striking at range, true contact 25% of frames): **the hand rule does not
  help there**: agreement 0.900 strict vs 0.887 adopted; F1 0.762 vs 0.746. It helps sustained hand contact (holds,
  grips, dance) and costs about a point on brief contact. Both rules are reported for every clip.

## Failures and fixes along the way
- **The wrong pair in a crowd.** The engaged pair is chosen where the two stand apart. A couple that never separates
  (close-embrace dance), a handheld camera that makes everyone appear to move, or a moment when one of the pair drops
  below the detector's confidence let onlookers win. Fixes: the crowd's median image motion is removed before judging
  who moves; the start pair must be at the scale of the clip's most engaged pair; a pair that never stands apart
  starts at its most separable engaged frame. One clip still needed a **one-click prompt** (a point on each person in
  the first frame), recorded with the clip: a user prompt, not ground truth.
- **A late seed on the wrong person.** SAM 3.1 is seeded at the first and last frames where the box tracker has the
  pair apart; the box tracker can drift onto an onlooker, or swap the two, by the end. Fixes: the late seed must match
  the frame's most engaged pair, person by person, and it is relabelled when torso clothing colour matches the first
  seed better swapped.
- **One mask holding both people** when one is mostly hidden at the seed frame: both are then seeded from their boxes,
  each with a positive point where the partner's box does not reach and a negative point on the partner's. The
  "both people masked" gate was added after this passed the earlier gates.
- **Brief touches were invisible** to the timeline (smoothed into 0.5 s runs) while their contact regions were still
  listed, which the narrator kept flagging. They are now their own event.
- **Documented failure:** a 304x224 VHS clip. The detector finds neither player (inverted, blurred bodies), and even
  from two drawn boxes SAM 3.1 drifts onto a standing onlooker. Retired; resolution below what this pipeline handles.

## Caveats
- SAM 3D Body was trained on Harmony4D (all 24 subjects), so absolute Harmony4D numbers are likely optimistic.
- The hand threshold was chosen after seeing the data (above); strict numbers are kept beside it.
- 12 clips of 10 s; no confidence intervals; not pre-registered.

## Reproduce
```
# one clip end to end (Harmony4D):     src/run_queue.sh with a queue.tsv line:  NAME  SEQ  CAM  first:last:step  note
# any video:  python src/engage_clip_make.py video --name NAME --src VIDEO_OR_URL --start S --dur 10
#             then a queue.tsv line:  NAME  <clips dir>/NAME  -  -  note
# stages:     python src/engage_demo_layers.py <stage> --seq SEQ_OR_CLIP_DIR [--cam camNN --frames a:b:s] --out DIR
#             lens, detect, track, pose, segment, samtrack, pose_pmpose, body3d, place, contact, context, score, render, check
```
