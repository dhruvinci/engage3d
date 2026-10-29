# Experiments

One folder per experiment. `PREREG.md` holds the question, method and pass criteria, committed before the run.
`RESULTS.md` holds the outcome against those criteria. Numbering is chronological.

| ID | Question | Verdict |
|---|---|---|
| [E00](E00-benchmark-baseline/) | How well does a single-camera 3D body model (SAM 3D Body) reproduce two-person relations? (Harmony4D) Do they match expert BJJ position labels? (ViCoS) | baseline: contact and body locks good; fine relations not; ViCoS 2/6 hypotheses |
| [E01](E01-joint-rigid-fit/) | Does a joint two-body rigid fit (2D keypoints + no interpenetration) fix placement? + oracle ablation, camera count, keypoint clicks | **fail**; oracle: relative depth is the missing quantity |
| [E02](E02-prompthmr/) | Does a whole-image multi-person model (PromptHMR) do better? | **fail** (better placement, much worse limbs in tangles) |
| [E03](E03-depth-map/) | Does a monocular depth map fix relative depth? | map alone **fail**; combined with the body model: best single-camera placement (exploratory) |
| [E04](E04-combined-depth-fresh-test/) | The combined depth on fresh views (pre-registered) | **pass** (distance 10.0 -> 3.9 cm, legs 0.42 -> 0.56); results in E03 |
| [E05](E05-vicos-combined-depth/) | Does the combined depth carry over to BJJ ground work? | **pass** (defining relations up, controls low) |
| [E06](E06-placement-model-v1/) | A learned two-person placement correction, held-out sequences | **pass** (distance 3.95 -> 3.17 cm) |
| [E07](E07-placement-model-v2-cameras/) | More camera angles of the same athletes (6k -> 27k views) | **fail by 0.04 cm** on one criterion; small consistent gains |
| [E08](E08-new-people/) | Does the correction transfer to new people? | **pass** (about 2/3 of the gain transfers) |
| [E09](E09-athlete-variety/) | Is athlete variety or data quantity the lever? | **fail** (new pair best on every measure, below thresholds; 2 pairs too few) |
| [E10](E10-limb-head/) | Can SAM 3D Body's own pose token fix limbs inside tangles? | **fail** (the token adds nothing inside tangles) |
| [E11](E11-decoder-finetune/) | Fine-tune SAM 3D Body's decoder on two-person contact (local pilot) | **fail** on the main criterion (2% below the cheap head, needed 15%); 8-9% better than the released model |
| [E12](E12-vicos-synced-phones/) | Ground work without 3D truth: do synchronised phones agree? | A1 **pass** (cross-phone agreement is a truth-free reliability score); B1 and C1 (MapAnything) **fail** |
| E13 | Geometry probe benchmark (VLMs vs vision encoders) | not run; folded into a later VLM project |
| [E14](E14-temporal-limbs/) | Can time fill in limbs hidden at a contact? | **fail** (more frames add nothing) |
| E15 | Hi4D and CHI3D (more interaction datasets) | dropped (2026-10-29): dataset access was not obtained (it requires an institutional registration) |
| [E16](E16-end-to-end-rfdetr/) | End to end with no ground-truth boxes: RF-DETR keypoints and masks, then SAM 3D Body | running (pre-registered) |
| [E17](E17-confidence/) | Can the pipeline flag the frames where its measurement is wrong? | C1, C3, C4 **pass**; C2 **fails** narrowly (flag AUROC 0.76-0.80) |
| [E17b](E17b-trust-flag-deeper/) | Making the trust flag usable: error bars, repair, clip summaries, the best camera | running (pre-registered) |
| [E18](E18-oracle-clicks/) | Do true 2D keypoints fix tangled limbs in SAM 3D Body? | O1 **fail** (-15%, needed -25%); O4 **pass** (new people -25%) |
| [E19](E19-contact-2d-keypoints/) | A 2D keypoint model trained on contact, feeding the clicks | **fail** (2D -21%, needed -30%; clicks still hurt) |
| [E20](E20-calmer-contact/) | Does it already work where contact is calmer (hugging, ballroom)? | **fail** as pre-registered; ballroom works, hugging hides limbs |
| [E21](E21-engagement-at-range/) | Engagement at range: sword fighting | **fail** on absolute bars; distance right to about 4% at 2.4 m |
| [E22](E22-layered-engagement/) | Six layers of understanding on whole clips, across a spectrum of engagement | demo, not pre-registered (0 identity switches; distance 1.5-5.1 cm on Harmony4D) |

E04 has no `RESULTS.md` (its results are in E03); E22 has no `PREREG.md` (a demo). Plan:
[`docs/RESEARCH_QUEUE.md`](../docs/RESEARCH_QUEUE.md).
