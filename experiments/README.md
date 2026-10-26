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

Queued (see [`docs/RESEARCH_QUEUE.md`](../docs/RESEARCH_QUEUE.md)):
- E10 SAM 3D Body limb scoping;
- E11 SAM 3D Body decoder fine-tune;
- E12 ViCoS without 3D truth;
- E13 geometry probe benchmark (VLMs vs vision encoders);
- E14 learned temporal model;
- E15 more interaction datasets;
- E16 real detection end to end;
- E17 per-frame confidence.
