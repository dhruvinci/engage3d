# Research queue, publishing and resources — living plan (created 2026-10-26)

Agreed with Dhruva on 2026-10-26: set up publishing and free resources **before** any new experiment, then run the
queue one experiment at a time (disk and the 6 GB GPU are the limits). Every experiment is pre-registered,
committed before it runs, and published with its results.

**Status, 2026-10-29.** E10-E14 and E17-E21 are done and published (see `experiments/README.md` for verdicts); E22 (a
demo) is published; E16 and E17b are done; E13 is folded into a later VLM project; E15 is dropped
(dataset access not obtained); E23 (engagement beyond contact) is drafted and awaits Dhruva's sign-off.

## 0. Setup (before any experiment)
| # | Step | Needs |
|---|---|---|
| S1 | Publishing system (section 3): GitHub repo + Hugging Face repos, a sync script with a privacy check, templates | Dhruva's OK on names / visibility |
| S2 | Free resources (section 4) | Dhruva's OK for archive-then-delete |
| S3 | Register for Hi4D and CHI3D (section 5): **dropped 2026-10-29**, registration requires an institutional account | Dhruva (applications are personal) |

## 1. Experiment queue (sequential; numbered after E00-E09, the first published set)
| # | Experiment | Question | Data | Compute (est.) | Disk (est.) | Publish |
|---|---|---|---|---|---|---|
| E10 | **SAM 3D Body limb scoping** (done 2026-10-26: **fail**, the token adds nothing inside tangles; `engagement/results/limbhead_results.md`) | How much limb error inside tangles can be removed from SAM 3D Body's own final pose token? (a small head: token -> limb correction, trained against ground truth) | ~40k Harmony4D views with saved tokens | CPU ~2-4 h | +0.5 GB | report + head weights (HF) |
| E11 | **SAM 3D Body decoder fine-tune** (top model priority; local pilot done 2026-10-26: **fail** on F1, -8% vs released, -2% vs the cheap head; next: oracle-click test, then RF-DETR keypoints in E16; `engagement/results/s3dft_results.md`) | Does fine-tuning the decoder on two-person contact data fix limbs in tangles, for new people too? Pilot locally; scale on a rented GPU if the pilot gains | Harmony4D (2 grappling pairs + MMA + hugging); + Hi4D / CHI3D when available | local pilot ~8-12 GPU-h; full run cloud (~1 day on one A100-class GPU, est. $20-50) | +3-6 GB (checkpoints) | model card + weights (HF, licence permitting), report |
| E12 | **ViCoS: depth understanding without 3D truth** (2026-10-26: A1 pass, combined depth most reliable, learned placement does not transfer; MapAnything ran and fails C1; `engagement/results/vicos_sync_results.md`) | (a) cross-phone agreement of predicted relations at the same instant (a reliability score on real ground work); (b) position-definition checks with the learned model; (c) feasibility of pseudo-3D from the synchronised phones with MapAnything | ViCoS (local) | GPU ~2-4 h | +2-4 GB (MapAnything weights + outputs) | report; metric code |
| E13 | **Geometry probe benchmark** (Qwen vs DINO vs SAM; reusable for future VLMs) (not run: folded into a later VLM fine-tuning project, 2026-10-29) | How much 3D understanding of two people in contact (whose pixel, who is in front, relative depth) can a fixed small decoder read out of each model's features? | Harmony4D-derived exact labels (rendered ownership + depth), ~5k-view probe set, held-out sequences | GPU ~4-6 h locally (vision towers); big-VLM layers in the cloud (~$10-20) | +5-10 GB features, deleted after | **benchmark code + leaderboard (public); benchmark dataset (private HF, on request)** |
| E14 | **Learned temporal model** (done 2026-10-27: **fail**, the errors persist across frames) | Does a model that reads time (a window of frames) beat per-frame placement, and how long a gap can it fill? Masked-gap protocol doubles as an eval for video tools (GVHMR, WHAM, TRAM, PromptHMR-Video) | Harmony4D at every 2nd frame (features exist) | CPU ~2-4 h; GPU only for denser frames | +1-2 GB | report + protocol |
| E15 | **Hi4D + CHI3D** (**dropped 2026-10-29**: access not obtained) | Variety learning curve with 20+ pairs (Harmony4D has only 2 grappling pairs); more limb-training data for E11 | Hi4D (20 pairs, 100 sequences), CHI3D (631 sequences) | GPU ~6-10 h processing | unknown until access: budget 30-60 GB, after S2 | report (data itself not redistributable) |
| E16 | **End-to-end with real detection** (done 2026-10-30: detection costs nothing where it finds the athletes (R5 pass) but finds 93%; masks and RF-DETR clicks do not help hidden limbs (R1-R4, R6 fail)) | All tests so far used ground-truth boxes. With detected boxes, how much worse is it? The honest real-world number | Harmony4D | GPU ~4-6 h | +3 GB | report |
| E17 | Confidence per frame (done 2026-10-27: flag AUROC 0.76-0.80; C2 fails narrowly) | Can the model say when it is wrong (ensemble spread vs error)? | existing | CPU ~1 h | — | report |
| E17b | **Trust flag, deeper** (done 2026-10-29: best camera of three halves the error (-46%, pass); error bars hold on the training pair (90.5%) but not new people (75%); repair and weighted summaries fail. The rest of this list waits on GPU passes or human review) | Make it more useful: error bars with a tested coverage guarantee (conformal calibration); the best camera per frame across 2-3 phones (simulated with Harmony4D's 20 cameras); trust-weighted summaries (time in contact, average distance); repairing flagged frames from trusted neighbours. Test it outside the lab: ViCoS synced phones (does the flag track disagreement between phones?); a blind human review of flagged vs trusted frames on public footage. Make it better: one flag per measurement (touching, who is on top, each limb); new signals (mirror consistency, RF-DETR keypoint uncertainty and detection confidence from E16); a stronger model; relative error at range | Harmony4D (existing features), ViCoS, public clips | CPU ~1 day | — | report + calibrated flag |

| E18 | **True 2D clicks into SAM 3D Body** (done 2026-10-27: -15% held-out / -25% new people; today's 2D clicks +19-25% worse) | | | | | |
| E19 | **ViTPose+ fine-tuned on contact** (done: 2D -21%, clicks still hurt +14%) | | | | | |
| E20 | **Calmer contact: hugging, ballroom** (done: ballroom works, hugging hides limbs) | | | | | |
| E21 | **Engagement at range: sword fighting** (done 2026-10-27: distance right to ~4% at 2.4 m; absolute bars fail; placement hurts depth at range) | | | | | |
| E22 | **Layered engagement** (done 2026-10-28: a demo, not pre-registered; six layers on whole clips without ground-truth boxes) | | | | | |
| E23 | **Engagement beyond contact** (spec v1 and pre-registration drafted 2026-10-28; awaiting Dhruva's sign-off on the definitions) | Centre of mass, base and stability, weight on the partner (proxy), leverage at contact, level and posture, head position, inside position, engagement over time: measurable from one camera? | E22's clips with truth | CPU | — | report |
| L1 | **Full-library auto-label (v1.1)** (not engage3d: an optional dataset run, last; Dhruva 2026-10-26) | Label every downloaded video at a sparse density so the library has a complete index before the videos are deleted | the 940 downloaded YouTube videos | GPU ~35-50 h (no 12.5 fps masks) + ~1 week of Astra | labels ~1-2 GB | private dataset v1.1 (HF) |

Not schedulable yet:
- **Hands and grips.** No dataset found with hand-level 3D truth of two people touching. CHI3D has SMPL-X, but access
  was not obtained. InterHand2.6M is one person's two hands. Inter-X has two-person motion capture
  with hands but no images (useful only for synthetic rendering). Blocked on data.
- **Own 3D-truth capture.** Dhruva is working on it (takes time; draft design: `docs/CAPTURE_DESIGN.v0.md`).
  Claude checks in on progress at milestones.

Removed: PromptHMR decoder fine-tune (2026-10-26). Our pipeline already places better than PromptHMR (2.9 vs
6.1 cm), and its whole-image encoding sees each athlete at about 200 px, a hard limit for limbs.

## 2. E14 design (learned temporal model)
- **Model:** a small temporal network (1D convolution or transformer) over a window of per-frame inputs: both bodies'
  canonical points after the combined depth, plus the depth-map and 2D features. It outputs the corrections for the
  centre frame. Window sizes 1 (= today's per-frame model), 5, 15 and 31 frames; same held-out sequences.
- **Gap-filling test:** hide k consecutive frames (and, separately, one athlete's limbs) and predict them from the
  neighbours. Plot error against gap length: how far time can carry information.
- **Tool evaluation:** the same masked-gap protocol and ground truth score external video models, so "time" tools
  can be compared on one scale.

## 3. Publishing system
**Where.**
- **GitHub** (`dhruvinci/engage3d`): code, the measurement spec, every experiment's
  pre-registration and results (`experiments/E<n>-<name>/{PREREG.md, RESULTS.md}`), `FINDINGS.md` (one line per
  finding, linked), licence (code: MIT; documents: CC BY 4.0), `CITATION.cff`, and data instructions (how to get each dataset).
  A pre-registration is committed and pushed before its run: the commit time is the proof.
- **Hugging Face** (`0xdhruva/...`):
  - model repos with model cards (training data, metrics, limits, licence). Released: placement model v1 (E06). Not
    released: v2 (E07 failed; no weights kept), the E10 head (failed; none kept), the E11 decoder (the SAM licence terms
    for derived weights are not yet checked);
  - **private** dataset repos for evals and benchmarks (Harmony4D-derived relation ground truth; no images, with a
    script to fetch them from Harmony4D). None created yet. All datasets are private (decided 2026-10-26);
    access on request;
  - Spaces: a layered-engagement gallery (E22, Harmony4D clips only) is built and awaits Dhruva's approval to publish;
  - **private** archive repos for closed-thread caches (section 4).
- **dhruva.life** links to both. The website session already has the briefs.

**How.** `kakashi-research` is not a git repository and holds private material, so the repo is a curated export:
- a sync script copies an allowlist of files (engagement code and docs) into `~/Code/<repo>`;
- it runs a privacy check (emails, tokens, personal names, Kakashi user data, absolute home paths) and refuses on a
  hit;
- it does not commit: the diff is reviewed, then committed by hand.

Weights and datasets go to Hugging Face through `huggingface_hub` (already logged in).

**What may be published (licences).**
| Source | Publishable | Not publishable |
|---|---|---|
| Harmony4D (MIT per dataset card) | derived labels, metrics, models trained on it | — |
| ViCoS (CC BY-NC-SA 4.0) | results, metric code; derived data only as NC-SA | images |
| Hi4D / CHI3D (research only; access not obtained) | results; models: check terms after access | data, derived data |
| MAMMA (two-person motion capture; no redistribution) | aggregate scores | media, derived data, models trained on it |
| SMPL / SMPL-X, SAM licences | code that loads them; derived metrics | the model files themselves |
| Kakashi-labelled videos from public sources | labels, intervals and masks in a private dataset repo; no media | media |
| Kakashi product (user) data | — | everything, including models trained on it (needs opt-in consent first) |

## 4. Resources
Disk: 423 GB; floor 30 GB free; about 43 GB free on 2026-10-29.

| Step | What | Frees | Status |
|---|---|---|---|
| A | My engagement intermediates whose results are written up: token re-run copy (tokens already inside the features), depth-test variant meshes, test-run mesh copies, ViCoS variant meshes (scores, records, maps kept) | ~6 GB | ready; consumers checked |
| B | Archive to **private** Hugging Face, verify checksums, then delete locally: Laya Vision caches (25 GB), Qwen VL-SEG phase 1 (12 GB), Kakashi viz runs (17 GB) | ~54 GB | **done 2026-10-26** (97 GB free after) |
| C | `dataset/media` is 85 GB, over its 70 GB cap | — | Dhruva's decision (never auto-deleted) |
After A+B (2026-10-26): about 100 GB free, since used by E10-E22's outputs.
Compute: one GPU job at a time; CPU jobs capped (`TORCH_THREADS`, at most one training next to a GPU job).

## 5. Dataset registration (dropped 2026-10-29)
Neither dataset was obtained: both require registering with an institutional (academic) account. E15 is dropped;
the two-person 3D truth here is Harmony4D (MAMMA is used for scoring only).

- **Hi4D** (ETH Zurich; 20 pairs, 100 sequences, 4D textured scans + multi-view video): application form at
  https://hi4d.ait.ethz.ch/ (project page https://yifeiyin04.github.io/Hi4D/). Access by email after approval.
- **CHI3D** (631 sequences, 728k 3D poses, contact annotations, SMPL-X/GHUM): https://ci3d.imar.ro/ ->
  Download (account login required); licence at https://ci3d.imar.ro/legal (research use).
