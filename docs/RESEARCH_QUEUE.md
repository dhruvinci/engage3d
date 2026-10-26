# Research queue, publishing and resources — living plan (created 2026-10-26)

Agreed with Dhruva on 2026-10-26: set up publishing and free resources **before** any new experiment, then run the
queue one experiment at a time (disk and the 6 GB GPU are the limits). Every experiment is pre-registered,
committed before it runs, and published with its results.

## 0. Setup (before any experiment)
| # | Step | Needs |
|---|---|---|
| S1 | Publishing system (section 3): GitHub repo + Hugging Face repos, a sync script with a privacy check, templates | Dhruva's OK on names / visibility |
| S2 | Free resources (section 4) | Dhruva's OK for archive-then-delete |
| S3 | Register for Hi4D and CHI3D (section 5) | Dhruva (applications are personal) |

## 1. Experiment queue (sequential; numbered after the published experiments E00-E09)
| # | Experiment | Question | Data | Compute (est.) | Disk (est.) | Publish |
|---|---|---|---|---|---|---|
| E10 | **SAM 3D Body limb scoping** | How much limb error inside tangles can be removed from SAM 3D Body's own final pose token? (a small head: token -> limb correction, trained against ground truth) | ~40k Harmony4D views with saved tokens | CPU ~2-4 h | +0.5 GB | report + head weights (HF) |
| E11 | **SAM 3D Body decoder fine-tune** (top model priority) | Does fine-tuning the decoder on two-person contact data fix limbs in tangles, for new people too? Pilot locally; scale on a rented GPU if the pilot gains | Harmony4D (2 grappling pairs + MMA + hugging); + Hi4D / CHI3D when available | local pilot ~8-12 GPU-h; full run cloud (~1 day on one A100-class GPU, est. $20-50) | +3-6 GB (checkpoints) | model card + weights (HF, licence permitting), report |
| E12 | **ViCoS: depth understanding without 3D truth** | (a) cross-phone agreement of predicted relations at the same instant (a reliability score on real ground work); (b) position-definition checks with the learned model; (c) feasibility of pseudo-3D from the synchronised phones with MapAnything | ViCoS (local) | GPU ~2-4 h | +2-4 GB (MapAnything weights + outputs) | report; metric code |
| E13 | **Geometry probe benchmark** (Qwen vs DINO vs SAM; reusable for future VLMs) | How much 3D understanding of two people in contact (whose pixel, who is in front, relative depth) can a fixed small decoder read out of each model's features? | Harmony4D-derived exact labels (rendered ownership + depth), ~5k-view probe set, held-out sequences | GPU ~4-6 h locally (vision towers); big-VLM layers in the cloud (~$10-20) | +5-10 GB features, deleted after | **benchmark code + leaderboard (public); benchmark dataset (private HF, on request)** |
| E14 | **Learned temporal model** | Does a model that reads time (a window of frames) beat per-frame placement, and how long a gap can it fill? Masked-gap protocol doubles as an eval for video tools (GVHMR, WHAM, TRAM, PromptHMR-Video) | Harmony4D at every 2nd frame (features exist) | CPU ~2-4 h; GPU only for denser frames | +1-2 GB | report + protocol |
| E15 | **Hi4D + CHI3D** (when access is granted) | Variety learning curve with 20+ pairs (Harmony4D has only 2 grappling pairs); more limb-training data for E11 | Hi4D (20 pairs, 100 sequences), CHI3D (631 sequences) | GPU ~6-10 h processing | unknown until access: budget 30-60 GB, after S2 | report (data itself not redistributable) |
| E16 | **End-to-end with real detection** | All tests so far used ground-truth boxes. With tracked boxes (SAM 3.1), how much worse is it? The honest real-world number | Harmony4D | GPU ~4-6 h | +3 GB | report |
| E17 | Confidence per frame (optional, low priority) | Can the model say when it is wrong (ensemble spread vs error)? | existing | CPU ~1 h | — | report |

Not schedulable yet:
- **Hands and grips.** No dataset found with hand-level 3D truth of two people touching. CHI3D has SMPL-X (hand
  detail to be checked after access). InterHand2.6M is one person's two hands. Inter-X has two-person motion capture
  with hands but no images (useful only for synthetic rendering). Blocked on data.
- **Own 3D-truth capture.** Dhruva is working on it (takes time; draft design: `engagement/CAPTURE_DESIGN.v0.md`).
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
- **GitHub** (`dhruvinci/<repo>`, name TBD, e.g. `engage3d`): code, the measurement spec, every experiment's
  pre-registration and results (`experiments/E<n>-<name>/{PREREG.md, RESULTS.md}`), `FINDINGS.md` (one line per
  finding, linked), licence (code: Apache-2.0 or MIT), `CITATION.cff`, and data instructions (how to get each dataset).
  A pre-registration is committed and pushed before its run: the commit time is the proof.
- **Hugging Face** (`0xdhruva/...`):
  - model repos (placement model v1/v2, the E10 head, E11 decoder) with model cards: training data, metrics, limits,
    licence;
  - **private** dataset repos for evals and benchmarks (Harmony4D-derived relation ground truth and the E13 probe
    labels; no images, with a script to fetch them from Harmony4D). All datasets are private (decided 2026-10-26);
    access on request;
  - later a Space with the E13 leaderboard;
  - **private** archive repos for closed-thread caches (section 4).
- **dhruva.life** links to both. The website session already has the briefs.

**How.** `kakashi-research` is not a git repository and holds private material, so the repo is a curated export:
- a sync script copies an allowlist of files (engagement code and docs) into `~/Code/<repo>`;
- it runs a privacy check (emails, tokens, personal names, Kakashi user data, absolute home paths) and refuses to
  commit on a hit;
- it commits.

Weights and datasets go to Hugging Face through `huggingface_hub` (already logged in).

**What may be published (licences).**
| Source | Publishable | Not publishable |
|---|---|---|
| Harmony4D (MIT per dataset card) | derived labels, metrics, models trained on it | — |
| ViCoS (CC BY-NC-SA 4.0) | results, metric code; derived data only as NC-SA | images |
| Hi4D / CHI3D (research only) | results; models: check terms after access | data, derived data |
| SMPL / SMPL-X, SAM licences | code that loads them; derived metrics | the model files themselves |
| Kakashi-labelled videos from public sources | labels, intervals and masks in a private dataset repo; no media | media |
| Kakashi product (user) data | — | everything (needs opt-in consent first) |

## 4. Resources
Disk: 423 GB; floor 30 GB free. **Now 43 GB free** (13 GB usable).

| Step | What | Frees | Status |
|---|---|---|---|
| A | My engagement intermediates whose results are written up: token re-run copy (tokens already inside the features), depth-test variant meshes, test-run mesh copies, ViCoS variant meshes (scores, records, maps kept) | ~6 GB | ready; consumers checked |
| B | Archive to **private** Hugging Face, verify checksums, then delete locally: Laya Vision caches (25 GB), Qwen VL-SEG phase 1 (12 GB), Kakashi viz runs (17 GB) | ~54 GB | needs Dhruva's OK |
| C | `dataset/media` is 85 GB, over its 70 GB cap | — | Dhruva's decision (never auto-deleted) |
After A+B: about 100 GB free, enough for E10-E16 and a Hi4D/CHI3D subset.
Compute: one GPU job at a time; CPU jobs capped (`TORCH_THREADS`, at most one training next to a GPU job).

## 5. Dataset registration
- **Hi4D** (ETH Zurich; 20 pairs, 100 sequences, 4D textured scans + multi-view video): application form at
  https://hi4d.ait.ethz.ch/ (project page https://yifeiyin04.github.io/Hi4D/). Access by email after approval.
- **CHI3D** (631 sequences, 728k 3D poses, contact annotations, SMPL-X/GHUM): https://ci3d.imar.ro/ ->
  Download (account login required); licence at https://ci3d.imar.ro/legal (research use).
