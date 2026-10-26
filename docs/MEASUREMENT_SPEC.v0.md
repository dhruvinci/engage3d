# Engagement measurement spec — v0 (draft for review, 2026-10-22)

**Scope (user, 2026-10-22):** any form of engagement between two people (grappling, hugging, dancing, striking,
caregiving, sport), not BJJ only. Loops (layer 2) are one family of measures; contact points and touching surfaces
(layer 1, to be refined below body-region level) matter as much. BJJ is one test domain.

**Purpose:** define what "understanding engagement" between two people means as quantities we compute, each with a
definition, a unit and an independent source of truth. Models are judged by how well they reproduce these quantities;
masks, keypoints and meshes are intermediate representations, not results.

**Status:** draft by Claude, not yet reviewed by the domain expert (the user). Definitions in §3 and §4 need a BJJ
practitioner's sign-off. Do not build scoring on a quantity until its definition is marked *agreed*.

## 1. Principles
- **Measurand first.** Each quantity serves a decision: a coach's feedback, a referee's score, a search query in a
  technique library, an injury-risk flag. A quantity with no consumer is dropped.
- **Truth from outside the pipeline.** Each quantity names a source of truth that no model in our pipeline produced:
  multi-view geometry, human expert labels, referee scoring, sensors. Reviewer dots on our own outputs are a test of
  ownership, not a source of truth for these quantities (they are placed where our output looked wrong).
- **Error bars, n >= 30, read the metric's definition** (carried over from movement-research).
- **Relations, not pixels.** BJJ positions are relations between named body parts of two people. They are computed
  from a two-person 3D body (SMPL: 24 joints, 6,890 vertices) in a gravity-aligned world frame.

## 2. Representation
Per frame, per athlete: an SMPL body (pose, shape, translation) in a world frame with z up and the floor at z = 0,
plus the camera. Body regions (16): head, chest, back, hips, and left/right upper arm, forearm, hand, thigh, shin,
foot. Regions come from SMPL skinning weights; chest/back are split by surface normal.

**Limb loops.** Each athlete has a *leg loop* (L ankle → L knee → L hip → R hip → R knee → R ankle, closed by the
ankle-to-ankle chord) and an *arm loop* (L wrist → L elbow → L shoulder → R shoulder → R elbow → R wrist, closed).
A body chain of the other athlete (a leg: hip → knee → ankle → foot; the torso: pelvis → neck; an arm; the head
and neck) **threads** a loop when it crosses the surface spanned by the loop. Threading is the core of every guard,
mount, back take and leg entanglement, and of the earlier observation that "the trapped foot of one athlete is
between the limbs of the other".

## 3. Quantities

### Layer 0 — each athlete alone
| ID | Quantity | Definition (v0) | Output |
|---|---|---|---|
| Q0.1 | Ground state | From pelvis height, torso angle to vertical, and chest normal: standing (pelvis > 0.7 m), crouched/kneeling, seated, supine, prone, on side, turtle (prone, knees and elbows under the torso) | class per frame |
| Q0.2 | Joint angles | Elbow and knee flexion, knee rotation, shoulder rotation, from the SMPL pose | degrees |

### Layer 1 — the pair's geometry
| ID | Quantity | Definition (v0) | Output |
|---|---|---|---|
| Q1.1 | Separation | Minimum mesh-to-mesh distance; pelvis-to-pelvis distance | m |
| Q1.2 | Contact map | 16 × 16 region pairs in contact (minimum distance < τ, τ calibrated against ground-truth mesh noise, ~2 cm) | binary matrix + contact area |
| Q1.3 | Top / bottom | In contact and on the ground: whose torso is higher, and whether it lies over the other's torso or hips (overlap of their floor-plane footprints) | A top / B top / neither |
| Q1.4 | Relative orientation | Angle between torso forward vectors and between head-to-hip axes: face to face, chest to back, perpendicular (T), head to feet (north-south) | angles + class |

### Layer 2 — threading (entanglement topology)
*Agreed with the domain expert, 2026-10-22:* threading matches how the user's coach explains control: you want more loops connected to your opponent than they have on you, and grappling is a game of creating and breaking those loops (it extends Danaher's inside-position game). Loop balance (Q2.6) is therefore the headline measure; position names (layer 3) are rules on top of it, added later.

| ID | Quantity | Definition (v0) | Example readings |
|---|---|---|---|
| Q2.1 | Leg-loop threading | Which of B's chains thread A's leg loop, and where along the chain (thigh, at the knee, shin, foot) | torso → guard or mount; one leg at the thigh → half guard or ashi garami past the knee line; head + one arm → triangle |
| Q2.2 | Arm-loop threading | Which of B's chains thread A's arm loop | head + arm → arm-triangle / head-and-arm; torso from behind → seatbelt; arm → underhook / overhook |
| Q2.3 | Mutual threading | Both directions at once | same-side legs both ways → 50/50 |
| Q2.4 | Lock closure | Ankles (or wrists) of the threading loop within ~15 cm, crossed | closed guard, triangle lock, body triangle |
| Q2.5 | Linking score | Gauss linking integral between the two leg loops: a continuous version of Q2.1 that degrades gracefully with noise | real number per frame |
| Q2.6 | **Loop balance** (headline) | Loops A holds on B (B's chains threading A's leg and arm loops, each weighted by its margin) minus loops B holds on A; with loop-creation and loop-breaking events over time | signed score per frame + event list |

### Layer 3 — positions (rules over layers 0–2)
The ViCoS classes, with who is dominant: standing, takedown, open guard, half guard, closed guard, 50/50, side
control, mount, back, turtle. v0 rules, for example: *mount* = B's torso threads A's leg loop, A on top, B supine;
*back* = B's torso threads A's leg loop, A's chest touching B's back; *side control* = B supine, chest-to-chest
contact, torsos roughly perpendicular, no leg-loop threading; *half guard* = one of B's legs threads A's leg loop,
B's torso does not. Leg-lock positions (outside ashi, inside sankaku / saddle, 50/50, cross ashi) need the expert's
definitions in terms of Q2.

### Layer 4 — time
| ID | Quantity | Definition (v0) |
|---|---|---|
| Q4.1 | Control episode | A dominant position held continuously >= 3 s (the IBJJF stabilisation rule for points) |
| Q4.2 | Transition | Change of position with a role flip: sweep, pass, takedown, back take, reversal |
| Q4.3 | Control time | Seconds in each dominant position per athlete |
| Q4.4 | Scramble intensity | Rate of contact-map and threading changes, relative pelvis velocity |

### Layer 5 — loading (submissions)
Q5.1: elbow extension under an armbar, knee extension and rotation under leg locks, shoulder rotation under kimura /
americana, from Q0.2 while the attacking threading holds. SMPL joints are not anatomical joints; a biomechanical
model (OpenCap / OpenSim) is the upgrade path.

## 4. Sources of truth
| Quantity | Harmony4D (multi-view 3D) | ViCoS (human position labels) | Own multi-view BJJ capture | Referee scoring | Independent black belts |
|---|---|---|---|---|---|
| Q0.1 ground state | standing / crouched only | indirect | yes | — | yes |
| Q0.2 joint angles | yes (standing) | — | yes | — | — |
| Q1.x pair geometry, contact | **yes** | — | yes | — | — |
| Q2.x threading | arm loops in clinch only | — | **yes** | — | yes |
| Q3 positions | — | **yes (120k frames, 2D)** | yes | indirect | yes |
| Q4 time, control | — | partial (sequential frames) | yes | **yes (points after 3 s control)** | yes |
| Q5 loading | standing only | — | yes | submission = tap | — |

**Measured 2026-10-22:** Harmony4D has no ground grappling. Across all 51 grappling sequences (sampled every 10th
frame of ~11,300) there is no frame with both athletes' pelvis below 0.45 m; the median pelvis height is ~1 m.
It validates contact maps, clinch arm-threading and joint angles, not guards, mounts or leg entanglements. Ground
positions are checked against ViCoS's human position labels now, and against our own capture later.

## 5. Open questions for the domain expert
1. Do the threading definitions in §2–§3 match how you would define guard, half guard, mount, back, and the leg-lock
   positions? Which leg-lock positions matter, and what distinguishes them in these terms?
2. Which decisions should these numbers serve first: coaching feedback, scoring, technique search,
   injury flags?
3. What accuracy is useful? Proposal: contact map region-pair F1 >= 0.8; threading >= 90% on frames where the truth
   is unambiguous; positions at or above agreement between two black belts.
4. Who could be the independent black belts, and can we record at a gym (3–4 phones + overhead 360)?

## 6. Status (2026-10-22): layers 0–2 on Harmony4D ground truth
Code: `scripts/engage_relations.py` (regions, contact map, threading with a margin), `scripts/engage_h4d_gt.py`,
`scripts/engage_render.py`, `scripts/h4d_ground_survey.py`. Results: `docs/engagement/results/`.

- 3,530 GT frames (16 grappling2 sequences, 2,789 frames; mma4, 741). Grappling: bodies within 2 cm in 75% of
  frames; most frequent contacts are forearm-forearm (39%), hand-hand (35%), hand-chest (34%): hand fighting.
- Threading, grappling, frames with a crossing / with margin >= 5 cm: torso through the other's arm loop (body
  lock) 359 / 352 (13%); arm through the other's arm loop 479 / 190 (7%); leg through the other's leg loop 40 / 15.
  The margin separates clear relations from grazing ones: body locks are robust, hand fighting mostly grazes.
- By eye (`h4d_gt_threading_examples.jpg`): body locks and a lift correct; a shin between the opponent's legs on a
  shot (penetration step) found unprompted; one false positive from an arm loop flattened into a line (now
  ambiguous by margin).
- Not yet done: Q2.5 linking score, Q0.1 ground-state classes, layer 3 position rules (need §5 answers), and the
  predicted side (monocular 3D from SAM 3D Body / PromptHMR on the same frames, scored against these GT relations).

## 7. First predicted-side results (2026-10-22)
Harmony4D (standing, 5,596 views): `results/h4d_s3d_results.md`. ViCoS (real BJJ ground work, pre-registered, 1,426
frames): `results/vicos_results.md`. In short: monocular 3D gets contact (F1 ~0.9) and body locks (0.80, 0.89 with a
4-camera vote) right; leg-between-legs is moderate (0.46); which parts touch (0.45), hand fighting (0.22) and loop
balance (sign 46%) are not reliable. On ViCoS the defining relations point the right way but are found in only
22-49% of mount / guard / back frames; 24-30% of mount, guard and side-control frames predict the same athlete twice.
