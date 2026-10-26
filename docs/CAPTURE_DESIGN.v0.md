# Capture design v0 — a "teacher" rig for two-person 3D truth (draft, 2026-10-25)

Status: draft from measured evidence (updated 2026-10-26 with the diversity test). Needs Dhruva's review.

## What the capture is for
Not for filming every match. A small, calibrated multi-camera rig records **3D truth** of two people in contact (the
**teacher**). That truth trains and tests the **single-camera model** (the **student**), which then runs on ordinary
phone video (Kakashi, YouTube). Harmony4D proved the format, but it has only two grappling pairs and no ground work.

## What the evidence says (Harmony4D + ViCoS, 2026-10-22..25)
| Finding | Consequence for the capture |
|---|---|
| One camera + a depth map + a small learned correction places two grapplers to 2.9-3.2 cm on held-out moments of familiar people (`placemodel_*_results.md`) | the student works; it needs teacher data |
| 4x more camera angles of the same two people closed only 9% more of the gap (`placemodel_v2_results.md`) | more cameras per session is not the lever |
| On new people (MMA) the model keeps about 2/3 of its gain (`newpeople_results.md`: configuration -15% vs -22% on familiar people) | the method generalises, but not fully |
| One extra pair vs the same amount of the same pair: the new pair is better on every measure but below the pre-registered thresholds; on a third pair, equal (`diversity_results.md`) | public data has too few pairs to measure variety: **the pilot must include >= 5 pairs, so the effect can be measured** |
| Even perfect body placement leaves leg-entanglement F1 at about 0.7 and hand fighting at 0.4: limb errors inside tangles (oracle) | the truth must resolve **limbs** in tangles, not only where each body is |
| No dataset has 3D truth of ground work; ViCoS position labels can only check direction | ground work is the main gap: half or more of capture time |
| Cameras at 1.2-1.5 m see depth along the floor poorly (rays 3-8 deg below horizontal) | at least one **elevated** view for ground work |
| Two cameras about 90 deg apart cut distance error 10.2 -> 3.3 cm even with naive averaging | at least 4 views around the mat for the truth itself |

## Rig (proposal)
- **Cameras:** 4-6 synchronised and calibrated. 4 around the mat at about 90 deg spacing (1.2-1.8 m high, every limb
  seen by at least two cameras), plus 1-2 elevated (>= 2.5 m, looking down 30-60 deg) for ground work. Each athlete
  should be >= 600 px tall in every view (4K at <= 6 m, or 1080p at <= 3.5 m). 30-60 fps, fast shutter.
- **Sync:** genlock or timecode if available; otherwise an audio clap + flash at the start and end of every take, with
  sub-frame alignment checked.
- **Calibration:** intrinsics per camera (ChArUco, once); extrinsics per session from a ChArUco board laid on the mat,
  plus bundle adjustment. Record the mat plane (floor height for ground contact).
- **Truth reconstruction:** a person box per athlete per view -> ViTPose 2D -> triangulation -> two-person SMPL fit
  with a no-interpenetration term and identity tracking. Quality gates: reprojection error per view, held-out-camera
  check (predict one camera from the others), and human spot checks of the most tangled frames. (The pipeline exists in
  pieces here; joint multi-view fitting is the open engineering task.)

## Content
- **Athletes:** as many different people as possible; varied body types and weight classes; gi and no-gi; both
  sexes. Harmony4D's two grappling pairs were too few to measure how variety scales (`diversity_results.md`). The
  pilot should have >= 5 pairs, so the learning curve over pairs can be measured before scaling.
- **Positions:** standing (grips, clinch, takedowns) and ground (guard types, passing, half guard, mount, side
  control, back, turtle, leg entanglements, scrambles). Planned drills plus free rolling.
- **Settings:** >= 2-3 gyms (lighting, mats, backgrounds), so the student does not learn one room.
- **Consent and licence:** written consent per athlete, covering research and commercial use. Data stored privately.
  Decide the release licence before the first session.

## Pilot before scaling
One session: 1 gym, 2-3 pairs, the full rig, about 1 hour (half standing, half ground). Success criteria:
- reprojection error below a set threshold;
- held-out-camera prediction good enough to separate limbs in tangles (target: limb positions within about 3 cm);
- the student, trained with the pilot data, improves on held-out pilot moments.
Then scale athletes and gyms.

## Open questions for Dhruva
1. Gyms and athletes reachable, and the time per session.
2. Budget: action cameras or phones on tripods vs dedicated cameras.
3. Use: research only, or also product (Kakashi)? This decides the consent and licence wording.
