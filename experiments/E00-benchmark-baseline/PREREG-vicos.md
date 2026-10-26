# Pre-registration: do predicted 3D relations match expert BJJ position labels? (ViCoS) — written 2026-10-22, before running

**Data.** ViCoS BJJ positions (Hudovernik & Skočaj 2022): 120,279 frames from real sparring, position labels assigned by
people; 2D poses auto-detected (frames where the detector lost an athlete have no pose; we can only use frames with both,
so the hardest entanglements are under-represented: a coverage caveat that travels with every number below).
Label convention, measured from the 2D poses (athlete N's torso higher in the image): mount N, side control N, turtle N:
athlete N is on top (100% / 100% / 99%); closed / half / open guard N: athlete N is the guard player, on the bottom
(0-1% higher); back N: not decidable by height (56-61%). Below, D = the numbered athlete, O = the other.

**Method.** Up to 80 frames per class (18 classes), spread over the 16 video streams, at least 30 frames apart within a
stream; boxes from each athlete's 2D keypoints (padded 20%); SAM 3D Body (body decoder) per athlete with its default
focal length (no intrinsics are known; both athletes share the same assumption, and movement-research found the distance
between two people at similar depth is robust to focal error). Relations from engage_relations.py, margin >= 5 cm counted.

**Hypotheses (relation rates on predicted bodies, per position, both numbered variants pooled):**
- H1 mount: O.torso through D.leg_loop in >= 60% of frames; in side control <= 20%.
- H2 closed guard: O.torso through D.leg_loop in >= 60% of frames.
- H3 half guard: one of O's legs through D.leg_loop in >= 40% of frames, more often than in closed guard.
- H4 50/50: mutual leg threading (a leg of each through the other's leg loop) more often than in any other position.
- H5 standing, takedown, turtle, side control: any torso or leg through either athlete's leg loop in <= 20% of frames.
- H6 back: a torso through the other athlete's leg loop (either direction, since D's role is unknown) in >= 40%.

**Falsifiers.** If H1/H2 rates are near the H5 rates, the predicted bodies do not carry the relation that defines these
positions (model failure, or the threading definition does not survive monocular error). If H5 fails (leg-loop
threading common in standing / side control), the measure has a false-positive problem.
