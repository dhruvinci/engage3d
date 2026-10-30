# Measurement spec v1: engagement beyond contact (drafted 2026-10-28; signed off by Dhruva 2026-10-30)

v0 (`MEASUREMENT_SPEC.v0.md`) measures contact, threading, distance and loop balance. v1 adds the combat-sports
concepts of **gravity, base, weight, leverage, level, head and inside position**. Every measure is computed the same way
on the true meshes and on the single-camera pipeline's bodies; only the inputs differ. Status: **signed off
2026-10-30**; the five items that were open are decided as marked **[decided]**, and the support rule was changed
after a check on the true bodies (M2).

## Shared frame: the floor
- The floor is a plane fitted (RANSAC, 2 cm inliers, then least squares) to the 40 lowest points of each body in every
  3rd frame of the clip. Truth: from the true meshes. Single camera: from the predicted bodies (no truth). Feasibility
  check on 4 clips: the two floors' normals differ by 0.2-1.7 deg and the camera's height above them by 1-11 cm.
- "Up" is the floor normal. Heights are distances above the floor. The ground plane coordinates are the floor's 2D axes.

## M1 Centre of mass (CoM), per person (m)
- Segment model: **de Leva (1996)**, adjusted Zatsiorsky-Seluyanov parameters (J. Biomech. 29:1223-1230): 14 segments
  (head-neck, trunk, upper arms, forearms, hands, thighs, shanks, feet) with mass fractions and the CoM position along each
  segment. Sex unknown: the mean of de Leva's male and female tables **[decided]**.
- Segment end points from the canonical skeleton both models share (neck, head, shoulders, elbows, wrists, hands,
  hips, knees, ankles, toes, pelvis).
- Check on truth only: a uniform-density volume CoM of the true mesh, to report how far the segment model is from it.

## M2 Base of support and stability margin (cm)
- Support points: body vertices within 3 cm of the body's own lowest vertex, when that vertex is within 6 cm of the
  floor (feet, knees, hands, hips, back: anything on the mat) **[decided]**. A fixed 3 cm from the fitted floor was
  dropped: Harmony4D's true bodies sit up to +-3 cm off it, which empties the base. A body whose lowest vertex is more
  than 6 cm above the floor has no support.
- Base: the convex hull of the support points in the floor plane.
- Stability margin: signed distance from the CoM's floor projection to the base's edge (positive inside). With fewer
  than 3 support points spanning 2 cm, the margin is minus the distance to the nearest support point.
- Also reported: "stable" = margin > 0.

## M3 Weight on the partner (proxy; flag)
Kinematics cannot measure force, so this is a geometric proxy, named as such.
- Supported by the partner = touching the partner (v0 contact, 3 cm) **and** the CoM projection outside the person's
  own base by more than 5 cm **[decided]**, i.e. they would fall without the partner. A body with no support counts
  as outside its base.
- Top pressure = supported by the partner **and** the CoM above the partner's CoM. Per-clip summary: share of frames.

## M4 Leverage at contact (cm)
- For each contact (v0 region pairs at 3 cm; hands at 10 cm as in E22), on the partner's limb that is touched: the
  lever arm = the distance from the contact point to that limb's nearest proximal joint (wrist contact on a forearm ->
  the elbow; forearm or upper arm -> elbow and shoulder; shin -> knee; thigh -> hip; head or neck contact -> the neck).
- Reported: the longest lever arm per joint per frame, and which joint. Direction (which way it bends the joint) needs
  forces and joint axes; it is **not** in v1 **[decided: length only]**.

## M5 Level and posture
- CoM height above the floor (cm), each person; level difference = A minus B.
- Trunk angle from vertical (deg): pelvis -> neck against the floor normal.
- Hips above: A's pelvis higher than B's pelvis (flag; wrestling's level change).

## M6 Head position
- Head lateral offset from the partner's centreline (cm): the head's distance from the partner's midsagittal plane
  (through the partner's pelvis and neck, normal to the shoulder line).
- Head inside = |offset| < 15 cm and in front of the partner's chest **[decided]**.
- Head height relative to the partner's head and to the partner's pelvis (cm).

## M7 Inside / outside limb position (flags)
- In the partner's torso frame (origin pelvis; x shoulder line; y up the spine; z forward): a wrist or hand is inside
  when |x| is less than half the partner's shoulder width and it is in front of the partner (z > 0) within 40 cm; a
  knee or foot is inside when |x| is less than half the partner's hip width and in front.
- Reported per limb and frame; complements loop balance (v0) for frames, hooks and underhooks without threading.

## M8 Engagement over time
- Connection time: share of the clip in sustained contact; number and length of contact episodes (E22 events).
- Closing speed (m/s): the rate at which the pelvis distance falls over the 0.5 s before each contact episode starts
  (striking, takedown entries).

Not in v1: frames as a named structure (an extended arm braced against the partner's torso or neck) **[decided:
later]**; grips (no hand-level truth exists).
