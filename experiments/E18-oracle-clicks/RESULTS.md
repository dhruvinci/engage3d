# E18 results: true 2D clicks lower tangled-limb error 15-25%; today's 2D model as clicks makes it 19-25% worse (pre-registration: `clicks_prereg.md`)

Run 2026-10-26 22:52 - 2026-10-27 02:27 on the laptop GPU, after the pre-registration was pushed (engage3d `7197e50`).
Code: `scripts/engage_h4d_clicks.py`. Outputs: `benchmarks/engage_h4d/clicks/` (`clicks.json`, per-view errors).
About 5 s per view (a 4K frame, two athletes, one backbone pass each, 8 click iterations).

## Criteria (median error of tangle points among the 8 limb joints, rigid-aligned over 14 points)
| | Comparison | Result | Verdict |
|---|---|---|---|
| O1 | held-out: `oracle2` vs `original` | 4.60 vs 5.39 cm: -14.7% (95% interval -16.9% to -12.7%); needed -25% | **fail** |
| O2 | held-out: share of `oracle2`'s gain reached by `vitpose2` | `vitpose2` 6.72 cm: +24.7% worse (interval +21% to +28%), so the share is negative | reported |
| O3 | held-out, other limb joints: `oracle2` vs `original` | 4.00 vs 4.24 cm: -5.7% | pass |
| O4 | new people (MMA): `oracle2` vs `original` | 4.69 vs 6.25 cm: -25.0% (interval -27.5% to -23.2%); needed -15% | pass |

## All numbers (median cm)
| | held-out: tangle | held-out: other | MMA: tangle | MMA: other |
|---|---|---|---|---|
| `original` (no clicks) | 5.39 | 4.24 | 6.25 | 4.61 |
| `vitpose2` (ViTPose+ clicks) | 6.72 | 4.62 | 7.42 | 4.97 |
| `oracle2` (true 2D, 2 clicks) | **4.60** | 4.00 | **4.69** | 4.30 |
| `oracle4` (true 2D, 4 clicks; exploratory) | 4.49 | 3.82 | 4.35 | 4.11 |
1,068 held-out views (3,760 tangle points); 1,484 MMA views (4,208). `original` reproduces E11's evaluation (5.39 vs 5.40).

## Reading (pre-registered: "O1 fails")
- **Even true 2D, at the model's trained budget of 2 clicks, removes only 15% of the held-out tangled-limb error**,
  short of the 25% bar; the rest is depth and ambiguity that 2D cannot settle, or more joints than 2 clicks can reach
  (4 clicks: -17%). On new people the true clicks remove 25%, more than on the familiar pair.
- **Today's 2D keypoints make SAM 3D Body worse inside tangles** (+25% held-out, +19% new people), and slightly worse
  elsewhere. Inside a tangle ViTPose+ takes the partner's limbs, and the clicks pull the body the wrong way. Clicks
  must not be used with an off-the-shelf 2D model in contact.
- Compared with E11: the decoder fine-tune gave -8% (held-out) and -9% (MMA); true clicks give -15% and -25%. The
  input route has the larger ceiling, if a 2D model can be made right inside tangles: E19 trains one.
