# E12 results: synchronised phones agree beyond the label; the combined depth is the more reliable pipeline; the learned placement does not carry over (pre-registration: `vicos_sync_prereg.md`)

Run 2026-10-26 20:24-21:48, after E11 had finished and after the pre-registration was pushed (engage3d `23788ed`).
Code: `scripts/engage_vicos_sync.py`, `engage_vicos_s3d_clicks.py --ids`, `engage_vicos_depth.py --save-maps`,
`engage_vicos_score.py`. Outputs: `benchmarks/engage_vicos/sync/` (`agree.json`, `score_*`).
1,223 images of 492 synchronised moments: bodies 43 min, depth maps 35 min (GPU, one at a time), the rest on the CPU.

## Criteria
| | Comparison | Result | Verdict |
|---|---|---|---|
| A1 | `blend`: same moment vs another moment with the same label (identity-free Dice) | 0.275 vs 0.142: +0.132 (95% interval +0.091 to +0.175) | **pass** |
| A2 | same-moment agreement: `blend` vs `clicks` | 0.275 vs 0.159 (+0.116) | **the combined depth is the more reliable pipeline** |
| B1 | `place` vs `blend`, defining relations | mount 0.33 -> 0.35, closed guard 0.40 -> 0.40, half guard 0.22 -> 0.28: up by >= 0.05 in 1 of 3 (needed 2); none down; controls <= 0.20 (standing 0.00, takedown 0.17, turtle 0.16, side control 0.08); same-moment agreement 0.282 vs 0.275 | **fail** |
| C1 | MapAnything feasibility | not run yet: installing its environment needs Dhruva (the auto-mode check blocked the package install) | pending |

## Agreement in full
| pipeline | same moment | baseline (same label) | difference [95%] | identity-aware difference [95%] | phone pairs |
|---|---|---|---|---|---|
| `clicks` | 0.159 | 0.098 | +0.061 [+0.026, +0.093] | +0.065 [+0.037, +0.091] | 459 |
| `blend` | 0.275 | 0.142 | **+0.132** [+0.091, +0.175] | +0.101 [+0.069, +0.132] | 607 |
| `place` | 0.282 | 0.187 | +0.095 [+0.056, +0.132] | +0.079 [+0.046, +0.112] | 605 |
Phone pairs with no relation in either image (not in the Dice): 376 / 263 / 277.

## Reading
- **A1 passes:** what one phone measures is about the moment, not just the position label; two phones of one moment
  agree about twice as much as the label alone explains. Cross-phone agreement is a usable reliability score for
  ground work where no 3D truth exists.
- **A2:** the combined depth (E05's default) more than doubles the moment-specific agreement (+0.061 -> +0.132): it
  is the more reliable pipeline, not only the one that finds more defining relations.
- **B1 fails:** the learned placement correction, trained on standing Harmony4D, adds only to half guard on ground
  work, and it raises the same-label baseline more than the same-moment agreement (+0.132 -> +0.095 beyond the label):
  it pulls bodies towards typical configurations rather than towards this moment. It stays a standing-work tool;
  ground work needs its own training data (3D truth for ground work: the capture, or pseudo-3D from part C).
- On this sample the combined depth reproduces E05's defining-relation rates (mount 0.33, closed guard 0.40, back 0.65;
  E05: 0.29, 0.33, 0.64).
- Caveat: the agreement says the measurements are consistent across viewpoints, not that they are right; a shared
  bias (both phones wrong the same way) also agrees.
