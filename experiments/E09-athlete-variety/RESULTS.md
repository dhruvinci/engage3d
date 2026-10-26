# Diversity test — results (pre-registration: `diversity_prereg.md`) — 2026-10-26

Data fetched from Harmony4D's training split by HTTP range reads (`engage_h4d_fetch.py`; 7,842 + 9,127 files; no full
download). The training split stores the world -> COLMAP similarity as `aria_from_colmap_transforms.pkl['aria01']`:
ground truth projected inside its own person boxes in 40/40 person-views. Pipeline as before (`div/queue.sh`). One
render fix was needed: a body almost on the lens made a >64 px triangle (`engage_depth_probe.render` now handles up to
256 px and skips larger ones).

Training sets (v2 settings): **A** = 12 development grappling2 sequences x 20 cameras (22,640 views, one pair);
**A+same** = A + 3,464 views of the same pair (`train/03_grappling2`, 16 sequences); **A+new** = A + 3,120 views of
the new pair (`train/02_grappling`, 29 sequences). Tests: MMA (third pair, 4,460 views) and the new pair's 6 held-out
sequences (001/017/028/037/044/050_grappling, 904 views). Outputs: `benchmarks/engage_h4d/div/{A,A_same,A_new}/`.

| | A | A+same | A+new |
|---|---|---|---|
| **MMA** configuration error (combined depth 9.10) | 7.73 cm | 7.59 | **7.54** |
| MMA relative depth (7.54) | 6.09 | 5.64 | **5.63** |
| MMA relative lateral | 3.16 | 3.15 | **3.03** |
| MMA distance error | 4.30 | 4.05 | **3.97** |
| MMA region pairs | 0.463 | 0.470 | **0.471** |
| **New-pair test** configuration error (combined depth 7.89) | 6.45 | 6.32 | **6.18** |
| new-pair relative depth | 4.34 | 3.97 | **3.41** |
| new-pair relative lateral | 2.64 | 2.51 | **2.36** |
| new-pair distance error | 2.52 | 2.55 | **2.38** |
| new-pair region pairs / body lock / legs | 0.549 / 0.835 / 0.377 | 0.549 / 0.846 / 0.426 | **0.558** / 0.839 / **0.481** |

- V1 (MMA, A+new better than A+same by >= 0.10 cm): 0.05 cm. **Fail.**
- V2 (new-pair test, by >= 0.20 cm): 0.14 cm. **Fail.**
- V3 (quantity alone, A -> A+same on MMA): -0.14 cm.

**Verdict (per the rule): neither passes.** The thresholds are not met, so this does not establish athlete variety
as the lever. What the numbers do show:
1. Extra data of either kind helps a little (MMA configuration -0.14 to -0.19 cm).
2. The new pair is better than the same pair on every measure, but by less than pre-registered. The one large effect
   is relative depth on the new pair's own unseen sequences (3.97 -> 3.41 cm).
3. On a third, unseen pair (MMA) variety and quantity were equivalent. With only two grappling pairs in all of
   Harmony4D, this test cannot measure how variety scales; it needs more pairs than exist publicly.
4. The model is near its own limit: a perfect rigid correction would give about 5.8 cm configuration error on
   familiar views; it is at 6.2-6.5. The rest is limb error inside tangles, which a whole-body correction cannot fix.

**Implications.** (a) The capture pilot should include enough different pairs (>= 5) to measure the variety effect
directly, rather than assume it. (b) The next model gain needs per-person limb accuracy in contact (fine-tuning the
per-person body model, e.g. SAM 3D Body's decoder, on contact data: backlog), not more placement data.
