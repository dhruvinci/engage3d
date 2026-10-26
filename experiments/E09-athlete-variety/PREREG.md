# Pre-registration: is athlete variety or data quantity the lever? — written 2026-10-25, before fetching the data

## Why
Placement v2 (`placemodel_v2_results.md`) gained little from 4x more views of the same two athletes. All local
Harmony4D grappling is one pair. Harmony4D's training split (checked remotely, by body-shape parameters) holds the
same pair again (`train/03_grappling2`, 16 sequences) and one new pair (`train/02_grappling`, 35 sequences).
Comparing "more of the same pair" with "the same amount from a new pair" separates quantity from variety.

## Data (fetched by `scripts/engage_h4d_fetch.py`, no full download; cameras 01/06/11/16)
- Same pair: `train/03_grappling2`, all 16 sequences, every 4th ground-truth frame (about 3,440 views).
- New pair: `train/02_grappling`, every 8th frame (about 3,900 views). Sequences at sorted positions 0, 6, 12, 18, 24
  and 30 are held out as the **new-pair test** (never trained on). The other 29 (about 3,300 views) are training data.
  Every 8th frame here and every 4th frame there gives about the same number of training views.
Each view goes through the same pipeline: SAM 3D Body, ViTPose+, MoGe-2, features (`--overlap`, then `--drop-overlap`).

## Models (v2 settings: width 512, 60 epochs, 3 seeds, mirroring, no tokens)
- **A:** the 12 development grappling2 sequences x 20 cameras (22,640 views; the new-people test's model).
- **A+same:** A + the same-pair views.
- **A+new:** A + the new pair's 29 training sequences.
None of them sees MMA, the grappling2 test sequences, or the new-pair test sequences.

## Evaluation
- **Unseen people:** MMA (016_mma4, 4,460 views; a third pair, different activity).
- **New-pair test:** the 6 held-out `02_grappling` sequences (about 600 views).
Measures: configuration error, relative depth, distance error and region pairs (`engage_bench_oracle.py`).

## Criteria
| | Comparison | Variety helps if |
|---|---|---|
| V1 | A+new vs A+same, MMA configuration error | A+new lower by >= 0.10 cm |
| V2 | A+new vs A+same, new-pair test configuration error | A+new lower by >= 0.20 cm |
| V3 | A+same vs A, MMA configuration error | reported (the effect of quantity alone) |
**Reading.**
- V1 and V2 pass: athlete variety is the lever, and the capture should prioritise many different athletes over many
  hours of few.
- V2 passes, V1 fails: a new pair helps on that pair but does not yet generalise; two pairs are too few to tell.
- Neither passes: neither quantity nor one extra pair moves this model; look at the model or its inputs instead.
