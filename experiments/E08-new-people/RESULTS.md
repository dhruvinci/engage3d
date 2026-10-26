# New-people test — results (pre-registration: `newpeople_prereg.md`) — 2026-10-25

Model: v2 settings trained only on the 12 development grappling2 sequences x 20 cameras (22,640 views; one pair of
athletes). Evaluated on every MMA view (016_mma4: a different pair, different activity; 4,460 views, 20 cameras).
Outputs: `benchmarks/engage_h4d/place_newpeople/` (`score_{blend,model}/`).

| MMA | combined depth | + model | change | criterion |
|---|---|---|---|---|
| N1 configuration error | 9.10 cm | 7.73 | -15% | <= -11%: **transfers** |
| N2 relative depth | 7.54 cm | 6.08 | -19% | <= -12%: **transfers** |
| N3 distance error | 5.07 cm | 4.30 | better | **pass** |
| N3 region pairs | 0.41 | 0.46 | better | **pass** |
| in contact | 0.93 | 0.95 | | |
| arms (hand fighting) | 0.24 | 0.28 | | |
| balance (as scored) | 40% | 45% | | |
(MMA has no confident body-lock cases.)

**Verdict: transfers** (all three pass). Learned from one pair, the correction keeps about two-thirds of its
familiar-people gain (configuration -15% vs -22%; relative depth -19% vs -24%) on new people doing a different
activity. The method generalises; the missing third is what athlete variety should buy (`diversity_prereg.md`).
