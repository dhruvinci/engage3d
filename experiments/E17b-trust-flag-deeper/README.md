# E17b — Making the trust flag usable: error bars, repair, clip summaries, the best camera

**Objective.** E17's truth-free flag finds the frames where a measurement is wrong (AUROC 0.76-0.80). Four uses of it,
each against a bar fixed in advance: error bars with a tested coverage; repairing flagged frames from trusted
neighbours; clip summaries that discount flagged frames; and choosing the best camera when three phones film.

**Setup.** E17 re-created exactly (same features, settings and seeds; confirmatory AUROC 0.7946 and 0.7632, equal to
E17's), this time saving every view; every rule fitted on the development sequences only. Confirmatory sets: the
held-out grappling pair on cameras never scored before, and MMA (new people). Criteria: [`PREREG.md`](PREREG.md).

**Results** (distance between the two people)
| | Part | Result | |
|---|---|---|---|
| C1 | 90% error bars, held-out pair | cover 90.5% (bar 85-95%); median half-width 7.2 cm vs 9.5 cm fixed | **pass** |
| C2 | 90% error bars, new people | cover 74.9% (bar >= 85%) | **fail** |
| C3 | repair flagged frames from neighbours | flagged frames worse: 4.14 -> 5.17 cm | **fail** |
| C4 | trust-weighted clip average | 4.86 vs 4.42 cm plain (no gain) | **fail** |
| C5 | best of three cameras by the flag | 1.82 cm vs 3.37 cm random: **-46%** (new people -46%, hugging -66%, ballroom -50%) | **pass** |

**Verdict.** C1 and C5 pass; C2-C4 fail.
**Learned.** The flag's best use is choosing a camera: with three phones it halves the error, about two-thirds of the
gain a perfect choice would give. Error bars hold on the pair the flag was trained on, not yet on new people. Repair
from neighbours fails for the reason E14 found: errors persist across frames, and the true distance moves within
0.4 s. At range (sword fighting) the flag does not work.
**Disclosure.** A smoke run (2 epochs, 3,000 views) used to test the code also scored the test sets, after the
pre-registration was pushed; nothing was changed after it, and it gave the same verdicts.

Files: `PREREG.md` (committed before the run), `RESULTS.md`, `data/` (`trust_b.json`, `run.log`). Code:
`src/engage_trust_b.py`.
