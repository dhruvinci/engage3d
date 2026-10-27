# E17 — Can the pipeline flag the frames where its measurement is wrong?

**Objective.** Ship every measurement with a trust flag: do truth-free signals (the ensemble's disagreement, how close
and overlapped the two bodies are, 2D confidence, the depth map's evidence) predict the frames with large errors?

**Setup.** A logistic flag per error type, trained on out-of-fold predictions of the development sequences; tested on
held-out grappling and MMA views never scored before, plus the other activities. Criteria fixed in advance, with a
disclosure: [`PREREG.md`](PREREG.md).

**Results** (distance between the two people; bad = more than 5 cm off)
| | AUROC | median error, all frames | trusted half | flagged half |
|---|---|---|---|---|
| held-out grappling (confirmatory) | **0.795** | 2.60 cm | **1.88 cm** | 4.14 cm |
| MMA, new people (confirmatory) | **0.763** | 4.15 cm | **2.55 cm** | 8.33 cm |
| hugging | 0.795 | 4.77 cm | 2.35 cm | 17.27 cm |
| ballroom | 0.772 | 2.72 cm | 1.87 cm | 4.48 cm |
| sword fighting (2.4 m apart) | 0.584 | 8.01 cm | 7.36 cm | 8.71 cm |
Limbs in contact: AUROC 0.792 held-out, 0.771 MMA.

**Verdict.** C1, C3 and C4 **pass**; C2 **fails** narrowly (trusting half cuts the held-out distance error by 28%; the
bar was 30%).
**Learned.** Without truth, the pipeline finds its bad frames about 4 times in 5, on new people and other activities
too; most of the signal is the placement ensemble's disagreement. Where errors are larger, trusting half the frames
removes 39-51% of the error. At range the flag does not work.
**Next.** Ship the flag with every measurement (show trusted frames, mark or smooth flagged ones); relative error at
range.

Files: `PREREG.md` (committed before the run), `RESULTS.md`.
