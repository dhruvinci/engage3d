# E10 — Can SAM 3D Body's own pose token fix limbs inside tangles?

**Objective.** Before fine-tuning the body model (E11), test the cheap route: a small head that reads SAM 3D Body's
final pose token and corrects each athlete's limbs.

**Setup.** Held-out test: the E06 views (1,068); new people: MMA (1,484 views). The head is compared with the same
head without the token, and with a shuffled token. Criteria fixed in advance: [`PREREG.md`](PREREG.md).

**Status.** Pre-registered 2026-10-26; results follow in `RESULTS.md`.
