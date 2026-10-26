# E19 — A 2D keypoint model trained on contact, feeding SAM 3D Body's clicks

**Objective.** E18: true 2D clicks lower tangled-limb error 15%, but today's 2D model (ViTPose+) as clicks raises it
25%, because inside a tangle it takes the partner's limbs. Can ViTPose+, fine-tuned on Harmony4D contact, turn that
harm into a gain?

**Setup.** Fine-tune on 16,000 contact-weighted athlete crops with the exact 2D joints (hidden ones included); test on
the E18 held-out views, in 2D and through SAM 3D Body's clicks. Criteria fixed in advance: [`PREREG.md`](PREREG.md).

**Status.** Pre-registered 2026-10-27; results follow in `RESULTS.md`.
