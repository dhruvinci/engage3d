# E11 — Fine-tune SAM 3D Body's decoder on two-person contact (local pilot)

**Objective.** E10 showed the decoder's final token holds no extra limb information inside tangles. Can training the
decoder itself on Harmony4D contact fix limbs inside tangles, for new people too?

**Setup.** Backbone frozen; only the body decoder (39.4M parameters) trains, on 10,000 athlete crops. Measured on the
E06 held-out views and on new people (MMA), against the released model and against E10's cheap fix. Criteria fixed in
advance: [`PREREG.md`](PREREG.md).

**Status.** Pre-registered 2026-10-26; results follow in `RESULTS.md`.
