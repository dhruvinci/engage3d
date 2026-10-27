# E19 — A 2D keypoint model trained on contact, feeding SAM 3D Body's clicks

**Objective.** E18: true 2D clicks lower tangled-limb error 15%, but today's 2D model (ViTPose+) as clicks raises it
25%, because inside a tangle it takes the partner's limbs. Can ViTPose+, fine-tuned on Harmony4D contact, turn that
harm into a gain?

**Setup.** ViTPose+ base fine-tuned on 16,000 contact-weighted athlete crops with the exact 2D joints (hidden ones
included), 3 epochs on the laptop GPU. Tested on the E18 held-out views, in 2D and through SAM 3D Body's clicks.
Criteria fixed in advance.

**Results** (held-out views)
| | 2D inside tangles | 3D tangled limbs via clicks |
|---|---|---|
| no clicks | | 5.39 cm |
| released ViTPose+ | 24.5 px | 6.72 cm (+25%) |
| **fine-tuned ViTPose+** | **19.2 px** (-21%) | 6.14 cm (+14%) |
| true 2D | 0 | 4.60 cm (-15%) |

**Verdict.** **Fail** (V1 2D -21%, needed -30%; V2 clicks still hurt; V3 other joints +5%). On new people the 2D
gain holds (-23%).
**Learned.** Contact training improves the 2D model and halves the harm of its clicks, but clicks only pay off when
the 2D is nearly exact inside tangles: the click goes exactly where the 2D model is most likely wrong.
**Next.** A stronger 2D model and more people (E16, RF-DETR keypoints), or 2D as a soft, confidence-weighted
constraint instead of hard clicks.

Files: `PREREG.md` (committed before the run), `RESULTS.md`. Code: `src/engage_vitpose_finetune.py`.
