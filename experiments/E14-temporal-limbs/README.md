# E14 — Can time fill in limbs hidden at a contact?

**Objective.** A limb hidden behind the partner now was usually visible a moment earlier. Does a small model reading a
window of frames correct hidden limbs better than one reading a single frame, and how long a gap can time alone bridge?

**Setup.** Windows of 1, 5, 15 and 31 frames (every 2nd frame, about 15 per second) of SAM 3D Body's per-frame points;
a masked-gap test hiding up to 1 s. Held-out grappling and MMA. Criteria fixed in advance (with an amendment before
the run): [`PREREG.md`](PREREG.md).

**Results** (median error of limb joints inside a tangle)
| | held-out | MMA (new people) |
|---|---|---|
| SAM 3D Body | 5.39 cm | 6.25 cm |
| window 1 (per-frame head) | **4.81 cm** | **6.01 cm** |
| window 15 (about 1 s) | 4.81 cm | 6.08 cm |
| window 31 (about 2 s) | 4.85 cm | 6.16 cm |
| centre +-3 frames hidden (0.5 s) | 11.14 cm | 15.07 cm |

**Verdict.** **Fail** on T1-T4: more frames add nothing (+0.1%, interval -1.9% to +2.3%).
**Learned.** SAM 3D Body's errors on a hidden limb persist across neighbouring frames: averaging the two neighbours
of a hidden frame is almost as good as the frame itself. Time alone bridges about one frame; grappling moves too fast
for longer gaps. The per-frame head, trained on all development views with the partner's points, is the best limb
correction so far (-11%).
**Next.** Hidden limbs need new evidence: image features trained on contact (E16) or a second camera.

Files: `PREREG.md` (committed before the run), `RESULTS.md`.
