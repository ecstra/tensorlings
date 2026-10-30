---
name: tensorlings-phase-4-paused-with-every-chapter-built
kind: doc
description: phase 4 has every chapter and quiz of Sections 8 to 10 built, the probe and digit projects wait on lesson numbers, reviews not yet run
updated: 2026-10-30
links: []
---

## Where it stands

Phase 4 is part built on `feat/tensorlings-phase-4`, which stacks on the phase 3 branch, and everything is committed. The owner paused work at 09:49 on 2026-10-30. The course has 267 exercises.

- Section 8: 60_word2vec to 63_autoencoders, quiz11, and the probe project, probe1 to 3.
- Section 9: 65_sampling to 70_vqvae, quiz12, and the digit diffusion project, digits1 to 3.
- Section 10: 72_sets to 74_equivariance, quiz13, and the shortest paths project, paths1 to 3, which is finished.

Each chapter was checked healthy with `run.py dev check` when it was committed, and every wrong answer run so far fails.

## Verified against written

- The probe project trained for real. probe1's SimCLR run took 21.5 minutes on MPS and saved `checkpoints/probe1.pt`. probe1 and probe2 pass against it.
- The digit project trained for real. digits2 took 20.6 minutes and saved `checkpoints/digits2.pt`, with a test loss of 0.0168. digits1 and digits2 pass against it.
- The shortest paths project is measured and done. Over 1,500 steps on graphs of 10 nodes, the network shaped like Bellman-Ford gets 100% within 0.05 on graphs of 10, 20 and 40 nodes. Generic message passing gets 99.5%, 0% and 0%, and the MLP gets 43.4% and 9.2%.

## Next step

Finish TODO-015.

- probe2 and probe3 still hold placeholders: PROBE_SCORE, SWEEP_ROWS and SWEEP_SENTENCE. Run `scratchpad/proto/measure_probe.py` against `checkpoints/probe1.pt`, which prints each row of the table. Two rows came back before the pause:
  - 100 labels: SimCLR probe 0.372, random 0.216, supervised 0.287.
  - 500 labels: SimCLR probe 0.446, random 0.274, supervised 0.455.
  - An earlier prototype gave 2,500 labels at 0.537, 0.387 and 0.666, and 10,000 at 0.579, 0.427 and 0.726.

  So SimCLR helps only with very few labels at this budget, and the lesson must say so.
- digits3 still holds placeholders: W0 to W4, C0 to C4, D0 to D4 and GUIDANCE_SENTENCE. Run `scratchpad/proto/measure_digits.py`. Its first row before the pause, at w = 0: clipped_ddim 0.72, clamped without recomputing 0.74, and plain ddim_sample 0.475. A prototype gave 90% at a guidance of 2, 97% at 3 and 99.5% at 5, in its own 1 + w scale. digits3's wrong answers have not been run.
- Then run the beginner reviews of chapters 60 to 75, which were stopped at the pause, and fix what they find.
- Then the full dev check, every wrong answer, the screen driver and `run.py init`. Move the checkpoints out of `checkpoints/` and write the handover.

## Open questions for the owner

- When to merge the stacked branches into main, and whether to add a remote to push to.

## Traps

- `checkpoints/` holds the agent's probe1 and digits2 models while phase 4 is open. Move them out at the end, as phase 3 did, so the owner trains their own.
- A shared helper that builds a tensor without a device breaks training on MPS. diffusion's `time_features` needed `device=t.device`, and probe's `nt_xent` moves its partners to the scores' device.
- The scratchpad scripts named here live in this session's scratchpad and are gone in a new session. Rebuild them from the module's reference functions if needed.
