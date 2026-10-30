---
name: tensorlings-phase-3-built-sections-5-to-7
kind: doc
description: phase 3 adds 105 exercises, Sections 5 to 7 with the CIFAR, adder and GPT projects and the owner's six gaps, on feat/tensorlings-phase-3
updated: 2026-10-30
links: []
---

## Where it stands

Phase 3 is built and checked on `feat/tensorlings-phase-3`, which stacks on the phase 2 branch and the ones under it. Nothing is merged, since merges wait for the owner's word. The course now has 215 exercises in 59 chapters plus the quizzes, Sections 1 to 7 of [the detailed curriculum](../docs/research/tensorlings-detailed-curriculum.md).

- Section 5: 30_convolution to 36_transfer, quiz7, and the CIFAR-10 project, cifar1 to 3.
- Section 6: 38_soft_design to 45_state_space, quiz8, and the adder project, adder1 to 3.
- Section 7: 47_attention to 58_bpe, quiz9 and quiz10, and the GPT project, gpt1 to 4.
- The owner's six gaps, per DEC-017: soft1 and soft2 in 16_activations, fft1, 38_soft_design, stft1, attention6, 48_linear_attention, induction1 and dynamic1. 48's exercises are named linattn1 and linattn2, since linear1 and linear2 were taken.

New pieces: CIFAR-10 and tiny Shakespeare in `DATASETS`. `run.py train <name>` runs a project file's `main()` on the best device and saves `checkpoints/<name>.pt`, which the check reads back on the CPU, as the design doc says. The owner's `exercises/` folder holds every chapter after `run.py init`.

## Verified against written

- `run.py dev check` reported all 215 exercises healthy with the reference checkpoints in place.
- `dev/wrong_answers.py` holds 995 wrong answers and every one fails.
- Five beginner reader reviews by Opus sub agents read every phase 3 exercise and re-ran the lesson tables. Their 54 findings were all fixed, by three fixers and the main agent. They included checks that let wrong answers pass: mask2's copies_right, greedy1 skipping END, transfer1 sharing the source's body, gpt3's train_step without zero_grad, and several functions that recorded for autograd. moe3's table only held without a capacity limit, and its lesson now says what dropping tokens does.
- The projects trained for real on MPS. cifar2 reached 85.6% on validation and 85.3% on test, adder2 got 100% of held out sums in 4,000 steps, and gpt3 reached a validation loss of 1.535 in about 5 minutes.
- The screen driver drew gpt4's lesson and result at 110 by 40.

## Next step

Phase 4, Sections 8 to 10 of the detailed curriculum, on a branch from this one's tip, per the plan build-tensorlings.

## Open questions for the owner

- When to merge the stacked branches into main, and whether to add a remote to push to.

## Traps

- The reference checkpoints were moved out of `checkpoints/`, so the owner trains their own and a check can't pass on the agent's model. Until a model is trained again, the dev check reports cifar2, cifar3, adder2, gpt3 and gpt4 as failing. Training all three reference models with the solutions' `main()` takes about 17 minutes on MPS.
- Forks that edit `tensorlings/chapters/__init__.py` and `dev/wrong_answers.py` at once work, as long as each keeps its module importable after every edit. A half written module stops every dev check for everyone.
- Machine load swings training times a lot here. Checks that train set time limits of 30 to 120 seconds, and conv4 needed 90.
- Long sums did not train for the adder in minutes, with or without cross attention, which is why cross2 copies digits instead. The adder holds out sums by `(a + 3 * b) % 11`, since a rule on the last digits alone hid whole digit pairs from training.
- A check that needs to know how one function called another swaps a scripted stand in into the learner's module, as the design doc describes.
