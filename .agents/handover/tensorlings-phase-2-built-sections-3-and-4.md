---
name: tensorlings-phase-2-built-sections-3-and-4
kind: doc
description: phase 2 adds 59 exercises, Section 3 with the names project and Section 4 with the broken training project, on feat/tensorlings-phase-2
updated: 2026-10-29
links: []
---

## Where it stands

Phase 2 is built and checked on `feat/tensorlings-phase-2`, which stacks on `feat/tensorlings-phase-1b` and the branches under it. Nothing is merged, since merges wait for the owner's word. The course now has 110 exercises in 30 chapters, Sections 1 to 4 of [the detailed curriculum](../docs/research/tensorlings-detailed-curriculum.md).

- Section 3: modules1, linear1 to 3, relu1 and 2, activations1, gelu1, activations2, mlp1 to 4, embedding1 to 4, init1 to 4, dropout1 to 3, batchnorm1 and 2, layernorm1, rmsnorm1, norm1, residual1 to 4, quiz4, quiz5, and the names project, names1 to 3.
- Section 4: sgd1, momentum1 and 2, rmsprop1, adam1, adam2, adamw1 and 2, schedule1, warmup1, schedule2, clipping1 and 2, debug1 to 5, quiz6, and the broken training project, broken1 and 2.

New pieces of the engine: names.txt is in `DATASETS` and downloads with `run.py data names`. An `Exercise` can set `imports_first`, and every exercise that makes an optimizer sets it to `OPTIMIZER_IMPORTS`. `dev/wrong_answers.py` takes exercise names to run only those. The dev check prints learner lines with rich markup off, so code like `[point]` shows.

The owner said on 2026-10-29 they will work through the course only once it is all built, so their files can be reset as needed, per the memory on resets.

## Verified against written

- `run.py dev check` reports all 110 exercises healthy, stubs failing, solutions passing, lesson lint clean.
- `dev/wrong_answers.py` holds 404 wrong answers, 209 of them new, and every one fails.
- Four beginner reader reviews by Opus sub agents read every new exercise, and three Opus fixers plus the main agent fixed every finding. Among them were checks that let a wrong `best_width`, a `one_hot` without the class count and a `ScaleThenShift` in the wrong order pass. Wrong claims were fixed too, about Smith's range test, how AdamW ties decay to the rate, Bengio's model reading words, and nanoGPT's warmup.
- Every number in a lesson was measured on this machine, and mlp4's table was remeasured on its check's own data.
- The screen driver drew adam1's lesson and result at 110 by 40.

## Next step

Phase 3, Sections 5, 6 and 7. Branch `feat/tensorlings-phase-3` from this branch's tip. Section 5 needs CIFAR-10, which the owner allowed, and tiny Shakespeare's checksum is 6fb458f1232090904fb40fe944165e91 for Section 7. The first project that trains for many minutes needs the checkpoint flow of the detailed curriculum and the device choice of DEC-016, neither built yet.

## Open questions for the owner

- When to merge the stacked branches into main, and whether to add a remote to push to.

## Traps

- The first optimizer `torch.optim` makes imports `torch._dynamo`, 10 to 20 seconds, see the course design doc. Set `imports_first=OPTIMIZER_IMPORTS` on any exercise that makes one or calls `clip_grad_norm_`.
- A class shared between starter files, like `ADAM_CLASS` in `adaptive.py`, is indented with `textwrap.indent` before it is joined into a `source_file`, which takes the indentation off the whole string.
- names3 and broken2 train 10,000 steps inside their checks, about 10 seconds on an idle laptop and minutes under load.
- A check of a function that changes something in place should say "not written yet" when nothing changed, as `check_optimizer_class` in `sgd.py` does, or the learner gets a wall of numbers.
- A wrong answer's piece must appear exactly once in the solution with the solution's own indentation, and a solution class body sits at 8 spaces.
