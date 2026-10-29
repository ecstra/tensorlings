---
name: tensorlings-phase-1b-built-sections-1-and-2
kind: doc
description: phase 1b adds 44 exercises, the rest of Section 1 with the micrograd project and all of Section 2 with the MNIST project, on feat/tensorlings-phase-1b
updated: 2026-10-29
links: []
---

# tensorlings phase 1b built, Sections 1 and 2

## Where it stands

Phase 1b is built and checked on `feat/tensorlings-phase-1b`, which stacks on `feat/split-into-a-package`, `feat/detailed-curriculum` and `feat/tensorlings-phase-1a`. Nothing is merged, since merges wait for the owner's word. The course now has 51 exercises, Sections 1 and 2 of [the detailed curriculum](../docs/research/tensorlings-detailed-curriculum.md).

- Section 1 rest: derivatives1 to 3, chain1 to 3, autograd1 to 4, backprop1 to 4, quiz2, and the micrograd project, micrograd1 to 3.
- Section 2: linreg1 to 3, descent1 to 4, minibatch1 to 3, softmax1 to 3, likelihood1 to 4, splits1, overfitting1 and 2, decay1 and 2, quiz3, and the MNIST project, mnist1 to 3.

The course moved out of one `run.py` into the `tensorlings/` package per DEC-015, the projects the table left open were picked per DEC-014, and devices follow DEC-016. `run.py init` now adds missing exercise files to an existing folder, and `run.py data mnist` downloads MNIST, which is already in `data/mnist` on this machine. The owner's exercises folder has every new file, fresh.

## Verified against written

- `run.py dev check` reports all 51 exercises healthy, stubs failing, solutions passing, lesson lint clean.
- `dev/wrong_answers.py` runs 195 wrong answers and every one fails with a readable message.
- Two beginner reader reviews by Opus sub agents, one per section, found 39 and 36 problems, and every one is fixed. Among them were checks that let a wrong `__pow__`, a wrong `log` and a nan learning rate through.
- The screen driver drew autograd2's lesson and result at 110 by 40 and overfitting1's table at 80 by 24.

## Next step

Phase 2, Sections 3 and 4. Drafts of `linear_layers.py`, `activations.py` and `mlp.py` for Section 3 sit untracked in `tensorlings/chapters/`, not registered and not yet checked. Branch `feat/tensorlings-phase-2` from this branch's tip, register them, and go on with embeddings, initialization, dropout, normalization, residuals, quiz4 and quiz5, and the names project, then Section 4.

names.txt and tiny Shakespeare were fetched once to check their checksums, 401b3c4cac9e28a586ed15e81256680d and 6fb458f1232090904fb40fe944165e91. They go into `DATASETS` in `tensorlings/data.py` when their phases need them.

## Open questions for the owner

- When to merge the stacked branches into main, and whether to add a remote to push to.
- The owner hasn't tried phase 1b yet. Their word on it shapes phase 2.

## Traps

- A check must never pass a tensor to `.backward()`, see the course design doc.
- Exercise files that train on real data need `time_limit` on their `Exercise`, as the MNIST project and overfitting2 have.
- A wrong answer's piece in `dev/wrong_answers.py` must appear exactly once in the solution, with the solution's own indentation, where function bodies sit at 4 spaces.
