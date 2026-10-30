---
name: tensorlings-phase-4-built-sections-8-to-10
kind: doc
description: phase 4 adds 52 exercises, Sections 8 to 10 with the probe, digit diffusion and shortest paths projects, on feat/tensorlings-phase-4
updated: 2026-10-30
links: []
---

## Where it stands

Phase 4 is built and checked on `feat/tensorlings-phase-4`, which stacks on the phase 3 branch and the ones under it. Nothing is merged, since merges wait for the owner's word. The course now has 267 exercises, Sections 1 to 10 of [the detailed curriculum](../docs/research/tensorlings-detailed-curriculum.md).

- Section 8: 60_word2vec to 63_autoencoders, quiz11, and the probe project, probe1 to 3. SimCLR trains a ResNet on CIFAR-10 with no labels, then a linear probe reads its features.
- Section 9: 65_sampling to 70_vqvae, quiz12, and the digit diffusion project, digits1 to 3. A U-Net learns MNIST digits by class, and guided DDIM samples them.
- Section 10: 72_sets to 74_equivariance, quiz13, and the shortest paths project, paths1 to 3. Message passing shaped like Bellman-Ford is set against a generic one and an MLP.

New pieces in the engine: checks run a training function with the device set to `meta` to catch a missing `.to(device)`, and a function whose reference gives a plain number now takes only a plain number. The design doc describes both.

## Verified against written

- `run.py dev check` reported all 267 exercises healthy with every reference checkpoint in place.
- `dev/wrong_answers.py` holds 1,349 wrong answers and every one fails.
- Four beginner reader reviews by Opus sub agents read every phase 4 exercise and reran the lesson tables. Their 51 findings were all fixed. Among them were checks that passed a constant from a failure message (probe3), a label drop that never happened at the check's seed (digits2), and lessons that claimed what the numbers did not show (diffusion2's reason for guessing the noise, equivariance3's 100%).
- The projects trained for real on MPS. probe1 took about 21 minutes and digits2 about 21.
  - probe2's probe scores 53.7% on validation from 2,500 labels. SimCLR beats a supervised model only at 100 labels, which probe3's table says plainly.
  - digits3's guided sampler reaches 97.5% at w = 4, read by a LeNet that gets 95.7% of real digits.
  - The Bellman-Ford shaped network gets 100% on graphs of 10, 20 and 40 nodes after training on 10, where the generic one drops to 0% past 10.
- The screen driver drew digits3's lesson and result at 110 by 40.
- The owner's `exercises/` folder holds every chapter after `run.py init`.

## Next step

Phase 5, Sections 11 and 12 of the detailed curriculum, on a branch from this one's tip, per the plan build-tensorlings.

## Open questions for the owner

- When to merge the stacked branches into main, and whether to add a remote to push to.

## Traps

- The reference checkpoints were moved out of `checkpoints/`, so the owner trains their own. Until a model is trained again, the dev check reports cifar2, cifar3, adder2, gpt3, gpt4, probe2, digits2 and digits3 as failing on the missing file. The reference ones take about 17 minutes for phase 3's three and 42 for phase 4's two on MPS.
- Several fixers adding to `dev/wrong_answers.py` at once work, since a failed Edit rereads the file. Their cases land in one hunk, so committing one fixer's share needs the other lines left out of the patch.
- A wrong answer can turn out to be a right one written another way, as probe2's float labels and adder1's draws did. When one passes, check that first.
