---
name: DEC-014-projects-for-the-sections-the-table-left-open
kind: decision
description: the agent picked a project for Sections 1, 2, 4, 6 and 8 to 12 from each section's own topics, on the owner's word
updated: 2026-10-29
links: [docs/research/tensorlings-detailed-curriculum.md]
date: 2026-10-29
area: course
status: standing
---

## Decision

Each section whose project the curriculum table left open gets one picked from that section's own topics and modeled on a known course. The owner left the choice to the agent on 2026-10-29 with "Whatever you feel is best, first draw up a properly detailed curriculum", and asked to see each one in its phase's handover. The picks are in [the detailed curriculum](../docs/research/tensorlings-detailed-curriculum.md).

- Section 1: finish the micrograd engine and check it against PyTorch.
- Section 2: softmax regression on MNIST, tuned on validation, as in CS231n's first assignment.
- Section 4: a broken run of the Section 3 name generator, fixed with the debugging recipe.
- Section 6: a model that adds numbers written as text.
- Section 8: contrastive features on CIFAR-10 measured by a linear probe.
- Section 9: a diffusion model that draws MNIST digits.
- Section 10: shortest paths with message passing against an MLP.
- Section 11: post training the learner's own GPT with LoRA SFT and DPO.
- Section 12: a faster GPT with a KV cache and int8 weights.

## Alternatives

For Section 1, a hand backprop of one big tensor expression, and a gradient checker on its own. The engine covers both, so they became its exercises. For Section 2, the same pipeline on generated data, which means less, and curve fitting with overfitting, which the generalization chapter already covers. Asking the owner before each project was offered and declined.

## Consequences

Sections 2, 5, 7 and 8 need datasets, which the owner allowed on 2026-10-29. Sections 11 and 12 build on the GPT from Section 7, so a learner who skips its project has no model to post train. The owner can swap any project after seeing it.
