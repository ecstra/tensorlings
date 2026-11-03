---
name: DEC-022-residual3-teaches-pre-norm-and-the-zero-start-apart
kind: decision
description: the pre norm block keeps PyTorch's default start, and the zero start is taught on a deep stack with no norm, where its papers show it
updated: 2026-11-03
links: []
date: 2026-11-03
area: course
status: standing
---

## Decision

residual3 teaches its two fixes for deep residual stacks separately, each with the measurement that shows it working. `PreNormBlock` keeps PyTorch's default start. It is measured against residual2's blocks at 51, 101 and 201 hidden layers, where residual2's turn nan from 101 layers on and pre norm trains to about 96%. The zero start is its own TODO, `zero_start_`, which zeroes the last layer and scales the first by 1 / sqrt(blocks) as Fixup does. It is measured on a 201 layer stack with no norm, where the default start turns nan and the zero start trains to 90% to 94%. quiz5's DropoutBlock follows the new PreNormBlock, and combine3 picks the zero start up again for a branch added to a trained model.

## Alternatives

The old residual3 put pre norm and the zero start in one block and credited both for the table's best scores. The final deep review measured each part on its own on 2026-11-03. With a pre norm in front, zeroing the last layer lowered accuracy a little at every depth and rate tried, and the plain pre norm block scored best. Keeping the joint block would have taught a claim the numbers don't support, which breaks the owner's rule that every demonstration shows its method working.

## Consequences

Later lessons that said the pre norm block starts as "do nothing" now point at residual3's zero start instead (combine3, design4, gpt2). Anything that builds on `PreNormBlock` gets the default start.
