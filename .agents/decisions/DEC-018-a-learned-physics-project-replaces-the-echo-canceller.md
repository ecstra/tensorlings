---
name: DEC-018-a-learned-physics-project-replaces-the-echo-canceller
kind: decision
description: on the owner's word, Section 13's first capstone is a particle simulator that learns springs, not a toy echo canceller
updated: 2026-10-30
links: []
date: 2026-10-30
area: course
status: standing
---

## Decision

Section 13's first capstone is 98_learned_physics, a simulator that learns how particles on springs move by passing messages on their relative positions, measured against simple baselines over long rollouts. It replaces 98_echo_canceller, which was built and then dropped on 2026-10-30. The owner said "I don't like echo canceller. Couldn't you find something cooler?" and picked learned physics from three offered projects.

## Alternatives

- Keeping the echo canceller. It was built on real spoken digits and worked, with an honest loss when the room changes, but the owner did not want it.
- Neural cellular automata, which grow a picture from one cell and regrow it after damage. Offered first, the owner chose physics instead.
- A Sudoku solver with a recurrent relational network. Offered, not chosen.

## Consequences

The project leans on Section 10, sets, message passing, equivariance and shortest paths, plus 95_design_method's steps. The spoken digits dataset and the STFT forward pointers in stft1 and fft1 went with the echo canceller. The course design's goal line now names a learned physics simulator where it named an echo canceller.
