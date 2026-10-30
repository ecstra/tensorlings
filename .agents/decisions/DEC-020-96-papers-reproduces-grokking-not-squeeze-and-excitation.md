---
name: DEC-020-96-papers-reproduces-grokking-not-squeeze-and-excitation
kind: decision
description: on the owner's word, the paper the learner turns into code and reproduces is grokking on modular arithmetic, not the squeeze and excitation image block
updated: 2026-10-30
links: []
date: 2026-10-30
area: course
status: standing
---

## Decision

96_papers turns Power and colleagues' grokking (2022) into code in paper1, and reproduces its late jump from memorizing to generalizing in paper2, across seeds. The squeeze and excitation block it was built on first lost about 2 points on CIFAR at the sizes a Mac trains in minutes, and the owner dislikes image models. Offered three choices on 2026-10-30, the owner picked "Grokking (Recommended)".

## Alternatives

- The Pre-LN transformer (Xiong and colleagues 2020), reproduced on the course GPT. Offered, not chosen.
- Keeping squeeze and excitation with a setup researched until its gain shows. Offered, not chosen.

## Consequences

papers.py no longer imports the CIFAR helpers, so nothing in Section 13 rests on images. paper2 must show grokking on every seed, by the owner's rule that a demonstration shows its method working.
