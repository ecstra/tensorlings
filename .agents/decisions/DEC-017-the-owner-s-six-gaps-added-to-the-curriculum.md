---
name: DEC-017-the-owner-s-six-gaps-added-to-the-curriculum
kind: decision
description: 11 exercises and a design chapter for soft relaxations, attention's why, linear attention, input dependent weights, spectra and induction
updated: 2026-10-29
links: []
date: 2026-10-29
area: course
status: standing
---

## Context

On 2026-10-29 the owner passed on a review of the curriculum by another agent and asked for its gaps to be fixed "as well as needed". Judged against the owner's goal, learning why and when to build a module so as to invent new ones, the review named six gaps. Each was checked against the detailed curriculum and held.

## Decision

The curriculum gains 11 exercises and one chapter.

- Soft relaxations come early. soft1 and soft2 in 16_activations show why `if`, `argmax` and indexing pass no gradient and how sigmoid and softmax stand in for them. The new chapter 38_soft_design opens Section 6 with the design loop, the exact algorithm, its hard step, and the soft step in its place, once for a gate and once for a lookup by key. 95_design_method's design3 now builds on it.
- Attention gets its why, attention6, key value recall where an RNN's fixed vector fails.
- Linear attention gets its own chapter, 48_linear_attention, the bridge from a vector state to full attention through a matrix state.
- Weights computed from the input are named as one idea. gating1 and combine1 and combine2 frame it, and dynamic1 in 56_moe ties gates, attention, SwiGLU and routing together.
- Spectra come before the echo canceller. fft1 in 30_convolution teaches the convolution theorem, stft1 in 44_causal_conv the STFT, and the echo project works on STFT frames.
- Composition is built, not only inspected. induction1 in 53_transformer_block shows one attention layer failing at copying the token after a match, and a shift then attention succeeding.

Every chapter from 38 on moves up one number, and every chapter from 48 on moves up two.

## Alternatives

Putting the design loop inside existing chapters, 41_gated_rnns and 47_attention, kept the numbering. The review asked for the loop before Section 6's blocks so the learner derives a gate and a lookup before seeing LSTMs and transformers, and a chapter of its own does that cleanly. Linear attention could have gone into 47_attention, which would have grown to eight exercises, so it got its own chapter.

## Consequences

Section 6 and 7 folders are numbered as the curriculum now lists them. soft1 and soft2 join Section 3, which was already built, so the owner's folder gains two files there when `run.py init` runs.
