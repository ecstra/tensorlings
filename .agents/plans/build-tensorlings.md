---
name: build-tensorlings
kind: plan
description: a rustlings style course from tensors to designing new modules, built in phases the owner reviews
updated: 2026-10-28
links: []
status: open
by: owner
born: 2026-10-27
---

## Why

"After the tensorlings I must be able to create my own modules, understand why each layer exists, when and why to use each layer and how to create transformers and modules from scratch (not just transformer but any module)."

"By the end of the tensorlings, I must be an AI expert like karpathy... Dont be biased. Be ground truthful."

"Phase 1 is fine but take like 5 exercises. Lets build the basics and see first (how excersices and tui look and function) once thats good ill just ack all after that till this is fully done."

The curriculum, the app and the writing rules are in [the course design](../docs/research/tensorlings-course-design.md).

## Steps

| # | Step | Points at | By | Status |
|---|---|---|---|---|
| 1 | Write the plan, docs, decisions and handover | | owner | done |
| 2 | Phase 1a, the app and the first 5 exercises, for the owner's review | | owner | pending |
| 3 | The owner reviews phase 1a, and from then on acks every phase | | owner | pending |
| 4 | Phase 1b, the rest of parts 1 and 2 | | owner | pending |
| 5 | Phase 2, parts 3 and 4 | | owner | pending |
| 6 | Phase 3, parts 5 to 7, with the GPT project | | owner | pending |
| 7 | Phase 4, parts 8 to 10 | | owner | pending |
| 8 | Phase 5, parts 11 and 12 | | owner | pending |
| 9 | Phase 6, part 13, with the echo canceller, quant and own module projects | | owner | pending |

## Done when

All 13 sections exist in `run.py`, `run.py dev check` reports every exercise healthy, every lesson passes the lesson lint and a beginner reader review, and the owner has worked through the course.
