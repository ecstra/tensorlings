---
name: the-owner-s-exercise-files-can-be-reset-freely
kind: memory
description: while the course is being built the owner only tests it, so their exercise files and progress can be reset without keeping a copy
updated: 2026-10-30
links: []
type: feedback
---

While the course is being built, the owner works through the exercises only to test them. When a task changes, reset their exercise files and progress with `run.py init` or `run.py reset`. Do not keep copies of their answers or ask first.

**Why:** the owner said "reset them, im just testing this. so resets are fine. no need to save them" after their old answers were moved aside instead.

On 2026-10-29 the owner added that they will work through all the sections only once the whole course is built, so resets stay free until then: "ill just do all sections once the entire implementation is done so you can reset as you need."

On 2026-10-30 the owner asked for the course to start clean, phase 1a included, since they had done those exercises back then: "make sure to reset it and properly do phase-1 as well". `run.py init` only adds missing files, so their phase 1a answers, progress and unlocked solutions had survived every phase. They were cleared that day.

**How to apply:** when a rewrite makes the owner's files stale, replace them and tell the owner it was done. When the course is handed over, set the folder up from nothing, with `exercises/`, `solutions/` and `.tensorlings-state.json` gone before `run.py init`, so no old answer or progress survives. Revisit this once the whole course is built and the owner starts it for real.
