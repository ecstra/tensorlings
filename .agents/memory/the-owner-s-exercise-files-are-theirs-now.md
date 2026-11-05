---
name: the-owner-s-exercise-files-are-theirs-now
kind: memory
description: the course is handed over, so the owner's exercise files, solutions and progress are their own work and are reset only on their word
updated: 2026-11-05
links: []
type: feedback
---

While the course is being built, the owner works through the exercises only to test them. When a task changes, reset their exercise files and progress with `run.py init` or `run.py reset`. Do not keep copies of their answers or ask first.

**Why:** the owner said "reset them, im just testing this. so resets are fine. no need to save them" after their old answers were moved aside instead.

On 2026-10-29 the owner added that they will work through all the sections only once the whole course is built, so resets stay free until then: "ill just do all sections once the entire implementation is done so you can reset as you need."

On 2026-10-30 the owner asked for the course to start clean, phase 1a included, since they had done those exercises back then: "make sure to reset it and properly do phase-1 as well". `run.py init` only adds missing files, so their phase 1a answers, progress and unlocked solutions had survived every phase. They were cleared that day.

On 2026-11-05 the whole course was built and reviewed, and the folder was set up from nothing for the owner to start for real. From then on their exercise files, unlocked solutions and `.tensorlings-state.json` hold their own work.

**How to apply:** now that the owner is working through the course, never reset, overwrite or delete their exercise files, solutions or progress without asking first. When a change makes one of their files stale, say which file, and let them choose between `run.py reset <name>` and keeping their answer. `run.py init` stays safe, since it only adds missing files.
