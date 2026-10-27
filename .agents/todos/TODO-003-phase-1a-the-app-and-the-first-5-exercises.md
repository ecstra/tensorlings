---
name: TODO-003-phase-1a-the-app-and-the-first-5-exercises
kind: todo
description: the full-screen tensorlings app and exercises tensors1 to tensors4 and matmul1, for the owner to judge
updated: 2026-10-27
links: []
status: open
by: owner
area: course
born: 2026-10-27
---

## What

"Phase 1 is fine but take like 5 exercises. Lets build the basics and see first (how excersices and tui look and function) once thats good ill just ack all after that till this is fully done. Handover and docs first, then build the 5 + tui."

## Steps

| # | Step | Points at | By | Status |
|---|---|---|---|---|
| 1 | Install rich and pin the dependencies | | owner | done |
| 2 | Build the full-screen app in run.py | | owner | done |
| 3 | Write tensors1 to tensors4 and matmul1 with their lessons | | owner | done |
| 4 | Run the dev check and the lesson lint | | owner | done |
| 5 | Drive the app in a pseudo terminal | | owner | done |
| 6 | Get a beginner reader review and fix what it finds | | owner | done |
| 7 | Replace the old AIlings course and write a new handover | | owner | done |
| 8 | Give the header room so the top of the screen is not cramped | | owner | done |
| 9 | Show the hint in its own panel instead of inside the result | | owner | done |
| 10 | Drop the dot separators from the header and panel titles | | owner | done |

## Done when

`run.py init` writes the 5 exercises, the app redraws in place with the lesson, the task, the result and the key bar, `run.py dev check` passes, and the owner has tried it.
