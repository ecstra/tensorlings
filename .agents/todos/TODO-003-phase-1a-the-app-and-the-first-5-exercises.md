---
name: TODO-003-phase-1a-the-app-and-the-first-5-exercises
kind: todo
description: the full-screen tensorlings app and exercises tensors1 to tensors4 and matmul1, for the owner to judge
updated: 2026-10-29
links: []
status: done
by: owner
area: course
born: 2026-10-27
done: 2026-10-29
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
| 11 | Move progress and the file under the result with labels, like rustlings | | owner | done |
| 12 | Put Exercise and Progress on one line at the top | | owner | done |
| 13 | Make the list's selected row one even highlight | | owner | done |
| 14 | Wrap the key bar so every key shows on a narrow terminal | | owner | done |
| 15 | Show only failing tests beside the lesson and count what the code printed | | owner | done |
| 16 | Open the full result on its own scrollable screen with tab | | owner | done |
| 17 | Put what the code printed in a box of its own | | owner | done |
| 18 | the owner's second round on lessons, tasks, results and the TUI | TODO-005 | owner | done |

## Done when

`run.py init` writes the 5 exercises, the app redraws in place with the lesson, the task, the result and the key bar, `run.py dev check` passes, and the owner has tried it.

Met on 2026-10-29. Phase 1a became seven exercises, tensors1 to quiz1, through TODO-005, and the owner worked through them and acked every phase after it.
