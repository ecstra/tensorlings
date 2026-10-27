---
name: TODO-002-make-ailings-work-like-rustlings
kind: todo
description: rebuild the AIlings layout and runner to match rustlings, with the lesson inside each exercise file
updated: 2026-10-27
links: []
status: done
by: owner
area: course
born: 2026-10-27
done: 2026-10-27
---

## What

"Why read the readme? rustlings do not work that way? make sure this works exactly like rustlings do. Chck rustslings once properly. i never read readme tbf."

"And make it as good as possible so I can understand the original issue (AI, why these specific layers, etc...) and search online how to teach that properly as well."

## Why

The first version put every lesson in a README.md the owner would never open. Rustlings puts the lesson and the task inside the exercise file and runs everything from watch mode.

## Steps

| # | Step | Points at | By | Status |
|---|---|---|---|---|
| 1 | Study rustlings source and exercise format | | owner | done |
| 2 | Research how to teach why layers exist | | owner | done |
| 3 | Redesign the layout to match rustlings | | owner | done |
| 4 | Rewrite the runner UI and commands | | owner | done |
| 5 | Rewrite the exercises with the lessons inside the files | | owner | done |
| 6 | Run the selftest and drive watch mode | | owner | done |
| 7 | Update the docs | | owner | done |

## Done when

Watch mode behaves like rustlings, every exercise file carries its own lesson and task, and `run.py selftest` reports every exercise healthy.

Met on 2026-10-27. The selftest became `run.py dev check`, as in rustlings, and reported all 22 exercises healthy. Watch mode, the list, reset, init and check-all were driven in a pseudo terminal on a scratch copy. The project is not a git repository yet, so no commit closes it.
