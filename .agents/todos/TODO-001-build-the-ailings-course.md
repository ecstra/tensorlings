---
name: TODO-001-build-the-ailings-course
kind: todo
description: a rustlings style course that teaches why and when to build a custom AI module
updated: 2026-10-27
links: []
status: done
by: owner
area: course
born: 2026-10-27
done: 2026-10-27
---

## What

"use a toy problem to teach me AI like rustlings do. Search online as well, do some reasearch as to how to teach well."

"Let run.py as huge as possible as you need if needed. So that way I start with either an empty model.py file or partially implemented ones from you. So the wrapper is obfoscuted for now."

"Again, research online properly first. And proper toy problem, like AIlings? idk"

## Why

The owner knows transformers and training loops but not why a module like attention exists, or how to decide when to build a custom one.

## Steps

| # | Step | Points at | By | Status |
|---|---|---|---|---|
| 1 | Research rustlings and teaching methods online | | owner | done |
| 2 | Design the curriculum around one toy world | | owner | done |
| 3 | Write run.py with the tasks, checks and runner | | owner | done |
| 4 | Write the exercise files and lessons | | owner | done |
| 5 | Verify every exercise against reference solutions | | owner | done |
| 6 | File the teaching research as a doc | | owner | done |

## Done when

`python run.py selftest` shows every stub failing and every reference solution passing, and the owner can start at exercise 00 with `python run.py`.

Met on 2026-10-27. The selftest reported all 25 exercises healthy. The project is not a git repository yet, so no commit closes it.
