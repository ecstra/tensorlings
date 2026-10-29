---
name: TODO-011-split-run-py-into-the-tensorlings-package
kind: todo
description: move the engine, exercise model, lint, progress, app, commands and chapters out of run.py into modules
updated: 2026-10-29
links: []
status: open
by: agent
area: course
born: 2026-10-29
---

## What

The owner said "you may use multiple files instead of one monolith, just saying, a proper code structure and organization wlll be better." DEC-015 settles the layout.

## Steps

| # | Step | Points at | By | Status |
|---|---|---|---|---|
| 1 | Write the engine, exercise model, lint, progress, app and command modules | | agent | pending |
| 2 | Move phase 1a's chapters into their own modules | | agent | pending |
| 3 | Make run.py the entry point and point the dev scripts at the package | | agent | pending |
| 4 | Update the course design doc for the new layout | | agent | pending |
| 5 | Run the dev check, the wrong answers and the screen driver | | agent | pending |

## Done when

`run.py` holds only the entry point, every other piece lives in its module in `tensorlings/`, and the dev check, the wrong answers and the screen driver give the same results as before the split.
