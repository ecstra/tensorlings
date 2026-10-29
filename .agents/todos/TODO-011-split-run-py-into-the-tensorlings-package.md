---
name: TODO-011-split-run-py-into-the-tensorlings-package
kind: todo
description: move the engine, exercise model, lint, progress, app, commands and chapters out of run.py into modules
updated: 2026-10-29
links: []
status: done
by: agent
area: course
born: 2026-10-29
done: 2026-10-29
---

## What

The owner said "you may use multiple files instead of one monolith, just saying, a proper code structure and organization wlll be better." DEC-015 settles the layout.

## Steps

| # | Step | Points at | By | Status |
|---|---|---|---|---|
| 1 | Write the engine, exercise model, lint, progress, app and command modules | | agent | done |
| 2 | Move phase 1a's chapters into their own modules | | agent | done |
| 3 | Make run.py the entry point and point the dev scripts at the package | | agent | done |
| 4 | Update the course design doc for the new layout | | agent | done |
| 5 | Run the dev check, the wrong answers and the screen driver | | agent | done |

## Done when

`run.py` holds only the entry point, every other piece lives in its module in `tensorlings/`, and the dev check, the wrong answers and the screen driver give the same results as before the split.

Met on 2026-10-29. A read only comparison found every lesson, file, solution, hint, chapter readme, the jargon and filler lists, the plain errors, the CSS and the page texts equal to the old `run.py`. The dev check passed all 7 exercises, all 40 wrong answers failed, and the screen driver drew the result tab on a copy.
