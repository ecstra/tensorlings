---
name: TODO-004-set-up-the-tensorling-folder-with-git-and-a-fresh-venv
kind: todo
description: the course lives in the renamed tensorling folder, under local git, with the venv rebuilt and the test tools in the repo
updated: 2026-10-27
links: []
status: open
by: owner
area: course
born: 2026-10-27
---

## What

"make sure code is in tensorlings folder with local git setup. Not in your memory. I changed the folder name and renamed venv, so resetup venv. Ill check once you do this. ill come back."

## Steps

| # | Step | Points at | By | Status |
|---|---|---|---|---|
| 1 | Install the pinned packages into the new venv | | owner | done |
| 2 | Move the screen test driver and the wrong answer checks into the repo | | owner | done |
| 3 | Ignore the venv and the learner's own exercise files in git | | owner | done |
| 4 | Start local git and commit the work on a branch | | owner | pending |
| 5 | Verify the course runs from the new folder | | owner | pending |

## Done when

`git log` on the work branch holds every file of the course and the brain, `.venv/bin/python run.py dev check` passes from the new folder, and the test tools run from the repo.
