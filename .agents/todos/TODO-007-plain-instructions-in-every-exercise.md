---
name: TODO-007-plain-instructions-in-every-exercise
kind: todo
description: every TODO says the operation in plain words with an example, then why it exists, after the owner could not follow tensors1
updated: 2026-10-28
links: []
status: done
by: owner
area: course
born: 2026-10-28
done: 2026-10-28
---

## What

"Honestly not good. I dont even understand the first one."

"What even is a batch? What is this even asking? Not great. Please rework. Like why is this so compllicated when all it could ask was "convert the given list to tensor" or something? and why is that even there? huh? unclear instructinos"

## Why

The TODOs described each task in deep learning words, like batch, example and feature, instead of saying the operation plainly, and never said why the task exists.

## Steps

| # | Step | Points at | By | Status |
|---|---|---|---|---|
| 1 | Rewrite every TODO as a plain instruction, an example call with its result, and a short why | | owner | done |
| 2 | Rename functions to plain names and trim deep learning words from the lessons | | owner | done |
| 3 | Lint the starting files' comments for terms no lesson explained | | owner | done |
| 4 | Run the dev check, the wrong answers and the screen driver | | owner | done |
| 5 | Reset the owner's exercise files to the new versions | | owner | done |
| 6 | try tensors1 and indexing1 in the new exercise method | TODO-008 | agent | done |

## Done when

Every TODO in the seven exercises reads as a plain instruction with an example, explains any deep learning word it uses and carries no why section, every check passes, and the owner has tried tensors1 and understood it.

Met on 2026-10-28 through TODO-008. Commits 94e3ec2 and 53c8d04 made the TODOs plain, and the owner then approved tensors1 and indexing1 in the explain then try format.
