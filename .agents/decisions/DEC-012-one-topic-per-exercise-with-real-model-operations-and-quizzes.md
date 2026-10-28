---
name: DEC-012-one-topic-per-exercise-with-real-model-operations-and-quizzes
kind: decision
description: each exercise teaches one section topic, every task is an operation real models run, and a quiz combines a group of chapters with nothing new
updated: 2026-10-28
links: [decisions/DEC-011-lessons-teach-every-tool-and-tasks-build-a-model.md]
date: 2026-10-28
area: course
status: superseded
superseded-by: DEC-013
---

## Decision

Each exercise teaches one topic of its section, as rustlings gives each language topic its own folder. Every task is an operation real models run, such as adding a bias to every row of a batch or picking rows of a table by token id, and never a puzzle with no use. A quiz closes a group of chapters, as rustlings' quizzes do, and combines their tools with nothing new. From DEC-011 two rules stay. Each lesson teaches every tool its tasks need, and the lint fails a lesson or hint that shows a solution word for word.

## Alternatives

DEC-011 built Karpathy's makemore bigram model through the first six exercises. It was never in the agreed curriculum, it pulled probabilities, sampling and a loss from Section 2 into Section 1, and it gave every task the same 27 by 27 table, so a batch of examples by features never came up. Karpathy himself teaches the bigram model second, after micrograd. Before that, plain grid tasks like a times table read as leetcode to the owner.

## Consequences

Exercises share no model data or helpers. A task is named for the model operation it is, and the lesson says in a short paragraph where models use it, without teaching another section's ideas. The owner asked for this on 2026-10-28.
