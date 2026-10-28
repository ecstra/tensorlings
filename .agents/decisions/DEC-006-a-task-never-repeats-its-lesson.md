---
name: DEC-006-a-task-never-repeats-its-lesson
kind: decision
description: the lesson shows one case and the task changes something about it, and the lint fails a lesson that holds a solution line
updated: 2026-10-28
links: []
date: 2026-10-28
area: course
status: standing
---

## Decision

A lesson teaches with one example, and the task changes one thing the example held fixed, such as the dimension, the shape or the operation. No line of a solution appears in its lesson, and the lesson lint checks that.

## Alternatives

Lessons that showed the exact line the task needed, as all six of phase 1a did. The owner called retyping a shown answer "meh", and research agrees it feels fluent but teaches little.

## Consequences

Tasks take more thought and need hints that climb to the answer. The lesson lint compares each solution's code lines against the lesson text.
