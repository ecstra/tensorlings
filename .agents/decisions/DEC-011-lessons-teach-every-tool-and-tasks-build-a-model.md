---
name: DEC-011-lessons-teach-every-tool-and-tasks-build-a-model
kind: decision
description: each lesson shows every tool its tasks use, and the tasks build a real model step by step instead of standalone puzzles
updated: 2026-10-28
links: [decisions/DEC-006-a-task-never-repeats-its-lesson.md]
date: 2026-10-28
area: course
status: standing
---

## Decision

The exercises build a real model step by step, as Karpathy's Zero to Hero does. The first six build his makemore bigram model and end with it rewritten as a layer. Each lesson teaches every tool its tasks need on its own small example, and the files give rustlings style scaffolding with the tensor work left as `TODO`s. The lint fails a lesson or hint that shows a task's solution word for word.

## Alternatives

DEC-006 had each task change something its lesson held fixed, with the lint ignoring variable names. That pushed the lessons to leave tools out, so a task needed `.numel()` that no lesson showed, and the tasks became puzzles like times tables and rescaled grids. The owner called them leetcode and asked for rustlings or Zero to Hero style.

## Consequences

Lessons may show a tool on a small example in the same form the task needs it, and the learning comes from applying it to the model. Every exercise in a chapter shares the model's data and helpers.
