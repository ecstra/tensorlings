---
name: DEC-007-lesson-hint-and-result-are-tabs
kind: decision
description: the main screen shows the lesson, the hints and the result as full height tabs, and a save switches to the result
updated: 2026-10-28
links: [decisions/DEC-004-the-app-is-built-on-textual.md]
date: 2026-10-28
area: course
status: standing
---

## Decision

The main screen holds three tabs, Lesson, Hint and Result, each with the whole screen. `1`, `2` and `3` switch them, and each save switches to Result by itself. The Result tab's label shows how many functions pass. The owner picked both choices.

## Alternatives

Stacked boxes with the lesson above the result kept both in view, but each box was too small on a laptop terminal, and the owner asked whether tabs were better. Staying on the current tab after a save lost, since the learner saves to see the outcome, as in rustlings.

## Consequences

The lesson and the result are never on screen together, so the result has to stand alone. Each failing function's explanation says what was checked and what came back.
