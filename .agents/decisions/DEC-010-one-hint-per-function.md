---
name: DEC-010-one-hint-per-function
kind: decision
description: each function in an exercise has one hint, the Hint tab shows the ones for unfinished functions, and nothing is locked
updated: 2026-10-28
links: [decisions/DEC-005-hints-open-in-three-levels-one-per-new-run.md]
date: 2026-10-28
area: course
status: standing
---

## Decision

Each function in an exercise has one hint. The Hint tab shows the hints for the functions that don't pass yet, and `2` or `h` opens it. No hint is locked behind a save. The owner asked for this.

## Alternatives

Three levels per exercise, each opening after one more run, DEC-005, lost. With several functions in a file one set of levels could not fit them all, and the owner found waiting for a save before the next level pointless.

## Consequences

Hints stop short of the full answer, since the solution file appears once the exercise passes. Every task function needs its own hint.
