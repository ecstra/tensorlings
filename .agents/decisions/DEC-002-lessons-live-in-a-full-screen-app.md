---
name: DEC-002-lessons-live-in-a-full-screen-app
kind: decision
description: tensorlings shows each lesson in a full-screen rich app and keeps the exercise files to code and TODOs
updated: 2026-10-28
links: []
date: 2026-10-27
area: course
status: superseded
superseded-by: DEC-004
---

## Decision

Each lesson, task and result shows in a full-screen app drawn with the `rich` package in the terminal's alternate screen. Exercise files hold only code, `TODO` markers and a closing list of what gets checked. The owner picked this option.

## Alternatives

Lessons as comments in the exercise files, exactly as rustlings does, and lessons in README files. The owner never opens READMEs, and the comment lessons read as confusing. A plain screen that only clears was also offered and lost, since the earlier screen scrolled instead of redrawing.

## Consequences

`rich` is a dependency, pinned in `requirements.txt`. Lessons must fit one screen or scroll inside their panel.
