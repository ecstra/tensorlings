---
name: DEC-015-the-course-lives-in-a-package-with-one-module-per-chapter
kind: decision
description: run.py becomes a thin entry point and the engine, app, lint and chapters live in the tensorlings package
updated: 2026-10-29
links: []
date: 2026-10-29
area: course
status: standing
---

## Decision

The course moves from one `run.py` into a `tensorlings/` package. `run.py` keeps the command and the check for missing packages, and the package splits the rest by job, with the engine that runs and checks a file, the exercise model, the lint, the progress, the app, the commands, and one module per chapter under `tensorlings/chapters/`. The chapters' order lives in `tensorlings/chapters/__init__.py`.

## Alternatives

One file for everything, which held phase 1a at 3,600 lines but would pass 70,000 at about 330 exercises. One module per section, which still leaves files of several thousand lines. The owner suggested several files over one monolith on 2026-10-29.

## Consequences

A chapter can be read and changed alone. The dev scripts load the package instead of `run.py`. A chapter module only imports from the engine and the exercise model, so adding a chapter touches its own module and the order in `__init__.py`.
