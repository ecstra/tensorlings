---
name: decisions-index
kind: doc
description: index of every decision taken, newest first
updated: 2026-10-28
links: [INDEX.md, spec/decisions.md]
---

# Decisions Index

Every decision taken, newest first.
Filed per [spec/decisions.md](../spec/decisions.md).
Built by the brain tool from the folder it lists.

- [DEC-008](DEC-008-the-terminal-keeps-the-mouse.md), the app runs without capturing the mouse, so the terminal selects text and cmd+c copies it
- [DEC-007](DEC-007-lesson-hint-and-result-are-tabs.md), the main screen shows the lesson, the hints and the result as full height tabs, and a save switches to the result
- [DEC-006](DEC-006-a-task-never-repeats-its-lesson.md), the lesson shows one case and the task changes something about it, and the lint fails a lesson that holds a solution line
- [DEC-005](DEC-005-hints-open-in-three-levels-one-per-new-run.md), each exercise has three hints, where to look, the idea, then the answer, and the next opens only after the learner runs the checks again
- [DEC-004](DEC-004-the-app-is-built-on-textual.md), tensorlings draws its full-screen app with Textual instead of a hand built rich layout
- [DEC-003](DEC-003-every-exercise-is-code.md), tensorlings has no prediction only exercises, every exercise is code to write or fix
- [DEC-002](DEC-002-lessons-live-in-a-full-screen-app.md), tensorlings shows each lesson in a full-screen rich app and keeps the exercise files to code and TODOs (superseded by DEC-004)
- [DEC-001](DEC-001-curriculum-from-a-survey-of-expert-courses.md), the tensorlings curriculum comes from what expert syllabi and textbooks agree on, not from one framing of modules
