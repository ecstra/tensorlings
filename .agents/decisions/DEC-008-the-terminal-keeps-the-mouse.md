---
name: DEC-008-the-terminal-keeps-the-mouse
kind: decision
description: the app runs without capturing the mouse, so the terminal selects text and cmd+c copies it
updated: 2026-10-28
links: [decisions/DEC-004-the-app-is-built-on-textual.md]
date: 2026-10-28
area: course
status: standing
---

## Decision

The app runs with Textual's mouse capture off. Dragging makes the terminal's own selection and cmd+c copies it. The wheel still scrolls the open tab, since terminals send arrow keys for the wheel in a full-screen app. The owner picked this.

## Alternatives

Keeping Textual's mouse capture lost. Cmd+C never reaches a terminal app on a Mac, so copying needed ctrl+c inside the app, and the owner wanted cmd+c. Copying every selection on mouse release was tried first, and the owner turned it off.

## Consequences

Tabs, list rows and buttons cannot be clicked, so every action has a key. Textual's own text selection and copying go unused.
