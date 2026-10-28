---
name: DEC-004-the-app-is-built-on-textual
kind: decision
description: tensorlings draws its full-screen app with Textual instead of a hand built rich layout
updated: 2026-10-28
links: [decisions/DEC-002-lessons-live-in-a-full-screen-app.md]
date: 2026-10-28
area: course
status: standing
---

## Decision

Each lesson, hint, result and the exercise list show in a full-screen app built on Textual, pinned in `requirements.txt`. Exercise files still hold only code, `TODO` markers and a closing list of what gets checked. The owner picked this after asking for a better TUI, "like claude code".

## Alternatives

Staying on the hand built `rich` layout from DEC-002 lost. It could scroll only one pane, had no mouse wheel, and every fix meant writing focus, scrolling and mouse handling by hand, which is what Textual already is. prompt_toolkit and urwid have no Markdown or footer.

## Consequences

The mouse is captured, so the app copies selected text with `pbcopy` as well as OSC 52. The app polls the exercise file's time, since Textual has no file watcher. Screens are tested headless with Textual's Pilot and SVG screenshots instead of a pseudo terminal.
