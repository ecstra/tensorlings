---
name: DEC-009-the-app-keeps-the-mouse-and-option-drag-copies
kind: decision
description: the app captures the mouse so tabs are clickable, and Option drag makes the terminal selection that cmd+c copies
updated: 2026-10-28
links: [decisions/DEC-008-the-terminal-keeps-the-mouse.md]
date: 2026-10-28
area: course
status: standing
---

## Decision

The app captures the mouse again, so tabs and list rows can be clicked. To copy with cmd+c, the learner holds Option while dragging, which makes VS Code's own terminal selection. The project's `.vscode/settings.json` turns on `terminal.integrated.macOptionClickForcesSelection` for that. In macOS Terminal the key is Fn. The owner asked for this after trying the other way.

## Alternatives

Leaving the mouse to the terminal, DEC-008, lost. Tabs could not be clicked, and the owner found cmd+c did not copy there either. Cmd+C cannot reach the app at all, because the terminal keeps it, so the app cannot copy its own selection on cmd+c. Copying on mouse release worked and the owner turned it off.

## Consequences

Copying needs Option held during the drag. A plain drag makes Textual's selection, which ctrl+c copies.
