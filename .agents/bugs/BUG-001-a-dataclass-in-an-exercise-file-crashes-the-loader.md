---
name: BUG-001-a-dataclass-in-an-exercise-file-crashes-the-loader
kind: bug
description: the checker loads the learner's file without putting it in sys.modules, so a @dataclass there crashes
updated: 2026-10-30
links: []
status: open
severity: breaks
area: engine
reported: 2026-10-30
---

## Problem

A learner who writes `@dataclass` in an exercise file gets a crash when the checker loads it. The dataclass machinery looks up the class's module in `sys.modules`, and `load_module` in `tensorlings/engine.py` runs the file without registering it there.

## Evidence

The own module builder hit it on 2026-10-30 while writing own1's plan as a dataclass, and switched to a `typing.NamedTuple` to get around it. Section 14's own module project invites the learner to write any module of their own, so a dataclass there is likely.
