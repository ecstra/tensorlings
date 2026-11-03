---
name: BUG-001-a-dataclass-in-an-exercise-file-crashes-the-loader
kind: bug
description: the checker loads the learner's file without putting it in sys.modules, so a @dataclass there crashes
updated: 2026-11-03
links: []
status: fixed
severity: breaks
area: engine
reported: 2026-10-30
---

## Problem

A learner who writes `@dataclass` in an exercise file gets a crash when the checker loads it. The dataclass machinery looks up the class's module in `sys.modules`, and `load_module` in `tensorlings/engine.py` runs the file without registering it there.

## Evidence

The own module builder hit it on 2026-10-30 while writing own1's plan as a dataclass, and switched to a `typing.NamedTuple` to get around it. Section 14's own module project invites the learner to write any module of their own, so a dataclass there is likely.

## Fix

The loaded file never sat in `sys.modules`, and `dataclasses` reads `sys.modules[cls.__module__].__dict__` while it builds a class. `load_module` now puts the file's module in `sys.modules` while its statements run and takes it out after, so checks run again and again leave nothing behind. Commit 1637a4e, 2026-11-03.

## Verification

A scratch file holding a `@dataclass` and one instance loaded with 0 crashes and the instance in place, where before it crashed with `AttributeError: 'NoneType' object has no attribute '__dict__'`. The count of `sys.modules` entries was the same before and after the load. tensors1, own1, own2 and combine5 still pass the dev check.
