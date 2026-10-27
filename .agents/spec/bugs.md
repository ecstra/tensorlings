---
name: spec-bugs
kind: doc
description: format and lifecycle for .agents/bugs, the defect tracker
updated: 2026-09-24
links: [spec/conventions.md, spec/todos.md, spec/reviews.md]
---

# Spec: bugs/

Defects: the product misbehaving. One bug per file, so status changes
edit fields instead of growing lines, and the file's git history is the
bug's history.

## A bug is not a todo

A bug has evidence of wrong behavior. A todo is work to do. Neither may
hold the other: a wish filed here is refiled as a todo, a defect filed
there as a bug, both through `brain refile --kind`. The old file stays,
marked `refiled`, and the two link to each other. The agent classifies
on intake without asking, and cross links carry any relationship
(`links:`, `parent:`).

## Format

Standard frontmatter, `kind: bug`, plus:

```
status: open | branched | fixed | wontfix | refiled
severity: breaks | bug | debt | nit
area: <the project's own area vocabulary, kept consistent>
reported: <YYYY-MM-DD>
became: <the id of the todo it was refiled as, written by the refile>
parent: <name of the bug or finding that spawned this one, if any>
```

`branched` means a branch is carrying the fix right now (name it in the
body). `wontfix` requires the reason written in the body. `refiled`
means the file turned out to be a todo, which the tool filed, named in
`became` and linked here, so a refiled bug always links to a todo.

Body sections, in order:

```
## Problem
What misbehaves, as observed.

## Evidence
Logs, field notes, screenshots described, and the steps that reproduce
it, which are evidence and never the fix's steps below.

## Steps
The same table a plan carries, per spec/plans.md, the tool's own, one
step of the fix per row. Optional, and the task chain reads it. Listed
as numbered lines on filing, they become the table, and a bug with
steps goes fixed once every one is done or skipped.

## Fix
Root cause, the commit hash, the date. Absent while open.

## Verification
How it was proven gone.
```

## Lifecycle

Born on intake (a field report, a hunter find, a review finding deferred
past its branch). Fixed bugs keep their file with status flipped and the
Fix section filled, nothing is deleted. The INDEX lists open bugs first,
grouped by severity, then fixed, wontfix and refiled.
