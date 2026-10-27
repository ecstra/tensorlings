---
name: spec-plans
kind: doc
description: format and lifecycle for .agents/plans, a goal with the steps that reach it
updated: 2026-09-25
links: [spec/conventions.md, spec/todos.md, spec/bugs.md, spec/reviews.md]
---

# Spec: plans/

Long horizon work. One file per plan, named for the goal, a goal the
owner and the agent agreed on with the steps that reach it. A plan's
steps each carry work of their own, a todo, a bug or a review where one
exists, and the plan's own Done when is the check beyond all of them.
The steps are a table the tool keeps and the agent grows as the work
teaches it what comes next, so a plan that starts as a goal alone is
legal and means "find out".

## Format

Standard frontmatter, `kind: plan`, plus:

```
status: open | done | dropped
by: owner | agent
born: <YYYY-MM-DD>
done: <YYYY-MM-DD, when closed>
```

Body sections, in order:

```
## Why
The goal and what it is for, in the words of whoever set it.

## Steps
The table below, the tool's own.

## Done when
The check that closes the plan, beyond every step being settled.
```

The steps table:

```
| # | Step | Points at | By | Status |
|---|---|---|---|---|
| 1 | auto ack, the hook line, the ledger's auto ack | TODO-014 | owner | done |
| 2 | compaction, the current task line, the handover | TODO-015 | owner | pending |
| 3 | an independent pass over main, the owner's run | | owner | pending |
```

A step is numbered from 1 and never renumbered or deleted. It points
at a todo or a bug by id, which a retitle never moves, or by name for
a file with no id, which doctor reports and a retitle moves the cell
along with, or at a review by name, or at nothing when it is a plain
line such as something the owner does themselves. `By` says who set
the step, `owner` or `agent`, so the agreed steps read apart from the
ones the agent added on its own. `Status` is `pending`, `done` or
`skipped`, and a step that points at an entry takes its status from
that entry, done when the todo is done, the bug fixed or wontfix, or
the review closed, skipped when the entry was dropped, and a step
whose todo or bug was refiled follows it to the entry filed in its
place. While the plan is open, every reader reads a pointed step off
its entry rather than off the cell, so a mark that went stale in the
file misleads nothing, and doctor repairs the cell. Once the plan is
closed its table is its record, read as the file holds it, marked no
more, and settled from the entries at the close. Only a step that
points at nothing is marked by hand, or one whose pointer reaches
nothing the brain can read, and a step that turned out wrong is
skipped rather than removed. Two entries whose steps point at each
other would wait on each other for good, so the second pointer is
refused.

## The tool

`brain new plan` files one from its prose, the steps it lists under
Steps turned into rows, or an empty table when it lists none.
`brain steps add` appends steps, numbered after the last, `brain steps
step` marks a plain step, and `brain steps` shows the table with its
marks. The tool refreshes a pointed step's mark whenever the entry it
points at changes status, while the plan is open, since a closed
plan's table is its record and is marked no more. Doctor reports and
repairs a mark that drifted, a pointer left on a refiled entry or
written by name, a list under Steps in the table's place and a plan
with no Steps section, which get the table, and it reports an id a
step names that reaches nothing, a became field naming nothing or the
wrong thing, two open entries whose steps wait on each other, and a
done plan with a step still pending, a dropped one closing as it is.
The same table, the same commands and the same checks serve a todo or
a bug whose work has steps, per [spec/todos.md](todos.md) and
[spec/bugs.md](bugs.md).

## Lifecycle

A plan closes with `status: done` once every step is done or skipped,
which the tool refuses before then, or `dropped` when the goal is
abandoned, and it carries its closing date either way. Nothing moves
and nothing is deleted, since the steps are the record of how the goal
was reached. One plan is the current work at a time, the root of the
task chain the brain keeps beside the owner's word, and
[spec/conventions.md](conventions.md) says where that chain lives.
