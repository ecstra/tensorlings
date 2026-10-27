---
name: spec-todos
kind: doc
description: format and lifecycle for .agents/todos, the work tracker
updated: 2026-09-24
links: [spec/conventions.md, spec/bugs.md]
---

# Spec: todos/

Work to do: features, changes, chores. One item per file. This is the
project's single tracker, the agent's and the owner's both, and the agent
uses it extensively: multi step work gets filed here instead of carried
in a session's head, so any session can pick it up.

## Format

Standard frontmatter, `kind: todo`, plus:

```
status: open | done | dropped | refiled
by: owner | agent
area: <the project's own area vocabulary, kept consistent>
born: <YYYY-MM-DD>
done: <YYYY-MM-DD, when closed>
became: <the id of the bug it was refiled as, written by the refile>
```

Body sections:

```
## What
The work, in the words of whoever asked. Owner wording is preserved
verbatim, cleaning touches structure only, never voice.

## Why
One or two lines when the what does not carry it. Optional.

## Steps
The same table a plan carries, per spec/plans.md, the tool's own.
Optional, and the task chain reads it, so multi step work carries one.
Listed as numbered lines on filing, they become the table, and a todo
with steps goes done once every one is done or skipped.

## Done when
The check that closes it. A closed item names the commit(s) here.
```

## The intake rule

Any todo the owner writes, anywhere (chat, a scratch note, a list), the
agent picks up, cleans, and files here with `by: owner`. When the agent
completes one, it marks it done here AND tells the owner which of their
own notes to tick, by name.

## Lifecycle

Done and dropped items keep their files with status flipped, nothing is
deleted and nothing grows inline paragraphs: completion notes go in Done
when, never appended to What. The INDEX lists every open item one per
line, grouped by area, and summarizes done, dropped and refiled by area
with counts (each closed file carries its own description, the id finds
it).
A todo that turns out to describe a defect is refiled as a bug through
the tool, `brain refile --kind bug`. The todo stays, marked `refiled`
with its closing date, names the bug in `became`, and the two link to
each other, so the id trail holds and no id is ever reused. A refiled
todo always links to a bug.
