---
name: agent-readme
kind: doc
description: what the .agents directory is and how a cold agent boots from it
updated: 2026-09-24
links: [INDEX.md, spec/conventions.md, house-rules.md]
---

# The Agent Directory

This directory is the project's brain: everything an agent needs to work
on this project, committed to the repo, all but what git ignores
there, so any agent on any machine resumes cold. Humans read docs/ at
the repo root when the project keeps one, agents read this.

## Booting

1. Read [INDEX.md](INDEX.md), the spine. What it holds, line by line,
   is in [spec/conventions.md](spec/conventions.md).
2. Read [house-rules.md](house-rules.md) before any branch, commit, or
   review work.
3. Obey [spec/](spec/conventions.md). Each content folder here has a spec
   file that states its format and lifecycle. A file that breaks its spec
   is a bug.

## The map

- `SYSTEM.md`: the rules every agent works by. Claude Code and Gemini
  CLI pull it in through the root `AGENTS.md`. Codex gets it from its
  session start hook. An agent that has neither opens it first.
- `spec/`: the law of this directory, one spec per content folder plus
  the shared conventions.
- `docs/`: the knowledge, categorized. What the product is and how it
  works.
- `memory/`: project facts learned along the way, one per file.
- `bugs/`: defects, one per file.
- `todos/`: work to do, one per file. The project's tracker.
- `plans/`: long horizon work, a goal with the steps that reach it, one
  per file, the steps a table the tool keeps.
- `decisions/`: decisions taken and why, one per file.
- `reviews/`: one folder per code review, the ledger inside.
- `handover/`: session handoffs, born with its first file.
- `skills/`: the kit's skills. `agent-brain` is the brain tool, the one
  way anything here is filed, changed, counted or checked, and
  `branch-review` is the code review the agent runs on the owner's
  word. Their files follow the Agent Skills standard rather than the
  specs above. A project may keep skills of its own here under other
  names, and doctor links them for Claude Code.
- `VERSION`: the kit release this brain is on, read by the tool.
- `.ack`: the owner's standing word to proceed and its end, the tool's
  own, which git ignores since it lives on one machine, and the
  conventions say the rest.
- `.task`: the current work, the chain from the plan down to the step
  the agent is on, which every hook prints, the tool's own and ignored
  the same way.

## Two absolutes

- Nothing secret ever enters this directory. All but what git ignores
  there is committed.
- Nothing here references any machine's local state. Machines point at
  this directory, never the reverse. What git ignores here, which the
  conventions list, is the one exception, since it never enters the
  repo.
