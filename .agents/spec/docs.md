---
name: spec-docs
kind: doc
description: format and folder rules for .agents/docs, the knowledge library
updated: 2026-09-22
links: [spec/conventions.md]
---

# Spec: docs/

The knowledge library. What the product is, how each system works, what
things cost. This is the one home of the project's durable knowledge:
humans get the front door (the root README, a small docs/ set if the
project keeps one), agents get the library. One fact lives in one place,
a doc never restates another doc's content, it links.

## Format

Standard frontmatter, `kind: doc`. The body is headings and terse prose.
Every claim about the code names the file it is true in.

## Categories

Docs live in category folders, each with enough members to earn it. A
sensible starting set, none mandatory:

- `foundation/`: what the product is built on. Architecture, stack,
  capacity.
- `systems/`: one doc per built system.
- `ops/`: how to run, ship, and operate it.
- `research/`: studied, not built.

The agent creates a new category folder when docs accumulate that fit
none of the existing ones (three or more is the bar). The tool files
the first doc into it with `brain new --category`, and the index picks
the folder up on the same run. A single stray doc goes in the nearest
existing category instead.

## Lifecycle

Docs are living: a code change that falsifies a doc updates the doc in
the same round, its `updated:` date moves through `brain set`, and the
index follows on the tool's next run, per the upkeep rule in
conventions.md. A doc for a removed feature is deleted, not archived,
git remembers.
