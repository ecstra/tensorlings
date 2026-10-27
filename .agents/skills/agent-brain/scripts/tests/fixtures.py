"""A small brain on disk, for tests that need one to work against."""

import tempfile
import unittest
from pathlib import Path

from brainlib import schema, store

INDEX_STUB = """---
name: {name}
kind: doc
description: {description}
updated: 2026-09-16
links: []
---

# {title}

Nothing here yet.
"""

TODO = """---
name: {name}
kind: todo
description: {description}
updated: 2026-09-16
links: []
status: {status}
by: owner
area: {area}
born: 2026-09-01
{closing}---

## What

{description}

## Done when

The check that closes it.
"""

BUG = """---
name: {name}
kind: bug
description: {description}
updated: 2026-09-16
links: []
status: open
severity: bug
area: auth
reported: 2026-09-02
---

## Problem

{description}

## Evidence

A log line.
"""

DECISION = """---
name: {name}
kind: decision
description: {description}
updated: 2026-09-16
links: []
date: 2026-09-03
area: storage
status: standing
---

## Decision

{description}
"""

DOC = """---
name: {name}
kind: doc
description: {description}
updated: 2026-09-16
links: []
---

# {title}

How it works.
"""

MEMORY = """---
name: {name}
kind: memory
description: {description}
updated: 2026-09-16
links: []
type: project
---

Stripe retries a failed webhook twice.
"""

LEDGER = """---
name: {name}
kind: review
description: {description}
updated: 2026-09-16
links: []
branch: feat/billing
status: open
---

## Summary

One branch.

## Batches

Batch 1, correctness.

## Findings

### F-01: a claim

- severity: bug
"""


def build(
    tmp: Path,
) -> Path:
    """Write a brain under `tmp` and hand back its `.agents/` path.

    The path is resolved, because `find_brain` resolves too and a macOS
    temp folder reaches the same directory by two spellings. The project
    around it keeps a .claude folder, so it carries the two front doors
    Claude Code reads, and no .gemini folder, so it owes no GEMINI.md.
    """
    root = tmp.resolve() / ".agents"

    for folder in ("docs", "memory", "todos", "bugs", "decisions", "plans", "reviews"):
        (root / folder).mkdir(parents=True, exist_ok=True)
        write(
            path=root / folder / "INDEX.md",
            text=INDEX_STUB.format(
                name=f"{folder}-index",
                description=f"index of {folder}",
                title=folder.title(),
            ),
        )

    (root / "spec").mkdir(parents=True, exist_ok=True)
    write(path=root / "SYSTEM.md", text="# System Rules\n\n## 1. Coding\n")
    write(path=root / "VERSION", text="1.0.0\n")
    write(
        path=root / "skills" / "agent-brain" / "SKILL.md",
        text="---\nname: agent-brain\ndescription: the brain tool\n---\n\nA stub, so the skill link has something to reach.\n",
    )
    write(
        path=root.parent / ".claude" / "output-styles" / "plain-and-short.md",
        text=(
            "---\nname: Plain & Short\ndescription: Answer first, then stop.\n"
            "keep-coding-instructions: true\n---\n\n# Plain & Short\n\n"
            "Answer first, then stop.\n"
        ),
    )
    write(
        path=root / "INDEX.md",
        text=INDEX_STUB.format(
            name="agent-index",
            description="the spine",
            title="Index",
        ),
    )

    write(
        path=root / "todos" / "TODO-001-export-to-csv.md",
        text=TODO.format(
            name="TODO-001-export-to-csv",
            description="export the table to csv",
            status="open",
            area="reports",
            closing="",
        ),
    )
    write(
        path=root / "todos" / "TODO-003-dark-mode.md",
        text=TODO.format(
            name="TODO-003-dark-mode",
            description="a dark theme",
            status="done",
            area="theme",
            closing="done: 2026-09-10\n",
        ),
    )
    write(
        path=root / "bugs" / "BUG-002-login-loop.md",
        text=BUG.format(
            name="BUG-002-login-loop",
            description="the sign in screen loops",
        ),
    )
    write(
        path=root / "decisions" / "DEC-001-postgres-over-sqlite.md",
        text=DECISION.format(
            name="DEC-001-postgres-over-sqlite",
            description="postgres won",
        ),
    )
    write(
        path=root / "docs" / "systems" / "login.md",
        text=DOC.format(
            name="login",
            description="the sign in flow",
            title="Login",
        ),
    )
    write(
        path=root / "memory" / "stripe-retries-twice.md",
        text=MEMORY.format(
            name="stripe-retries-twice",
            description="stripe retries a failed webhook twice",
        ),
    )
    write(
        path=root / "reviews" / "2026-09-feat-billing" / "ledger.md",
        text=LEDGER.format(
            name="2026-09-feat-billing",
            description="the billing review",
        ),
    )

    write(
        path=root.parent / "AGENTS.md",
        text="# AGENTS.md\n",
    )
    write(
        path=root.parent / "CLAUDE.md",
        text="@AGENTS.md\n",
    )

    return root


def write(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


OLD_RULES = """# CLAUDE.md

## 0. Project Rules

> The shell is a Flutter kit. Read [theme.md](.agent/docs/foundation/theme.md) first.

### 0.1 Gates

Run `flutter test` before a review.

## 1. Coding Guideline

The old copy of the kit's rules, replaced by the kit on upgrade.

## 2. Coding Style

More old rules.
"""

OLD_DOC = """---
name: theme
kind: doc
description: the design system
updated: 2026-08-13
links: [agent-readme]
---

# Theme

The palette lives in one place, see [the readme](../../README.md).
"""

OLD_BUG = """---
name: BUG-001-scrollbar
kind: bug
description: the notification stack shows no scrollbar on phones
updated: 2026-08-13
links: [theme]
status: open
severity: bug
area: phone
reported: 2026-07-15
---

## Problem

No scrollbar on phones.

## Evidence

Seen on an iPhone 12.
"""


def ship_review(
    root: Path,
) -> None:
    """Put the review skill into a test brain built before the kit shipped it, with the stamp that ships it."""
    since = next(skill.since for skill in schema.KIT_SKILLS if skill.name == "branch-review")
    write(
        path=root / "skills" / "branch-review" / "SKILL.md",
        text="---\nname: branch-review\ndescription: the code review\n---\n\nA stub, so the second skill link has something to reach.\n",
    )
    write(
        path=root / "VERSION",
        text=f"{since}\n",
    )


def build_old_project(
    root: Path,
) -> Path:
    """A project on the layout before 1.0.0: the brain in .agent/, the rules in CLAUDE.md, a copy in Agents.md.

    Its .gitignore carries Windows line endings, since a real one did.
    """
    write(path=root / "CLAUDE.md", text=OLD_RULES)
    write(path=root / "Agents.md", text=OLD_RULES.replace("# CLAUDE.md", "# Agents.md", 1))
    write(path=root / "README.md", text="# Shell\n\nThe design system is in [theme.md](.agent/docs/foundation/theme.md).\n")
    write(path=root / "CONTRIBUTING.md", text="# Contributing\n\nThe project's own words.\n")
    (root / ".gitignore").write_bytes(b"build/\r\n.agent/.worktrees/\r\n")
    write(
        path=root / ".agent" / "README.md",
        text=old_doc(name="agent-readme", description="the old map", title="The Agent Directory", body="Old.", links="[agent-index]"),
    )
    write(
        path=root / ".agent" / "house-rules.md",
        text=old_doc(name="house-rules-agent", description="the old house rules", title="House Rules", body="Old."),
    )
    write(
        path=root / ".agent" / "INDEX.md",
        text=old_doc(name="agent-index", description="the spine", title="Index", body="- [agent-readme](README.md), the map", links="[agent-readme]"),
    )
    write(
        path=root / ".agent" / "spec" / "conventions.md",
        text=old_doc(name="conventions", description="the old shared rules", title="Conventions", body="Old."),
    )
    write(path=root / ".agent" / "docs" / "foundation" / "theme.md", text=OLD_DOC)
    write(
        path=root / ".agent" / "docs" / "INDEX.md",
        text=old_doc(name="docs-index", description="the docs", title="Docs", body="- [theme](foundation/theme.md), the design system"),
    )
    write(path=root / ".agent" / "bugs" / "BUG-001-scrollbar.md", text=OLD_BUG)
    write(
        path=root / ".agent" / "bugs" / "INDEX.md",
        text=old_doc(name="bugs-index", description="the bugs", title="Bugs", body="- [BUG-001-scrollbar](BUG-001-scrollbar.md), no scrollbar"),
    )

    return root


def old_doc(
    name: str,
    description: str,
    title: str,
    body: str,
    links: str = "[]",
) -> str:
    """One brain file of the layout before 1.0.0, its links written as names the way that layout did."""
    return f"---\nname: {name}\nkind: doc\ndescription: {description}\nupdated: 2026-08-13\nlinks: {links}\n---\n\n# {title}\n\n{body}\n"


class BrainCase(unittest.TestCase):
    """A fresh fixture brain per test, in a temp folder cleaned up after.

    `self.root` is the brain, `self.brain` the store over it. A case that
    needs more calls `super().setUp()` first and adds its own.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = build(Path(self.tmp.name))
        self.brain = store.Brain(root=self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()
