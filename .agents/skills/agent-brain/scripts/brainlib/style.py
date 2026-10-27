"""The chat style, written once for Claude Code and copied for the others.

`.claude/output-styles/plain-and-short.md` is the source, since Claude
Code reads it as an output style. Codex and Gemini CLI have no such
thing, so each gets the same text as a plain file inside its own config
folder, and doctor keeps the copies identical to the source.
"""

import os
from pathlib import Path
from typing import Optional

from . import frontmatter, store

SOURCE = Path(".claude") / "output-styles" / "plain-and-short.md"

COPIES: tuple[Path, ...] = (
    Path(".codex") / "plain-and-short.md",
    Path(".gemini") / "plain-and-short.md",
)


class StyleError(Exception):
    """The source style file is missing or cannot be read."""


def project(
    brain: store.Brain,
) -> Path:
    return brain.root.parent


def folders(
    brain: store.Brain,
) -> list[Path]:
    """Each agent's own folder, which holds its hook config and, under Claude Code's, the style and the sub agent files the tool writes."""
    found: list[Path] = []
    for relative in (SOURCE, *COPIES):
        folder = project(brain=brain) / relative.parts[0]
        if folder not in found:
            found.append(folder)

    return found


def source_body(
    brain: store.Brain,
) -> str:
    """The style text without the frontmatter only Claude Code reads."""
    source = project(brain) / SOURCE
    if not store.readable_file(source):
        raise StyleError(f"{SOURCE} is missing, so there is no style to copy")

    try:
        document = frontmatter.read(str(source))
    except frontmatter.FrontmatterError as error:
        raise StyleError(f"{SOURCE}: {error}") from error

    return document.body.lstrip("\n")


def stale(
    brain: store.Brain,
) -> list[Path]:
    """The copies that are missing or differ from the source."""
    return stale_against(brain=brain, wanted=source_body(brain))


def expected(
    brain: store.Brain,
) -> list[Path]:
    """The copies the project owes, one per agent folder it keeps.

    A project with no `.codex/` or `.gemini/` folder runs no such agent,
    so it owes no copy there and the tool never conjures the folder.
    """
    return [
        relative
        for relative in COPIES
        if store.quietly(lambda: (project(brain) / relative.parent).is_dir())
    ]


def stale_against(
    brain: store.Brain,
    wanted: str,
) -> list[Path]:
    """The copies owed that do not say `wanted`, for a caller that read the source once."""
    return [relative for relative in expected(brain) if copy_text(project(brain) / relative) != wanted]


def copy_text(
    target: Path,
) -> Optional[str]:
    """What a copy says, read the way it was written, or None when it cannot be read.

    Text mode folds line endings both ways, so a copy written on Windows
    compares equal to the source rather than reading as stale forever. A
    copy that cannot be opened or decoded is stale, and gets rewritten.
    """
    if not store.readable_file(target):
        return None

    try:
        return target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def write(
    brain: store.Brain,
) -> list[Path]:
    """Rewrite the stale copies from the source, and say which ones."""
    wanted = source_body(brain)
    changed = stale_against(brain=brain, wanted=wanted)

    for relative in changed:
        write_one(target=project(brain) / relative, text=wanted)

    return changed


def write_one(
    target: Path,
    text: str,
) -> None:
    """Write one copy as a fresh real file, whole or not at all, into a folder that is there.

    A copy the process may not write goes first, since the copies are
    the tool's and nothing in one is worth keeping.
    """
    if store.readable_file(target) and not os.access(str(target), os.W_OK):
        target.unlink()
    frontmatter.write_text(path=str(target), text=text)
