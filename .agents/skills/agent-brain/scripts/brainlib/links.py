"""What a file points at, what points back, and whether any of it holds.

A frontmatter link is a path relative to `.agents/`. Older brains wrote
names instead, so resolving still accepts an id or a name, and the
upgrade turns them into paths.
"""

import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

from . import frontmatter, sections, store

# The two ways markdown writes a link: inline, `[text](target)`, and by
# reference, `[label]: target` on a line of its own. Group 1 is the
# target in both, so one rewrite serves both.
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\((<[^>\n]*>|[^)\s]+)\)")
REFERENCE_LINK = re.compile(r"^\[[^\]]+\]:[ \t]+(<[^>\n]*>|\S+)", re.MULTILINE)

# The scratch folder a symlink is put back through, the prefix and the
# random tail TemporaryDirectory adds. Doctor knows the exact shape, so
# one left by an interrupted run is found and nothing else is touched.
SCRATCH_PREFIX = ".putback."
SCRATCH_SHAPE = re.compile(r"^\.putback\.[A-Za-z0-9_]+$")


def resolve(
    brain: store.Brain,
    reference: str,
) -> Optional[Path]:
    """The file a link means, whether it was written as a path or a name."""
    direct = inside(brain=brain, reference=reference)
    if direct is not None and store.readable_file(direct):
        return direct

    try:
        entry = brain.find(reference)
    except store.Ambiguous:
        entry = None

    if entry is not None:
        return entry.path

    for path, document in kit_documents(brain):
        if document.get("name") == reference:
            return path

    return None


def inside(
    brain: store.Brain,
    reference: str,
) -> Optional[Path]:
    """The file a path reference names, or None when it lies outside the brain.

    An absolute path replaces the root when joined, and `..` climbs out
    of it, so both are checked against the root before anything is read.
    A dotted path is nobody's entry, so a link to one reaches nothing.
    """
    if not reference:
        return None

    candidate = brain.root / reference
    relative = within(brain=brain, path=candidate)
    if relative is None or store.hidden(Path(relative)):
        return None

    return candidate


def within(
    brain: store.Brain,
    path: Path,
) -> Optional[str]:
    """The path relative to the brain, or None when it lies outside.

    Nothing is resolved. A symlink filed inside the brain is inside it,
    and `..` is folded lexically, so a path cannot climb out unseen and
    a symlink loop on disk cannot raise here.
    """
    root = Path(os.path.normpath(str(brain.root)))
    folded = Path(os.path.normpath(str(path)))

    try:
        return str(folded.relative_to(root))
    except ValueError:
        return None


def canonical(
    brain: store.Brain,
    reference: str,
) -> Optional[str]:
    """The path form of a link, or None when it points at nothing."""
    path = resolve(brain=brain, reference=reference)
    if path is None:
        return None

    return within(brain=brain, path=path)


def outgoing(
    entry: store.Entry,
) -> list[str]:
    return entry.document.get_list("links")


def backlinks(
    brain: store.Brain,
    target: store.Entry,
) -> list[store.Entry]:
    """Every entry whose frontmatter links reach `target`."""
    found: list[store.Entry] = []

    for entry in brain.entries():
        if entry.path == target.path:
            continue

        resolved = [resolve(brain=brain, reference=link) for link in outgoing(entry)]
        if target.path in resolved:
            found.append(entry)

    return found


def broken_in_body(
    path: Path,
    document: frontmatter.Document,
) -> list[str]:
    """Markdown links in the prose that do not reach a file from where they stand.

    A body link is relative to its file, so an absolute one is broken
    however the machine it was written on is laid out. A link that
    starts inside code is not a link, while one whose text holds a code
    span is.
    """
    found: list[str] = []

    for target in body_targets(document.body):
        if "://" in target or target.startswith("#"):
            continue

        file_part = target.split("#")[0]
        if not file_part or Path(file_part).is_absolute() or not store.quietly((path.parent / file_part).exists):
            found.append(target)

    return found


def body_targets(
    body: str,
) -> list[str]:
    """Every link target in the prose, inline and reference style, code stepped over."""
    spans = sections.code_spans(body)

    return [
        bare(match.group(1))
        for pattern in (MARKDOWN_LINK, REFERENCE_LINK)
        for match in pattern.finditer(body)
        if not sections.in_code(spans=spans, position=match.start())
    ]


def bare(
    target: str,
) -> str:
    """The target without the angle brackets markdown allows around one."""
    if target.startswith("<") and target.endswith(">"):
        return target[1:-1]

    return target


def documents(
    brain: store.Brain,
) -> list[tuple[Path, frontmatter.Document]]:
    """Every file that carries links: the entries and the kit's own files."""
    found = [(entry.path, entry.document) for entry in brain.entries()]

    return found + kit_documents(brain)


@dataclass
class Planned:
    """One file about to have its links moved, with what it was before.

    `links` is the header list with the move made, or None when the
    header does not point there. `body` is the prose with its markdown
    links moved, or None when none of them point there.
    """

    path: Path
    document: frontmatter.Document
    links: Optional[list[str]]
    body: Optional[str]
    text: str
    link: Optional[str]


def rewrite_references(
    brain: store.Brain,
    old: str,
    new: str,
    stamp: str = "",
) -> list[Path]:
    """Point every frontmatter link at `old` to `new`, and say which files changed."""
    return [item.path for item in move_links(brain=brain, old=old, new=new, stamp=stamp)]


def move_links(
    brain: store.Brain,
    old: str,
    new: str,
    stamp: str = "",
) -> list[Planned]:
    """Point every link at `old` to `new`, in headers and in prose, stamping each file touched.

    Paths are root relative. A header link is rewritten in path form, a
    markdown link in the prose keeps its relative form from the file it
    sits in. A file whose links change has changed, so it gets the date
    like any other edit, per the upkeep rule. The writes go all together
    or not at all: every file is read before any is written, and a file
    that cannot be written puts the ones before it back as they were,
    symlink included, so no link points at a path the rename then never
    made. What comes back is each file written with what it held before,
    so a caller that fails later can put them back too.
    """
    stamp = stamp or date.today().isoformat()
    planned: list[Planned] = []
    for path, document in documents(brain):
        moved = moved_links(brain=brain, document=document, old=old, new=new)
        body = moved_body(brain=brain, path=path, document=document, old=old, new=new)
        if moved is None and body is None:
            continue

        planned.append(Planned(path=path, document=document, links=moved, body=body, text=frontmatter.load(str(path)), link=link_target(path)))

    written: list[Planned] = []

    try:
        for item in planned:
            if item.links is not None:
                item.document.set_list(key="links", values=item.links)
            if item.body is not None:
                item.document.body = item.body
            item.document.set(key="updated", value=stamp)
            frontmatter.write(path=str(item.path), document=item.document)
            written.append(item)
    except OSError as error:
        stuck = put_back_all(written)
        if stuck:
            raise OSError(f"{store.explain(error)}, and {names_of(stuck)} could not be put back") from error
        raise
    finally:
        brain.refresh()

    return written


def put_back_all(
    written: list[Planned],
) -> list[Path]:
    """Restore every file it can, and hand back the ones it could not."""
    stuck: list[Path] = []

    for item in written:
        try:
            put_back(item)
        except OSError:
            stuck.append(item.path)

    return stuck


def names_of(
    paths: list[Path],
) -> str:
    return ", ".join(store.near(str(path)) for path in paths)


def link_target(
    path: Path,
) -> Optional[str]:
    if not store.quietly(path.is_symlink):
        return None

    return os.readlink(str(path))


def put_back(
    item: Planned,
) -> None:
    """Restore a file the way it was, a symlink as the symlink it was.

    The symlink is made in a scratch folder beside the file and moved
    into place in one step, so there is no moment with nothing at the
    path, and no fixed name for a leftover to collide with.
    """
    path = item.path
    if item.link is None:
        frontmatter.write_text(path=str(path), text=item.text)
        return

    with tempfile.TemporaryDirectory(dir=str(path.parent), prefix=SCRATCH_PREFIX) as scratch:
        beside = os.path.join(scratch, path.name)
        os.symlink(item.link, beside)
        os.replace(beside, str(path))


def moved_body(
    brain: store.Brain,
    path: Path,
    document: frontmatter.Document,
    old: str,
    new: str,
) -> Optional[str]:
    """The file's prose with every markdown link at `old` pointed at `new`, or None when none is.

    A body link is relative to the file it sits in, so the new target is
    written relative to that file too, anchor kept. Everything else in
    the prose is left character for character.
    """
    old_target = os.path.normpath(str(brain.root / old))
    folder = path.parent
    spans: list[tuple[int, int]] = []

    def moved(match: re.Match) -> str:
        target = bare(match.group(1))
        if sections.in_code(spans=spans, position=match.start()):
            return match.group(0)

        if not target or "://" in target or target.startswith("#") or Path(target).is_absolute():
            return match.group(0)

        file_part, hash_sign, anchor = target.partition("#")
        if os.path.normpath(str(folder / file_part)) != old_target:
            return match.group(0)

        replacement = os.path.relpath(str(brain.root / new), str(folder)) + hash_sign + anchor
        if target != match.group(1):
            replacement = f"<{replacement}>"
        head = match.group(0)[: match.start(1) - match.start(0)]
        tail = match.group(0)[match.end(1) - match.start(0) :]

        return head + replacement + tail

    body = document.body
    for pattern in (MARKDOWN_LINK, REFERENCE_LINK):
        spans = sections.code_spans(body)
        body = pattern.sub(moved, body)

    return body if body != document.body else None


def moved_links(
    brain: store.Brain,
    document: frontmatter.Document,
    old: str,
    new: str,
) -> Optional[list[str]]:
    """The header's links with `old` pointed at `new`, or None when none point there."""
    current = document.get_list("links")
    moved = [new if link == old or canonical(brain=brain, reference=link) == old else link for link in current]
    if moved == current:
        return None

    return list(dict.fromkeys(moved))


def kit_documents(
    brain: store.Brain,
) -> list[tuple[Path, frontmatter.Document]]:
    """The kit's own files that carry frontmatter, so names still resolve."""
    return brain.kit_documents()
