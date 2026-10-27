"""Answering questions about the brain from the files, never from memory.

Every function returns plain data, so the command line can print it as
text for a person or as JSON for a hook. Counting happens here, on the
files as they are right now.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Optional

from . import doctor, frontmatter, links, schema, store, task


def version(
    brain: store.Brain,
) -> str:
    stamp = brain.root / schema.VERSION_FILE
    if not store.readable_file(stamp):
        return "unversioned"

    try:
        return store.stamp_text(stamp)
    except (OSError, UnicodeDecodeError):
        return "unreadable"


def status(
    brain: store.Brain,
) -> dict:
    """The whole picture in one structure: version, health, and counts."""
    findings = doctor.run(brain)
    counts = {kind.name: counted(entries=brain.entries_of(kind), kind=kind) for kind in schema.KINDS.values()}
    reviews = [entry for entry in brain.entries_of(schema.kind_named("review")) if entry.field("status") == "open"]
    handovers = brain.entries_of(schema.kind_named("handover"))

    return {
        "version": version(brain),
        "findings": len(findings),
        "fixable": sum(1 for finding in findings if finding.fixable),
        "counts": counts,
        "open_reviews": [entry.name for entry in reviews],
        "latest_handover": handovers[-1].relative if handovers else None,
        "task": chain_or_none(brain),
        "unreadable": [str(path.relative_to(brain.root)) for path in brain.unreadable],
    }


def chain_or_none(
    brain: store.Brain,
) -> Optional[list[dict]]:
    """The chain as it stands, or None when the file does not read as one, which the text form says in words."""
    written = task.read(brain)
    if written is None:
        return None

    return [asdict(item) for item in task.current(brain=brain, chain=written)]


def counted(
    entries: list[store.Entry],
    kind: schema.Kind,
) -> dict[str, int]:
    """The kind's files by status, or just how many are filed when the kind has no status."""
    if any(field.name == "status" for field in kind.fields):
        return tally(entries=entries, field="status")

    return {"filed": len(entries)} if entries else {}


def tally(
    entries: list[store.Entry],
    field: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}

    for entry in entries:
        value = entry.field(field) or schema.UNKNOWN
        counts[value] = counts.get(value, 0) + 1

    return counts


def render_status(
    data: dict,
) -> str:
    health = "doctor clean" if data["findings"] == 0 else f"doctor {data['findings']} findings, {data['fixable']} fixable"
    lines = [f"kit {data['version']}    {health}"]

    for kind_name in ("plan", "todo", "bug", "decision", "review", "doc", "memory", "handover"):
        counts = data["counts"].get(kind_name, {})
        total = sum(counts.values())
        if kind_name in ("doc", "memory", "handover"):
            summary = str(total) if total else "none"
        else:
            summary = ", ".join(f"{count} {value}" for value, count in counts.items()) or "none"

        lines.append(f"{schema.kind_named(kind_name).folder:<10} {summary}")

    if data["open_reviews"]:
        lines.append(f"{'reviewing':<10} {', '.join(data['open_reviews'])}")

    if data["latest_handover"]:
        lines.append(f"{'handover':<10} {data['latest_handover']}")

    if data["unreadable"]:
        lines.append(f"{'unreadable':<10} {', '.join(data['unreadable'])}")

    return "\n".join(lines)


def rows(
    brain: store.Brain,
    kind: schema.Kind,
    filters: Optional[dict[str, str]] = None,
) -> list[dict]:
    """One row per entry of `kind`, narrowed by field=value filters."""
    wanted = filters or {}
    found: list[dict] = []

    for entry in brain.entries_of(kind):
        if any(entry.field(field) != value for field, value in wanted.items()):
            continue

        row = {
            "id": entry.identifier or entry.name,
            "path": entry.relative,
            "description": entry.field("description") or "",
        }
        for column in kind.columns:
            row[column] = entry.field(column) or ""

        found.append(row)

    return found


def render_rows(
    kind: schema.Kind,
    found: list[dict],
) -> str:
    if not found:
        return f"no {kind.folder} match"

    columns = ["id", *kind.columns]
    widths = {column: max(len(column), *(len(row[column]) for row in found)) for column in columns}
    lines: list[str] = []

    for row in found:
        cells = [row[column].ljust(widths[column]) for column in columns]
        lines.append("  ".join(cells) + "  " + row["description"])

    return "\n".join(lines)


def show(
    brain: store.Brain,
    key: str,
    meta_only: bool = False,
) -> Optional[str]:
    entry = brain.find(key)
    if entry is None:
        return None

    if meta_only:
        return "\n".join(entry.document.lines)

    return frontmatter.render(entry.document)


def path_of(
    brain: store.Brain,
    key: str,
) -> Optional[Path]:
    entry = brain.find(key)
    if entry is None:
        return None

    return entry.path


def links_of(
    brain: store.Brain,
    key: str,
) -> Optional[dict]:
    entry = brain.find(key)
    if entry is None:
        return None

    return {
        "path": entry.relative,
        "links": [links.canonical(brain=brain, reference=link) or link for link in links.outgoing(entry)],
        "backlinks": [other.relative for other in links.backlinks(brain=brain, target=entry)],
    }


def render_links(
    data: dict,
) -> str:
    lines = [f"{data['path']}", "", "links to"]
    lines += [f"  {link}" for link in data["links"]] or ["  nothing"]
    lines += ["", "linked from"]
    lines += [f"  {link}" for link in data["backlinks"]] or ["  nothing"]

    return "\n".join(lines)
