"""Review ledgers, the one body the tool edits by design.

A ledger's batches and findings are structured records inside the body,
per spec/reviews.md, so numbering a finding or flipping its status is
bookkeeping rather than prose. Everything outside those records stays as
it was written.
"""

import re
from datetime import date
from typing import Optional

from . import autoack, frontmatter, schema, sections, store, writes

FINDING = re.compile(r"^### (F-\d+): (.+?)[ \t]*$", re.MULTILINE)
FINDING_ID = re.compile(r"F-\d+")
BULLET = re.compile(r"^- ([a-z-]+):[ \t]*(.*?)[ \t]*$", re.MULTILINE)
# One row of the batch table, on one line: the number, then three cells.
BATCH_ROW = re.compile(
    r"^\|[ \t]*(\d+)[ \t]*\|[ \t]*(.*?)[ \t]*\|[ \t]*(.*?)[ \t]*\|[ \t]*(.*?)[ \t]*\|[ \t]*$",
    re.MULTILINE,
)

BATCH_HEADER = "| # | Theme | Status | Ack |\n|---|---|---|---|\n"

# The batch table's header line and the rule under it, however the words
# are cased. It ends at the end of the rule line.
TABLE_HEADER = re.compile(
    r"^\|[ \t]*#[ \t]*\|[ \t]*theme[ \t]*\|[ \t]*status[ \t]*\|[ \t]*ack[ \t]*\|[ \t]*\n\|[ \t]*:?-+:?[ \t]*\|[^\n]*$",
    re.MULTILINE | re.IGNORECASE,
)

FIELDS = ("severity", "found-by", "batch", "status", "fix", "reason")
NEEDS_REASON = ("wontfix", "deferred", "refuted")
FOUND_BY = ("review", "hunter")
FINDING_STATUSES = ("open", "fixed", "wontfix", "deferred", "refuted")
BATCH_STATUSES = ("pending", "fixing", "verified", "done")
LEVELS = schema.REVIEW_LEVELS
DEFAULT_LEVEL = "high"

# House rule 7, held here so the report can say when the pass is due:
# the closing pass files what it finds into a batch of this theme, so
# the ledger knows the pass ran, and low and a sweep branch skip it.
CLOSING_THEME = "closing pass"
NO_CLOSING_LEVELS = ("low",)
SWEEP_PREFIX = "sweep/"

# .agents/spec/reviews.md sets this cap on a review's batches and says why.
MAX_BATCHES = 5

# The next step the report gives once nothing else stands, and close
# refuses until the report gives it.
CLOSE_STEP = "close the ledger"

# The ack a batch takes while the owner's standing word holds, recorded
# as auto with the day, so the trail says which batches ran unattended.
AUTO_ACK = "auto"


class LedgerError(Exception):
    """The ledger change was refused, with the reason in the message."""


def open_ledger(
    brain: store.Brain,
    branch: str,
    summary: str,
    level: str = DEFAULT_LEVEL,
    stamp: str = "",
) -> store.Entry:
    if level not in LEVELS:
        raise LedgerError(f"a review runs at {' or '.join(LEVELS)}, not '{level}'")

    summary = one_line(value=summary, what="summary")
    if summary.startswith("#"):
        raise LedgerError("the summary is prose, not a heading")

    stamp = stamp or date.today().isoformat()
    title = f"{stamp[:7]} {branch}"
    body = (
        "## Summary\n\n"
        f"{summary}\n\n"
        "## Batches\n\n"
        f"{BATCH_HEADER}\n"
        "## Findings\n"
    )

    return writes.create(
        brain=brain,
        kind=schema.kind_named("review"),
        title=title,
        description=f"review of {branch}",
        values={"branch": branch, "level": level},
        body=body,
        stamp=stamp,
    )


def add_batch(
    brain: store.Brain,
    key: str,
    theme: str,
) -> int:
    theme = cell(value=theme, what="theme")
    if not theme:
        raise LedgerError("a batch needs a theme")

    entry = ledger_entry(brain=brain, key=key)
    rows = batches(entry=entry)
    has_room = room_for(
        rows=rows,
        theme=theme,
    )
    if not has_room:
        raise LedgerError(f"a review has {MAX_BATCHES} batches at most, file the finding into the batch whose theme fits")

    number = max((row["number"] for row in rows), default=0) + 1
    row = "\n" + batch_row(number=number, theme=theme, status="pending", ack="")

    body = entry.document.body
    table_end = end_of_table(body)
    entry.document.body = body[:table_end] + row + body[table_end:]
    save(brain=brain, entry=entry)

    return number


def room_for(
    rows: list[dict],
    theme: str,
) -> bool:
    """Whether a ledger with the batches `rows` has room for one more batch themed `theme`, the closing pass's batch standing outside the cap."""
    if theme.casefold() == CLOSING_THEME:
        return True

    counted_batches = [row for row in rows if row["theme"].casefold() != CLOSING_THEME]

    return len(counted_batches) < MAX_BATCHES


def set_batch(
    brain: store.Brain,
    key: str,
    number: int,
    status: str = "",
    ack: str = "",
    theme: str = "",
) -> None:
    if status and status not in BATCH_STATUSES:
        raise LedgerError(f"a batch is {' or '.join(BATCH_STATUSES)}, not '{status}'")

    auto = is_auto(ack)
    ack = cell(value=ack, what="ack")
    theme = cell(value=theme, what="theme")
    if not (status or ack or theme):
        raise LedgerError("nothing to set, give a status, an ack or a theme")

    entry = ledger_entry(brain=brain, key=key)
    body = entry.document.body
    replaced = False

    for match in batch_rows(body):
        row_number, old_theme, old_status, old_ack = match.groups()
        if int(row_number) != number:
            continue

        # a batch already counted keeps its place under the cap whatever it
        # is renamed to, so only the closing pass's batch can join the count
        was_counted = old_theme.casefold() != CLOSING_THEME
        has_room = was_counted or room_for(
            rows=batches(entry=entry),
            theme=theme or old_theme,
        )
        if not has_room:
            raise LedgerError(f"a review has {MAX_BATCHES} batches at most besides the {CLOSING_THEME}, so batch {number} keeps its theme")

        new_status = status or old_status
        if auto and new_status == "pending":
            raise LedgerError("auto ack is the word at a batch's start, give it with --status fixing")

        own = in_batch(
            items=findings(entry=entry),
            number=number,
        )
        starting = old_status == "pending" and status in ("fixing", "verified", "done")

        if auto and old_ack and not is_auto(old_ack):
            raise LedgerError(f"batch {number} already carries the owner's ack {old_ack}")

        # the word is asked for once the ledger and the batch are known, so a
        # bad key or number is reported as that, and a stamp already carried
        # stays past the start while the word still has to stand
        if auto:
            fresh = standing_word(brain=brain)
            new_ack = old_ack if old_ack and not starting else fresh
        else:
            new_ack = ack or old_ack

        if starting and own and not ack and is_auto(new_ack):
            new_ack = standing_word(brain=brain)

        if status in ("fixing", "verified", "done"):
            still_open = [item["id"] for item in own if is_open(item)]
            if status != "fixing" and still_open:
                raise LedgerError(f"{', '.join(still_open)} still open, batch {number} is {status} once every finding has landed")

            if own and not new_ack:
                if autoack.current(brain=brain) is not None:
                    raise LedgerError(f"batch {number} holds findings and runs on the standing word, give --ack auto")

                raise LedgerError(f"batch {number} holds findings and runs on the owner's ack, give --ack")

        row = batch_row(
            number=number,
            theme=theme or old_theme,
            status=new_status,
            ack=new_ack,
        )
        body = body[: match.start()] + row + body[match.end() :]
        replaced = True
        break

    if not replaced:
        raise LedgerError(f"{entry.name} has no batch {number}")

    entry.document.body = body
    save(brain=brain, entry=entry)


def is_auto(
    ack: str,
) -> bool:
    """Whether the ack asks for the standing word, anything that starts with auto in any case."""
    return ack.strip().casefold().startswith(AUTO_ACK)


def standing_word(
    brain: store.Brain,
) -> str:
    """The auto ack as the batch records it, refused as a ledger error when the word does not stand."""
    try:
        return autoack.stamp(brain=brain)
    except autoack.AckError as off:
        raise LedgerError(str(off)) from off


def batch_row(
    number: int,
    theme: str,
    status: str,
    ack: str,
) -> str:
    """One row of the batch table, an empty cell written as one space."""
    cells = [str(number), theme, status, ack]

    return "|" + "|".join(f" {value} " if value else " " for value in cells) + "|"


def add_finding(
    brain: store.Brain,
    key: str,
    claim: str,
    severity: str,
    found_by: str = "review",
    batch: int = 0,
) -> str:
    return add_findings(
        brain=brain,
        key=key,
        claims=[claim],
        severity=severity,
        found_by=found_by,
        batch=batch,
    )[0]


def add_findings(
    brain: store.Brain,
    key: str,
    claims: list[str],
    severity: str,
    found_by: str = "review",
    batch: int = 0,
) -> list[str]:
    """File the claims in one load and one write, numbered on from the last finding, so a pass lands whole or not at all."""
    severity = checked(field="severity", value=severity)
    found_by = checked(field="found-by", value=found_by)
    batch_cell = checked(field="batch", value=str(batch) if batch else "")
    claims = [one_line(value=claim, what="claim") for claim in claims]
    if not claims or not all(claims):
        raise LedgerError("a finding needs a claim")

    entry = ledger_entry(brain=brain, key=key)
    known_batch(
        entry=entry,
        value=batch_cell,
    )
    first = max((int(record["id"][2:]) for record in findings(entry)), default=0) + 1
    identifiers = [f"F-{first + offset:02d}" for offset in range(len(claims))]
    blocks = "".join(
        f"\n### {identifier}: {claim}\n\n"
        f"- severity: {severity}\n"
        f"- found-by: {found_by}\n"
        f"- batch: {batch_cell}\n"
        "- status: open\n"
        "- fix:\n"
        for identifier, claim in zip(identifiers, claims)
    )

    body = reopened(
        body=entry.document.body,
        number=int(batch_cell) if batch_cell else None,
    )
    section_start = section_span(body=body, title="Findings")[0]
    section_end = sections.heading_after(body=body, start=section_start, depth=2)
    entry.document.body = body[:section_end].rstrip("\n") + "\n" + blocks + ("\n" if section_end < len(body) else "") + body[section_end:]
    save(brain=brain, entry=entry)

    return identifiers


def set_findings(
    brain: store.Brain,
    key: str,
    identifiers: list[str],
    changes: dict[str, str],
) -> None:
    """Make the same changes to every finding named in one load and one write, so the call lands whole or not at all.

    Each finding is held to every rule a change to it alone is held to,
    and a call that names no finding, no change or a finding twice is
    refused.
    """
    if not identifiers:
        raise LedgerError("nothing to set, name a finding like F-03")

    if not changes:
        raise LedgerError("nothing to set, give a change like status=fixed")

    changes = {field: checked(field=field, value=value) for field, value in changes.items()}

    entry = ledger_entry(brain=brain, key=key)
    known_batch(
        entry=entry,
        value=changes.get("batch", ""),
    )
    body = entry.document.body

    for position, identifier in enumerate(identifiers):
        if identifier in identifiers[:position]:
            raise LedgerError(f"{identifier} is named twice, name each finding once")

        body = changed(
            body=body,
            ledger_name=entry.name,
            identifier=identifier,
            changes=changes,
        )

    entry.document.body = body
    save(brain=brain, entry=entry)


def changed(
    body: str,
    ledger_name: str,
    identifier: str,
    changes: dict[str, str],
) -> str:
    """The body with one finding's changes made, refused when the ledger `ledger_name` has no such finding or the changes break a rule it is held to."""
    span = finding_span(body=body, identifier=identifier)
    if span is None:
        raise LedgerError(f"{ledger_name} has no finding {identifier}")

    start, end = span
    block = body[start:end]
    before = bullets(block)
    after = dict(before)
    after.update(changes)

    # A fix an earlier call left on a finding that was not fixed names no
    # commit of this fix, so a finding moving into fixed takes the call's own.
    moves_into_fixed = after.get("status") == "fixed" and before.get("status") != "fixed"
    named_commit = changes.get("fix") if moves_into_fixed else after.get("fix")
    if after.get("status") == "fixed" and not named_commit:
        raise LedgerError("a fixed finding names its commit, fix=<hash>")

    new_status = changes.get("status", "")
    if new_status in NEEDS_REASON and not changes.get("reason"):
        raise LedgerError(f"a {new_status} finding says why, reason=<one line>")

    if "reason" in changes and not changes["reason"] and after.get("status") in NEEDS_REASON:
        raise LedgerError(f"a {after['status']} finding keeps its reason, give a new one or a new status")

    for field, value in changes.items():
        block = with_bullet(block=block, field=field, value=value)

    if new_status and new_status not in NEEDS_REASON:
        block = without_bullet(block=block, field="reason")

    leaves_fixed = before.get("status") == "fixed" and new_status not in ("", "fixed")
    if leaves_fixed and "fix" not in changes:
        block = without_bullet(
            block=block,
            field="fix",
        )

    body = body[:start] + block + body[end:]
    moved = batch_of(after) != batch_of(before)
    if not (is_open(item=after) or moved):
        return body

    return reopened(
        body=body,
        number=batch_of(item=after),
    )


def reopened(
    body: str,
    number: Optional[int],
) -> str:
    """The body with the batch put back to fixing when a finding lands in it after it was verified or done.

    A late finding, a hunter's, a reopened one or one moved in already
    fixed, needs its fix verified like the rest, so the batch goes
    through fixing and verified again. The ack that started the batch
    stays, since the owner's word covered the batch and what it holds.
    """
    if number is None:
        return body

    for match in batch_rows(body):
        if int(match.group(1)) != number or match.group(3) not in ("verified", "done"):
            continue

        row = batch_row(
            number=number,
            theme=match.group(2),
            status="fixing",
            ack=match.group(4),
        )
        return body[: match.start()] + row + body[match.end() :]

    return body


def with_bullet(
    block: str,
    field: str,
    value: str,
) -> str:
    """The block with its `- field:` line saying `value`.

    The first such line outside code is rewritten. A block that lost the
    line gets it back, after the last bullet it still has, so a finding
    edited by hand can be put right through the tool.
    """
    line = f"- {field}: {value}"
    spans = sections.code_spans(block)
    lines = [match for match in BULLET.finditer(block) if not sections.in_code(spans=spans, position=match.start())]

    for match in lines:
        if match.group(1) == field:
            return block[: match.start()] + line + block[match.end() :]

    if lines:
        at = lines[-1].end()
        return block[:at] + "\n" + line + block[at:]

    heading_end = block.index("\n") if "\n" in block else len(block)
    return block[:heading_end] + "\n\n" + line + block[heading_end:]


def without_bullet(
    block: str,
    field: str,
) -> str:
    """The block without its `- field:` line, the first one outside code, so a value that no longer applies does not linger."""
    spans = sections.code_spans(block)

    for match in BULLET.finditer(block):
        if match.group(1) != field or sections.in_code(spans=spans, position=match.start()):
            continue

        line_end = match.end() + 1 if block[match.end() : match.end() + 1] == "\n" else match.end()
        return block[: match.start()] + block[line_end:]

    return block


def close(
    brain: store.Brain,
    key: str,
    stamp: str = "",
) -> None:
    entry = ledger_entry(brain=brain, key=key)
    if entry.field("status") == "closed":
        raise LedgerError(f"{entry.name} is already closed")

    items = findings(entry)
    still_open = [item for item in items if is_open(item)]
    if still_open:
        names = ", ".join(item["id"] for item in still_open)
        raise LedgerError(f"{names} still open, a ledger closes when every finding has landed")

    step = next_step(
        brain=brain,
        entry=entry,
        rows=batches(entry),
        items=items,
    )
    if step != CLOSE_STEP:
        raise LedgerError(f"the ledger closes after {step}")

    counts = tally(items)
    summary_end = end_of_section(body=entry.document.body, title="Summary")
    line = f"Closed with {counts}.\n\n"
    entry.document.body = (
        entry.document.body[:summary_end].rstrip("\n") + "\n\n" + line + entry.document.body[summary_end:]
    )
    entry.document.set(key="status", value="closed")
    save(brain=brain, entry=entry, stamp=stamp)

    writes.refresh_after(brain=brain, entry=entry, stamp=stamp)


def report(
    brain: store.Brain,
    key: str,
) -> str:
    """The ledger's state in a few lines, counted from the file, with what comes next."""
    entry = ledger_entry(
        brain=brain,
        key=key,
    )
    rows = batches(entry)
    items = findings(entry)
    noun = "batch" if len(rows) == 1 else "batches"
    lines = [f"{entry.name}, {entry.field('status')}, {len(rows)} {noun}, {tally(items)}"]

    for row in rows:
        own = in_batch(
            items=items,
            number=row["number"],
        )
        ack = f", ack {row['ack']}" if row["ack"] else ""
        lines.append(f"batch {row['number']}, {row['theme']}: {row['status']}{ack}, {tally(own)}")

    step = next_step(
        brain=brain,
        entry=entry,
        rows=rows,
        items=items,
    )
    lines.append("next: " + step)

    return "\n".join(lines)


def in_batch(
    items: list[dict],
    number: int,
) -> list[dict]:
    return [item for item in items if batch_of(item) == number]


def batch_of(
    item: dict,
) -> Optional[int]:
    """The batch a finding says it is in, as the table numbers it, or None when the cell is blank or not a plain number."""
    cell_text = (item.get("batch") or "").strip()

    return int(cell_text) if cell_text.isdecimal() and cell_text.isascii() else None


def is_open(
    item: dict,
) -> bool:
    """Whether a finding still needs work, a blank status counting as open the way close counts it."""
    return (item.get("status") or "open") == "open"


def needs_a_batch(
    item: dict,
) -> bool:
    """Whether a finding has to sit in a batch, an open one and a fixed one, since a fix is verified by its batch, where a wontfix, a deferral or a refuted claim needs none."""
    return is_open(item) or item.get("status") == "fixed"


def next_step(
    brain: store.Brain,
    entry: store.Entry,
    rows: list[dict],
    items: list[dict],
) -> str:
    """What the review does next, read from the batches and the findings as they stand.

    An open or a fixed finding filed into no batch the table has comes
    first, since nothing moves until it is placed and a fix is verified
    by its batch. Then the batches in order. A pending one waits for the
    owner's word, the ack that starts it and is recorded on it, one
    that carries the ack already is there to start, and so is one with
    none while the owner's standing word holds. A stamp from an
    earlier word counts as no ack while the word is off, since a start
    with findings in the batch takes the word again. A batch
    past pending needs its open findings fixed, a late one included,
    then its verification, and verification closes it. A batch that
    holds findings with no ack recorded, which a hand edit leaves, or a
    finding filed into an empty batch closed without one, asks for that
    ack before anything else, the standing word's while one holds. Then
    the walk moves on to the closing pass or the close.
    """
    if entry.field("status") == "closed":
        return "nothing, the ledger is closed"

    known = {row["number"] for row in rows}
    stray = [item["id"] for item in items if needs_a_batch(item) and batch_of(item) not in known]
    if stray:
        return f"put {', '.join(stray)} in a batch"

    for row in rows:
        number = row["number"]
        if row["status"] == "pending":
            standing = autoack.current(brain=brain) is not None
            if row["ack"] and not (is_auto(row["ack"]) and not standing):
                return f"start batch {number}, it has the word"

            if standing:
                return f"start batch {number} on the standing word"

            return f"batch {number} on the owner's word"

        own = in_batch(
            items=items,
            number=number,
        )
        if own and not row["ack"]:
            if autoack.current(brain=brain) is not None:
                return f"record the standing word on batch {number}, it holds findings and has none"

            return f"record the owner's ack of batch {number}, it holds findings and has none"

        still_open = [item["id"] for item in own if is_open(item)]
        if still_open:
            return f"fix {', '.join(still_open)} in batch {number}"

        if row["status"] == "fixing":
            return f"verify batch {number}"

        if row["status"] == "verified":
            return f"mark batch {number} done" + ("" if own else ", it holds nothing")

    owed = wants_closing_pass(
        entry=entry,
        rows=rows,
    )
    if owed:
        return "the closing pass"

    return CLOSE_STEP


def wants_closing_pass(
    entry: store.Entry,
    rows: list[dict],
) -> bool:
    """Whether the review still owes its closing pass.

    The pass is owed at every level above low, on any branch that is
    not a sweep, until a batch of the closing theme records that it
    ran. A ledger with no level recorded, one opened before the level
    was written, is read at the default level.
    """
    level = entry.field("level") or DEFAULT_LEVEL
    if not rows or level in NO_CLOSING_LEVELS:
        return False

    if (entry.field("branch") or "").startswith(SWEEP_PREFIX):
        return False

    return not any(row["theme"].casefold() == CLOSING_THEME for row in rows)


def known_batch(
    entry: store.Entry,
    value: str,
) -> None:
    """Refuse a batch number the table does not carry, so no finding is filed where the report cannot see it."""
    if value and int(value) not in {row["number"] for row in batches(entry)}:
        raise LedgerError(f"{entry.name} has no batch {value}")


def findings(
    entry: store.Entry,
) -> list[dict]:
    body = entry.document.body
    spans = sections.code_spans(body)
    found: list[dict] = []

    for match in finding_headings(body=body, spans=spans):
        end = sections.heading_after(
            body=body,
            start=match.end(),
            spans=spans,
        )
        block = body[match.start() : end]
        record = {"id": match.group(1), "claim": match.group(2)}
        record.update(bullets(block))
        found.append(record)

    return found


def bullets(
    block: str,
) -> dict[str, str]:
    """The `- name: value` lines of a block that sit outside code, each value as written.

    A name that appears twice counts once, by its first line, which is
    the line a change rewrites, so reading and writing agree.
    """
    spans = sections.code_spans(block)
    found: dict[str, str] = {}

    for match in BULLET.finditer(block):
        if not sections.in_code(spans=spans, position=match.start()):
            found.setdefault(match.group(1), match.group(2))

    return found


def finding_headings(
    body: str,
    spans: Optional[list[tuple[int, int]]] = None,
) -> list[re.Match]:
    """Every finding heading outside code, in order, the code spans passed in when the caller already has them."""
    if spans is None:
        spans = sections.code_spans(body)

    return [match for match in FINDING.finditer(body) if not sections.in_code(spans=spans, position=match.start())]


def batches(
    entry: store.Entry,
) -> list[dict]:
    return [
        {"number": int(match.group(1)), "theme": match.group(2), "status": match.group(3), "ack": match.group(4)}
        for match in batch_rows(entry.document.body)
    ]


def batch_rows(
    body: str,
) -> list[re.Match]:
    """Every row of the batch table, the lines right under its header and nothing else."""
    return batch_table(body)[1]


def batch_table(
    body: str,
) -> tuple[re.Match, list[re.Match]]:
    """The batch table's header and its rows.

    The header is the first one in the Batches section outside code, and
    the rows are the lines directly under it, the way markdown reads a
    table. A row shaped line anywhere else in the section is refused
    rather than read or ignored, since a table a hand edit pulled apart
    would otherwise be numbered twice.
    """
    start, end = batches_section(body)
    spans = sections.code_spans(body)
    header = None

    for match in TABLE_HEADER.finditer(string=body, pos=start, endpos=end):
        if not sections.in_code(spans=spans, position=match.start()):
            header = match
            break

    if header is None:
        raise LedgerError("the ledger has no batch table")

    rows: list[re.Match] = []
    position = header.end()

    while position < end and body[position] == "\n":
        line_start = position + 1
        row = BATCH_ROW.match(string=body, pos=line_start, endpos=end)
        if row is None:
            if body[line_start:end].split("\n", 1)[0].lstrip().startswith("|"):
                raise LedgerError("a line of the batch table is not a whole row, a row is | number | theme | status | ack |")

            break

        rows.append(row)
        position = row.end()

    read = {row.start() for row in rows}
    for match in BATCH_ROW.finditer(string=body, pos=start, endpos=end):
        if match.start() not in read and not sections.in_code(spans=spans, position=match.start()):
            raise LedgerError("a batch row sits apart from the table, put it directly under the header")

    return header, rows


def batches_section(
    body: str,
) -> tuple[int, int]:
    """Where the Batches section's text starts and ends."""
    return section_span(body=body, title="Batches")


def section_span(
    body: str,
    title: str,
) -> tuple[int, int]:
    """Where the named section's text starts and ends, refused when the ledger has none."""
    found = sections.span(body=body, title=title)
    if found is None:
        raise LedgerError(f"the ledger has no {title} section")

    return found


def tally(
    items: list[dict],
) -> str:
    counts: dict[str, int] = {}
    for item in items:
        status = "open" if is_open(item) else item["status"]
        counts[status] = counts.get(status, 0) + 1

    total = len(items)
    noun = "finding" if total == 1 else "findings"
    parts = ", ".join(f"{count} {status}" for status, count in counts.items())

    return f"{total} {noun}" + (f", {parts}" if parts else "")


def one_line(
    value: str,
    what: str,
) -> str:
    """The value trimmed, refused when it would break the record it goes into."""
    value = value.strip()
    if "\n" in value or "\r" in value:
        raise LedgerError(f"the {what} is one line")

    return value


def checked(
    field: str,
    value: str,
) -> str:
    """A finding's value as the record takes it, refused when the spec has no room for it."""
    if field not in FIELDS:
        raise LedgerError(f"a finding has no '{field}' field, its fields are {', '.join(FIELDS)}")

    value = one_line(value=value, what=field)
    if field == "severity" and value not in schema.SEVERITIES:
        raise LedgerError(f"severity is {' or '.join(schema.SEVERITIES)}, not '{value}'")

    if field == "found-by" and value not in FOUND_BY:
        raise LedgerError(f"found-by is review or hunter, not '{value}'")

    if field == "status" and value not in FINDING_STATUSES:
        raise LedgerError(f"a finding is {' or '.join(FINDING_STATUSES)}, not '{value}'")

    if field == "batch" and value and not re.fullmatch(pattern=r"[1-9][0-9]*", string=value):
        raise LedgerError(f"batch is a number from 1, or blank, not '{value}'")

    return value


def cell(
    value: str,
    what: str,
) -> str:
    """One line with no pipe, since the batch table's columns are pipes."""
    value = one_line(value=value, what=what)
    if "|" in value:
        raise LedgerError(f"the {what} cannot hold a pipe, the batch table's columns are pipes")

    return value


def ledger_entry(
    brain: store.Brain,
    key: str,
) -> store.Entry:
    try:
        entry = brain.find(key)
    except store.Ambiguous as clash:
        raise LedgerError(str(clash)) from clash

    if entry is None or entry.kind.name != "review":
        raise LedgerError(f"no review ledger answers to '{key}'")

    return entry


def save(
    brain: store.Brain,
    entry: store.Entry,
    stamp: str = "",
) -> None:
    entry.document.set(key="updated", value=stamp or date.today().isoformat())
    frontmatter.write(path=str(entry.path), document=entry.document)
    brain.refresh()


def finding_span(
    body: str,
    identifier: str,
) -> Optional[tuple[int, int]]:
    """Where one finding's block starts and ends, up to the next heading outside code."""
    for match in finding_headings(body):
        if match.group(1) != identifier:
            continue

        return match.start(), sections.heading_after(body=body, start=match.end())

    return None


def end_of_table(
    body: str,
) -> int:
    """The offset at the end of the last batch row's line, or of the table header's, before the line break."""
    header, rows = batch_table(body)

    return rows[-1].end() if rows else header.end()


def end_of_section(
    body: str,
    title: str,
) -> int:
    """The offset where the named section ends, at the next heading outside code."""
    return section_span(body=body, title=title)[1]
