"""The steps a piece of work is made of, a table the tool keeps on a plan, a todo or a bug.

The Steps section holds a table, one row per step, numbered from 1 and
never renumbered, each pointing at a todo, a bug or a review or standing
on its own. A plan always carries the table, a todo or a bug carries
one when its work has steps. Whoever files the entry lists the steps as
numbered lines, and the tool turns them into the table. Rows are added
and marked through here, never deleted, a step that turned out wrong is
skipped, so the file reads as the record of how the work went. A step
that points at an entry takes its mark from that entry, and a step is
marked by hand only when it points at nothing, or at nothing the brain
can read, a dead key or an id two files carry.
"""

import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

from . import frontmatter, schema, sections, store

STEP_STATUSES: tuple[str, ...] = ("pending", "done", "skipped")
BY: tuple[str, ...] = ("owner", "agent")

# the kinds that carry a steps table
STEPPED_KINDS: tuple[str, ...] = ("plan", "todo", "bug")

# the kinds a step may point at, and what counts as done for each
POINTED_KINDS: dict[str, tuple[str, ...]] = {
    "todo": ("done",),
    "bug": ("fixed", "wontfix"),
    "review": ("closed",),
}
POINTED_GONE: dict[str, tuple[str, ...]] = {
    "todo": ("dropped", "refiled"),
    "bug": ("refiled",),
    "review": (),
}

# the statuses that keep work open, for the chain and for adding steps
OPEN_STATUSES: dict[str, tuple[str, ...]] = {
    "plan": ("open",),
    "todo": ("open",),
    "bug": ("open", "branched"),
    "review": ("open",),
}

# the status that closes each stepped kind once every step is settled
CLOSING: dict[str, str] = {
    "plan": "done",
    "todo": "done",
    "bug": "fixed",
}

TABLE_HEADER = "| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n"
HEADER_LINE = re.compile(r"^\|[ \t]*#[ \t]*\|[ \t]*Step[ \t]*\|[ \t]*Points at[ \t]*\|[ \t]*By[ \t]*\|[ \t]*Status[ \t]*\|[ \t]*\n\|[-| \t]+\|[ \t]*$", re.MULTILINE)
# a cell runs to the next pipe, and a pipe written as \| is part of the cell,
# the one escape there is, so every other backslash is the cell's own and a
# backslash never ends a cell before a pipe, and a cell neither starts nor
# ends with blank space, the blank space after it being read only when it
# holds something, so the padding around it and every escape in it have one
# reading and a row that does not match fails at once rather than through
# every split of them, an empty cell leaving its group unmatched
CELL = r"[ \t]*(?:((?:\\\||[^|\n \t\\]|\\(?!\|))(?:\\\||[^|\n\\]|\\(?!\|))*?(?<![ \t]))[ \t]*)?\|"
ROW = re.compile(
    rf"^\|[ \t]*(\d+)[ \t]*\|{CELL}{CELL}{CELL}{CELL}[ \t]*$",
    re.MULTILINE,
)
LISTED = re.compile(r"^(\d+)\.[ \t]+(.+?)[ \t]*$", re.MULTILINE)

# a step line may open with the key of the entry it points at
POINTER = re.compile(r"^([A-Za-z0-9][\w.-]*)[,:][ \t]+(.+)$")
KEY_START = re.compile(r"^([A-Za-z0-9-]+)")

# a heading meant as the Steps section that the readers would never find
STEPS_LOOKALIKE = re.compile(r"^#{1,6}[ \t]+steps:?[ \t]*$", re.IGNORECASE | re.MULTILINE)

# a date opening a step line is prose, though it is shaped like a review's name
DATE_SHAPE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:$|T\d)")


class StepsError(Exception):
    """A steps table that does not read as one, or a step that cannot be added or marked as asked, with the reason in the message."""


@dataclass(frozen=True)
class Step:
    """One row of the table."""

    number: int
    text: str
    points_at: str
    by: str
    status: str


def table(
    body: str,
) -> list[Step]:
    """Every row of the steps table, in order, refused when the section or the table is not there or a row is broken."""
    rows = [
        Step(
            number=int(row.group(1)),
            text=unescaped(row.group(2) or ""),
            points_at=unescaped(row.group(3) or ""),
            by=unescaped(row.group(4) or ""),
            status=unescaped(row.group(5) or ""),
        )
        for row in located(body)[1]
    ]

    for expected, step in enumerate(rows, 1):
        if step.number != expected:
            raise StepsError(f"the steps are numbered 1 onward in order, and row {step.number} sits where {expected} should")

        if step.status not in STEP_STATUSES:
            raise StepsError(f"step {step.number} is {' or '.join(STEP_STATUSES)}, not '{step.status}'")

        if step.by not in BY:
            raise StepsError(f"step {step.number} is set by {' or '.join(BY)}, not '{step.by}'")

    return rows


def located(
    body: str,
) -> tuple[re.Match, list[re.Match]]:
    """The table's header and its rows, the lines right under the header outside code, the one reading every reader and every writer of the table shares.

    A row shaped line anywhere else in the section, outside code, is
    refused rather than read or ignored, since a table a hand edit
    pulled apart would otherwise be written back without the lines
    between.
    """
    found = span(body)
    if found is None:
        raise StepsError("there is no Steps section")

    start, end = found
    spans = sections.code_spans(body)
    header: Optional[re.Match] = None
    for match in HEADER_LINE.finditer(string=body, pos=start, endpos=end):
        if not sections.in_code(spans=spans, position=match.start()):
            header = match
            break

    if header is None:
        raise StepsError("the Steps section has no table, its header is | # | Step | Points at | By | Status |, and a numbered list there becomes one through brain steps <key> add with a step to add, or through brain doctor --fix on a plan or a todo, and doctor says what to do on a bug")

    rows: list[re.Match] = []
    position = header.end()
    while position < end and body[position] == "\n":
        line_start = position + 1
        row = ROW.match(string=body, pos=line_start, endpos=end)
        if row is None:
            if body[line_start:end].split("\n", 1)[0].lstrip().startswith("|"):
                raise StepsError("a line of the steps table is not a whole row, a row is | number | step | points at | by | status |, a pipe inside a cell is written \\|, and a backslash never ends a cell before its pipe")

            break

        rows.append(row)
        position = row.end()

    read = {row.start() for row in rows}
    for match in ROW.finditer(string=body, pos=start, endpos=end):
        if match.start() not in read and not sections.in_code(spans=spans, position=match.start()):
            raise StepsError("a step row sits apart from the table, put it directly under the header")

    for match in LISTED.finditer(string=body, pos=start, endpos=end):
        if not sections.in_code(spans=spans, position=match.start()):
            raise StepsError(f"step lines go in the table, and {match.group(0).strip()!r} sits outside it")

    return header, rows


def blank(
    body: str,
) -> bool:
    """Whether the body has no Steps section, or one with nothing under its heading, which a todo or a bug reads as no steps at all."""
    where = span(body)

    return where is None or not body[where[0] : where[1]].strip()


def span(
    body: str,
) -> Optional[tuple[int, int]]:
    """Where the Steps section's text starts and ends, up to the next heading of level one or two, so a subheading inside it belongs to the section and reads as a line the table cannot carry."""
    for match in sections.heading_matches(body):
        if match.group(1).lower() == "steps":
            return match.end(), sections.heading_after(body=body, start=match.end(), depth=2)

    return None


def rows_of(
    brain: store.Brain,
    entry: store.Entry,
) -> list[Step]:
    """The entry's steps as they stand, each pointed one read off the entry it points at rather than off the file while the entry is open, so a mark that went stale in the file never misleads a reader, and as the file holds them once the entry is closed, since a closed entry's table is its record, none for a todo or a bug without a Steps section or with an empty one, and refused with the path named when the table does not read."""
    rows = stored_rows(entry)
    if not is_open(entry):
        return rows

    own = own_keys(entry)

    return [derived_row(brain=brain, step=step, own=own) for step in rows]


def is_open(
    entry: store.Entry,
) -> bool:
    return (entry.field("status") or "") in OPEN_STATUSES.get(entry.kind.name, ())


def own_keys(
    entry: store.Entry,
) -> set[str]:
    """The keys a step elsewhere would use to point at the entry."""
    return {entry.identifier, entry.name} - {None}


def answers_of(
    entry: store.Entry,
) -> set[str]:
    """The keys a step's cell may hold for the entry and reach it by, its own keys and, for a review, its folder's name, which answers as its name does."""
    answers = own_keys(entry)
    if entry.kind.name == "review":
        answers.add(entry.path.parent.name)

    return answers


def stored_rows(
    entry: store.Entry,
) -> list[Step]:
    """The entry's steps as the file holds them, the path named when they do not read, which a closed entry is read by, its table being its record, and which the writers rewrite in place."""
    try:
        return stored_rows_of(kind=entry.kind.name, body=entry.document.body)
    except StepsError as broken:
        raise StepsError(f"{entry.relative}: {broken}") from broken


def stored_rows_of(
    kind: str,
    body: str,
) -> list[Step]:
    """The steps a body of the kind holds as written, none for a todo or a bug with no Steps section or an empty one."""
    if kind != "plan" and blank(body):
        return []

    return table(body)


def has_table(
    body: str,
) -> bool:
    """Whether the Steps section already holds the table's header, outside code."""
    found = span(body)
    if found is None:
        return False

    spans = sections.code_spans(body)

    return any(not sections.in_code(spans=spans, position=match.start()) for match in HEADER_LINE.finditer(string=body, pos=found[0], endpos=found[1]))


def current_step(
    rows: list[Step],
) -> Optional[int]:
    """The first step not yet done or skipped, or None when every step is settled or there are none."""
    for step in rows:
        if step.status == "pending":
            return step.number

    return None


def render_row(
    step: Step,
) -> str:
    """One row of the table, an empty cell written as one space and a pipe in a cell written as \\|, the way markdown reads it."""
    cells = [str(step.number), escaped(step.text), escaped(step.points_at), step.by, step.status]

    return "|" + "|".join(f" {value} " if value else " " for value in cells) + "|"


def escaped(
    text: str,
) -> str:
    """A cell's text with its pipes escaped, the one escape the table has, so a backslash in a step is written as it is."""
    return text.replace("|", "\\|")


def unescaped(
    text: str,
) -> str:
    return text.replace("\\|", "|")


def rewrite(
    body: str,
    rows: list[Step],
) -> str:
    """The body with its steps table replaced by these rows, the prose and the code around it untouched."""
    header, old_rows = located(body)
    table_end = old_rows[-1].end() if old_rows else header.end()
    text = "".join("\n" + render_row(step) for step in rows)

    return body[: header.end()] + text + body[table_end:]


def listed(
    body: str,
) -> list[str]:
    """The steps of a Steps section as someone filed it, one per numbered line with its indented continuation lines joined on, refused when the section holds anything else, since a line the table cannot carry is never dropped without a word."""
    where = span(body)
    if where is None:
        return []

    found: list[str] = []
    for line in body[where[0] : where[1]].split("\n"):
        if not line.strip():
            continue

        match = LISTED.match(line)
        if match is not None:
            if int(match.group(1)) != len(found) + 1:
                raise StepsError(f"the steps are numbered 1 onward in order, and {match.group(1)} sits where {len(found) + 1} should")

            found.append(match.group(2))
            continue

        if found and line[0] in " \t":
            found[-1] = f"{found[-1]} {line.strip()}"
            continue

        raise StepsError(f"the Steps section holds a line that is not a numbered step, one per line with its wrapped lines indented under it: {line.strip()!r}")

    return found


def tabled(
    brain: store.Brain,
    body: str,
    by: str,
    kind: schema.Kind,
    own: Optional[set[str]] = None,
) -> str:
    """The body with the numbered lines under Steps turned into the table, or as it was when it has no Steps section or already holds the table, its steps checked against the keys the entry takes over when a refile files it."""
    if span(body) is None:
        return body

    if has_table(body):
        return checked(brain=brain, body=body, own=own)

    return converted(brain=brain, body=body, by=by, kind=kind, own=own)


def checked(
    brain: store.Brain,
    body: str,
    own: Optional[set[str]] = None,
) -> str:
    """The body with a table someone typed by hand read, its pointers resolved, its marks derived and a step line left outside it refused, so a filed table passes every check a listed step does."""
    rows = table(body)
    fresh: list[Step] = []
    for step in rows:
        if not step.text.strip():
            raise StepsError(f"step {step.number} has no text")

        target = pointed(brain=brain, key=step.points_at) if step.points_at else None
        if step.points_at and target is None:
            raise unreachable(brain=brain, number=step.number, key=step.points_at)

        pointer = pointer_of(target) if target is not None else ""
        if pointer and doubled(brain=brain, key=pointer):
            raise doubled_pointer(number=step.number, key=step.points_at, pointer=pointer)

        if own and pointer in own:
            raise StepsError(f"step {step.number} points at {pointer} itself, and a step points at other work")

        if not pointer:
            word = opening_key(brain=brain, number=step.number, text=step.text)
            if word:
                raise StepsError(f"step {step.number} opens with {word} in its Step cell, and a row points through its Points at cell")

        fresh.append(derived_row(brain=brain, step=Step(number=step.number, text=step.text, points_at=pointer, by=step.by, status=step.status), own=own))

    if own:
        refuse_cycles(brain=brain, rows=fresh, own=own)

    return rewrite(body=body, rows=fresh)


def refuse_lookalike(
    body: str,
) -> None:
    """Refuse a heading meant as a plan's Steps section that no reader would find, so its steps are never left behind as prose, a plan being the one kind whose Steps section is required, where a todo or a bug may hold such a heading as prose."""
    spans = sections.code_spans(body)
    for match in STEPS_LOOKALIKE.finditer(body):
        if not sections.in_code(spans=spans, position=match.start()):
            raise StepsError(f"the steps go under a level two heading, ## Steps, and {match.group(0).strip()!r} is not one")


def converted(
    brain: store.Brain,
    body: str,
    by: str,
    kind: schema.Kind,
    status: str = "pending",
    own: Optional[set[str]] = None,
) -> str:
    """The body with the list under Steps turned into the table, each row marked as given unless it points at an entry, which gives the mark, and a row pointing at the entry's own keys refused."""
    rows = [row_from(brain=brain, number=number, line=line, by=by, own=own) for number, line in enumerate(listed(body), 1)]
    for step in rows:
        if own and step.points_at in own:
            raise StepsError(f"step {step.number} points at {step.points_at} itself, and a step points at other work")

    if own and status == "pending":
        refuse_cycles(brain=brain, rows=rows, own=own)

    # a closed entry's list becomes its record, every row settled by the entry's own status, the pointed ones too
    if status != "pending":
        rows = [Step(number=step.number, text=step.text, points_at=step.points_at, by=step.by, status=status) for step in rows]

    return with_table(body=body, rows=rows, kind=kind)


def settled_as(
    entry: store.Entry,
) -> str:
    """The mark a step of a closed entry takes when its list becomes the table: done for an entry done or fixed, skipped for one dropped, refiled or wontfix, pending while it is open."""
    if is_open(entry):
        return "pending"

    return "done" if (entry.field("status") or "") in ("done", "fixed") else "skipped"


def with_table(
    body: str,
    rows: list[Step],
    kind: schema.Kind,
) -> str:
    """The body with its Steps section holding the table and nothing else, added where the kind's sections put it when the prose had none."""
    found = span(body)
    table_text = TABLE_HEADER.rstrip("\n") + "".join("\n" + render_row(step) for step in rows) + "\n"
    if found is not None:
        start, end = found
        return body[:start] + "\n\n" + table_text + "\n" + body[end:]

    if kind.name == "plan":
        refuse_lookalike(body)

    titles = [section.title for section in kind.sections]
    for title in titles[titles.index("Steps") + 1 :]:
        following = sections.span(body=body, title=title)
        if following is not None:
            heading_start = body.rfind("\n", 0, following[0]) + 1
            return body[:heading_start] + "## Steps\n\n" + table_text + "\n" + body[heading_start:]

    return body.rstrip("\n") + "\n\n## Steps\n\n" + table_text


def row_from(
    brain: store.Brain,
    number: int,
    line: str,
    by: str,
    own: Optional[set[str]] = None,
) -> Step:
    """One step from one line, which may open with the key of the entry it points at, refused when that key reaches nothing of the right kind, its mark derived with the holder's own keys so a refile is followed only where the successor does not point back."""
    if by not in BY:
        raise StepsError(f"a step is set by {' or '.join(BY)}, not '{by}'")

    line = line.strip()
    if "\n" in line or "\r" in line:
        raise StepsError(f"step {number} is one line, and a line break would split its row")

    points_at = ""
    match = POINTER.match(line)
    if match is not None:
        target = pointed(brain=brain, key=match.group(1))
        if target is not None:
            points_at = pointer_of(target)
            line = match.group(2).strip()
            if doubled(brain=brain, key=points_at):
                raise doubled_pointer(number=number, key=match.group(1), pointer=points_at)
        elif looks_like_a_key(match.group(1)) and not known(brain=brain, key=match.group(1)):
            raise unreachable(brain=brain, number=number, key=match.group(1))

    if not points_at:
        word = opening_key(brain=brain, number=number, text=line)
        if word:
            raise StepsError(f"step {number} opens with {word} but not as '{word}, what to do', so it would point at nothing")

    if not line:
        raise StepsError(f"step {number} has no text")

    return derived_row(brain=brain, step=Step(number=number, text=line, points_at=points_at, by=by, status="pending"), own=own)


def opening_key(
    brain: store.Brain,
    number: int,
    text: str,
) -> str:
    """The key a plain step's text opens with when that key names an entry a step could point at, so the caller can refuse a pointer that was meant and not made, and refused outright when it is shaped like a key the brain does not know."""
    opening = KEY_START.match(text)
    word = opening.group(1) if opening is not None else ""
    if not looks_like_a_key(word):
        return ""

    if pointed(brain=brain, key=word) is not None:
        return word

    if not known(brain=brain, key=word):
        raise unreachable(brain=brain, number=number, key=word)

    return ""


def unreachable(
    brain: store.Brain,
    number: int,
    key: str,
) -> StepsError:
    """The refusal for a pointer that reaches nothing, doubled when two files carry the key and unknown otherwise."""
    if doubled(brain=brain, key=key):
        return StepsError(f"step {number} points at '{key}', which two files carry, so it reaches nothing while the {doubled_word(key)} is doubled")

    return StepsError(f"step {number} points at '{key}', which is not a todo, a bug or a review the brain knows")


def doubled_pointer(
    number: int,
    key: str,
    pointer: str,
) -> StepsError:
    """The refusal for a key that reaches one file whose own pointer, the id or the name the cell would store, two files carry."""
    return StepsError(f"step {number} points at '{key}', and its {doubled_word(pointer)} {pointer} two files carry, so it would reach nothing")


def doubled_word(
    key: str,
) -> str:
    """What a doubled key is called: a review's name, or an id."""
    return "name" if schema.is_review_folder(key.lower()) else "id"


def opens_with_a_key(
    line: str,
) -> bool:
    """Whether a line is a numbered step whose text opens with an id shaped key or a review's name, the shape a pointer takes in a list."""
    numbered = LISTED.match(line)
    if numbered is None:
        return False

    opened = POINTER.match(numbered.group(2))

    return opened is not None and looks_like_a_key(opened.group(1))


def looks_like_a_key(
    word: str,
) -> bool:
    """Whether a word opening a step line is meant as an id, in any case, or a review's name rather than a word of prose, a plain date being prose."""
    if DATE_SHAPE.match(word):
        return False

    return schema.FILENAME_WITH_ID.match(f"{word.upper()}-x.md") is not None or schema.is_review_folder(word.lower())


def known(
    brain: store.Brain,
    key: str,
) -> bool:
    """Whether the key names an entry of any kind, so a step opening with a decision's id reads as prose rather than a pointer that reaches nothing."""
    return found(brain=brain, key=key) is not None


def found(
    brain: store.Brain,
    key: str,
) -> Optional[store.Entry]:
    """The entry the key names, an id or a review's name answering in any case, or None when it names nothing or more than one file."""
    for candidate in (key, key.upper(), key.lower()):
        try:
            entry = brain.find(candidate)
        except store.Ambiguous:
            return None

        if entry is not None:
            return entry

    return None


def doubled(
    brain: store.Brain,
    key: str,
) -> bool:
    """Whether the key names more than one file, in any case, the doubled id or name a merge leaves."""
    try:
        for candidate in (key, key.upper(), key.lower()):
            brain.find(candidate)
    except store.Ambiguous:
        return True

    return False


def pointed(
    brain: store.Brain,
    key: str,
) -> Optional[store.Entry]:
    """The todo, bug or review a step points at by its id or its name, a review's folder answering as its name does, or None when the key reaches nothing of those kinds, or reaches one by a stem or a path a step never uses."""
    entry = found(brain=brain, key=key)
    if entry is None or entry.kind.name not in POINTED_KINDS:
        return None

    if not {key, key.upper(), key.lower()} & answers_of(entry):
        return None

    return entry


def pointer_of(
    entry: store.Entry,
) -> str:
    """The key a step stores for the entry, the id for a todo or a bug, which a retitle never moves, and the name for a review, or for a todo or a bug with no id, a file the tool never wrote, whose cells a retitle moves with the name."""
    return entry.identifier or entry.name


def derived_row(
    brain: store.Brain,
    step: Step,
    own: Optional[set[str]] = None,
    seen: Optional[set[str]] = None,
) -> Step:
    """The row a pointed step should carry, its mark read off the entry it points at, and its pointer moved on to the entry a refile filed in its place, unless that entry's steps point back at the holder named by its own keys, or the row as it is when it points at nothing the brain can read."""
    if not step.points_at:
        return step

    entry = pointed(brain=brain, key=step.points_at)
    if entry is None:
        return step

    status = entry.field("status") or ""
    if status == "refiled":
        seen = (seen or set()) | {step.points_at}
        successor = refiled_to(brain=brain, entry=entry)
        if successor is not None and hands_on_to(brain=brain, successor=successor, own=own, seen=seen):
            moved = Step(number=step.number, text=step.text, points_at=pointer_of(successor), by=step.by, status=step.status)
            return derived_row(brain=brain, step=moved, own=own, seen=seen)

    if status in POINTED_KINDS[entry.kind.name]:
        mark = "done"
    elif status in POINTED_GONE[entry.kind.name]:
        mark = "skipped"
    else:
        mark = "pending"

    return Step(number=step.number, text=step.text, points_at=kept_pointer(brain=brain, step=step, entry=entry), by=step.by, status=mark)


def kept_pointer(
    brain: store.Brain,
    step: Step,
    entry: store.Entry,
) -> str:
    """The pointer a derived row stores, the entry's own id or name, unless two files carry that and the cell as written still reaches the one it means."""
    pointer = pointer_of(entry)

    return step.points_at if doubled(brain=brain, key=pointer) else pointer


def hands_on_to(
    brain: store.Brain,
    successor: store.Entry,
    own: Optional[set[str]],
    seen: set[str],
) -> bool:
    """Whether a step at a refiled entry moves on to the entry filed in its place: not through a loop of refiles, not to an id two files carry, and, with the holder's own keys in hand, not to the holder itself nor to a successor whose steps lead back here or do not read, since either could leave the two waiting on each other."""
    handed = pointer_of(successor)
    if handed in seen or doubled(brain=brain, key=handed):
        return False

    if not own:
        return True

    if handed in own or unreadable_reached(brain=brain, key=handed) is not None:
        return False

    return not points_back(brain=brain, key=handed, own=own)


def refiled_to(
    brain: store.Brain,
    entry: store.Entry,
) -> Optional[store.Entry]:
    """The entry a refiled todo or bug was filed as, named by id in the became field the refile writes, which a retitle never moves, or for a file from before that field the last link of the other kind whose own links name it back, or None."""
    wanted = "bug" if entry.kind.name == "todo" else "todo"
    named = entry.field("became")
    if named:
        target = found(brain=brain, key=named)
        return target if target is not None and target.kind.name == wanted else None

    for link in reversed(entry.document.get_list("links")):
        target = found(brain=brain, key=link)
        if target is None or target.kind.name != wanted:
            continue

        if any(found(brain=brain, key=back) is entry for back in target.document.get_list("links")):
            return target

    return None


def became_slips(
    brain: store.Brain,
    entry: store.Entry,
) -> list[str]:
    """What is wrong with the entry's became field, in words: set while the entry is not refiled, naming nothing, naming two files, or naming an entry of the wrong kind."""
    named = entry.field("became")
    if not named:
        return []

    if entry.field("status") != "refiled":
        return [f"became names {named} while the entry is not refiled"]

    wanted = "bug" if entry.kind.name == "todo" else "todo"
    if doubled(brain=brain, key=named):
        return [f"became names {named}, which two files carry, so the steps that point here follow nothing until one is refiled"]

    target = found(brain=brain, key=named)
    if target is None:
        return [f"became names {named}, which reaches nothing, so the steps that point here follow nothing"]

    if target.kind.name != wanted:
        return [f"became names {named}, which is a {target.kind.name} where a refiled {entry.kind.name} becomes a {wanted}"]

    return []


def became_gaps(
    brain: store.Brain,
    entry: store.Entry,
    fields: dict[str, str],
) -> list[str]:
    """What a became value falls short of when it is set through brain set: it names one entry of the other kind, and the entry is refiled."""
    named = fields.get("became", "")
    if not named:
        return []

    wanted = "bug" if entry.kind.name == "todo" else "todo"
    if fields.get("status") != "refiled":
        return ["became goes with status refiled, since it names what the entry was refiled as"]

    if doubled(brain=brain, key=named):
        return [f"became names {named}, which two files carry"]

    target = found(brain=brain, key=named)
    if target is None or target.kind.name != wanted:
        return [f"became names {named}, which is not a {wanted} the brain knows"]

    return []


def points_back(
    brain: store.Brain,
    key: str,
    own: set[str],
    seen: Optional[set[str]] = None,
) -> bool:
    """Whether the entry the key names, while it is open, has a step that points, at any remove, at one of the own keys, which would leave the two waiting on each other for good, a closed entry's steps waiting on nothing."""
    entry = pointed(brain=brain, key=key)
    if entry is None or not is_open(entry):
        return False

    seen = (seen or set()) | {key}
    try:
        keys = pointers_in(entry.document.body) if entry.kind.name in STEPPED_KINDS else []
    except StepsError:
        return False

    # a cell is read by the entry it reaches, so one holding an id two files carry, which reaches nothing, points back at neither
    for pointed_key in keys:
        target = pointed(brain=brain, key=pointed_key)
        if target is None:
            continue

        if any(answer in own and not doubled(brain=brain, key=answer) for answer in own_keys(target)):
            return True

        if pointed_key not in seen and points_back(brain=brain, key=pointed_key, own=own, seen=seen):
            return True

    return False


def pointers_in(
    body: str,
) -> list[str]:
    """The keys the Steps section points at, read off the table, or off a numbered list not yet turned into one, whose lines open with the key the way filing reads them, since the list is the table in waiting and a loop through it would stand once it is turned, refused when the table does not read, and none for a list that does not read as one, which doctor reports."""
    if has_table(body):
        return [step.points_at for step in table(body) if step.points_at]

    if table_shaped(body):
        raise StepsError(TABLE_LINES_WITHOUT_THE_HEADER)

    return [step.points_at for step in listed_rows(body) if step.points_at]


TABLE_LINES_WITHOUT_THE_HEADER = "the Steps section holds table lines without the header the tool reads, | # | Step | Points at | By | Status |"


def refuse_cycles(
    brain: store.Brain,
    rows: list[Step],
    own: set[str],
) -> None:
    """Refuse a row whose entry's steps lead back to the holder, or whose entry's steps cannot be read on the way, since a guess either way could leave the two waiting on each other for good."""
    for step in rows:
        if not step.points_at:
            continue

        broken = unreadable_reached(brain=brain, key=step.points_at)
        if broken is not None:
            where = step.points_at if broken == step.points_at else f"{step.points_at}, whose steps lead to {broken}"
            raise StepsError(f"step {step.number} points at {where}, whose steps do not read, so the tool cannot tell whether they point back here, and brain doctor names what is wrong with them")

        if points_back(brain=brain, key=step.points_at, own=own):
            raise StepsError(f"step {step.number} points at {step.points_at}, whose steps point back here, and the two would wait on each other for good")


def unreadable_reached(
    brain: store.Brain,
    key: str,
    seen: Optional[set[str]] = None,
) -> Optional[str]:
    """The key, as a step wrote it, of the first open entry on the way from the one named whose Steps holds a table that does not read, or None when every table on the way reads, a list on the way walked by the keys its lines open with."""
    entry = pointed(brain=brain, key=key)
    if entry is None or not is_open(entry) or entry.kind.name not in STEPPED_KINDS:
        return None

    try:
        keys = pointers_in(entry.document.body)
    except StepsError:
        return key

    seen = (seen or set()) | {key}
    for pointed_key in keys:
        if pointed_key in seen:
            continue

        broken = unreadable_reached(brain=brain, key=pointed_key, seen=seen)
        if broken is not None:
            return broken

    return None


def reopening_gaps(
    brain: store.Brain,
    entry: store.Entry,
    fields: dict[str, str],
) -> list[str]:
    """What reopening a closed entry falls short of: its steps and another open entry's would point at each other, which a close let stand and a reopening would set waiting for good."""
    status = fields.get("status", "")
    was_open = is_open(entry)
    if entry.kind.name not in STEPPED_KINDS or was_open or status not in OPEN_STATUSES[entry.kind.name]:
        return []

    body = entry.document.body
    unreadable = f"reopening {entry.name} waits on its steps table, which does not read"
    if has_table(body):
        try:
            rows = table(body)
        except StepsError as broken:
            return [f"{unreadable}, {broken}, so the tool cannot tell whether it and another entry would wait on each other, and brain doctor names what is wrong with it"]
    elif table_shaped(body):
        return [f"{unreadable}, {TABLE_LINES_WITHOUT_THE_HEADER}, so the tool cannot tell whether it and another entry would wait on each other, and brain doctor names what is wrong with it"]
    else:
        rows = listed_rows(body)

    for step in rows:
        broken_key = unreadable_reached(brain=brain, key=step.points_at) if step.points_at else None
        if broken_key is not None:
            return [f"reopening {entry.name} waits on the steps of {broken_key}, which do not read, so the tool cannot tell whether it and {entry.name} would wait on each other, and brain doctor names what is wrong with them"]

    try:
        refuse_cycles(brain=brain, rows=rows, own=own_keys(entry))
    except StepsError as loop:
        return [f"reopening {entry.name} would leave it and another entry waiting on each other, {loop}"]

    return []


def table_shaped(
    body: str,
) -> bool:
    """Whether the Steps section holds a line shaped like a table's, outside code, while no header the tool reads sits there, a table typed with another header, a separator the tool does not read or an indent, an indented line under a numbered one being that step's wrapped line whatever it opens with."""
    where = span(body)
    if where is None:
        return False

    spans = sections.code_spans(body)
    position = where[0]
    under_a_step = False
    for line in body[where[0] : where[1]].split("\n"):
        start = position
        position += len(line) + 1
        if not line.strip() or sections.in_code(spans=spans, position=start):
            continue

        if LISTED.match(line):
            under_a_step = True
            continue

        if line.lstrip().startswith("|") and not (under_a_step and line[0] in " \t"):
            return True

    return False


def listed_rows(
    body: str,
) -> list[Step]:
    """The numbered lines under Steps, as rows to judge, each numbered as written, its indented wrapped lines joined on the way listed joins them, and pointing at the key it opens with, whatever else sits among them, since a pointer counts for a loop whether or not the list reads whole, which is doctor's to report."""
    where = span(body)
    if where is None:
        return []

    spans = sections.code_spans(body)
    numbered: list[tuple[int, str]] = []
    position = where[0]
    for line in body[where[0] : where[1]].split("\n"):
        start = position
        position += len(line) + 1
        if not line.strip() or sections.in_code(spans=spans, position=start):
            continue

        match = LISTED.match(line)
        if match is not None:
            numbered.append((int(match.group(1)), match.group(2)))
        elif numbered and line[0] in " \t":
            number, text = numbered[-1]
            numbered[-1] = (number, f"{text} {line.strip()}")

    rows: list[Step] = []
    for number, text in numbered:
        opened = POINTER.match(text.strip())
        rows.append(Step(number=number, text=text, points_at=opened.group(1) if opened is not None else "", by="agent", status="pending"))

    return rows


def looping(
    brain: store.Brain,
    entry: store.Entry,
    rows: list[Step],
) -> list[str]:
    """The steps of an open entry that point at an open entry whose steps point back, one line each, which doctor reports since no write should have let them stand."""
    if not is_open(entry):
        return []

    own = own_keys(entry)

    return [
        f"step {step.number} points at {step.points_at}, whose steps point back here, and the two wait on each other for good"
        for step in rows
        if step.points_at and points_back(brain=brain, key=step.points_at, own=own)
    ]


def add(
    brain: store.Brain,
    key: str,
    lines: list[str],
    by: str = "agent",
) -> list[int]:
    """Append steps to the entry, numbered after the last, and hand back their numbers."""
    entry = stepped_entry(brain=brain, key=key)
    if not is_open(entry):
        raise StepsError(f"{entry.name} is {entry.field('status') or ''}, a step is added while it is open")

    own = own_keys(entry)
    body = entry.document.body
    if span(body) is None:
        body = with_table(body=body, rows=[], kind=entry.kind)
    elif not has_table(body):
        body = converted(brain=brain, body=body, by=entry.field("by") or "agent", kind=entry.kind, own=own)

    rows = table(body)
    added = [row_from(brain=brain, number=len(rows) + offset, line=line, by=by, own=own) for offset, line in enumerate(lines, 1)]
    if not added:
        raise StepsError("no steps given, one per line")

    for step in added:
        if step.points_at in own:
            raise StepsError(f"step {step.number} points at {entry.name} itself, and a step points at other work")

    refuse_cycles(brain=brain, rows=added, own=own)

    entry.document.body = rewrite(body=body, rows=rows + added)
    save(brain=brain, entry=entry)

    return [step.number for step in added]


def mark(
    brain: store.Brain,
    key: str,
    number: int,
    status: str,
) -> None:
    """Mark a step done, skipped or pending again, one that points at nothing, or one whose pointer reaches nothing the brain can read, a doubled id or a dead key, since a step that reaches its entry takes its mark from there."""
    if status not in STEP_STATUSES:
        raise StepsError(f"a step is {' or '.join(STEP_STATUSES)}, not '{status}'")

    entry = stepped_entry(brain=brain, key=key)
    if not is_open(entry):
        raise StepsError(f"{entry.name} is {entry.field('status')}, a step is marked while it is open")

    rows = rows_of(brain=brain, entry=entry)
    step = step_numbered(entry=entry, rows=rows, number=number)
    target = pointed(brain=brain, key=step.points_at) if step.points_at else None
    if target is not None:
        raise StepsError(f"step {number} points at {step.points_at}, {follows_its_entry(brain=brain, target=target, key=step.points_at, asked=status)}")

    rows[number - 1] = Step(number=number, text=step.text, points_at=step.points_at, by=step.by, status=status)
    entry.document.body = rewrite(body=entry.document.body, rows=rows)
    save(brain=brain, entry=entry)


def follows_its_entry(
    brain: store.Brain,
    target: store.Entry,
    key: str,
    asked: str,
) -> str:
    """Why a pointed step takes no mark by hand, and what gives it the mark asked for instead, the entry named as the cell names it, since that key is the one that resolves."""
    kind = target.kind.name
    status = target.field("status") or ""
    # a step still at a refiled entry is one whose hand off is held, since the entry filed in its place leads back here or its steps do not read, or one that has no successor, and a set to open would undo the refile
    if status == "refiled" and asked == "pending":
        successor = refiled_to(brain=brain, entry=target)
        if successor is None:
            became = target.field("became") or ""
            if became and doubled(brain=brain, key=became):
                return f"which was refiled as {became}, which two files carry, so its mark stands skipped while the {doubled_word(became)} is doubled, and follows {became} then"

            return "which was refiled as nothing the brain knows, so its mark stands skipped"

        handed = pointer_of(successor)
        if successor.field("status") == "refiled":
            return f"which sits in a loop of refiles through {key} and {handed}, so its mark stands skipped until the loop is undone"

        if doubled(brain=brain, key=handed):
            return f"which was refiled as {handed}, which two files carry, so its mark stands skipped while the {doubled_word(handed)} is doubled, and follows {handed} then"

        broken = unreadable_reached(brain=brain, key=handed)
        if broken is not None:
            where = handed if broken == handed else f"{handed}, whose steps lead to {broken}"

            return f"which was refiled as {where}, whose steps do not read, so its mark stands skipped until they are put right, which brain doctor names, and follows {handed} then"

        return f"which was refiled as {handed}, whose steps lead back here, so its mark stands skipped until {handed} closes or its steps no longer lead back, and follows {handed} then"

    if is_open(target):
        if asked == "pending":
            return f"so its mark follows that entry, and is pending while {key} is open"

        settles = POINTED_KINDS[kind] if asked == "done" else POINTED_GONE[kind]
        if not settles:
            return f"so its mark follows that entry, and a step at a {kind} is never {asked}"

        # a refile never skips the step, it hands it on to the entry filed in the refiled one's place, where it takes that entry's mark
        gone = [status for status in settles if status != "refiled"]
        if asked == "skipped" and not gone:
            return f"so its mark follows that entry, and follows the entry filed in its place once {key} is refiled, a {kind} being never dropped"

        if asked == "skipped":
            return f"so its mark follows that entry, and goes skipped once {key} is {' or '.join(gone)}, or follows the entry filed in its place once {key} is refiled"

        return f"so its mark follows that entry, and goes {asked} once {key} is {' or '.join(settles)}"

    if status in POINTED_KINDS[kind] or status in POINTED_GONE[kind]:
        mark = "done" if status in POINTED_KINDS[kind] else "skipped"
        if asked == "pending" and kind == "review":
            return f"which is {status}, so its mark follows that entry, and brain set {key} status=open reopens it"

        if asked == "pending":
            return f"which is {status}, so its mark follows that entry, and brain set {key} status=open reopens it, unless its steps and another open entry's would wait on each other, or its table or one they lead to does not read, which the set refuses"

        if asked == mark:
            return f"which is {status}, so its mark follows that entry and is {mark} already"

        return f"which is {status}, so its mark follows that entry and stands {mark}, not {asked}"

    return "whose status is not one the tool knows, which brain doctor reports, and its mark follows that entry"


def step_numbered(
    entry: store.Entry,
    rows: list[Step],
    number: int,
) -> Step:
    if number < 1 or number > len(rows):
        raise StepsError(f"{entry.name} has {len(rows)} steps, and there is no step {number}")

    return rows[number - 1]


def view(
    brain: store.Brain,
    key: str,
) -> str:
    """The entry's steps in a few lines: its state, then one line per step with its mark."""
    entry = stepped_entry(brain=brain, key=key)
    rows = rows_of(brain=brain, entry=entry)
    settled = sum(1 for step in rows if step.status in ("done", "skipped"))
    lines = [f"{entry.name}, {entry.field('status')}, {settled} of {len(rows)} steps settled"]
    for step in rows:
        pointer = f", {step.points_at}" if step.points_at else ""
        lines.append(f"{step.number}. {step.text}{pointer}, {step.status}")

    return "\n".join(lines)


def stepped_entry(
    brain: store.Brain,
    key: str,
) -> store.Entry:
    try:
        entry = brain.find(key)
    except store.Ambiguous as clash:
        raise StepsError(str(clash)) from clash

    if entry is None or entry.kind.name not in STEPPED_KINDS:
        raise StepsError(f"no plan, todo or bug answers to '{key}'")

    return entry


def save(
    brain: store.Brain,
    entry: store.Entry,
    stamp: str = "",
) -> None:
    entry.document.set(key="updated", value=stamp or date.today().isoformat())
    frontmatter.write(path=str(entry.path), document=entry.document)
    brain.refresh()


def settle(
    brain: store.Brain,
    entry: store.Entry,
) -> None:
    """Write the rows an entry's readers would derive into its document, without saving, so a status move leaves the file as the readers see it, whatever status it moves from."""
    try:
        rows = stored_rows(entry)
    except StepsError:
        return

    own = own_keys(entry)
    fresh = [derived_row(brain=brain, step=step, own=own) for step in rows]
    if fresh != rows:
        entry.document.body = rewrite(body=entry.document.body, rows=fresh)


def refresh(
    brain: store.Brain,
    entry: store.Entry,
    stamp: str = "",
) -> bool:
    """Bring every pointed step's row of an open entry in line with the entry it points at, and say whether anything changed, a closed entry's table being its record and left as it is."""
    if not is_open(entry):
        return False

    rows = stored_rows(entry)
    own = own_keys(entry)
    fresh = [derived_row(brain=brain, step=step, own=own) for step in rows]
    if fresh == rows:
        return False

    entry.document.body = rewrite(body=entry.document.body, rows=fresh)
    save(brain=brain, entry=entry, stamp=stamp)

    return True


def repoint(
    brain: store.Brain,
    keys: set[str],
    to: str,
    stamp: str = "",
) -> list[str]:
    """Write every cell that points at one of the keys, in any case, as the pointer given, in closed holders as much as open ones, a list not yet turned into a table having its opening keys written the same way since it is the table in waiting, and name the holders that changed."""
    wanted = {key.lower() for key in keys}
    changed: list[str] = []
    # a holder that cannot be written is passed over for the ones after it, whose folders may well be writable, and its failure is raised once every holder has had its turn
    first_failure: Optional[OSError] = None
    for kind_name in STEPPED_KINDS:
        for holder in brain.entries_of(schema.kind_named(kind_name)):
            try:
                rows = stored_rows(holder)
            except StepsError:
                relisted = repointed_list(body=holder.document.body, wanted=wanted, to=to)
                if relisted != holder.document.body:
                    holder.document.body = relisted
                    try:
                        save(brain=brain, entry=holder, stamp=stamp)
                    except OSError as error:
                        first_failure = first_failure or error
                        continue

                    changed.append(holder.name)

                continue

            fresh = [
                Step(number=step.number, text=step.text, points_at=to, by=step.by, status=step.status)
                if step.points_at.lower() in wanted and step.points_at != to
                else step
                for step in rows
            ]
            if fresh != rows:
                holder.document.body = rewrite(body=holder.document.body, rows=fresh)
                try:
                    save(brain=brain, entry=holder, stamp=stamp)
                except OSError as error:
                    first_failure = first_failure or error
                    continue

                changed.append(holder.name)

    if first_failure is not None:
        raise first_failure

    return changed


def repointed_list(
    body: str,
    wanted: set[str],
    to: str,
) -> str:
    """The body with every numbered line under Steps that opens with one of the keys, in any case, opening with the pointer given instead, and as it was when the section holds a table or nothing."""
    where = span(body)
    if where is None or has_table(body):
        return body

    def relined(line: re.Match) -> str:
        opened = POINTER.match(line.group(2))
        if opened is None or opened.group(1).lower() not in wanted:
            return line.group(0)

        return f"{line.group(0)[: line.start(2) - line.start(0)]}{to}{line.group(2)[len(opened.group(1)) :]}"

    start, end = where

    return body[:start] + LISTED.sub(relined, body[start:end]) + body[end:]


def refresh_pointing(
    brain: store.Brain,
    key: str,
    stamp: str = "",
) -> list[str]:
    """Refresh every plan, todo and bug with a step that points at the entry, named by any key the brain answers to, and name the ones that changed."""
    entry = found(brain=brain, key=key)
    if entry is None or entry.kind.name not in POINTED_KINDS:
        return []

    wanted = {answer.lower() for answer in answers_of(entry)}
    changed: list[str] = []
    # a holder that cannot be written is passed over for the ones after it, and its failure is raised once every holder has had its turn
    first_failure: Optional[OSError] = None
    for kind_name in STEPPED_KINDS:
        for holder in brain.entries_of(schema.kind_named(kind_name)):
            try:
                rows = stored_rows(holder)
            except StepsError:
                continue

            if not any(step.points_at.lower() in wanted for step in rows):
                continue

            try:
                if refresh(brain=brain, entry=holder, stamp=stamp):
                    changed.append(holder.name)
            except OSError as error:
                first_failure = first_failure or error

    if first_failure is not None:
        raise first_failure

    return changed


def all_settled(
    rows: list[Step],
) -> bool:
    """Whether every step is done or skipped, which holds of no steps at all, a goal alone closing on its Done when."""
    return all(step.status in ("done", "skipped") for step in rows)


def gaps(
    brain: store.Brain,
    entry: store.Entry,
    fields: dict[str, str],
) -> list[str]:
    """What an entry closing falls short of, the one rule its table sets: done, or fixed for a bug, means every step is done or skipped, each pointed step read off its entry, while dropped, wontfix and refiled close as they are."""
    return closing_gaps(brain=brain, kind=entry.kind, fields=fields, body=entry.document.body, own=own_keys(entry))


def closing_gaps(
    brain: store.Brain,
    kind: schema.Kind,
    fields: dict[str, str],
    body: str,
    own: Optional[set[str]] = None,
) -> list[str]:
    """The closing rule on a body and its fields, for an entry being filed as much as one being changed."""
    status = fields.get("status", "")
    if kind.name not in CLOSING or status != CLOSING[kind.name]:
        return []

    if kind.name != "plan" and blank(body):
        return []

    try:
        rows = [derived_row(brain=brain, step=step, own=own) for step in table(body)]
    except StepsError as broken:
        return [str(broken)]

    if not all_settled(rows):
        return [f"a {kind.name} goes {status} once every step is done or skipped, and one of its steps still stands"]

    return []
