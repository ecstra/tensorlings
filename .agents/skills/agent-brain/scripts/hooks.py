"""The hook every agent runs, so the brain is checked without anyone asking.

Claude Code, Codex and Gemini CLI each call this with their own event
JSON on standard input and the event name as the first argument. Five
events matter, the start of a session, each prompt the owner sends, a
tick after the agent's tool calls, a write that is about to create a
file inside .agents/, and a write that just landed there. The session
start, the prompt and the tick carry the clock, since a model does not
know the time and forgets what it was told, where the owner's standing
word to proceed stands, on, held or off, and the work the agent is on,
the chain from the plan down to the step.

    python3 .agents/skills/agent-brain/scripts/hooks.py session-start --agent claude
    python3 .agents/skills/agent-brain/scripts/hooks.py prompt --agent claude
    python3 .agents/skills/agent-brain/scripts/hooks.py tick --agent claude
    python3 .agents/skills/agent-brain/scripts/hooks.py pre-write --agent codex
    python3 .agents/skills/agent-brain/scripts/hooks.py post-write --agent gemini

Exit 2 blocks the write, with the reason on standard error, which all
three agents honour.
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO, NoReturn, Optional, Union

from brainlib import autoack, clock, doctor, frontmatter, links, reads, schema, steps, store, task

BLOCK = 2
FAILED = 1

# The file markers of a Codex patch. A patch written with Windows line
# endings carries a carriage return before every newline, so the path
# is taken up to the last visible character rather than the line's end.
MARKER = re.compile(r"^\*\*\* (Add File|Update File|Delete File|Move to): (.+?)\s*$")
PATCH_OPENS = "*** Begin Patch"

# Where Codex puts the patch text: `command` per its hook reference, the
# others in case that name moves.
PATCH_KEYS = ("command", "patch", "input")

SKILLS = "skills"


@dataclass
class Touch:
    """One file a write event reaches, with what the write would do to it.

    `content` is the whole new text when the tool writes a file whole,
    `old_string` and `new_string` the edit a tool makes in place, and
    `hunks` the blocks a patch applies, each line still carrying its
    leading space, minus or plus. Whichever the agent's tool provides,
    the hook works out the file as the write would leave it.
    """

    path: Path
    content: Optional[str] = None
    old_string: Optional[str] = None
    new_string: Optional[str] = None
    replace_all: bool = False
    hunks: tuple[tuple[str, ...], ...] = ()
    moved_from: Optional[Path] = None
    deleted: bool = False


class HookParser(argparse.ArgumentParser):
    """An argument parser whose refusal is the non blocking error, never the blocking exit.

    argparse exits 2 on an argument it does not know, and 2 is the exit
    every agent takes as a block, so a hook script older than the event
    its config names, one a checkout of an older branch put in place,
    would erase every prompt until the session restarts.
    """

    def error(
        self,
        message: str,
    ) -> NoReturn:
        print(f"the brain hook could not run: {message}", file=sys.stderr)
        raise SystemExit(FAILED)


def main(
    argv: Optional[list[str]] = None,
) -> int:
    parser = HookParser(prog="hooks")
    parser.add_argument("event", choices=("session-start", "prompt", "tick", "pre-write", "post-write"))
    parser.add_argument("--agent", choices=("claude", "codex", "gemini"), default="claude")
    parser.add_argument("--with-rules", action="store_true", help="add SYSTEM.md to the context")
    args = parser.parse_args(argv)

    try:
        return run(args)
    except (store.StoreError, frontmatter.FrontmatterError, OSError, ValueError) as error:
        print(f"the brain hook could not run: {store.explain(error)}", file=sys.stderr)
        return FAILED


def run(
    args: argparse.Namespace,
) -> int:
    """Dispatch one event. A brain the hook cannot read fails visibly, not silently.

    Exit 1 is the non blocking error every agent shows to the person,
    which is the right outcome for a broken brain: the write goes ahead
    and the reason is on screen, rather than every write being refused
    until someone finds a traceback.
    """
    event = read_event()
    brain = locate(event)
    if brain is None:
        return 0

    if args.event == "session-start":
        return session_start(brain=brain, agent=args.agent, with_rules=args.with_rules)

    if args.event == "prompt":
        return prompt(brain=brain, agent=args.agent)

    if args.event == "tick":
        return tick(brain=brain, agent=args.agent)

    if args.event == "pre-write":
        return pre_write(brain=brain, event=event, agent=args.agent)

    return post_write(brain=brain, event=event, agent=args.agent)


def read_event() -> dict:
    """The event JSON the agent piped in, or nothing when it piped nothing.

    A closed standard input leaves `sys.stdin` as None, which is not an
    event either.
    """
    if sys.stdin is None or sys.stdin.isatty():
        return {}

    raw = stdin_bytes().decode("utf-8", errors="replace")
    if not raw.strip():
        return {}

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def stdin_bytes() -> bytes:
    """Standard input as bytes, read under the text layer so the locale's encoding never gets a say.

    The event is JSON, which is UTF-8 whatever the locale, and on a
    prompt it carries the owner's own text. A byte layer that is not
    there, a text stream handed in by a test, is read as the text it
    holds.
    """
    buffer: Optional[BinaryIO] = getattr(sys.stdin, "buffer", None)
    if buffer is None:
        return sys.stdin.read().encode("utf-8")

    return buffer.read()


def locate(
    event: dict,
) -> Optional[store.Brain]:
    start = Path(event.get("cwd") or Path.cwd())

    try:
        return store.find_brain(start)
    except store.StoreError:
        return None


def session_start(
    brain: store.Brain,
    agent: str,
    with_rules: bool,
) -> int:
    text = the_moment(brain=brain) + "\n\nThe brain, counted from the files right now:\n" + reads.render_status(reads.status(brain=brain))
    if with_rules:
        rules = brain.root / "SYSTEM.md"
        if store.readable_file(rules):
            text += "\n\nThe rules this project works by, from .agents/SYSTEM.md:\n\n" + text_of(rules)

        chat_style = brain.root.parent / f".{agent}" / "plain-and-short.md"
        if store.readable_file(chat_style):
            text += f"\n\nHow to talk in this project, from .{agent}/plain-and-short.md:\n\n" + text_of(chat_style)

    emit_context(agent=agent, text=text, event_name="SessionStart")
    return 0


def prompt(
    brain: store.Brain,
    agent: str,
) -> int:
    """On each prompt the owner sends, the clock, the standing word and the current work, so all three are in front of the model when it reads the prompt."""
    emit_context(agent=agent, text=the_moment(brain=brain), event_name=PROMPT.get(agent, "UserPromptSubmit"))
    return 0


def tick(
    brain: store.Brain,
    agent: str,
) -> int:
    """After the agent's tool calls, the clock, the standing word and the current work again, off and none included, so a long stretch of work never loses any of them and sees the word end."""
    emit_context(agent=agent, text=the_moment(brain=brain), event_name=TICK.get(agent, "PostToolUse"))
    return 0


def the_moment(
    brain: store.Brain,
) -> str:
    """The clock, the standing word and the current work, the lines the session start, the prompt hook and the tick carry."""
    return clock.line() + "\n" + autoack.describe(brain=brain) + "\n" + task.describe(brain=brain)


def text_of(
    path: Path,
) -> str:
    """The file as text for the agent's context, or one line saying why there is none."""
    try:
        return path.read_bytes().decode("utf-8", errors="replace")
    except OSError as error:
        return f"({store.near(str(path))} cannot be read: {error.strerror or error})"


def pre_write(
    brain: store.Brain,
    event: dict,
    agent: str,
) -> int:
    touched = touches(event=event, agent=agent)
    for touch in touched:
        verdict = refusal(brain=brain, touch=touch)
        if verdict is None:
            continue

        print(verdict, file=sys.stderr)
        return BLOCK

    # a patch that reaches several brain files is judged across them too, since each alone reads as fine against the disk
    if len(touched) > 1:
        verdict = loop_across(brain=brain, writes=table_writes_of(brain=brain, touched=touched))
        if verdict is not None:
            print(verdict, file=sys.stderr)
            return BLOCK

    emit_nothing(agent)
    return 0


def refusal(
    brain: store.Brain,
    touch: Touch,
) -> Optional[str]:
    """Why this write may not touch the brain by hand, or None if it may.

    A new brain file, an index, a dotted path, the standing word, the
    current work, a rename, the header block of an existing file and
    the rows of its steps table are the tool's. The prose below a
    header and around the table is the agent's to edit, and doctor
    reads it afterwards.
    """
    if touch.moved_from is not None:
        source = entry_path(brain=brain, path=touch.moved_from)
        if source is not None:
            return f"{source} is renamed with brain refile, not by hand"

    path = touch.path
    inside = links.within(brain=brain, path=path)
    if inside is None:
        return None

    # the word's name in any case, since a disk that ignores case would take
    # .ACK for the file itself, and the suffix as the store reads it
    if inside.casefold() == autoack.FILE and not touch.deleted:
        return f"{autoack.FILE} is the owner's standing word, set with brain ack on and never written by hand"

    if inside.casefold() == task.FILE and not touch.deleted:
        return f"{task.FILE} is the current work, set with brain task on and never written by hand"

    if not markdown_named(path):
        return None

    if Path(inside).parts[0] == SKILLS:
        return None

    if store.hidden(Path(inside)):
        return f"{inside} would be a dotted path, which the brain never lists or checks, so it holds none"

    if Path(inside).name == schema.INDEX:
        if schema.is_kit_path(inside):
            return f"{inside} is generated, run brain index instead of writing it"

        return f"{inside} would be an index where none belongs, the tool builds one per tracker, docs/ and memory/"

    if schema.is_kit_path(inside):
        return None

    if store.quietly(path.exists):
        return header_change(touch=touch, inside=inside) or table_change(brain=brain, touch=touch, inside=inside)

    if Path(inside).parts[0] == "spec":
        return f"{inside} would nest inside spec/, which holds its files directly"

    return (
        f"{inside} would be a new brain file written by hand. File it with "
        "brain new so it gets its id, its header block and its index line, "
        "with the prose on standard input, or rename an existing one with "
        "brain refile."
    )


def entry_path(
    brain: store.Brain,
    path: Path,
) -> Optional[str]:
    """The brain relative path of an existing entry file, or None for anything else."""
    inside = links.within(brain=brain, path=path)
    if inside is None or not markdown_named(path) or not store.quietly(path.exists):
        return None

    relative_path = Path(inside)
    if relative_path.parts[0] == SKILLS or store.hidden(relative_path) or schema.is_kit_path(inside):
        return None

    return inside


def markdown_named(
    path: Path,
) -> bool:
    """Whether the path carries the markdown suffix the way the store reads it, without regard to case on Windows."""
    return os.path.normcase(path.suffix) == ".md"


HEADER_IS_THE_TOOLS = (
    "the header block changes through brain set or brain refile, not by hand. "
    "The prose below it is yours to edit."
)


def header_change(
    touch: Touch,
    inside: str,
) -> Optional[str]:
    """Why this write to an existing brain file may not go ahead, or None if it may.

    The file is judged as the write would leave it: the whole new text,
    the text after the edit, or the file after the patch's hunks. Its
    header block, every line between the fences except the updated date
    the upkeep rule asks the agent to move, must come out as it went in.
    A file whose header does not read is already broken, so the write is
    taken as the repair and held to one thing, leaving a header that
    reads, then doctor judges the rest. A file the hook cannot open at
    all is not judged. A hunk that fits nowhere is judged by its shape
    either way: refused when it carries a fence or a header field, since
    the hook cannot tell where it would land.
    """
    try:
        text = frontmatter.load(str(touch.path))
    except frontmatter.FrontmatterError:
        return None

    current: Optional[frontmatter.Document] = None
    trouble = ""
    try:
        current = frontmatter.parse(text)
    except frontmatter.FrontmatterError as error:
        trouble = str(error)

    after, loose = outcome_of(touch=touch, text=text)
    names = field_names(inside=inside, lines=current.lines if current is not None else [])
    for hunk in loose:
        line = header_line_in(hunk=hunk, names=names)
        if line is not None:
            return f"{inside}: a hunk the hook cannot place touches the header block with '{line}', and {HEADER_IS_THE_TOOLS}"

    if after is None:
        return None

    try:
        wanted = frontmatter.parse(after)
    except frontmatter.FrontmatterError as error:
        if current is None:
            return still_broken(inside=inside, trouble=trouble, error=str(error))

        return f"{inside}: after that write the header would not read, {error}, and {HEADER_IS_THE_TOOLS}"

    if current is not None and header_lines(wanted) != header_lines(current):
        return f"{inside}: {HEADER_IS_THE_TOOLS}"

    return None


TABLE_IS_THE_TOOLS = "the steps table is the tool's"


def table_change(
    brain: store.Brain,
    touch: Touch,
    inside: str,
) -> Optional[str]:
    """Why this write may not touch the steps table of a plan, a todo or a bug, or None if it may.

    The rows the file holds are the tool's: a mark moves through the
    task or the steps command or follows the entry a step points at, a
    step is never reworded, repointed or deleted, the table is never
    taken away, and a table the write would leave unreadable is refused
    too. A row may follow the last one, held to what add holds a step
    to, pending, pointing at an entry the brain knows or at nothing,
    and on an open entry. A numbered list under Steps, new or written
    to, is the table in waiting, held to what filing holds it to, and a
    file whose table does not read now is doctor's, not the hook's. A
    hunk that fits nowhere and carries a row is refused, since the hook
    cannot tell where it would land.
    """
    write = table_write_of(brain=brain, touch=touch, inside=inside)
    if write is None:
        return None

    if isinstance(write, str):
        return write

    # a list's rows were held to what filing holds them to as they were turned, and only the loops a patch closes across files are left to judge
    if write.listed:
        return None

    why = changed_row(brain=brain, before=write.before, rows=write.rows, key=write.key, status=write.status, closed=write.closed) or added_row(
        brain=brain,
        before=write.before,
        rows=write.rows,
        key=write.key,
        status=write.status,
        closed=write.closed,
        own=write.own,
    )
    if why is None:
        return None

    return f"{inside}: {TABLE_IS_THE_TOOLS}, {why}, and the prose around it is yours to edit"


@dataclass(frozen=True)
class TableWrite:
    """A write to a stepped file as the hook reads it.

    The key is the name a command names the file by, own the keys a
    step would point at it with, its id and its name as the store reads
    them, or the header's name alone when the store holds no entry at
    the path, and node the file's own place in the pointer graph, its
    path folded in case, so two files that carry one id stay two
    nodes. The rows are the ones
    the file holds and the ones the write would leave, which a list
    under Steps gives as the rows the tool would turn it into.
    """

    inside: str
    key: str
    own: set[str]
    node: str
    status: str
    closed: bool
    before: list[steps.Step]
    rows: list[steps.Step]
    listed: bool

    @property
    def added(self) -> list[steps.Step]:
        return self.rows[len(self.before):]


def table_write_of(
    brain: store.Brain,
    touch: Touch,
    inside: str,
) -> Union[TableWrite, str, None]:
    """The steps table a write reaches, as it is and as the write would leave it, a refusal when the write cannot be judged as a table, or None when there is no table to judge, a file the hook cannot read being taken as its repair the way the header check takes it."""
    try:
        text = frontmatter.load(str(touch.path))
        current = frontmatter.parse(text)
    except frontmatter.FrontmatterError:
        return None

    kind = str(current.get("kind") or "")
    if kind not in steps.STEPPED_KINDS:
        return None

    status = str(current.get("status") or "")
    closed = status not in steps.OPEN_STATUSES[kind]
    entry = entry_at(brain=brain, inside=inside)
    key = entry.name if entry is not None else Path(inside).stem

    # a table that does not read now is doctor's, and a list on disk is the table in waiting, judged whole as the write leaves it
    try:
        before = steps.stored_rows_of(kind=kind, body=current.body)
    except steps.StepsError:
        if steps.has_table(current.body):
            return None

        before = []

    after, loose = outcome_of(touch=touch, text=text)
    for hunk in loose:
        line = row_line_in(hunk)
        if line is not None:
            return f"{inside}: a hunk the hook cannot place touches the steps table with '{line}', and {TABLE_IS_THE_TOOLS}"

        line = step_line_in(hunk)
        if line is not None:
            return f"{inside}: a hunk the hook cannot place carries '{line}', which reads as a step or the Steps heading and could land in the list the tool turns into the table, and {TABLE_IS_THE_TOOLS}"

        # on a closed entry any numbered line could land under Steps as a step the record never had, and a table row would be refused there
        line = numbered_line_in(hunk) if closed else None
        if line is not None:
            return f"{inside}: a hunk the hook cannot place carries '{line}', which reads as a step, and {key} is {status}, a closed table being its record, so no step is added to it"

    if after is None:
        return None

    # the header check ran first on the same outcome and refused a write whose header would not read, so this parse holds
    body = frontmatter.parse(after).body

    own = steps.own_keys(entry) if entry is not None else {str(current.get("name") or key)}
    write = TableWrite(
        inside=inside,
        key=key,
        own=own,
        node=inside.casefold(),
        status=status,
        closed=closed,
        before=before,
        rows=[],
        listed=False,
    )

    if not steps.has_table(body):
        if before:
            return f"{inside}: {TABLE_IS_THE_TOOLS}, its rows are never taken away, and the prose around it is yours to edit"

        # a plan's table stands even empty, and its Steps section with it, since a plan with no table is doctor's finding
        if kind == "plan" and steps.has_table(current.body):
            return f"{inside}: {TABLE_IS_THE_TOOLS}, and a plan's table is never taken away, empty or not, and the prose around it is yours to edit"

        return listed_change(brain=brain, write=write, kind=kind, was=current.body, body=body)

    try:
        rows = steps.stored_rows_of(kind=kind, body=body)
    except steps.StepsError as error:
        return f"{inside}: after that write the steps table would not read, {error}"

    return replace(write, rows=rows)


def listed_change(
    brain: store.Brain,
    write: TableWrite,
    kind: str,
    was: str,
    body: str,
) -> Union[TableWrite, str, None]:
    """A numbered list under Steps, new or written to, as the rows the tool would turn it into, held to what filing holds a list to since the tool would refuse it then, a refusal when it does not pass or the entry is closed, or None when the section is empty or the write leaves it as it was."""
    if steps.blank(body) or steps_text(was) == steps_text(body):
        return None

    if write.closed:
        return f"{write.inside}: {TABLE_IS_THE_TOOLS}, {write.key} is {write.status}, and a closed table is its record, so no step is added to it, and the prose around it is yours to edit"

    try:
        turned = steps.converted(brain=brain, body=body, by="agent", kind=schema.kind_named(kind), own=write.own)
    except steps.StepsError as refused:
        return f"{write.inside}: {TABLE_IS_THE_TOOLS}, {refused}, and the prose around it is yours to edit"

    return replace(write, rows=steps.table(turned), listed=True)


def steps_text(
    body: str,
) -> str:
    """What the Steps section holds, as written, so a write elsewhere in the file leaves a list on disk unjudged."""
    where = steps.span(body)

    return body[where[0] : where[1]].strip() if where is not None else ""


def table_writes_of(
    brain: store.Brain,
    touched: list[Touch],
) -> list[TableWrite]:
    """The steps tables a patch reaches, one per brain file it touches that reads as a table, the refusals having been judged touch by touch already."""
    writes: list[TableWrite] = []
    for touch in touched:
        inside = entry_path(brain=brain, path=touch.path)
        if inside is None:
            continue

        write = table_write_of(brain=brain, touch=touch, inside=inside)
        if isinstance(write, TableWrite):
            writes.append(write)

    return writes


def loop_across(
    brain: store.Brain,
    writes: list[TableWrite],
) -> Optional[str]:
    """Why a row a patch adds may not close a loop, or None when none does.

    Each file alone reads as fine against the disk, since the other
    rows of the patch are not written yet, so the pointers of the brain
    as it stands and the rows the patch adds are walked together, and a
    row whose entry leads back to its own file is refused, whether the
    way back is another row of the patch or one already on disk.
    """
    edges = pointer_graph(brain)
    for write in writes:
        for step in write.added:
            target = resolved(brain=brain, key=step.points_at)
            if target is not None:
                edges.setdefault(write.node, set()).add(target)

    for write in writes:
        for step in write.added:
            target = resolved(brain=brain, key=step.points_at)
            if target is not None and leads_to(edges=edges, start=target, goal=write.node):
                return f"{write.inside}: {TABLE_IS_THE_TOOLS}, step {step.number} points at {step.points_at}, whose steps lead back here once the patch lands, and the entries would wait on each other for good, and the prose around it is yours to edit"

    return None


def pointer_graph(
    brain: store.Brain,
) -> dict[str, set[str]]:
    """Every pointer the open steps tables hold, a list not yet turned into one read by the keys its lines open with, as edges from the holder's file to the file pointed at, a closed entry's steps waiting on nothing.

    A file is its own node, the path folded in case, so two files that
    carry one id after a merge stay two nodes rather than one.
    """
    edges: dict[str, set[str]] = {}
    for kind_name in steps.STEPPED_KINDS:
        for entry in brain.entries_of(schema.kind_named(kind_name)):
            if not steps.is_open(entry):
                continue

            try:
                keys = steps.pointers_in(entry.document.body)
            except steps.StepsError:
                continue

            for key in keys:
                target = resolved(brain=brain, key=key)
                if target is not None:
                    edges.setdefault(entry.relative.casefold(), set()).add(target)

    return edges


def resolved(
    brain: store.Brain,
    key: str,
) -> Optional[str]:
    """The node of the entry a row's pointer reaches, its path folded in case, or None when it reaches nothing."""
    if not key:
        return None

    target = steps.pointed(brain=brain, key=key)

    return target.relative.casefold() if target is not None else None


def leads_to(
    edges: dict[str, set[str]],
    start: str,
    goal: str,
) -> bool:
    """Whether the pointers lead from one entry to another, at any remove."""
    seen: set[str] = set()
    ahead = [start]
    while ahead:
        node = ahead.pop()
        if node == goal:
            return True

        if node in seen:
            continue

        seen.add(node)
        ahead.extend(edges.get(node, ()))

    return False


def changed_row(
    brain: store.Brain,
    before: list[steps.Step],
    rows: list[steps.Step],
    key: str,
    status: str,
    closed: bool,
) -> Optional[str]:
    """Why the first row that no longer reads as filed may not change that way, or None when every row the file had stands.

    A closed table is its record before anything else, a step whose
    pointer reaches an entry takes its mark from there, and one whose
    pointer reaches nothing is marked by hand like a plain one.
    """
    for step in before:
        fresh = rows[step.number - 1] if step.number <= len(rows) else None
        if fresh is None:
            return f"step {step.number} is never deleted, a step that turned out wrong is skipped"

        if fresh.status != step.status:
            if closed:
                return f"{key} is {status}, and a closed table is its record"

            if step.points_at and steps.pointed(brain=brain, key=step.points_at) is not None:
                return f"step {step.number} points at {step.points_at} and takes its mark from that entry"

            return f"step {step.number} is marked {fresh.status} with {marking_command(key=key, number=step.number, status=fresh.status)}"

        # a cell left empty by a write from before the tool refused one is filled in by hand, since no command rewords a step, and held to what a new row's text is held to
        if fresh.text != step.text:
            if step.text.strip():
                return f"step {step.number} is worded as it was filed"

            why = opens_with_a_key(brain=brain, step=fresh) if not fresh.points_at else None
            if why is not None:
                return why

        if fresh.points_at != step.points_at:
            return f"step {step.number} points where it was filed to point"

        if fresh.by != step.by:
            return f"step {step.number} is set by whom it was filed by"

    return None


def opens_with_a_key(
    brain: store.Brain,
    step: steps.Step,
) -> Optional[str]:
    """Why a row's text may not open with a key, since a row points through its Points at cell, or None when it does not."""
    try:
        word = steps.opening_key(brain=brain, number=step.number, text=step.text)
    except steps.StepsError as refused:
        return str(refused)

    if word:
        return f"step {step.number} opens with {word} in its Step cell, and a row points through its Points at cell"

    return None


def marking_command(
    key: str,
    number: int,
    status: str,
) -> str:
    """The commands that give a step the mark asked for: the task command when the work stands on the entry, the steps command in any case."""
    by_task = {"done": "brain task done", "skipped": "brain task skip", "pending": f"brain task step {number}"}[status]

    return f"{by_task} or brain steps {key} step {number} {status}"


def entry_at(
    brain: store.Brain,
    inside: str,
) -> Optional[store.Entry]:
    """The stepped entry the store holds at the path being written, matched in any case since a disk that ignores case takes the write however the path is spelled, so the keys a step would use to point at it and the key a command names it by are the store's, a plan's slug carrying no id however it starts, or None when the store holds no entry there."""
    wanted = inside.casefold()
    for kind_name in steps.STEPPED_KINDS:
        for entry in brain.entries_of(schema.kind_named(kind_name)):
            if entry.relative.casefold() == wanted:
                return entry

    return None


def added_row(
    brain: store.Brain,
    before: list[steps.Step],
    rows: list[steps.Step],
    key: str,
    status: str,
    closed: bool,
    own: set[str],
) -> Optional[str]:
    """Why the first row added after the last may not come in that way, held to what add holds a step to, or None when each may."""
    added = rows[len(before):]
    if added and closed:
        return f"{key} is {status}, and a closed table is its record, so no step is added to it"

    for step in added:
        if not step.text.strip():
            return f"step {step.number} has no text"

        target = steps.pointed(brain=brain, key=step.points_at) if step.points_at else None
        if step.points_at and target is None:
            return str(steps.unreachable(brain=brain, number=step.number, key=step.points_at))

        if target is not None and steps.doubled(brain=brain, key=steps.pointer_of(target)):
            return str(steps.doubled_pointer(number=step.number, key=step.points_at, pointer=steps.pointer_of(target)))

        if target is not None and steps.pointer_of(target) in own:
            return f"step {step.number} points at {key} itself, and a step points at other work"

        if step.status != "pending" and step.points_at:
            return f"step {step.number} is added pending, and takes its mark from {step.points_at}"

        if step.status != "pending":
            return f"step {step.number} is added pending, and marked once it is {step.status} with {marking_command(key=key, number=step.number, status=step.status)}"

        if not step.points_at:
            why = opens_with_a_key(brain=brain, step=step)
            if why is not None:
                return why

    try:
        steps.refuse_cycles(brain=brain, rows=added, own=own)
    except steps.StepsError as loop:
        return str(loop)

    return None


def row_line_in(
    hunk: tuple[str, ...],
) -> Optional[str]:
    """The first line of a hunk, kept, removed or added, that is a line of the steps table, a row, its header or the five celled separator under it, blank space around it aside, since a patch tool may place it with that trimmed, a table of another shape in the prose being left alone."""
    for line in hunk:
        text = line[1:].strip()
        if steps.ROW.match(text) or TABLE_LINE.match(text):
            return line[1:]

    return None


TABLE_LINE = re.compile(r"^\|[ \t]*#[ \t]*\|[ \t]*Step[ \t]*\|[ \t]*Points at[ \t]*\|[ \t]*By[ \t]*\|[ \t]*Status[ \t]*\|$|^\|(?:[ \t]*:?-+:?[ \t]*\|){5}$")


def step_line_in(
    hunk: tuple[str, ...],
) -> Optional[str]:
    """The first line of a hunk, kept, removed or added, that is a numbered step opening with a key or a Steps heading, blank space around it aside, since a list the hook cannot place could be the one under Steps, which the tool turns into the table."""
    for line in hunk:
        text = line[1:].strip()
        if steps.opens_with_a_key(text) or STEPS_HEADING.match(text):
            return line[1:]

    return None


def numbered_line_in(
    hunk: tuple[str, ...],
) -> Optional[str]:
    """The first added line of a hunk that is a numbered line, blank space around it aside."""
    for line in hunk:
        if line.startswith("+") and steps.LISTED.match(line[1:].strip()):
            return line[1:]

    return None


STEPS_HEADING = re.compile(r"^#{1,6}[ \t]+steps[ \t]*$", re.IGNORECASE)


def outcome_of(
    touch: Touch,
    text: str,
) -> tuple[Optional[str], tuple[tuple[str, ...], ...]]:
    """The file as the write would leave it, and the hunks of a patch that fit nowhere."""
    if touch.hunks:
        return patched(text=text, hunks=touch.hunks)

    return outcome(touch=touch, text=text), ()


def still_broken(
    inside: str,
    trouble: str,
    error: str,
) -> str:
    """The refusal for a write that leaves a broken header broken, the reason said once when it is the same."""
    if trouble == error:
        return f"{inside} does not read as a header now, {trouble}, and that write would leave it so"

    return f"{inside} does not read as a header now, {trouble}, and that write would not make it one, {error}"


def field_names(
    inside: str,
    lines: list[str],
) -> set[str]:
    """The fields this file's header carries or could carry, the updated date aside.

    What the file holds now plus what its kind's spec allows, so a loose
    hunk that adds an optional field the file does not yet carry is
    still read as header work, while a prose line with a colon whose
    name is neither is not. A file whose header does not read offers no
    lines, and its kind's fields still count.
    """
    names = {name for name in (frontmatter.split_field(line)[0] for line in lines) if name is not None}
    kind = schema.kind_for_folder(Path(inside).parts[0])
    if kind is not None:
        names |= {field.name for field in schema.fields_for(kind)}

    return names - {"updated"}


def header_line_in(
    hunk: tuple[str, ...],
    names: set[str],
) -> Optional[str]:
    """The first line of a hunk, kept, removed or added, that is a fence or a header field."""
    for line in hunk:
        text = line[1:]
        if text == frontmatter.FENCE or frontmatter.split_field(text)[0] in names:
            return text

    return None


def outcome(
    touch: Touch,
    text: str,
) -> Optional[str]:
    """The file's text as the write would leave it, or None when nothing says.

    Line endings are folded to one kind first, since the file is read
    that way and an agent on Windows hands over the other kind.
    """
    if touch.content is not None:
        return unix(touch.content)

    if touch.old_string is not None:
        old, new = unix(touch.old_string), unix(touch.new_string or "")
        if touch.replace_all:
            return text.replace(old, new)

        return text.replace(old, new, 1)

    return None


def unix(
    text: str,
) -> str:
    return text.replace("\r\n", "\n")


def patched(
    text: str,
    hunks: tuple[tuple[str, ...], ...],
) -> tuple[Optional[str], tuple[tuple[str, ...], ...]]:
    """The text after the hunks that fit are applied, one after another, and the ones that did not.

    Each hunk is placed against the text as the hunks before it left it,
    the way the patch is applied. A hunk that fits nowhere is handed
    back loose, so the caller can judge it by its shape, and the rest
    are still applied. If none fits, the patch will not apply and there
    is no text to judge.
    """
    lines = text.split("\n")
    loose: list[tuple[str, ...]] = []

    for hunk in hunks:
        before = before_lines(hunk)
        start = block_start(lines=lines, block=before)
        if start is None:
            loose.append(hunk)
            continue

        lines[start : start + len(before)] = [line[1:] for line in hunk if line[:1] in (" ", "+")]

    if len(loose) == len(hunks):
        return None, tuple(loose)

    return "\n".join(lines), tuple(loose)


def before_lines(
    hunk: tuple[str, ...],
) -> list[str]:
    return [line[1:] for line in hunk if line[:1] in (" ", "-")]


def block_start(
    lines: list[str],
    block: list[str],
) -> Optional[int]:
    """Where a run of lines first occurs, or None when it does not, or is empty."""
    if not block:
        return None

    for start in range(len(lines) - len(block) + 1):
        if lines[start : start + len(block)] == block:
            return start

    return None


def header_lines(
    document: frontmatter.Document,
) -> list[str]:
    return [line for line in document.lines if frontmatter.split_field(line)[0] != "updated"]


def post_write(
    brain: store.Brain,
    event: dict,
    agent: str,
) -> int:
    inside = [touch for touch in touches(event=event, agent=agent) if is_inside(brain=brain, path=touch.path)]
    findings = doctor.run(brain) if inside else []
    if not findings:
        emit_nothing(agent)
        return 0

    lines = [finding.line() for finding in findings]
    text = "brain doctor after that write:\n" + "\n".join(lines) + "\nRun brain doctor --fix for the fixable ones."
    emit_context(agent=agent, text=text, event_name=AFTER_WRITE.get(agent, "PostToolUse"))
    return 0


def is_inside(
    brain: store.Brain,
    path: Path,
) -> bool:
    return links.within(brain=brain, path=path) is not None


def touches(
    event: dict,
    agent: str,
) -> list[Touch]:
    """The files a write event touches, absolute, in the shape each agent sends.

    Claude Code and Gemini CLI name one file and hand over its new text
    or the edit they make in place. Codex hands over a patch that can
    add, update, delete or move several files, and only Codex is read
    that way, so a file whose text quotes a patch is not taken for one.
    A relative path is taken against the directory the event names, not
    the one this process happens to run in. The folder part is then
    spelled the way the disk spells it, so a project reached through a
    symlink still compares equal to the brain's real root.
    """
    base = Path(event.get("cwd") or Path.cwd())
    tool_input = event.get("tool_input") or {}
    found: list[Touch] = []

    if agent == "codex":
        for key in PATCH_KEYS:
            for text in texts_in(tool_input.get(key)):
                if PATCH_OPENS in text:
                    return patch_touches(base=base, patch=text)

        return found

    for key in ("file_path", "path", "filePath"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            found.append(
                Touch(
                    path=absolute(base=base, value=value.strip()),
                    content=text_field(tool_input=tool_input, key="content"),
                    old_string=text_field(tool_input=tool_input, key="old_string"),
                    new_string=text_field(tool_input=tool_input, key="new_string"),
                    replace_all=replaces_more_than_one(tool_input),
                )
            )

    return found


def replaces_more_than_one(
    tool_input: dict,
) -> bool:
    """Whether the edit means every occurrence, in the words each tool uses.

    Claude Code says replace_all, Gemini CLI says allow_multiple or
    counts them in expected_replacements.
    """
    if tool_input.get("replace_all") or tool_input.get("allow_multiple"):
        return True

    expected = tool_input.get("expected_replacements")

    return isinstance(expected, int) and expected > 1


def texts_in(
    value: object,
) -> list[str]:
    """The strings a field holds, whether it is one string or a list of them.

    A shell style call hands the patch over as one element of a list.
    """
    if isinstance(value, str):
        return [value]

    if isinstance(value, list):
        return [element for element in value if isinstance(element, str)]

    return []


def text_field(
    tool_input: dict,
    key: str,
) -> Optional[str]:
    value = tool_input.get(key)

    return value if isinstance(value, str) else None


def patch_touches(
    base: Path,
    patch: str,
) -> list[Touch]:
    """Every file a Codex patch names, an add carrying its text, an update its hunks.

    A hunk opens at `@@` and holds context, removed and added lines,
    each with its leading character. It closes at the next `@@`, the
    next file marker, or the end of file mark. A bare empty line inside
    a hunk is a blank context line, as apply_patch reads it, and blank
    lines trailing a hunk are the patch's spacing, not the file's. An
    added file reads a bare empty line as a blank line of its own.
    """
    found: list[Touch] = []
    hunk: list[str] = []
    updating: Optional[int] = None
    adding: Optional[int] = None

    def close_hunk() -> None:
        while hunk and hunk[-1] == " ":
            hunk.pop()
        if updating is not None and hunk:
            found[updating] = replace(found[updating], hunks=found[updating].hunks + (tuple(hunk),))
        hunk.clear()

    for raw in patch.splitlines():
        line = raw.rstrip("\r")
        marker = MARKER.match(line)
        if marker is not None:
            close_hunk()
            action, target = marker.groups()
            path = absolute(base=base, value=target)
            if action == "Move to" and updating is not None:
                found.append(Touch(path=path, moved_from=found[updating].path))
                continue

            found.append(
                Touch(
                    path=path,
                    content="" if action == "Add File" else None,
                    deleted=action == "Delete File",
                )
            )
            updating = len(found) - 1 if action == "Update File" else None
            adding = len(found) - 1 if action == "Add File" else None
            continue

        if adding is not None and (line[:1] == "+" or line == ""):
            found[adding] = replace(found[adding], content=(found[adding].content or "") + line[1:] + "\n")
            continue

        if line.startswith("@@") or line.startswith("*** End"):
            close_hunk()
            continue

        if updating is not None and line[:1] in (" ", "-", "+"):
            hunk.append(line)
        elif updating is not None and line == "":
            hunk.append(" ")

    close_hunk()

    return found


def absolute(
    base: Path,
    value: str,
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path

    return settled(path)


def settled(
    path: Path,
) -> Path:
    """The path with its folder resolved on disk and the file itself left alone.

    Only the folder is followed, so a symlinked file still counts where
    it sits. A folder that cannot be resolved, a loop on the 3.9 floor
    or a path with a stray null byte in it, keeps its lexical spelling
    and is judged as written.
    """
    try:
        return path.parent.resolve() / path.name
    except (OSError, RuntimeError, ValueError):
        return path


# What each agent calls the event after a write, which the reply names.
# Claude Code and Codex say PostToolUse, Gemini CLI says AfterTool.
AFTER_WRITE = {"claude": "PostToolUse", "codex": "PostToolUse", "gemini": "AfterTool"}

# What each agent calls the event on a prompt, and the one the tick rides
# on. Claude Code fires PostToolBatch once after each round of tool
# calls. Codex fires PostToolUse and Gemini CLI AfterTool after each
# tool, and the shipped configs match the shell tool alone, so there
# the tick comes after each shell call.
PROMPT = {"claude": "UserPromptSubmit", "codex": "UserPromptSubmit", "gemini": "BeforeAgent"}
TICK = {"claude": "PostToolBatch", "codex": "PostToolUse", "gemini": "AfterTool"}

# The events on which Codex takes plain text on standard output as
# developer context. On its tool events plain text is dropped and only
# the JSON object reaches the model.
CODEX_PLAIN = ("SessionStart", "UserPromptSubmit")


def emit_context(
    agent: str,
    text: str,
    event_name: str,
) -> None:
    """Hand text back the way this agent takes context from a hook.

    Codex takes plain text on standard output at the start of a session
    and on a prompt, and the JSON object after a tool. Claude Code and
    Gemini CLI take one JSON object naming the event.
    """
    if agent == "codex" and event_name in CODEX_PLAIN:
        write_utf8(text + "\n")
        return

    payload = {"hookSpecificOutput": {"hookEventName": event_name, "additionalContext": text}}
    print(json.dumps(payload))


def write_utf8(
    text: str,
) -> None:
    """Put text on standard output as UTF-8 bytes, under the text layer, so the locale's encoding never gets a say.

    The rulebook carries characters a Windows code page cannot encode,
    and the agent reads the pipe as UTF-8 whatever the locale. A byte
    layer that is not there, a text stream handed in by a test, takes
    the text as it is.
    """
    buffer: Optional[BinaryIO] = getattr(sys.stdout, "buffer", None)
    if buffer is None:
        sys.stdout.write(text)
        return

    sys.stdout.flush()
    buffer.write(text.encode("utf-8"))
    buffer.flush()


def emit_nothing(
    agent: str,
) -> None:
    """Say there is nothing to add, in the form the agent wants.

    On a pass, exit 0, Gemini CLI parses standard output as JSON, so it
    gets an empty object. On a refusal, exit 2, every agent takes the
    reason from standard error and reads nothing else. Claude Code takes
    silence on a pass, and Codex drops plain text on a tool event, so
    both get nothing.
    """
    if agent == "gemini":
        print("{}")


if __name__ == "__main__":
    sys.exit(main())
