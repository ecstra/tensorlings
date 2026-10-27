"""The current work, a chain from the plan down to the step the agent is on.

One chain at a time, kept in a local file the tool writes and every
hook prints, so the model always knows the task, the plan it serves,
the step inside it and the review running on it. A plan, a todo, a bug
or a review is the root. A todo, a bug or a review pushes under the
step that points at it, a review pushes under the todo or bug the work
is on too, and a link whose entry closes drops off with everything
under it. The step a link stands on is the first pending row of the
entry's steps table, read each time with a pointed step's mark taken
from its entry, so nothing is stored beside the table and `brain task
done` moves the work by marking. The rule in SYSTEM.md 8.3 is that
multi step work starts with `brain task on`.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from . import clock, frontmatter, schema, steps, store

FILE = ".task"
NONE = "task none"
BROKEN = f"{FILE} does not read as a chain and brain task off --all clears it"
FOLDER = f"{FILE} is a folder the tool never made, move it away by hand"

# what a link's entry is this time, and only a closed one drops the link
OPEN, CLOSED, AWAY, DOUBLED, UNREADABLE, UNKNOWN = "open", "closed", "away", "doubled", "unreadable", "unknown"

# the word each kind is printed under
WORDS: dict[str, str] = {
    "plan": "plan",
    "todo": "task",
    "bug": "bug",
    "review": "review",
}


class TaskError(Exception):
    """The chain could not be set as asked, with the reason in the message."""


@dataclass(frozen=True)
class Item:
    """One link of the chain, the entry being worked on."""

    kind: str
    key: str


def path_of(
    brain: store.Brain,
) -> Path:
    return brain.root / FILE


def read(
    brain: store.Brain,
) -> Optional[list[Item]]:
    """The chain as written, an empty list when there is no file, or None when the file does not read as a chain, a folder in its place included."""
    path = path_of(brain)
    if not store.readable_file(path):
        return [] if not store.quietly(path.exists) and not store.quietly(path.is_symlink) else None

    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
        items = [Item(kind=str(link["kind"]), key=str(link["key"])) for link in data["chain"]]
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError):
        return None

    if not all(item.kind in steps.OPEN_STATUSES and item.key for item in items):
        return None

    return items


def write(
    brain: store.Brain,
    chain: list[Item],
) -> None:
    """Record the chain, whole or not at all, or take the file away when the chain is empty, and refuse a folder standing in the file's place."""
    path = path_of(brain)
    if folder_in_the_way(brain):
        raise TaskError(FOLDER)

    if not chain:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return

    text = json.dumps({"chain": [asdict(item) for item in chain], "since": clock.now().isoformat()}) + "\n"
    frontmatter.write_text(path=str(path), text=text)


def folder_in_the_way(
    brain: store.Brain,
) -> bool:
    return store.quietly(path_of(brain).is_dir)


def written(
    brain: store.Brain,
) -> list[Item]:
    """The chain as written, refused by name when a folder stands in the file's place or the file does not read as a chain."""
    if folder_in_the_way(brain):
        raise TaskError(FOLDER)

    chain = read(brain)
    if chain is None:
        raise TaskError(BROKEN)

    return chain


def current(
    brain: store.Brain,
    chain: Optional[list[Item]] = None,
) -> list[Item]:
    """The chain as it stands, with a link whose entry has closed dropped along with everything under it.

    Only an entry found and closed drops its link. One the brain cannot
    read this time, one two files carry the id of, and one that is not
    in this checkout all stay, since none of those is a closed todo, and
    the line says which it is. A reading never writes the file, so the
    hook that reads after every tool call cannot write over a chain set
    meanwhile, and the file keeps a closed link until the next command
    that moves the chain, which is why off takes the innermost link as
    written and never one more.
    """
    if chain is None:
        chain = read(brain) or []

    return pruned(brain=brain, chain=chain)


def pruned(
    brain: store.Brain,
    chain: list[Item],
) -> list[Item]:
    """The chain cut at the first link whose entry is found closed."""
    kept: list[Item] = []
    for item in chain:
        if state_of(brain=brain, item=item) == CLOSED:
            break

        kept.append(item)

    return kept


def state_of(
    brain: store.Brain,
    item: Item,
) -> str:
    """What the link's entry is this time: open, closed, unreadable in a file or folder the brain could not read, doubled by a merge, away in another checkout, or in a status the tool does not know."""
    return looked_up(brain=brain, item=item)[0]


def looked_up(
    brain: store.Brain,
    item: Item,
) -> tuple[str, Optional[store.Entry]]:
    """The link's state and its entry, which is there for an open or a closed one and for one in an unknown status."""
    try:
        entry = find_link(brain=brain, item=item)
    except store.Ambiguous:
        return DOUBLED, None

    if entry is None:
        return (UNREADABLE if unreadable(brain=brain, item=item) else AWAY), None

    if steps.is_open(entry):
        return OPEN, entry

    if (entry.field("status") or "") in known_statuses(entry.kind):
        return CLOSED, entry

    return UNKNOWN, entry


def find_link(
    brain: store.Brain,
    item: Item,
) -> Optional[store.Entry]:
    """The entry a link means, looked up among its own kind, by the file first and by the header's name second.

    A file of another kind cannot stand in the way, the file itself
    wins over a copy that kept its name, and one renamed by hand still
    answers by its header until the doctor's repair moves the link.
    Two files of the kind that answer alike are refused.
    """
    entries = brain.entries_of(schema.kind_named(item.kind))
    for keyed in (file_key, header_key):
        matches = [entry for entry in entries if keyed(entry) == item.key]
        if len(matches) > 1:
            paths = ", ".join(entry.relative for entry in matches)
            raise store.Ambiguous(f"'{item.key}' names {len(matches)} files: {paths}")

        if matches:
            return matches[0]

    return None


def file_key(
    entry: store.Entry,
) -> str:
    """The key the file itself gives: the id a todo or a bug carries in its name, a review's folder, a plan's stem."""
    if entry.identifier:
        return entry.identifier

    return entry.path.parent.name if entry.kind.name == "review" else entry.path.stem


def header_key(
    entry: store.Entry,
) -> Optional[str]:
    """The name a plan's or a review's header says, which parts from the file's key only through a rename by hand."""
    return None if entry.identifier else entry.name


def known_statuses(
    kind: schema.Kind,
) -> tuple[str, ...]:
    return next((field.values for field in kind.fields if field.name == "status"), ())


def entry_of(
    brain: store.Brain,
    item: Item,
) -> Optional[store.Entry]:
    return looked_up(brain=brain, item=item)[1]


def why_not_open(
    item: Item,
    state: str,
) -> str:
    """The reason a link cannot be worked on right now, in the words the chain's line and its refusals share."""
    if state == DOUBLED:
        return "two files carry the id" if item.kind in ("todo", "bug") else "two files answer to the name"

    if state == AWAY:
        return "not in this checkout"

    if state == UNKNOWN:
        return "the status is not one the tool knows, which brain doctor reports"

    return "cannot be read right now"


def unreadable(
    brain: store.Brain,
    item: Item,
) -> bool:
    """Whether the link's entry may be there unread, in a file of its own folder or in that folder itself, which the brain could not read this time.

    A plan is its name, so only the file of that name counts, while an
    id heads a file name and is followed by the title.
    """
    folder = brain.folder(schema.kind_named(item.kind))
    for path, trouble in brain.unreadable.items():
        if trouble == store.UNLISTED and (path == folder or folder in path.parents):
            return True

        if item.kind == "review":
            if path.parent.parent == folder and path.parent.name == item.key:
                return True
            continue

        if folder not in path.parents:
            continue

        if path.stem == item.key or (item.kind != "plan" and path.stem.startswith(f"{item.key}-")):
            return True

    return False


def resolve(
    brain: store.Brain,
    key: str,
) -> store.Entry:
    """The one open plan, todo, bug or review the key answers to, refused otherwise."""
    try:
        entry = brain.find(key)
    except store.Ambiguous as clash:
        raise TaskError(str(clash)) from clash

    if entry is None:
        raise TaskError(f"nothing in the brain answers to '{key}'")

    if entry.kind.name not in steps.OPEN_STATUSES:
        raise TaskError(f"the work is a plan, a todo, a bug or a review, and {entry.name} is a {entry.kind.name}")

    if not steps.is_open(entry):
        raise TaskError(f"{entry.name} is {entry.field('status')}, and the work is on something open")

    return entry


def key_of(
    entry: store.Entry,
) -> str:
    """The key the chain stores, the one the file itself gives, which a copy that kept the header's name cannot double."""
    return file_key(entry)


def turn_on(
    brain: store.Brain,
    key: str,
    replace: bool = False,
) -> list[Item]:
    """Start work on the entry: as the root of a fresh chain, pushed under the innermost link where the rules allow, or in place of the chain when told to replace it. The entry already innermost is left where it is."""
    entry = resolve(brain=brain, key=key)
    item = Item(kind=entry.kind.name, key=key_of(entry))

    chain: list[Item] = []
    if not replace:
        chain = current(brain=brain, chain=written(brain))

    if chain and chain[-1] == item:
        return chain

    if item in chain:
        raise TaskError(f"{item.key} is on the chain already, above {chain[-1].key}, and brain task off drops back to it")

    if chain and not may_push(brain=brain, chain=chain, entry=entry):
        raise TaskError(f"the work is on {describe_item(brain=brain, item=chain[-1])}, finish it or drop it with brain task off first")

    chain.append(item)
    write(brain=brain, chain=chain)

    return chain


def may_push(
    brain: store.Brain,
    chain: list[Item],
    entry: store.Entry,
) -> bool:
    """Whether the entry may go under the innermost link: a review under a todo or a bug, or the entry the innermost link's current step points at under that link."""
    innermost = chain[-1]
    if innermost.kind in ("todo", "bug") and entry.kind.name == "review":
        return True

    if innermost.kind not in steps.STEPPED_KINDS:
        return False

    holder = entry_of(brain=brain, item=innermost)
    if holder is None:
        return False

    try:
        rows = steps.rows_of(brain=brain, entry=holder)
    except steps.StepsError:
        return False

    step = steps.current_step(rows)
    if step is None:
        return False

    pointer = rows[step - 1].points_at

    return bool(pointer) and pointer in steps.own_keys(entry)


def mark(
    brain: store.Brain,
    status: str,
) -> list[Item]:
    """Mark the step the innermost link stands on done or skipped, so the work moves to the next pending one."""
    chain, entry = innermost(brain)
    rows = steps.rows_of(brain=brain, entry=entry)
    step = steps.current_step(rows)
    if step is None:
        raise TaskError(f"{entry.name} has no step pending" if rows else f"{entry.name} has no steps, add some with brain steps {key_of(entry)} add")

    steps.mark(brain=brain, key=entry.relative, number=step, status=status)

    return chain


def reopen(
    brain: store.Brain,
    number: int,
) -> list[Item]:
    """Mark a step of the innermost link pending again, so the work stands on it, or on an earlier step still pending."""
    chain, entry = innermost(brain)
    rows = steps.rows_of(brain=brain, entry=entry)
    if not rows:
        raise TaskError(f"{entry.name} has no steps, add some with brain steps {key_of(entry)} add")

    step = steps.step_numbered(entry=entry, rows=rows, number=number)
    if step.status == "pending":
        current_step = steps.current_step(rows)
        where = "it" if current_step == number else f"step {current_step}"
        raise TaskError(f"step {number} is pending already, and the work stands on {where}")

    steps.mark(brain=brain, key=entry.relative, number=number, status="pending")

    return chain


def innermost(
    brain: store.Brain,
) -> tuple[list[Item], store.Entry]:
    """The chain as written and the entry of its innermost link, which must be open and carry steps of its own.

    The link is taken as written, like off takes it, so a link whose
    entry closed meanwhile is refused rather than passed over, which
    would land the mark on the step above it that nobody did.
    """
    chain = written(brain)
    if not chain:
        raise TaskError("no work is on, start it with brain task on")

    item = chain[-1]
    if item.kind == "review":
        raise TaskError("a review's step is its batch, read off the ledger, so it takes no mark here")

    state, entry = looked_up(brain=brain, item=item)
    if state == CLOSED:
        raise TaskError(f"{item.key} has closed and holds the place until brain task off drops it")

    if entry is None or state != OPEN:
        raise TaskError(f"{item.key}, {why_not_open(item=item, state=state)}")

    return chain, entry


def turn_off(
    brain: store.Brain,
    everything: bool = False,
) -> list[Item]:
    """Drop the innermost link as written, or the whole chain, and hand back what remains, a link whose entry closed meanwhile going with it."""
    if everything:
        write(brain=brain, chain=[])
        return []

    chain = pruned(brain=brain, chain=written(brain)[:-1])
    write(brain=brain, chain=chain)

    return chain


def renamed(
    brain: store.Brain,
    kind: str,
    old: str,
    new: str,
) -> None:
    """Follow a retitle in the chain, where a plan is stored by its name, and leave a file that does not read alone."""
    stored = read(brain)
    if not stored:
        return

    chain = [Item(kind=item.kind, key=new) if item.kind == kind and item.key == old else item for item in stored]
    if chain != stored:
        write(brain=brain, chain=chain)


def describe(
    brain: store.Brain,
) -> str:
    """The chain as lines, one per link, outermost first, or the one line that says no work is on."""
    if folder_in_the_way(brain):
        return f"{NONE}, {FOLDER}"

    stored = read(brain)
    if stored is None:
        return f"{NONE}, {BROKEN}"

    chain = current(brain=brain, chain=stored)
    if not chain:
        return NONE

    return "\n".join(f"{WORDS[item.kind]} {describe_item(brain=brain, item=item)}" for item in chain)


def describe_item(
    brain: store.Brain,
    item: Item,
) -> str:
    """One link in words: the key, then where it stands, the step and its text, the batch and its status, or what is missing."""
    state, entry = looked_up(brain=brain, item=item)
    if entry is None or state == UNKNOWN:
        return f"{item.key}, {why_not_open(item=item, state=state)}"

    if item.kind == "review":
        return f"{item.key}, {batch_text(entry)}"

    where = step_text(brain=brain, entry=entry)

    return f"{item.key}, {where}" if where else item.key


def step_text(
    brain: store.Brain,
    entry: store.Entry,
) -> str:
    """Where the work stands in the entry's steps, nothing for a todo or a bug with none."""
    try:
        rows = steps.rows_of(brain=brain, entry=entry)
    except steps.StepsError:
        return "the steps table cannot be read"

    if not rows:
        return "no steps yet" if entry.kind.name == "plan" else ""

    step = steps.current_step(rows)
    if step is None:
        return f"every step settled, close it with {closing_command(brain=brain, entry=entry)}"

    return f"step {step} of {len(rows)}, {rows[step - 1].text}"


def closing_command(
    brain: store.Brain,
    entry: store.Entry,
) -> str:
    """The set command that closes the entry, with the closing date a plan or a todo wants, and the section a bug or a todo wants filled named.

    The entry goes by its key, or by its path when a file of another
    kind answers to the key too, which a merge can bring in, so the
    command goes through as written.
    """
    key = key_of(entry)
    try:
        brain.find(key)
    except store.Ambiguous:
        key = entry.relative

    if entry.kind.name == "bug":
        return f"brain set {key} status=fixed once Fix names the cause and the commit"

    command = f"brain set {key} status={steps.CLOSING[entry.kind.name]} done={clock.now().date().isoformat()}"

    return f"{command} once Done when names what closed it" if entry.kind.name == "todo" else command


def batch_text(
    entry: store.Entry,
) -> str:
    """Where a review stands, the first batch not yet done with its status, read off the ledger."""
    # here and not above, since the ledger imports the writer and the writer imports this module, and an import at the top would run the writer's body while this module is half built
    from . import ledger

    try:
        rows = ledger.batches(entry)
    except ledger.LedgerError:
        return "the batch table cannot be read"

    for row in rows:
        if row["status"] != "done":
            return f"batch {row['number']} of {len(rows)}, {row['status']}"

    return f"every batch done, {len(rows)} in all" if rows else "no batches yet"


def push_review(
    brain: store.Brain,
    name: str,
) -> Optional[str]:
    """Put a review that just opened on the chain, and say why when the rules keep it off, since the ledger opened either way."""
    try:
        turn_on(brain=brain, key=name)
    except (TaskError, steps.StepsError, OSError) as reason:
        return store.explain(reason)

    return None
