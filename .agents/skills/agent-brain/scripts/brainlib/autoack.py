"""Auto ack, the owner's standing word to proceed, with an end it never runs past.

The owner says "auto ack" and the agent records it here through the
tool, with an end by time, by task or both, the earliest winning. There
is no default end. A word given without one is not recorded, and the
agent proposes an end to the owner and waits for the answer. While the
word stands the ledger takes an ack of "auto" and the review skill
passes its gates without waiting. The state is the tool's own, one
dotted file at the root of the brain, which git ignores since it
belongs to this machine and this stretch of work, and the brain's
conventions say the rest.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from . import clock, frontmatter, schema, store

FILE = ".ack"
OFF = "auto ack off"
BROKEN = f"{FILE} does not read as a word and brain ack off clears it"
FOLDER = f"{FILE} is a folder the tool never made, move it away by hand"

# the last year a word may end in on its own wall clock, since Windows
# stops placing a moment on its clock at the year 3000, and the instant
# that year ends on the clock furthest behind UTC, the envelope every
# recorded end fits in
LAST_YEAR = 2999
LATEST = datetime(3000, 1, 1, 11, 59, 59, 999999, tzinfo=timezone.utc)

# the kinds a word can end with, and the statuses that keep each one open
OPEN_STATUSES = {
    "todo": ("open",),
    "bug": ("open", "branched"),
    "review": ("open",),
}

DURATION = re.compile(r"^(\d+(?:\.\d+)?)\s*([mhd])$")
CLOCK_TIME = re.compile(r"^(?:(today|tomorrow)\s+)?(\d{1,2}):(\d{2})$")
DAY_TIME = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:\s+(\d{1,2}):(\d{2}))?$")


class AckError(Exception):
    """The auto ack could not be set or read, with the reason in the message."""


@dataclass(frozen=True)
class Grant:
    """One standing word, when it was given, the moment it ends, and the task whose closing ends it, at least one of the two."""

    since: datetime
    until: Optional[datetime]
    task: Optional[str] = None


def path_of(
    brain: store.Brain,
) -> Path:
    return brain.root / FILE


def turn_on(
    brain: store.Brain,
    hours: Optional[float] = None,
    until_text: Optional[str] = None,
    task: Optional[str] = None,
    at: Optional[datetime] = None,
) -> Grant:
    """Record the word with its end, the earliest of the ends given, refused when none is given or a folder sits where the word goes."""
    if a_folder(path=path_of(brain=brain)):
        raise AckError(FOLDER)

    at = at or clock.now()
    ends: list[datetime] = []
    try:
        if hours is not None:
            if hours <= 0:
                raise AckError("auto ack lasts for a time above zero")

            ends.append(utc(at) + timedelta(hours=hours))

        if until_text is not None:
            ends.append(utc(parse_until(text=until_text, at=at)))
    except OverflowError as far:
        raise AckError(f"auto ack would end past the year {LAST_YEAR}, give a nearer end") from far

    if task is not None:
        task = task_key(brain=brain, key=task)

    if not ends and task is None:
        raise AckError("auto ack needs an end, a time with --for or --until, a task with --until-done, or both")

    until = min(ends) if ends else None
    if until is not None and until <= utc(at):
        raise AckError("auto ack would end before it starts, give a time still to come")

    # the envelope first, since an end inside it is one every machine can
    # place on its clock, and then the year as the word's own clock shows it
    if until is not None and (until > LATEST or on_the_wall(at=at, moment=until).year > LAST_YEAR):
        raise AckError(f"auto ack would end past the year {LAST_YEAR}, give a nearer end")

    grant = Grant(
        since=at,
        until=until,
        task=task,
    )
    write(brain=brain, grant=grant)

    return grant


def turn_off(
    brain: store.Brain,
) -> bool:
    """Take the word back, whatever sits where the word goes short of a folder. True when there was something to take."""
    path = path_of(brain=brain)
    if a_folder(path=path):
        raise AckError(FOLDER)

    if not store.quietly(path.exists) and not store.quietly(path.is_symlink):
        return False

    try:
        path.unlink()
    except FileNotFoundError:
        return False

    return True


def a_folder(
    path: Path,
) -> bool:
    """Whether a folder of its own sits where the word goes, which the tool never made and never removes, where a link to one is cleared like any link."""
    return store.quietly(path.is_dir) and not store.quietly(path.is_symlink)


def trouble_text(
    brain: store.Brain,
) -> str:
    """What to say about something where the word goes that does not read as one, and how the owner clears it."""
    return FOLDER if a_folder(path=path_of(brain=brain)) else BROKEN


def current(
    brain: store.Brain,
    at: Optional[datetime] = None,
) -> Optional[Grant]:
    """The word as it stands, or None when none was given, its time ran out, its task is done or gone, or its task cannot be read or told apart right now.

    A word its task has ended, or whose task is gone from the brain or
    doubled by a merge, is taken back on the spot, so it never comes
    back when the task reads open again after a reopen, or when a new
    entry takes the same id on another branch. Where the brain's folder
    cannot be written the file stays, `describe` says so, and the word
    would stand again on a reopen until `brain ack off` clears it. A
    task the brain cannot read right now, one a rebase stopped on,
    holds the word off with the file in place, so the word is back
    once the entry is.
    """
    grant = read(brain=brain)
    if grant is None:
        return None

    at = at or clock.now()
    if grant.until is not None and utc(at) >= grant.until:
        return None

    if grant.task is not None:
        state = task_state(brain=brain, key=grant.task)
        if state in ("done", "gone"):
            forget(brain=brain, grant=grant)

        if state != "open":
            return None

    return grant


def forget(
    brain: store.Brain,
    grant: Grant,
) -> None:
    """Drop the file of a word its task has ended, as long as it still holds that word.

    This runs on every read of the word, from hooks in parallel among
    them, so a file another read dropped first, one a newer word has
    since replaced, or a root that cannot be written right now, is no
    reason to fail the read. The word is off either way, and the next
    read tries again. Until one succeeds the file stays, and would be
    the word again if the task were reopened, so `describe` says it
    could not be cleared.
    """
    if read(brain=brain) != grant:
        return

    try:
        path_of(brain=brain).unlink()
    except OSError:
        pass


def stamp(
    brain: store.Brain,
    at: Optional[datetime] = None,
) -> str:
    """The ack a ledger records while the word stands, refused when it does not, or is held."""
    at = at or clock.now()
    if current(brain=brain, at=at) is None:
        held = held_word(brain=brain, at=at)
        if held is not None:
            raise AckError(f"auto ack is held, {held_text(brain=brain, grant=held)}, the owner's word starts a batch")

        if unreadable_word(brain=brain):
            raise AckError(f"auto ack is off, {trouble_text(brain=brain)}, the owner's word starts a batch")

        raise AckError("auto ack is off, the owner's word starts a batch")

    return f"auto {at:%Y-%m-%d}"


def unreadable_word(
    brain: store.Brain,
) -> bool:
    """Whether something sits where the word goes that does not read as one, a broken file, a folder or a dangling link, which the owner is told about."""
    path = path_of(brain=brain)
    if not store.quietly(path.is_symlink) and not store.quietly(path.exists):
        return False

    return not store.readable_file(path) or read(brain=brain) is None


def held_word(
    brain: store.Brain,
    at: datetime,
) -> Optional[Grant]:
    """A word still within its time whose task the brain cannot read right now, or whose ledger this checkout lacks, which holds it off without ending it."""
    grant = read(brain=brain)
    if grant is None or grant.task is None:
        return None

    if grant.until is not None and utc(at) >= grant.until:
        return None

    return grant if task_state(brain=brain, key=grant.task) in ("unsure", "away") else None


def held_text(
    brain: store.Brain,
    grant: Grant,
) -> str:
    """Why the word is held, in the words the status line and the ledger's refusal share."""
    if task_state(brain=brain, key=grant.task) == "away":
        return f"{grant.task} is not in this checkout, brain ack off clears it"

    return f"{grant.task} cannot be read right now"


def stale_word(
    brain: store.Brain,
    at: datetime,
) -> Optional[Grant]:
    """A word within its time whose task has ended and whose file is still on disk, which means `forget` could not drop it."""
    grant = read(brain=brain)
    if grant is None or grant.task is None:
        return None

    if grant.until is not None and utc(at) >= grant.until:
        return None

    return grant if task_state(brain=brain, key=grant.task) in ("done", "gone") else None


def describe(
    brain: store.Brain,
    at: Optional[datetime] = None,
) -> str:
    """One line on the word, off, held, or on with what is left of it, for the hooks and the status line."""
    at = at or clock.now()
    grant = current(brain=brain, at=at)
    if grant is None:
        held = held_word(brain=brain, at=at)
        if held is not None:
            return f"auto ack held, {held_text(brain=brain, grant=held)}"

        stale = stale_word(brain=brain, at=at)
        if stale is not None:
            return f"{OFF}, {FILE} could not be cleared since the brain's folder cannot be written, and the word would stand again if {stale.task} were reopened"

        if unreadable_word(brain=brain):
            return f"{OFF}, {trouble_text(brain=brain)}"

        return OFF

    if grant.until is None:
        return f"auto ack on until {grant.task} is done"

    text = f"auto ack on, {left(since=at, until=grant.until)} left, until {when(at=at, moment=grant.until)}"
    if grant.task is not None:
        text += f", or until {grant.task} is done"

    return text


def left(
    since: datetime,
    until: datetime,
) -> str:
    """What is left of the word in whole minutes, rounded up, so a word just given shows what was asked and the last minute never reads as none."""
    seconds = (utc(until) - utc(since)).total_seconds()
    minutes = int(-(-seconds // 60))
    hours, minutes = divmod(minutes, 60)
    if hours and minutes:
        return f"{hours} h {minutes} min"

    if hours:
        return f"{hours} h"

    return f"{minutes} min"


def when(
    at: datetime,
    moment: datetime,
) -> str:
    """A moment named the way a person says it, the time today, tomorrow, or the date and time, on the wall clock the word runs on."""
    moment = on_the_wall(at=at, moment=moment)
    days = (moment.date() - at.date()).days
    if days == 0:
        return f"{moment:%H:%M} today"

    if days == 1:
        return f"{moment:%H:%M} tomorrow"

    return f"{moment:%Y-%m-%d %H:%M}"


def utc(
    moment: datetime,
) -> datetime:
    """The same instant in UTC, so that arithmetic and comparison never run on a wall clock."""
    return moment.astimezone(timezone.utc)


def machine_fold(
    at: datetime,
) -> Optional[int]:
    """Which side of a repeated hour this machine's clock reads the moment on, 0 or 1, or None when the moment is not this machine's.

    The clock gives the moment with a fixed offset and no zone name, so
    the test is whether its wall clock, read as this machine's local
    time, is the same instant, on either side of a repeated hour, since
    the clock drops which side it was on. A moment that carries a
    named zone brings its own rules and is never this machine's.
    """
    if at.tzinfo is None:
        return 0

    if type(at.tzinfo) is not timezone:
        return None

    naive = at.replace(tzinfo=None)
    for fold in (0, 1):
        if naive.replace(fold=fold).astimezone() == at:
            return fold

    return None


def on_this_machine(
    at: datetime,
) -> bool:
    """Whether the moment is expressed in this machine's own zone, whose daylight saving rules then apply."""
    return machine_fold(at) is not None


def on_the_wall(
    at: datetime,
    moment: datetime,
) -> datetime:
    """The moment on the wall clock the word runs on, this machine's when the word was given on it."""
    if on_this_machine(at):
        return moment.astimezone()

    return moment.astimezone(at.tzinfo)


def from_the_wall(
    at: datetime,
    naive: datetime,
) -> datetime:
    """A wall clock reading placed in the zone the word runs on, this machine's rules when the word was given on it.

    A reading inside a repeated hour lands on the same side of it as
    the moment the word is given at, so an end asked for in the second
    pass is not placed an hour early in the first. A reading this
    machine cannot place at all, before 1970 or past the year 3000 on
    Windows, is refused as the past or the far end it is.
    """
    fold = machine_fold(at)
    if fold is None:
        return naive.replace(tzinfo=at.tzinfo, fold=at.fold)

    try:
        return naive.replace(fold=fold).astimezone()
    except OSError as unplaceable:
        if naive < at.replace(tzinfo=None):
            raise AckError("auto ack would end before it starts, give a time still to come") from unplaceable

        raise AckError(f"auto ack would end past the year {LAST_YEAR}, give a nearer end") from unplaceable


def parse_duration(
    text: str,
) -> float:
    """Hours from a duration such as 8h, 90m or 2d."""
    match = DURATION.match(text.strip().lower())
    if match is None:
        raise AckError(f"'{text}' is not a duration like 8h, 90m or 2d")

    amount_text, unit = match.groups()
    amount = float(amount_text)
    if unit == "m":
        return amount / 60

    if unit == "d":
        return amount * 24

    return amount


def parse_until(
    text: str,
    at: datetime,
) -> datetime:
    """A moment from 22:00, tomorrow 22:00, 2026-09-24 22:00 or 2026-09-24, on the wall clock the word runs on.

    A bare time already past today means the same time tomorrow, since
    a word given in the evening is meant for the night ahead. A date
    alone means that whole day, through its last moment.
    """
    text = text.strip().lower()
    wall = at.replace(tzinfo=None)
    try:
        match = CLOCK_TIME.match(text)
        if match is not None:
            day_word, hour, minute = match.groups()
            moment = wall.replace(
                hour=int(hour),
                minute=int(minute),
                second=0,
                microsecond=0,
            )
            if day_word == "tomorrow" or (day_word is None and moment <= wall):
                moment += timedelta(days=1)

            return from_the_wall(at=at, naive=moment)

        match = DAY_TIME.match(text)
        if match is not None:
            day_text, hour, minute = match.groups()
            day = datetime.strptime(day_text, "%Y-%m-%d")
            if hour is None:
                return from_the_wall(at=at, naive=day + timedelta(days=1) - timedelta(microseconds=1))

            return from_the_wall(
                at=at,
                naive=day.replace(
                    hour=int(hour),
                    minute=int(minute),
                ),
            )
    except ValueError as bad:
        raise AckError(f"'{text}' is not a moment on the clock or the calendar, {bad}") from bad

    raise AckError(f"'{text}' is not a moment like 22:00, tomorrow 22:00 or 2026-09-24 22:00")


def task_key(
    brain: store.Brain,
    key: str,
) -> str:
    """The id, or the name of a review, of the one open todo, bug or review the key answers to, refused otherwise.

    The id is what stays through a retitle, so the word follows the
    entry wherever its name goes. A todo or a bug is stored by its id
    and nothing else, so a stored key with no id prefix is always a
    review's name, which `task_kind` relies on.
    """
    try:
        entry = brain.find(key)
    except store.Ambiguous as clash:
        raise AckError(clash_text(brain=brain, key=key, clash=clash)) from clash

    if entry is None:
        raise AckError(f"nothing in the brain answers to '{key}', so auto ack cannot end with it")

    if entry.kind.name not in OPEN_STATUSES:
        raise AckError(f"auto ack ends with a todo, a bug or a review, and {entry.name} is a {entry.kind.name}")

    if not is_open(entry):
        raise AckError(f"{entry.name} is already {entry.field('status')}, so auto ack cannot end with it")

    if entry.kind.has_ids and entry.identifier is None:
        raise AckError(f"{entry.relative} carries no id, so auto ack cannot end with it until it is repaired")

    if entry.identifier is None and not entry.document.get("name"):
        raise AckError(f"{entry.relative} carries no name, so auto ack cannot end with it until it is repaired")

    stored = entry.identifier or entry.name
    try:
        brain.find(stored)
    except store.Ambiguous as clash:
        raise AckError(clash_text(brain=brain, key=stored, clash=clash)) from clash

    return stored


def clash_text(
    brain: store.Brain,
    key: str,
    clash: store.Ambiguous,
) -> str:
    """What to say when a key names several entries, the doctor's clash when it is the id, or a review's name, they all share, the store's own advice otherwise."""
    if all((entry.identifier or entry.name) == key for entry in brain.by_key().get(key, [])):
        return f"{key} names more than one entry, so auto ack cannot end with it until the doctor clears the clash"

    return str(clash)


def is_open(
    entry: store.Entry,
) -> bool:
    """Whether the entry still counts as open for its kind, a bug on a branch included."""
    return (entry.field("status") or "open") in OPEN_STATUSES.get(entry.kind.name, ("open",))


TaskState = Literal["open", "done", "gone", "unsure", "away"]


def task_state(
    brain: store.Brain,
    key: str,
) -> TaskState:
    """Whether the task is open, done, gone from the brain, unsure, when the brain cannot read it right now, or away, a review ledger this checkout lacks.

    Every state but open leaves the word off. Done and gone take it
    back for good, and so does an id that a merge has doubled, since
    the brain can then no longer say which entry the owner named.
    Unsure keeps the file, since an entry that is unreadable while a
    rebase stops on it may well read open again. A review ledger that
    is missing is away, never gone, since it lives on its branch and is
    back once that branch is checked out again, and no later review
    takes its name.
    """
    try:
        entry = brain.find(key)
    except store.Ambiguous:
        return "gone"

    if entry is None:
        if unreadable_task(brain=brain, key=key):
            return "unsure"

        return "away" if task_kind(key) is schema.KINDS["review"] else "gone"

    return "open" if is_open(entry) else "done"


def unreadable_task(
    brain: store.Brain,
    key: str,
) -> bool:
    """Whether the task may still be there unread, in a file the brain could not read this time or a folder it could not list."""
    folder = brain.folder(task_kind(key))
    for path, trouble in brain.unreadable.items():
        if path.stem == key or path.stem.startswith(f"{key}-") or path.parent.name == key:
            return True

        if trouble == store.UNLISTED and (path == folder or folder in path.parents):
            return True

    return False


def task_kind(
    key: str,
) -> schema.Kind:
    """The kind a stored task key names, by its id prefix, a review when it has none."""
    for name in OPEN_STATUSES:
        kind = schema.KINDS[name]
        if kind.id_prefix and key.startswith(f"{kind.id_prefix}-"):
            return kind

    return schema.KINDS["review"]


def write(
    brain: store.Brain,
    grant: Grant,
) -> None:
    data = {
        "since": grant.since.isoformat(),
        "until": grant.until.isoformat() if grant.until is not None else None,
        "task": grant.task,
    }
    frontmatter.write_text(
        path=str(path_of(brain=brain)),
        text=json.dumps(data, indent=2) + "\n",
    )


def read(
    brain: store.Brain,
) -> Optional[Grant]:
    """The recorded word, or None when there is none or the file cannot be read as one.

    A word carries its zone on every moment and has an end by time, by
    task or both, and a file that says otherwise is no word.
    """
    path = path_of(brain=brain)
    if not store.readable_file(path):
        return None

    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
        grant = Grant(
            since=moment_of(data["since"]),
            until=None if data.get("until") is None else moment_of(data["until"]),
            task=data.get("task"),
        )
    except (OSError, ValueError, KeyError, TypeError, OverflowError, RecursionError):
        return None

    if grant.until is None and grant.task is None:
        return None

    if grant.task is not None and not isinstance(grant.task, str):
        return None

    return grant


def moment_of(
    text: str,
) -> datetime:
    """A moment from the state file, refused when it carries no zone or lies past what every machine can place on its clock.

    The far end raises an OverflowError on its own, from the conversion
    the comparison needs, and the reader takes that as the refusal.
    """
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        raise ValueError(f"'{text}' carries no zone")

    if utc(moment) > LATEST:
        raise ValueError(f"'{text}' lies past the year 2999")

    return moment
