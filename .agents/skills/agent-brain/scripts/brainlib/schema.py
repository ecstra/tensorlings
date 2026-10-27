"""The rules in .agents/spec/, written so the tool can check them.

The prose specs stay the law a person reads. This module is the same law
in the form doctor and the write commands need, so a rule lives in one
place here rather than being rediscovered by every command.
"""

import re
from dataclasses import dataclass
from typing import Optional

DATE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")
SLUG = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

# A bug's severities, worst first, the one order every reader uses.
SEVERITIES: tuple[str, ...] = ("breaks", "bug", "debt", "nit")

FILENAME_WITH_ID = re.compile(r"^(BUG|TODO|DEC)-(\d{3})-(.+)\.md$")

# A review folder is <yyyy-mm>-<branch-slug>, per spec/reviews.md.
REVIEW_FOLDER = re.compile(r"^\d{4}-(0[1-9]|1[0-2])-[a-z0-9]+(-[a-z0-9]+)*$")

ID_DIGITS = 3


@dataclass(frozen=True)
class Field:
    """One frontmatter field a kind carries."""

    name: str
    required: bool = True
    values: tuple[str, ...] = ()
    is_date: bool = False


# How wide a review's finding pass goes, the one list the ledger's
# header field and the tool's --level share.
REVIEW_LEVELS: tuple[str, ...] = ("low", "medium", "high", "max")


@dataclass(frozen=True)
class Section:
    """One heading a kind's body carries."""

    title: str
    required: bool = True


@dataclass(frozen=True)
class Rule:
    """What a file in one status must carry, in the words used when it does not.

    One rule names one thing: a field that must be filled, a section
    that must be filled, or a link to a file of another kind. The writer
    refuses a status change that breaks a rule and doctor reports a file
    that sits in a status breaking one, with the same words.
    """

    status: str
    says: str
    field: str = ""
    section: str = ""
    link_kind: str = ""


@dataclass(frozen=True)
class Kind:
    """A kind of brain file, its home, what it must carry, and the rules its statuses hold to."""

    name: str
    folder: str
    fields: tuple[Field, ...] = ()
    sections: tuple[Section, ...] = ()
    id_prefix: str = ""
    frontmatter_kind: str = ""
    rules: tuple[Rule, ...] = ()
    # The fields `brain list` shows beside the id, and the description of
    # the folder's INDEX.md, empty for a folder that carries none.
    columns: tuple[str, ...] = ()
    index_description: str = ""

    @property
    def kind_value(self) -> str:
        return self.frontmatter_kind or self.name

    @property
    def has_ids(self) -> bool:
        return bool(self.id_prefix)

    @property
    def indexed(self) -> bool:
        return bool(self.index_description)


COMMON_FIELDS: tuple[Field, ...] = (
    Field(name="name"),
    Field(name="kind"),
    Field(name="description"),
    Field(name="updated", is_date=True),
    Field(name="links"),
)

KINDS: dict[str, Kind] = {
    "doc": Kind(
        name="doc",
        folder="docs",
        index_description="index of the knowledge library, one line per doc",
    ),
    "memory": Kind(
        name="memory",
        folder="memory",
        fields=(
            Field(
                name="type",
                values=("project", "feedback", "reference"),
            ),
        ),
        columns=("type",),
        index_description="index of project memory, one line per fact",
    ),
    "bug": Kind(
        name="bug",
        folder="bugs",
        id_prefix="BUG",
        columns=("status", "severity", "area"),
        index_description="the defect tracker's index, open bugs first",
        fields=(
            Field(
                name="status",
                values=("open", "branched", "fixed", "wontfix", "refiled"),
            ),
            Field(
                name="severity",
                values=SEVERITIES,
            ),
            Field(name="area"),
            Field(name="reported", is_date=True),
            Field(name="became", required=False),
            Field(name="parent", required=False),
        ),
        sections=(
            Section(title="Problem"),
            Section(title="Evidence"),
            Section(title="Steps", required=False),
            Section(title="Fix", required=False),
            Section(title="Verification", required=False),
        ),
        rules=(
            Rule(
                status="fixed",
                says="a bug goes fixed once its Fix section names the cause and the commit",
                section="Fix",
            ),
            Rule(
                status="wontfix",
                says="a wontfix needs its reason written under Problem",
                section="Problem",
            ),
            Rule(
                status="refiled",
                says="a refiled bug links to the todo it became",
                link_kind="todo",
            ),
        ),
    ),
    "todo": Kind(
        name="todo",
        folder="todos",
        id_prefix="TODO",
        columns=("status", "area"),
        index_description="the work tracker's index, open items first",
        fields=(
            Field(
                name="status",
                values=("open", "done", "dropped", "refiled"),
            ),
            Field(
                name="by",
                values=("owner", "agent"),
            ),
            Field(name="area"),
            Field(name="born", is_date=True),
            Field(name="done", required=False, is_date=True),
            Field(name="became", required=False),
        ),
        sections=(
            Section(title="What"),
            Section(title="Why", required=False),
            Section(title="Steps", required=False),
            Section(title="Done when"),
        ),
        rules=(
            Rule(
                status="done",
                says="a todo goes done with its closing date, done=YYYY-MM-DD",
                field="done",
            ),
            Rule(
                status="done",
                says="a todo goes done once Done when names what closed it",
                section="Done when",
            ),
            Rule(
                status="dropped",
                says="a todo goes dropped with its closing date, done=YYYY-MM-DD",
                field="done",
            ),
            Rule(
                status="refiled",
                says="a todo goes refiled with its closing date, done=YYYY-MM-DD",
                field="done",
            ),
            Rule(
                status="refiled",
                says="a refiled todo links to the bug it became",
                link_kind="bug",
            ),
        ),
    ),
    "decision": Kind(
        name="decision",
        folder="decisions",
        id_prefix="DEC",
        columns=("status", "date", "area"),
        index_description="index of every decision taken, newest first",
        fields=(
            Field(name="date", is_date=True),
            Field(name="area"),
            Field(
                name="status",
                values=("standing", "superseded"),
            ),
            Field(name="superseded-by", required=False),
        ),
        sections=(
            Section(title="Decision"),
            Section(title="Alternatives", required=False),
            Section(title="Consequences", required=False),
        ),
        rules=(
            Rule(
                status="superseded",
                says="a superseded decision names the one that replaced it",
                field="superseded-by",
            ),
        ),
    ),
    "plan": Kind(
        name="plan",
        folder="plans",
        columns=("status", "by"),
        index_description="index of the plans, each a goal with its steps, open first",
        fields=(
            Field(
                name="status",
                values=("open", "done", "dropped"),
            ),
            Field(
                name="by",
                values=("owner", "agent"),
            ),
            Field(name="born", is_date=True),
            Field(name="done", required=False, is_date=True),
        ),
        sections=(
            Section(title="Why"),
            Section(title="Steps"),
            Section(title="Done when"),
        ),
        rules=(
            Rule(
                status="done",
                says="a plan goes done with its closing date, done=YYYY-MM-DD",
                field="done",
            ),
            Rule(
                status="dropped",
                says="a plan goes dropped with its closing date, done=YYYY-MM-DD",
                field="done",
            ),
        ),
    ),
    "review": Kind(
        name="review",
        folder="reviews",
        columns=("status", "branch"),
        index_description="index of code review ledgers, active first",
        fields=(
            Field(name="branch"),
            Field(
                name="status",
                values=("open", "closed"),
            ),
            Field(
                name="level",
                required=False,
                values=REVIEW_LEVELS,
            ),
        ),
        sections=(
            Section(title="Summary"),
            Section(title="Batches"),
            Section(title="Findings"),
        ),
    ),
    "handover": Kind(
        name="handover",
        folder="handover",
        frontmatter_kind="doc",
    ),
}

# Files the kit owns at the root of .agents/. They carry no project
# content, so doctor judges them against the kit rather than a kind.
KIT_FILES: tuple[str, ...] = (
    "SYSTEM.md",
    "INDEX.md",
    "README.md",
    "house-rules.md",
)


@dataclass(frozen=True)
class Definition:
    """One sub agent definition a skill brings under .claude/agents/, its path relative to the project root.

    A skill gains definitions after it first ships, so each carries
    the release it came with, and a project stamped older than that
    with a file of the same name made it itself.
    """

    path: str
    since: str


@dataclass(frozen=True)
class KitSkill:
    """One skill the kit ships under skills/, a folder with a SKILL.md.

    Claude Code finds it through a link under .claude/skills/, which
    doctor makes, and the port and the upgrade treat the folder as the
    kit's own, replaced whole, from the release named in `since` on. A
    project stamped older than that with a folder of the same name made
    it itself, and the upgrade refuses to write over it. The same holds
    for the sub agent definitions a skill brings, each from its own
    release on.
    """

    name: str
    hook: str
    since: str
    definitions: tuple[Definition, ...] = ()


KIT_SKILLS: tuple[KitSkill, ...] = (
    KitSkill(
        name="agent-brain",
        hook="the brain tool, the one way anything here is filed, changed or checked",
        since="0",
    ),
    KitSkill(
        name="branch-review",
        hook="the code review, run on the owner's word with sub agents in parallel",
        since="1.2.0",
        definitions=(
            Definition(
                path=".claude/agents/review-finder.md",
                since="1.2.0",
            ),
            Definition(
                path=".claude/agents/review-fixer.md",
                since="1.4.0",
            ),
            Definition(
                path=".claude/agents/review-hunter.md",
                since="1.2.0",
            ),
            Definition(
                path=".claude/agents/review-verifier.md",
                since="1.2.0",
            ),
        ),
    ),
)
KIT_SKILL_NAMES: tuple[str, ...] = tuple(skill.name for skill in KIT_SKILLS)

# A brain with no stamp to read is on the oldest release as far as the
# tool can tell, so it carries only what every release shipped, and the
# upgrade never writes over a folder that might be the project's own.
NO_VERSION = "0"


def ships(
    skill: KitSkill,
    stamp: tuple[int, ...],
) -> bool:
    """Whether the release a brain is stamped with carries the skill."""
    return carries(
        since=skill.since,
        stamp=stamp,
    )


def carries(
    since: str,
    stamp: tuple[int, ...],
) -> bool:
    """Whether a brain stamped `stamp` is on the release named in `since` or a later one."""
    return as_tuple(version=since) <= stamp


def as_tuple(
    version: str,
) -> tuple[int, ...]:
    """The version as numbers to compare, padded to three so 1.0 is 1.0.0, or a ValueError when the text is not one."""
    parts = version.split(".")
    if not all(part.isdecimal() and part.isascii() for part in parts):
        raise ValueError(f"'{version}', not a version like 1.0.0")

    numbers = [int(part) for part in parts]

    return tuple(numbers + [0] * (3 - len(numbers)))


# SYSTEM.md is the rulebook. An agent takes it as it stands, so it is the
# one markdown file at the root of the directory with no frontmatter. The
# tool's own docs under skills/ follow the Agent Skills standard instead.
NO_FRONTMATTER: tuple[str, ...] = ("SYSTEM.md",)

# The folders that carry their own INDEX.md, per spec/conventions.md,
# read off the kinds so the table lives in one place.
INDEXED_FOLDERS: tuple[str, ...] = tuple(kind.folder for kind in KINDS.values() if kind.indexed)

# The files the kit writes at the top of a project: the front doors every
# agent reads first, the contributing note and the ignore file.
FRONT_DOORS: tuple[str, ...] = ("AGENTS.md", "CLAUDE.md", "GEMINI.md")
ROOT_FILES: tuple[str, ...] = (*FRONT_DOORS, "CONTRIBUTING.md")
IGNORE_FILE = ".gitignore"

INDEX = "INDEX.md"
LEDGER = "ledger.md"

# Where the skills live under .agents/, one folder each with this file
# in it, the layout Claude Code loads a skill from through its link.
SKILLS = "skills"
SKILL_FILE = "SKILL.md"

# The kit's version stamp at the root of .agents/, written by a port or
# an upgrade and read by the doctor and the status line.
VERSION_FILE = "VERSION"

# The one word for a status a file does not carry, wherever it is counted.
UNKNOWN = "unknown"


def is_hidden(
    relative: str,
) -> bool:
    """Whether any part of a path inside .agents/ starts with a dot.

    A dotfile or anything inside a dot folder is nobody's: an editor's
    swap file, a scratch folder a write left behind, a tool's cache. It
    is never an entry, never the kit's, never linked, listed or checked.
    """
    return any(part.startswith(".") for part in relative.split("/"))


def is_kit_path(
    relative: str,
) -> bool:
    """Whether a path inside .agents/ names a file the kit owns.

    That is a root kit file, a spec, or the index of a folder that
    carries one. Everything else in the brain is an entry or a stray,
    and a dotted path is neither.
    """
    if is_hidden(relative):
        return False

    parts = relative.split("/")

    if len(parts) == 1:
        return parts[0] in KIT_FILES

    if len(parts) == 2 and parts[0] == "spec":
        return parts[1].endswith(".md") and parts[1] != INDEX

    return len(parts) == 2 and parts[0] in INDEXED_FOLDERS and parts[1] == INDEX


class UnknownKind(Exception):
    """A kind name the spec does not know, with the known ones in the message."""


def kind_named(
    name: str,
) -> Kind:
    if name not in KINDS:
        known = ", ".join(sorted(KINDS))
        raise UnknownKind(f"unknown kind '{name}', expected one of {known}")

    return KINDS[name]


def kind_for_folder(
    folder: str,
) -> Optional[Kind]:
    for kind in KINDS.values():
        if kind.folder == folder:
            return kind

    return None


def fields_for(
    kind: Kind,
) -> tuple[Field, ...]:
    return COMMON_FIELDS + kind.fields


def is_slug(
    value: str,
) -> bool:
    return bool(SLUG.match(value))


def is_date(
    value: str,
) -> bool:
    return bool(DATE.match(value))


def is_review_folder(
    value: str,
) -> bool:
    return bool(REVIEW_FOLDER.match(value))


def format_id(
    prefix: str,
    number: int,
) -> str:
    return f"{prefix}-{number:0{ID_DIGITS}d}"


def slugify(
    title: str,
) -> str:
    """Turn a title into the filename slug the conventions ask for."""
    lowered = title.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", lowered)

    return cleaned.strip("-")
