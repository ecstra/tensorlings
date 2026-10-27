"""Checking the whole brain, fixing what is mechanical, flagging the rest.

Every check is a function that reads the brain and reports findings. A
finding that a machine can put right carries its own repair, and `run`
applies those before it reports, so what comes back is only what needs a
person or an agent to decide.
"""

import errno
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Optional, Union

from . import autoack, frontmatter, index, links, schema, sections, steps, store, style, task, writes

SKIP_DIRS: frozenset[str] = frozenset({schema.SKILLS, ".worktrees"})

# A header block the parser refuses, taken off by its fences alone so
# the prose under it is still read.
HEADER_BLOCK = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)

MACHINE_PATH = re.compile(r"(/Users/|/home/|[A-Z]:\\)")

SECRET_SHAPES = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),
    re.compile(r"(?i)\b(password|passwd|secret|token)\s*[:=]\s*\S{6,}"),
)

SKILL_LINKS = Path(".claude") / schema.SKILLS
SKILL_HOMES = Path(".agents") / schema.SKILLS

# How many repair passes one --fix may take. Two is the longest chain
# the checks can form, a name put right and then its index rebuilt.
PASSES = 5


@dataclass
class Finding:
    """One thing wrong, where it is, and whether the tool can put it right."""

    path: str
    check: str
    message: str
    repair: Optional[Callable[[], None]] = None

    @property
    def fixable(self) -> bool:
        return self.repair is not None

    def line(self) -> str:
        where = self.path or "."
        mark = "fixable" if self.fixable else "flag"

        return f"{where}: {self.check}: {self.message} [{mark}]"

    def as_dict(self) -> dict:
        """The finding as plain data, for the JSON form of doctor."""
        return {
            "path": self.path,
            "check": self.check,
            "message": self.message,
            "fixable": self.fixable,
        }


def run(
    brain: store.Brain,
    fix: bool = False,
) -> list[Finding]:
    """Every finding in the brain, after repairs when `fix` is set.

    Repairs run in passes. One pass applies every repair on offer, then
    the brain is read again, since one repair changes what another check
    sees, an index built before a name was put right for one. It stops
    when nothing fixable is left or a pass changed nothing. A repair
    that fails turns its finding into a flag that carries the reason.
    """
    found = check_all(brain)
    if not fix:
        return found

    failed: dict[tuple[str, str, str], str] = {}
    attempted: set[tuple[str, str, str]] = set()

    for _ in range(PASSES):
        pending = [finding for finding in found if finding.fixable and key_of(finding) not in failed]
        if not pending:
            break

        for finding in pending:
            attempted.add(key_of(finding))
            try:
                finding.repair()
            except (OSError, frontmatter.FrontmatterError, task.TaskError, steps.StepsError) as error:
                failed[key_of(finding)] = store.explain(error)

        brain.refresh()
        again = check_all(brain)
        if lines_of(again) == lines_of(found):
            break

        found = again

    return [flagged(finding=finding, failed=failed, attempted=attempted) for finding in found]


def key_of(
    finding: Finding,
) -> tuple[str, str, str]:
    """What makes a finding the same one across passes: its file, check and words."""
    return finding.path, finding.check, finding.message


def lines_of(
    found: list[Finding],
) -> list[str]:
    return [finding.line() for finding in found]


def flagged(
    finding: Finding,
    failed: dict[tuple[str, str, str], str],
    attempted: set[tuple[str, str, str]],
) -> Finding:
    """The finding as it stands, or as a flag when its repair failed or got nowhere.

    A finding still on offer as fixable after its repair ran is one no
    number of runs will clear, so it goes back as a flag that says so
    rather than an invitation to run --fix again.
    """
    key = key_of(finding)
    reason = failed.get(key)
    if reason is not None:
        return Finding(path=finding.path, check=finding.check, message=f"{finding.message}, and the repair failed: {reason}")

    if finding.fixable and key in attempted:
        return Finding(path=finding.path, check=finding.check, message=f"{finding.message}, and the repair ran without clearing it")

    return finding


def check_all(
    brain: store.Brain,
) -> list[Finding]:
    found: list[Finding] = []

    for check in CHECKS:
        found.extend(check(brain))

    return found


def markdown_files(
    brain: store.Brain,
) -> list[Path]:
    """The files the frontmatter checks look at: the entries and the kit's files.

    A file that is neither is a stray, reported once by its own check
    rather than measured against rules written for something else. The
    style and hygiene scans read every file regardless, since a pasted
    secret is most likely in exactly the file nobody filed.
    """
    accounted = brain.known()

    return [path for path in all_markdown(brain) if path in accounted or schema.is_kit_path(relative(brain=brain, path=path))]


def all_markdown(
    brain: store.Brain,
    with_skills: bool = False,
) -> list[Path]:
    """Every markdown file under the brain, the rulebook and dotfiles excepted, the skills only when asked.

    The skills folder is the tool's own, its files follow the Agent
    Skills standard rather than the specs, so only the prose and
    boundary checks (style and hygiene) read them.
    """
    found: list[Path] = []

    for path in sorted(brain.root.rglob("*.md")):
        relative_path = path.relative_to(brain.root)
        parts = relative_path.parts
        if store.hidden(relative_path):
            continue

        if parts[0] in SKIP_DIRS and not (with_skills and parts[0] == schema.SKILLS):
            continue

        if path.name in schema.NO_FRONTMATTER and len(parts) == 1:
            continue

        found.append(path)

    return found


def check_strays(
    brain: store.Brain,
) -> list[Finding]:
    """Files under the brain that neither the store nor the kit accounts for."""
    accounted = brain.known()
    found: list[Finding] = []

    for path in all_markdown(brain):
        where = relative(brain=brain, path=path)
        if path in accounted or schema.is_kit_path(where):
            continue

        found.append(Finding(path=where, check="stray", message=stray_reason(where)))

    return found


def check_leftovers(
    brain: store.Brain,
) -> list[Finding]:
    """What an interrupted write left beside its target, safe to remove.

    Only the tool's own shapes count, a real file named the way a write
    in flight is named and, inside the brain, a real folder named the
    way a put back scratch is named. A person's own dotfile is nobody's
    to delete, and a suffix of eight letters looks like the tail the
    writer draws, so a temp file counts only when its target is one the
    tool would write, a markdown or JSON file, VERSION, the word, the
    current work, a file that exists beside it, or one of the kit's own
    files at the top of the project. The places the tool writes to in
    every run are looked at, the whole brain with its skills, each
    agent's own folder down to its sub agent files, and the top of the
    project. The one time move from `.agent/` rewrites files anywhere
    in the project, cleans up after itself unless killed outright, and
    is not looked for.
    """
    found: list[Finding] = []
    project = style.project(brain=brain)
    candidates = [path for path in brain.root.rglob(".*") if path.relative_to(brain.root).parts[0] != ".worktrees"]
    candidates += list(project.glob(".*"))
    for folder in style.folders(brain):
        if store.quietly(folder.is_dir):
            candidates += list(folder.rglob(".*"))

    for path in sorted(candidates):
        if store.quietly(path.is_symlink):
            continue

        in_flight = store.quietly(path.is_file) and tools_own_write(path=path, project=project)
        scratch = brain.root in path.parents and store.quietly(path.is_dir) and links.SCRATCH_SHAPE.match(path.name) is not None
        if not in_flight and not scratch:
            continue

        found.append(
            Finding(
                path=where_of(brain=brain, path=path),
                check="leftover",
                message="left by an interrupted write, nothing in it is the brain's",
                repair=remove(path),
            )
        )

    return found


def tools_own_write(
    path: Path,
    project: Path,
) -> bool:
    """Whether the file is a write in flight of the tool's, by its shape and by a target the tool would write."""
    target = frontmatter.target_of(path.name)
    if target is None:
        return False

    if path.parent == project:
        return target in (*schema.ROOT_FILES, schema.IGNORE_FILE)

    if target.endswith((".md", ".json")) or target in (schema.VERSION_FILE, autoack.FILE, task.FILE):
        return True

    return store.quietly((path.parent / target).exists)


def where_of(
    brain: store.Brain,
    path: Path,
) -> str:
    """A path from the brain when it is inside it, else from the project."""
    if brain.root in path.parents:
        return relative(brain=brain, path=path)

    return str(path.relative_to(brain.root.parent))


def remove(
    path: Path,
) -> Callable[[], None]:
    """A repair that removes a leftover, a folder with whatever the tool left in it."""

    def repair() -> None:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    return repair


def stray_reason(
    where: str,
) -> str:
    parts = where.split("/")

    if parts[0] == "reviews" and parts[-1] == schema.LEDGER:
        return f"a ledger sits at reviews/<folder>/{schema.LEDGER}, one folder down, no more and no less"

    if parts[0] == "reviews":
        return f"a review folder holds {schema.LEDGER} and nothing else"

    if parts[-1] == schema.INDEX:
        return "no index belongs here, the tool builds one per tracker, docs/ and memory/"

    if parts[0] == "spec":
        return "spec/ holds its files directly, nothing nested"

    return "not a file the brain lists, file it with brain new or move it out"


def relative(
    brain: store.Brain,
    path: Path,
) -> str:
    return str(path.relative_to(brain.root))


def check_frontmatter(
    brain: store.Brain,
) -> list[Finding]:
    found: list[Finding] = []

    for path in markdown_files(brain):
        where = relative(brain=brain, path=path)

        try:
            document = frontmatter.read(str(path))
        except frontmatter.FrontmatterError as error:
            found.append(Finding(path=where, check="frontmatter", message=str(error)))
            continue

        for item in schema.COMMON_FIELDS:
            value = document.get(item.name)
            if value:
                continue

            found.append(common_field_gap(path=path, where=where, name=item.name, present=value is not None))

        updated = document.get("updated")
        if updated and not schema.is_date(updated):
            found.append(
                Finding(path=where, check="frontmatter", message=f"updated reads '{updated}', not YYYY-MM-DD")
            )

    return found


def common_field_gap(
    path: Path,
    where: str,
    name: str,
    present: bool,
) -> Finding:
    """A common field that is missing or blank. The date and the links can be filled in."""
    if name == "updated":
        return Finding(path=where, check="frontmatter", message="no updated date", repair=stamp_today(path))

    if name == "links":
        return Finding(
            path=where,
            check="frontmatter",
            message="links is blank, none is written as []",
            repair=set_field(path=path, key="links", value="[]"),
        )

    state = "is blank" if present else "field is missing"

    return Finding(path=where, check="frontmatter", message=f"{name} {state}")


def stamp_today(
    path: Path,
) -> Callable[[], None]:
    def repair() -> None:
        document = frontmatter.read(str(path))
        document.set(key="updated", value=date.today().isoformat())
        frontmatter.write(path=str(path), document=document)

    return repair


def check_names(
    brain: store.Brain,
) -> list[Finding]:
    found: list[Finding] = []
    seen: dict[str, str] = {}

    for path, document in links.kit_documents(brain):
        name = document.get("name") or ""
        if name:
            seen[name] = relative(brain=brain, path=path)

    for entry in brain.entries():
        expected = writes.entry_name(path=entry.path, kind=entry.kind)
        actual = entry.document.get("name") or ""
        if not actual:
            continue

        if actual != expected:
            found.append(
                Finding(
                    path=entry.relative,
                    check="name",
                    message=f"name is '{actual}', the file says '{expected}'",
                    repair=set_name(brain=brain, entry=entry, actual=actual, expected=expected),
                )
            )

        if actual in seen:
            found.append(
                Finding(
                    path=entry.relative,
                    check="name",
                    message=f"name '{actual}' is also carried by {seen[actual]}",
                )
            )
        seen[actual] = entry.relative

    return found


def set_name(
    brain: store.Brain,
    entry: store.Entry,
    actual: str,
    expected: str,
) -> Callable[[], None]:
    """A repair that sets the name the file says, and moves the links that hold the old name with it: a plan's or a review's link on the chain, and the step cells that point at a review.

    The chain and the cells move first: the new name resolves by the
    file whether or not the field follows, and a chain that cannot be
    written leaves the field as it was, so the finding stays to report
    the failure. The links stay where they are when another file
    answers to the old name, a copy that kept it, since they mean that
    file.
    """
    set_the_field = set_field(path=entry.path, key="name", value=expected)
    # a review is pointed at by its name, and so is a todo or a bug with no id, a file the tool never wrote, so their cells follow the name too
    pointed_by_name = entry.kind.name == "review" or (entry.kind.name in ("todo", "bug") and entry.identifier is None)
    follows = (entry.kind.name == "plan" or pointed_by_name) and not any(
        other.path != entry.path and actual in (other.name, task.file_key(other)) for other in brain.entries()
    )

    def repair() -> None:
        if follows and entry.kind.name in ("plan", "review"):
            task.renamed(brain=brain, kind=entry.kind.name, old=actual, new=expected)

        if follows and pointed_by_name:
            steps.repoint(brain=brain, keys={actual}, to=expected)

        set_the_field()

    return repair


def set_field(
    path: Path,
    key: str,
    value: str,
) -> Callable[[], None]:
    """A repair that changes one field and stamps the file, per the upkeep rule."""

    def repair() -> None:
        document = frontmatter.read(str(path))
        document.set(key=key, value=value)
        document.set(key="updated", value=date.today().isoformat())
        frontmatter.write(path=str(path), document=document)

    return repair


def check_kinds(
    brain: store.Brain,
) -> list[Finding]:
    found: list[Finding] = []

    for entry in brain.entries():
        actual = entry.document.get("kind")
        if not actual or actual == entry.kind.kind_value:
            continue

        found.append(
            Finding(
                path=entry.relative,
                check="kind",
                message=f"kind is '{actual}' but the file sits in {entry.kind.folder}/",
            )
        )

    return found


def check_fields(
    brain: store.Brain,
) -> list[Finding]:
    found: list[Finding] = []

    for entry in brain.entries():
        fields = entry.document.fields()
        try:
            writes.check_own(kind=entry.kind, fields=fields)
        except writes.WriteError as error:
            found.append(Finding(path=entry.relative, check="fields", message=str(error)))

        found += [Finding(path=entry.relative, check="fields", message=slip) for slip in common_slips(fields)]

    for path, document in links.kit_documents(brain):
        where = relative(brain=brain, path=path)
        found += [Finding(path=where, check="fields", message=slip) for slip in common_slips(document.fields())]

    return found


def common_slips(
    fields: dict[str, str],
) -> list[str]:
    """What the house style bans in the common header values, since each flows into an index."""
    return [
        f"{item.name} {slip}"
        for item in schema.COMMON_FIELDS
        for slip in sections.style_slips(fields.get(item.name, ""))
    ]


def check_sections(
    brain: store.Brain,
) -> list[Finding]:
    found: list[Finding] = []

    for entry in brain.entries():
        body = entry.document.body
        for title in sections.missing(body=body, kind=entry.kind):
            # a plan's missing Steps section is the steps check's finding, which carries the repair that adds the empty table
            if entry.kind.name == "plan" and title == "Steps":
                continue

            found.append(
                Finding(path=entry.relative, check="sections", message=f"no {title} heading")
            )

        for message in sections.lifecycle_gaps(kind=entry.kind, fields=entry.document.fields(), body=body):
            found.append(Finding(path=entry.relative, check="lifecycle", message=message))

    return found


def check_steps(
    brain: store.Brain,
) -> list[Finding]:
    """The steps table a plan, a todo or a bug carries, per spec/plans.md.

    The table must read as one, a list in its place turned into one on
    a plan or a todo and a plan's missing section given an empty table,
    every step that points at an entry must reach one, a pointed step's
    mark must be the one its entry gives and its pointer the entry's
    id, moved on from a refiled one, both repaired, a became field must
    name what took the entry's place, two entries must not wait on each
    other, and an entry done or fixed with a step still pending is
    reported, since a closed table is its record.
    """
    found: list[Finding] = []

    for kind_name in steps.STEPPED_KINDS:
        for entry in brain.entries_of(schema.kind_named(kind_name)):
            found += check_table(brain=brain, entry=entry)

    return found


def check_table(
    brain: store.Brain,
    entry: store.Entry,
) -> list[Finding]:
    found: list[Finding] = []
    try:
        rows = steps.stored_rows_of(kind=entry.kind.name, body=entry.document.body)
    except steps.StepsError as broken:
        message = str(broken)
        if entry.kind.name == "bug" and not steps.has_table(entry.document.body):
            message += ", a bug's Steps being the fix's steps and a reproduction going under Evidence, so move it there by hand, or make it a numbered list and on an open bug run brain steps <key> add to keep it as the fix's"

        offered = convert_steps(brain=brain, entry=entry)
        if isinstance(offered, str):
            return [Finding(path=entry.relative, check="steps", message=f"{message}, and the list is refused as it stands, {offered}")]

        return [Finding(path=entry.relative, check="steps", message=message, repair=offered)]

    found += [Finding(path=entry.relative, check="steps", message=message) for message in steps.became_slips(brain=brain, entry=entry)]
    found += [Finding(path=entry.relative, check="steps", message=f"step {step.number} has no text, which no write should have let stand, so fill it in by hand") for step in rows if not step.text.strip()]

    if not steps.is_open(entry):
        pending = [step.number for step in rows if step.status == "pending"]
        if pending and entry.field("status") == steps.CLOSING[entry.kind.name]:
            numbers = ", ".join(str(number) for number in pending)
            # nothing points at a plan, so only a todo or a bug can have its reopen refused for a loop, while steps on the way that do not read refuse it for any kind
            if entry.kind.name == "plan":
                way_round = ", and when the set to open is refused for steps that do not read, mend them first"
            else:
                way_round = ", and when the set to open is refused, mend the steps it says do not read, or close the entry whose steps point back here first"

            found.append(
                Finding(
                    path=entry.relative,
                    check="steps",
                    message=f"the entry is {entry.field('status')} with step {numbers} still pending, so reopen it and close it again once the step is settled{way_round}",
                )
            )

        return found

    derived: list[steps.Step] = []
    own = steps.own_keys(entry)
    for step in rows:
        fresh = steps.derived_row(brain=brain, step=step, own=own)
        derived.append(fresh)
        if step.points_at and steps.pointed(brain=brain, key=step.points_at) is None:
            if steps.doubled(brain=brain, key=step.points_at):
                why = f"which two files carry, so it reaches nothing while the {steps.doubled_word(step.points_at)} is doubled"
            else:
                why = "which reaches no todo, bug or review"

            found.append(
                Finding(
                    path=entry.relative,
                    check="steps",
                    message=f"step {step.number} points at '{step.points_at}', {why}, and brain steps <key> step {step.number} done or skipped marks it by hand meanwhile",
                )
            )
        elif fresh.points_at != step.points_at:
            target = steps.pointed(brain=brain, key=step.points_at)
            moved = target is not None and target.field("status") == "refiled"
            if moved:
                why = refiled_word(brain=brain, target=target, kept=fresh.points_at)
            elif target is not None and target.kind.name == "review":
                why = f"and the tool keeps the name {fresh.points_at}, which a review keeps as its branch's"
            elif target is not None and target.identifier is None:
                why = f"and the tool keeps the name {fresh.points_at}, the file carrying no id, which doctor reports"
            else:
                why = f"and the tool keeps the id {fresh.points_at}, which a retitle never moves"
            found.append(
                Finding(
                    path=entry.relative,
                    check="steps",
                    message=f"step {step.number} points at {step.points_at}, {why}",
                    repair=refresh_steps(brain=brain, entry=entry),
                )
            )
        elif fresh.status != step.status:
            found.append(
                Finding(
                    path=entry.relative,
                    check="steps",
                    message=f"step {step.number} is marked {step.status} while {step.points_at} says {fresh.status}",
                    repair=refresh_steps(brain=brain, entry=entry),
                )
            )

    found += [Finding(path=entry.relative, check="steps", message=message) for message in steps.looping(brain=brain, entry=entry, rows=derived)]

    return found


def refiled_word(
    brain: store.Brain,
    target: store.Entry,
    kept: str,
) -> str:
    """Why a cell at a refiled entry reads as the tool would write it, handed on to the entry filed in its place, at whatever remove, or kept at a refiled one, whose successor's steps lead back here or do not read, or which has no successor the brain knows."""
    landed = steps.pointed(brain=brain, key=kept)
    if landed is None or landed.field("status") != "refiled":
        return f"which was refiled as {kept}"

    successor = steps.refiled_to(brain=brain, entry=landed)
    if successor is None:
        became = landed.field("became") or ""
        if became and steps.doubled(brain=brain, key=became):
            return f"which was refiled as {became}, which two files carry, so the tool keeps the key {kept} and the step stands skipped while the {steps.doubled_word(became)} is doubled"

        return f"which was refiled as nothing the brain knows, so the tool keeps the key {kept} and the step stands skipped"

    handed_on = steps.pointer_of(successor)
    if successor.field("status") == "refiled":
        return f"which sits in a loop of refiles through {kept} and {handed_on}, so the tool keeps the key {kept} and the step stands skipped until the loop is undone"

    if steps.doubled(brain=brain, key=handed_on):
        return f"which was refiled as {handed_on}, which two files carry, so the tool keeps the key {kept} and the step stands skipped while the {steps.doubled_word(handed_on)} is doubled"

    broken = steps.unreadable_reached(brain=brain, key=handed_on)
    if broken is not None:
        where = handed_on if broken == handed_on else f"{handed_on}, whose steps lead to {broken}"

        return f"which was refiled as {where}, whose steps do not read, so the tool keeps the key {kept} and the step stands skipped until they are put right"

    return f"which was refiled as {handed_on}, whose steps lead back here, so the tool keeps the key {kept} and the step stands skipped"


def convert_steps(
    brain: store.Brain,
    entry: store.Entry,
) -> Union[Callable[[], None], str, None]:
    """A repair that turns a list under Steps into the table, its rows marked by the entry's own status, the reason when the tool refuses the list as it stands, so the finding says why no run of --fix clears it, or none when there is no list to turn.

    A bug gets no repair, since a list under its Steps from before the
    table may be the reproduction, which the spec puts under Evidence,
    and only whoever filed it can say which it is, though the reason
    the section would not turn is named for it too, so the line to
    move by hand is known.
    """
    body = entry.document.body
    if steps.has_table(body):
        return None

    own = steps.own_keys(entry)
    try:
        steps.converted(brain=brain, body=body, by=entry.field("by") or "agent", kind=entry.kind, status=steps.settled_as(entry), own=own)
    except steps.StepsError as refused:
        return str(refused)

    if entry.kind.name == "bug":
        return None

    def repair() -> None:
        fresh = brain.read(path=entry.path, kind=entry.kind)
        fresh.document.body = steps.converted(brain=brain, body=fresh.document.body, by=fresh.field("by") or "agent", kind=fresh.kind, status=steps.settled_as(fresh), own=own)
        steps.save(brain=brain, entry=fresh)

    return repair


def refresh_steps(
    brain: store.Brain,
    entry: store.Entry,
) -> Callable[[], None]:
    """A repair that brings an entry's pointed marks in line with what they point at, stamping the file."""

    def repair() -> None:
        steps.refresh(brain=brain, entry=brain.read(path=entry.path, kind=entry.kind))

    return repair


def check_filenames(
    brain: store.Brain,
) -> list[Finding]:
    found: list[Finding] = []
    numbers: dict[str, str] = {}

    for entry in brain.entries():
        if entry.kind.name == "review":
            if not schema.is_review_folder(entry.slug):
                found.append(
                    Finding(
                        path=entry.relative,
                        check="filename",
                        message=f"a review folder is named <yyyy-mm>-<branch-slug>, not '{entry.slug}'",
                    )
                )
            continue

        for folder in Path(entry.relative).parts[1:-1]:
            if not schema.is_slug(folder):
                found.append(
                    Finding(path=entry.relative, check="filename", message=f"sits in a folder that is not a slug, '{folder}'")
                )

        if entry.kind.has_ids and entry.identifier is None:
            found.append(
                Finding(
                    path=entry.relative,
                    check="filename",
                    message=f"a {entry.kind.name} is named {entry.kind.id_prefix}-###-<slug>.md",
                )
            )
            continue

        if not schema.is_slug(entry.slug):
            found.append(
                Finding(path=entry.relative, check="filename", message=f"'{entry.slug}' is not a slug")
            )

        if entry.identifier is None:
            continue

        if entry.identifier in numbers:
            found.append(
                Finding(
                    path=entry.relative,
                    check="filename",
                    message=f"{entry.identifier} is also {numbers[entry.identifier]}",
                )
            )
        numbers[entry.identifier] = entry.relative

    return found


def check_links(
    brain: store.Brain,
) -> list[Finding]:
    found: list[Finding] = []
    documents = links.documents(brain)

    for path, document in documents:
        where = relative(brain=brain, path=path)
        current = document.get_list("links")
        canonical: list[str] = []

        for link in current:
            resolved = links.canonical(brain=brain, reference=link)
            if (resolved or link) in canonical:
                continue

            if resolved is None:
                found.append(Finding(path=where, check="link", message=f"'{link}' reaches nothing"))
            canonical.append(resolved or link)

        if canonical != current:
            found.append(
                Finding(
                    path=where,
                    check="link",
                    message="links not written as paths relative to .agents/, each once",
                    repair=set_links(path=path, values=canonical),
                )
            )

    for path, document in documents:
        for target in links.broken_in_body(path=path, document=document):
            found.append(
                Finding(path=relative(brain=brain, path=path), check="link", message=f"body link '{target}' reaches nothing")
            )

    return found


def set_links(
    path: Path,
    values: list[str],
) -> Callable[[], None]:
    def repair() -> None:
        document = frontmatter.read(str(path))
        document.set_list(key="links", values=values)
        document.set(key="updated", value=date.today().isoformat())
        frontmatter.write(path=str(path), document=document)

    return repair


def check_indexes(
    brain: store.Brain,
) -> list[Finding]:
    """One build of the brain serves the check and every repair it hands out.

    A brain whose skills folder cannot be listed has no spine the tool
    can build, so the build leaves it out and the spine is reported as
    unbuildable and left as it is rather than rewritten without the
    project's skills.
    """
    wanted = index.build(brain)
    found = [
        Finding(
            path=relative(brain=brain, path=path),
            check="index",
            message="differs from what the tool builds for its folder",
            repair=write_index(path=path, text=wanted[path]),
        )
        for path in index.stale_among(wanted)
    ]
    if brain.root / schema.INDEX not in wanted:
        found.append(Finding(path=schema.INDEX, check="index", message="cannot be rebuilt while the skills folder cannot be listed"))

    return found


def write_index(
    path: Path,
    text: str,
) -> Callable[[], None]:
    def repair() -> None:
        index.write_one(path=path, text=text)

    return repair


def check_style(
    brain: store.Brain,
) -> list[Finding]:
    """The prose rules over every file but the indexes.

    An index is the tool's rendering of other files' header values and
    of the folders they sit in, checked at their source by the fields
    and filename checks, so a slip is named on the file a person can
    put right rather than on one nobody may edit.
    """
    found: list[Finding] = []

    for path in all_markdown(brain=brain, with_skills=True):
        if path.name == schema.INDEX:
            continue

        prose = prose_in(path)
        if prose is None:
            continue

        where = relative(brain=brain, path=path)
        found += [Finding(path=where, check="style", message=slip) for slip in sections.style_slips(prose)]

    return found


def prose_in(
    path: Path,
) -> Optional[str]:
    """The file's prose, its header block and its code stepped over, or None when it cannot be read.

    A header the tool can read is taken off by the parser, one it
    cannot, the Agent Skills shape in `skills/` for one, by its fences.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    try:
        body = frontmatter.parse(text).body
    except frontmatter.FrontmatterError:
        body = HEADER_BLOCK.sub("", text, count=1)

    return sections.prose_of(body)


def check_hygiene(
    brain: store.Brain,
) -> list[Finding]:
    found: list[Finding] = []

    for path in all_markdown(brain=brain, with_skills=True):
        try:
            text = path.read_bytes().decode("utf-8", errors="replace")
        except OSError:
            continue

        where = relative(brain=brain, path=path)

        if MACHINE_PATH.search(sections.prose_of(text)):
            found.append(Finding(path=where, check="hygiene", message="names a machine's local path"))

        for shape in SECRET_SHAPES:
            if shape.search(text):
                found.append(Finding(path=where, check="hygiene", message="carries something shaped like a secret"))
                break

    return found


def check_layout(
    brain: store.Brain,
) -> list[Finding]:
    found: list[Finding] = []
    project = brain.root.parent

    if not store.readable_file(brain.root / schema.VERSION_FILE):
        found.append(Finding(path=schema.VERSION_FILE, check="layout", message="no kit version stamp, run the upgrade"))
    elif store.stamp_of(brain.root) is None:
        found.append(Finding(path=schema.VERSION_FILE, check="layout", message="the kit version stamp is not a version like 1.0.0, write the kit's release in it"))

    shared = skills_folder_outside(project)
    shipped = store.kit_skills_shipped(brain.root)
    wanted = list(shipped)
    listed = True
    try:
        wanted += store.project_skills(
            brain=brain.root,
            shipped=shipped,
        )
    except OSError as error:
        reason = error.strerror or str(error)
        found.append(
            Finding(
                path=schema.SKILLS,
                check="layout",
                message=f"the skills folder cannot be listed, so only the kit's skill links were checked: {reason}",
            )
        )
        listed = False

    for name in wanted:
        trouble = skill_link_trouble(
            project=project,
            name=name,
            shared=shared,
            kits=name in shipped,
        )
        if trouble is None:
            continue

        message = trouble.message
        repair = trouble.repair
        if shared and repair is not None:
            message += SHARED_FOLDER
            repair = None

        found.append(
            Finding(
                path=str(trouble.path),
                check="layout",
                message=message,
                repair=repair,
            )
        )

    if listed:
        strays = stray_skill_links(
            project=project,
            wanted=wanted,
            shared=shared,
        )
        found.extend(strays)

    found.extend(check_style_copies(brain))

    return found


CANNOT_SEE = ", so Claude Code cannot see the skill"
REAL_FOLDER_WITHOUT_SKILL = "a folder with no SKILL.md sits where the skill link belongs" + CANNOT_SEE
ANOTHER_SKILL = "a skill of its own sits where the skill link belongs, so Claude Code loads that one instead"
STALE_COPY = "a copy of the skill sits where the skill link belongs and is behind the brain's, so Claude Code loads the old text"
SHARED_FOLDER = ", and the skills folder sits outside the project, so the link is yours to make"
SHARED_REMOVAL = ", and the skills folder sits outside the project, so the link is yours to remove"
SKILL_GONE = "the skill link points at a skill that is gone"


@dataclass(frozen=True)
class Trouble:
    """What is wrong with one skill link: the message, the path the finding names, and the repair when doctor may make one."""

    message: str
    path: Path
    repair: Optional[Callable[[], None]] = None


def skills_folder_outside(
    project: Path,
) -> bool:
    """Whether `.claude/skills` resolves to a folder outside the project.

    Such a folder is shared with other checkouts on the machine, and a
    link made here would point every one of them at this project's
    skills, so doctor names the trouble and leaves the link to a person.
    """
    folder = resolved(project / SKILL_LINKS)
    root = resolved(project)

    return folder is not None and root is not None and not folder.is_relative_to(root)


def resolved(
    path: Path,
) -> Optional[Path]:
    """The path resolved, or None when it reaches nothing.

    A link that loops raises RuntimeError up to 3.12 and resolves to
    itself from 3.13 on, a folder the process cannot search raises
    OSError, and a name the system refuses raises ValueError. Each is
    a path that reaches nothing, answered here once for every check
    that resolves one.
    """
    try:
        return path.resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def skill_missing(
    name: str,
) -> str:
    return f"the skill itself is missing from {SKILL_HOMES / name}, reinstall the kit"


def stray_skill_links(
    project: Path,
    wanted: list[str],
    shared: bool,
) -> list[Finding]:
    """A link under .claude/skills that doctor once made for a project skill now gone from the brain, with its removal as the repair.

    Only a link that points into the brain's skills folder and dangles,
    leading to nothing at all, is doctor's to remove. One that leads to
    a folder or a file still there is somebody's and is left alone,
    whatever it holds, and so is one the process cannot reach, and so
    is a link the project made elsewhere, and so is everything when
    .claude/skills is itself a link into the brain, since its entries
    are then the brain's own. In a skills folder shared outside the
    project the trouble is named and the removal is left to a person.
    The caller skips this pass when the skills folder could not be
    listed.
    """
    links = project / SKILL_LINKS
    found: list[Finding] = []

    homes = resolved(project / SKILL_HOMES)
    folder = resolved(links)
    if homes is None or folder is None or folder.is_relative_to(homes):
        return found

    try:
        entries = sorted(links.iterdir())
    except OSError:
        return found

    for link in entries:
        if link.name in wanted or not store.quietly(link.is_symlink) or not skill_gone(link):
            continue

        target = resolved(link)
        if target is None or not target.is_relative_to(homes):
            continue

        found.append(
            Finding(
                path=str(SKILL_LINKS / link.name),
                check="layout",
                message=SKILL_GONE + (SHARED_REMOVAL if shared else ""),
                repair=None if shared else remove_link(link),
            )
        )

    return found


def skill_gone(
    link: Path,
) -> bool:
    """Whether the link dangles, leading to nothing at all, as against to something the process cannot reach."""
    try:
        link.stat()
    except (FileNotFoundError, NotADirectoryError):
        return True
    except (OSError, RuntimeError, ValueError):
        return False

    return False


def remove_link(
    link: Path,
) -> Callable[[], None]:
    def repair() -> None:
        link.unlink()

    return repair


def skill_link_trouble(
    project: Path,
    name: str,
    shared: bool,
    kits: bool,
) -> Optional[Trouble]:
    """What is wrong with the named skill's link, or None when Claude Code loads this skill through it.

    What Claude Code needs is `.claude/skills/<name>/SKILL.md`, and it
    must be this skill: a link that resolves to the skill's own folder,
    or a real copy of its SKILL.md, the way a ZIP download or a checkout
    without symlinks leaves one. A link that lands elsewhere is dead or
    points at another skill and is remade, except in a skills folder
    shared outside the project, where a link or a real copy of this
    skill from another checkout is how the sharing works and passes. A
    real folder whose SKILL.md names a skill the stamp says the kit
    shipped, `kits`, but differs from the brain's is a copy that fell
    behind and is refreshed, since the folder is the kit's. Under a
    project skill's name, a kit name the stamp predates included, the
    same folder holds a person's own text, and it is left for them like
    a folder naming another skill, and like a real folder with no
    SKILL.md at all. A link can only be made when the skill itself is
    in the brain to point at, and a copy with nothing left to point at
    is all there is, so it passes.
    """
    link = project / SKILL_LINKS / name
    home = project / SKILL_HOMES / name
    at_link = SKILL_LINKS / name
    remake = make_skill_link(
        project=project,
        name=name,
    )

    if not store.readable_file(home / schema.SKILL_FILE):
        if store.readable_file(link / schema.SKILL_FILE):
            return None

        return Trouble(
            message=skill_missing(name),
            path=SKILL_HOMES / name,
        )

    if store.quietly(link.is_symlink):
        reached = reaches(
            link=link,
            home=home,
        )
        if reached:
            return None

        if shared and carries_skill(folder=link, name=name):
            return None

        return Trouble(
            message="the skill link does not reach the skill" + CANNOT_SEE,
            path=at_link,
            repair=remake,
        )

    if store.quietly(link.is_dir):
        if not store.readable_file(link / schema.SKILL_FILE):
            return Trouble(
                message=REAL_FOLDER_WITHOUT_SKILL,
                path=at_link,
            )

        copied = store.same_bytes(
            one=link / schema.SKILL_FILE,
            other=home / schema.SKILL_FILE,
        )
        if copied:
            return None

        if shared and carries_skill(folder=link, name=name):
            return None

        if kits and carries_skill(folder=link, name=name):
            return Trouble(
                message=STALE_COPY,
                path=at_link,
                repair=refresh_skill_copy(
                    home=home,
                    copy=link,
                ),
            )

        return Trouble(
            message=ANOTHER_SKILL,
            path=at_link,
        )

    if store.quietly(link.exists):
        return Trouble(
            message="something that is not a link sits where the skill link belongs" + CANNOT_SEE,
            path=at_link,
            repair=remake,
        )

    return Trouble(
        message="the skill link is missing" + CANNOT_SEE,
        path=at_link,
        repair=remake,
    )


def carries_skill(
    folder: Path,
    name: str,
) -> bool:
    """Whether the folder's SKILL.md is a skill of this name, by the name its frontmatter gives."""
    try:
        document = frontmatter.read(str(folder / schema.SKILL_FILE))
    except frontmatter.FrontmatterError:
        return False

    return document.get("name") == name


def refresh_skill_copy(
    home: Path,
    copy: Path,
) -> Callable[[], None]:
    """A repair that brings a real folder copy of a skill up to the brain's, for a checkout where a link is not an option."""

    def repair() -> None:
        shutil.copytree(home, copy, dirs_exist_ok=True)

    return repair


def reaches(
    link: Path,
    home: Path,
) -> bool:
    """Whether the link resolves to the skill's own folder."""
    target = resolved(link)

    return target is not None and target == resolved(home)


def check_style_copies(
    brain: store.Brain,
) -> list[Finding]:
    """Codex and Gemini CLI carry the chat style as copies of the Claude Code one.

    A copy is owed only where the project keeps that agent's folder. The
    source is read once, and each stale copy gets a repair that writes
    only itself.
    """
    if not style.expected(brain):
        return []

    try:
        wanted = style.source_body(brain)
    except style.StyleError as error:
        return [Finding(path=str(style.SOURCE), check="layout", message=str(error))]

    return [
        Finding(
            path=str(relative),
            check="layout",
            message="the chat style copy does not match the Claude Code one",
            repair=write_copy(target=style.project(brain) / relative, text=wanted),
        )
        for relative in style.stale_against(brain=brain, wanted=wanted)
    ]


def write_copy(
    target: Path,
    text: str,
) -> Callable[[], None]:
    def repair() -> None:
        style.write_one(target=target, text=text)

    return repair


def make_skill_link(
    project: Path,
    name: str,
) -> Callable[[], None]:
    """A repair that puts the named skill's symlink in place, clearing a dead link or a stray file first.

    A real folder is never removed here. It is somebody's, and the check
    leaves it for a person.
    """

    def repair() -> None:
        link = project / SKILL_LINKS / name
        link.parent.mkdir(parents=True, exist_ok=True)
        if not os.access(str(link.parent), os.W_OK):
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(link.parent))

        if link.is_symlink() or (link.exists() and not link.is_dir()):
            link.unlink()

        target = skill_target(
            project=project,
            link=link,
            name=name,
        )
        os.symlink(target, str(link))

    return repair


def skill_target(
    project: Path,
    link: Path,
    name: str,
) -> str:
    """The named skill's folder, relative to where the link really sits.

    On the shipped layout that is `../../.agents/skills/<name>`. When
    `.claude/skills` is itself a symlink the link lands elsewhere, and a
    target counted from there is the one that still resolves.
    """
    return os.path.relpath(str(project / SKILL_HOMES / name), str(link.parent.resolve()))


# Two agents read the front doors' `@` imports, each by its own rules.
# Each has a front door of its own and a folder a project keeps for it,
# and AGENTS.md is read by every agent whose folder the project keeps.
CLAUDE_CODE = "Claude Code"
GEMINI_CLI = "Gemini CLI"
OWN_DOORS: dict[str, tuple[str, str]] = {
    "CLAUDE.md": (CLAUDE_CODE, ".claude"),
    "GEMINI.md": (GEMINI_CLI, ".gemini"),
}
RULES = "AGENTS.md"
NO_RULES = "does not import AGENTS.md, so the rules never load"

# A path runs to the next whitespace in both agents, so a full stop or
# a comma glued to it becomes part of the name and the file never loads.
GLUED = ".,;:!?)"
ESCAPED_SPACE = "\\ "
EMPHASIS_END = re.compile(f"[*_]+[{re.escape(GLUED)}]*$")

# The markdown lexer both agents find code with, read line by line for
# the blocks and in order inside a paragraph for the spans and comments.
LINE = re.compile(
    r"^.*$",
    re.MULTILINE,
)
FENCE_OPEN = re.compile(r"(`{3,}|~{3,})(.*)$")
FENCE_CLOSE = re.compile(r"(`{3,}|~{3,})\s*$")
LIST_MARKER = re.compile(r"([-*+]|\d{1,9}[.)])(?:([ \t]+)|$)")
HEADING = re.compile(r"#{1,6}(?:[ \t]|$)")
SETEXT_UNDERLINE = re.compile(r"(?:=+|-+)[ \t]*$")
INLINE_OPEN = re.compile(r"<!--|(?<!\\)`+")
COMMENT_OPEN = "<!--"
COMMENT_CLOSE = "-->"
PARAGRAPH_BREAK = re.compile(r"\n[ \t\r]*\n")
NOT_A_LINE_END = re.compile(r"[^\n]")
CODE_INDENT = 4

# A stretch of a front door the lexer holds apart from its text, as its
# start, its end and whether it is code or an HTML comment.
Region = tuple[int, int, str]
IN_CODE = "code"
IN_COMMENT = "comment"


@dataclass(frozen=True)
class Blocks:
    """What the reading of the blocks finds: the code and comment regions, and the offsets where a block starts that no span or emphasis reaches into."""

    regions: list[Region]
    starts: list[int]


# What a front door that is not there costs, since the rules then never
# load in the agent that reads it.
MISSING: dict[str, str] = {
    RULES: "is not there, so no agent finds the rules, and the next upgrade writes it back",
    "CLAUDE.md": "is not there, so Claude Code loads no rules, and the next upgrade writes it back",
    "GEMINI.md": "is not there, so Gemini CLI loads no rules, and the next upgrade writes it back",
}


def check_front_doors(
    brain: store.Brain,
) -> list[Finding]:
    """The imports in the root files the agents read, each reaching a file, none glued to punctuation, and the rules imported.

    A slip here unloads the rulebook in two of the three agents with
    nothing said, which is what bit the kit once, so doctor reads the
    front doors the way the parsers do. CLAUDE.md is read only while the
    project keeps a .claude folder and GEMINI.md only while it keeps a
    .gemini folder, since a project without one runs no such agent. A
    front door that is a link to AGENTS.md is that file, read once under
    its own name. A front door that is not there is named, AGENTS.md
    always and the other two while their agent's folder is there.
    """
    found: list[Finding] = []
    project = brain.root.parent

    for name in schema.FRONT_DOORS:
        own = OWN_DOORS.get(name)
        kept = own is None or keeps(
            project=project,
            folder=own[1],
        )
        if not kept:
            continue

        spelled = spelling_of(
            folder=project,
            name=name,
        )
        if spelled is None:
            found.append(
                import_finding(
                    name=name,
                    message=MISSING[name],
                )
            )
            continue

        if spelled != name:
            found.append(
                import_finding(
                    name=name,
                    message=f"is spelled {spelled}, and only {name} is read, so rename it",
                )
            )
            continue

        door = resolved(project / name)
        if name != RULES and door is not None and door == resolved(project / RULES):
            continue

        try:
            text = (project / name).read_bytes().decode(
                encoding="utf-8-sig",
                errors="replace",
            )
        except OSError as error:
            found.append(
                import_finding(
                    name=name,
                    message=f"cannot be read: {store.explain(error)}",
                )
            )
            continue

        found += import_slips(
            name=name,
            text=text,
            project=project,
        )

    return found


def spelling_of(
    folder: Path,
    name: str,
) -> Optional[str]:
    """The name an entry goes by in the folder, another case of it when only that is there, or None when neither is.

    The folder is listed rather than the entry opened, since a file
    system blind to case opens Claude.md for CLAUDE.md where Linux does
    not, and the answer has to be the same on every machine. A folder
    that cannot be listed leaves the read to say what is wrong.
    """
    try:
        entries = sorted(os.listdir(folder))
    except OSError:
        return name

    if name in entries:
        return name

    for entry in entries:
        if entry.casefold() == name.casefold():
            return entry

    return None


def import_slips(
    name: str,
    text: str,
    project: Path,
) -> list[Finding]:
    """The import findings of one front door's text, read against the project's files, so the upgrade can ask the same of the kit's own text.

    The door is judged once by each agent that reads it here. Each
    agent's findings stay whole, a slip written twice named twice, and a
    later agent's list drops only as many copies of a message as an
    earlier one already gave, so a slip both agents see is named once.
    The upgrade counts on that, telling the kit's slips in AGENTS.md
    from the project's by how many of each there are. The code and the
    comments are found once, since both agents find them with the same
    lexer.
    """
    hidden = hidden_regions(text)
    found: list[Finding] = []

    for agent in readers_of(
        name=name,
        project=project,
    ):
        slips = READERS[agent](
            name=name,
            text=text,
            hidden=hidden,
            project=project,
        )
        named = Counter(known.message for known in found)
        for slip in slips:
            if named[slip.message] > 0:
                named[slip.message] -= 1
                continue

            found.append(slip)

    return found


def readers_of(
    name: str,
    project: Path,
) -> list[str]:
    """The agents that read the front door in this project, in the order doctor judges it.

    CLAUDE.md and GEMINI.md each belong to their own agent. AGENTS.md
    is read by every agent whose folder the project keeps, and by Claude
    Code's rules when it keeps neither folder.
    """
    own = OWN_DOORS.get(name)
    if own is not None:
        return [own[0]]

    kept = [
        agent
        for agent, folder in OWN_DOORS.values()
        if keeps(
            project=project,
            folder=folder,
        )
    ]

    return kept or [CLAUDE_CODE]


def keeps(
    project: Path,
    folder: str,
) -> bool:
    """Whether the project keeps the agent's folder, and so runs that agent."""
    return store.quietly((project / folder).is_dir)


# Claude Code opens an import at an `@` at the start or after whitespace
# of the lexer's text, or right after an emphasis marker that opens a
# pair, and a backslash before a space carries the path on. Any other
# backslash ends the path.
CLAUDE_IMPORT = re.compile(r"(?<![^\s*_])@((?:[^\s\\]|\\ )+)")
MARKERS = "*_"
MARKER_RUN = re.compile(r"\*+|_+")
TRAILING_RUN = re.compile(f"(\\*+|_+)[{re.escape(GLUED)}]*$")
WHITESPACE = re.compile(r"\s")


@dataclass(frozen=True)
class Opened:
    """One import Claude Code opens: the path it reads, the token as written up to the next whitespace, whether an emphasis marker stands right before it, and whether a backslash cut the path short."""

    path: str
    written: str
    after_marker: bool
    cut_at_backslash: bool


def claude_opens(
    prose: str,
    starts: list[int],
) -> list[Opened]:
    """The imports Claude Code opens in the lexer's text, emphasis paired the way markdown pairs it.

    The runs are paired within their block, so an opener on one line of
    a paragraph pairs with a closer on a later line of it, and never
    across a blank line, a list item's start, a heading or a fence. Only
    a run that `paired_runs` uses up as an opener lets an `@` right
    after it open an import, and only a run it uses up as a closer comes
    off the end of a path. A run left without a partner is plain text,
    so a lone marker opens nothing and stays in the path.
    """
    opened: list[Opened] = []

    for block in blocks_of(
        prose=prose,
        starts=starts,
    ):
        pairs = paired_runs(block)

        for match in CLAUDE_IMPORT.finditer(block):
            after_marker = match.start() > 0 and block[match.start() - 1] in MARKERS
            if after_marker and match.start() not in pairs.opener_ends:
                continue

            path = match.group(1)
            tail = TRAILING_RUN.search(path)
            if tail is not None and match.start(1) + tail.start() in pairs.closer_starts:
                path = path[: tail.start()]

            rest = WHITESPACE.split(
                string=block[match.end() :],
                maxsplit=1,
            )[0]
            opened.append(
                Opened(
                    path=path,
                    written=match.group(1) + rest,
                    after_marker=after_marker,
                    cut_at_backslash=bool(rest),
                )
            )

    return opened


def blocks_of(
    prose: str,
    starts: list[int],
) -> list[str]:
    """The prose cut into its blocks, at each blank line and where `starts` says a block begins, each block's lines joined again."""
    found: list[str] = []
    current: list[str] = []
    begins = set(starts)

    for match in LINE.finditer(prose):
        line = match.group()
        if current and (match.start() in begins or not line.strip()):
            found.append("\n".join(current))
            current = []

        if line.strip():
            current.append(line)

    if current:
        found.append("\n".join(current))

    return found


@dataclass(frozen=True)
class Pairs:
    """The marker runs in one block that markdown pairs as emphasis, by where each opener ends and where each closer starts."""

    opener_ends: frozenset[int]
    closer_starts: frozenset[int]


def paired_runs(
    block: str,
) -> Pairs:
    """The emphasis in one block, paired the way markdown pairs it, a line end counting as whitespace.

    A run is a maximal stretch of `*` or of `_`. It can open when a
    character that is not whitespace follows it, and it can close when
    one stands before it, while a run of `_` with a letter or a digit on
    both sides does neither, since it sits inside a word. The runs are
    walked from left to right and the ones that can open are kept. A run
    that can close pairs with the nearest kept run of the same marker
    and length, both used up, the closer tried before the opener when a
    run can do both. A run that finds no partner stays plain text.
    """
    kept: list[re.Match] = []
    opener_ends: set[int] = set()
    closer_starts: set[int] = set()

    for run in MARKER_RUN.finditer(block):
        can_open, can_close = flanks_of(
            block=block,
            run=run,
        )
        partner = None
        if can_close:
            partner = nearest_partner(
                kept=kept,
                run=run,
            )

        if partner is not None:
            kept.remove(partner)
            opener_ends.add(partner.end())
            closer_starts.add(run.start())
            continue

        if can_open:
            kept.append(run)

    return Pairs(
        opener_ends=frozenset(opener_ends),
        closer_starts=frozenset(closer_starts),
    )


def nearest_partner(
    kept: list[re.Match],
    run: re.Match,
) -> Optional[re.Match]:
    """The last kept opener of the same marker and length as the run, or None when there is none."""
    for opener in reversed(kept):
        if opener.group() == run.group():
            return opener

    return None


def flanks_of(
    block: str,
    run: re.Match,
) -> tuple[bool, bool]:
    """Whether the marker run can open emphasis and whether it can close it, by the characters on either side."""
    before = block[run.start() - 1] if run.start() > 0 else ""
    after = block[run.end()] if run.end() < len(block) else ""
    if run.group().startswith("_") and before.isalnum() and after.isalnum():
        return False, False

    return after != "" and not after.isspace(), before != "" and not before.isspace()


def claude_code_slips(
    name: str,
    text: str,
    hidden: Blocks,
    project: Path,
) -> list[Finding]:
    """The front door's imports as Claude Code reads them.

    It runs on the markdown lexer's text, so code and HTML comments are
    skipped and emphasis markers around an import are not part of it,
    judged by pairs as `claude_opens` says. An `@` at the start or after
    whitespace opens an import, the path runs to the next whitespace, a
    backslash before a space keeps the space in the path, any other
    backslash ends it, and the path is cut at its first `#`. A file that
    is missing is skipped without a word, which is why doctor names it.
    """
    found: list[Finding] = []
    loads_rules = False
    prose = blanked(
        text=text,
        regions=hidden.regions,
        kinds=(IN_CODE, IN_COMMENT),
    )

    for opened in claude_opens(
        prose=prose,
        starts=hidden.starts,
    ):
        if opened.cut_at_backslash:
            backslashed = is_import(
                bare=with_slashes(bare_of(opened.written)),
                project=project,
            )
            if backslashed:
                found.append(
                    import_finding(
                        name=name,
                        message=f"imports @{opened.written} with a backslash in its path, which ends the path there, so write it with slashes",
                    )
                )
            continue

        path = opened.path.partition("#")[0]
        wanted = path.replace(
            ESCAPED_SPACE,
            " ",
        )
        imported = is_import(
            bare=bare_of(wanted),
            project=project,
        )
        if not imported:
            continue

        if path[-1] in GLUED:
            found.append(
                glued_on(
                    name=name,
                    path=path,
                )
            )
            continue

        if path.startswith("~"):
            continue

        if path.startswith("/"):
            found.append(
                from_root(
                    name=name,
                    path=path,
                )
            )
            continue

        back_in = climbs_back_in(
            path=wanted,
            project=project,
        )
        if back_in:
            found.append(
                back_in_by_name(
                    name=name,
                    path=path,
                )
            )
            continue

        spelled = spelled_on_disk(
            base=project,
            path=wanted,
        )
        loads_rules = loads_rules or reaches_rules(
            target=project / (spelled or wanted),
            project=project,
        )
        if spelled is None:
            found.append(
                import_finding(
                    name=name,
                    message=f"imports @{path}, which is not there",
                )
            )
        elif spelled != wanted:
            found.append(
                spelled_otherwise(
                    name=name,
                    path=path,
                    spelled=spelled,
                )
            )

    if name != RULES and not loads_rules:
        found.append(
            import_finding(
                name=name,
                message=NO_RULES,
            )
        )

    return found


# Gemini CLI opens an import only at an `@` at the start or after
# whitespace of the raw text, and takes every character up to the next
# whitespace. The path Claude Code would read from the same `@` shows
# where Gemini CLI stopped at an escaped space.
GEMINI_IMPORT = re.compile(r"(?<!\S)@(\S+)")
ESCAPED_PATH = re.compile(r"(?:[^\s\\]|\\ )+")


def gemini_cli_slips(
    name: str,
    text: str,
    hidden: Blocks,
    project: Path,
) -> list[Finding]:
    """The front door's imports as Gemini CLI reads them.

    It skips code regions and reads the rest raw, HTML comments
    included. Only an `@` at the start or after whitespace opens an
    import, so one right after `*` or `_` opens nothing, and the path is
    every character up to the next whitespace. A backslash ends it at
    the space and is part of the name anywhere else, a `#` fragment
    stays in the name it looks for, and emphasis markers after the path
    are part of the name.
    """
    found: list[Finding] = []
    loads_rules = False
    prose = blanked(
        text=text,
        regions=hidden.regions,
        kinds=(IN_CODE,),
    )

    for match in GEMINI_IMPORT.finditer(prose):
        path = match.group(1)
        escaped = ESCAPED_PATH.match(
            string=prose,
            pos=match.start(1),
        )
        if escaped is not None and ESCAPED_SPACE in escaped.group():
            escaped_import = is_import(
                bare=bare_of(escaped.group()).replace(
                    ESCAPED_SPACE,
                    " ",
                ),
                project=project,
            )
            if escaped_import:
                found.append(
                    import_finding(
                        name=name,
                        message=f"imports @{escaped.group()} with an escaped space, which Gemini CLI does not read, so the path ends at the backslash there",
                    )
                )
                continue

        if "\\" in path:
            backslashed = is_import(
                bare=with_slashes(bare_of(path)),
                project=project,
            )
            if backslashed:
                found.append(
                    import_finding(
                        name=name,
                        message=f"imports @{path} with a backslash in its path, which Gemini CLI reads as part of the name, so write it with slashes",
                    )
                )
                continue

        imported = is_import(
            bare=bare_of(path),
            project=project,
        )
        if not imported:
            continue

        if EMPHASIS_END.search(path):
            found.append(
                import_finding(
                    name=name,
                    message=f"imports @{path} with emphasis markers glued on, which Gemini CLI reads as part of the name",
                )
            )
            continue

        if path[-1] in GLUED:
            found.append(
                glued_on(
                    name=name,
                    path=path,
                )
            )
            continue

        if path.startswith("~"):
            continue

        if path.startswith("/"):
            found.append(
                from_root(
                    name=name,
                    path=path,
                )
            )
            continue

        back_in = climbs_back_in(
            path=path,
            project=project,
        )
        if back_in:
            found.append(
                back_in_by_name(
                    name=name,
                    path=path,
                )
            )
            continue

        spelled = spelled_on_disk(
            base=project,
            path=path,
        )
        loads_rules = loads_rules or reaches_rules(
            target=project / (spelled or path),
            project=project,
        )
        if spelled is None and "#" in path:
            found.append(
                import_finding(
                    name=name,
                    message=f"imports @{path}, which Gemini CLI looks for with the fragment in its name, so it is not there",
                )
            )
        elif spelled is None:
            found.append(
                import_finding(
                    name=name,
                    message=f"imports @{path}, which is not there",
                )
            )
        elif spelled != path:
            found.append(
                spelled_otherwise(
                    name=name,
                    path=path,
                    spelled=spelled,
                )
            )

    if CLAUDE_CODE in readers_of(
        name=name,
        project=project,
    ):
        found += unopened_after_markers(
            name=name,
            text=text,
            hidden=hidden,
            project=project,
        )

    if name != RULES and not loads_rules:
        found.append(
            import_finding(
                name=name,
                message=NO_RULES,
            )
        )

    return found


def unopened_after_markers(
    name: str,
    text: str,
    hidden: Blocks,
    project: Path,
) -> list[Finding]:
    """The imports Claude Code opens right after an emphasis marker, which Gemini CLI never opens, named in a file both agents read."""
    found: list[Finding] = []
    prose = blanked(
        text=text,
        regions=hidden.regions,
        kinds=(IN_CODE, IN_COMMENT),
    )

    for opened in claude_opens(
        prose=prose,
        starts=hidden.starts,
    ):
        if not opened.after_marker or opened.cut_at_backslash:
            continue

        path = opened.path.partition("#")[0]
        imported = is_import(
            bare=bare_of(path),
            project=project,
        )
        if imported:
            found.append(
                import_finding(
                    name=name,
                    message=f"imports @{path} right after an emphasis marker, where Gemini CLI opens no import",
                )
            )

    return found


READERS: dict[str, Callable[..., list[Finding]]] = {
    CLAUDE_CODE: claude_code_slips,
    GEMINI_CLI: gemini_cli_slips,
}


def import_finding(
    name: str,
    message: str,
) -> Finding:
    return Finding(
        path=name,
        check="import",
        message=message,
    )


def glued_on(
    name: str,
    path: str,
) -> Finding:
    return import_finding(
        name=name,
        message=f"imports @{path} with the punctuation glued on, so the parser looks for that name and the file never loads, end the path with a space or a line break",
    )


def from_root(
    name: str,
    path: str,
) -> Finding:
    """An import that opens with a slash is never looked up, since it is there on one machine and missing on the next, and so never counts as the rules import."""
    return import_finding(
        name=name,
        message=f"imports @{path} from the root of the disk, which is there on one machine and missing on the next, so write it from the project's folder",
    )


def back_in_by_name(
    name: str,
    path: str,
) -> Finding:
    """An import that climbs out of the project's folder and comes back in by its name is never looked up, since another checkout of the project may go by another name, and so never counts as the rules import."""
    return import_finding(
        name=name,
        message=f"imports @{path} out of the project's folder and back in by its name, which another checkout may not share, so write it from the project's folder",
    )


def climbs_back_in(
    path: str,
    project: Path,
) -> bool:
    """Whether the path climbs above the project's folder and comes back into it by retracing the names it climbed out of.

    It is decided from the path's parts and the names of the project's
    own folder and its parents, never from the disk. The path is walked
    over those names, a `..` stepping up one and a name stepping down
    into that name, and it comes back in only when, after climbing above
    the project, its steps down lead to the project's own folder again.
    A path that climbs out and stays out is looked up like any other.
    """
    home = list(project.parts)
    trail = list(home)
    climbed_out = False

    for part in path.split("/"):
        if part in ("", "."):
            continue

        if part == "..":
            if len(trail) > 1:
                trail.pop()
            climbed_out = climbed_out or len(trail) < len(home)
            continue

        trail.append(part)
        if climbed_out and trail == home:
            return True

    return False


def spelled_otherwise(
    name: str,
    path: str,
    spelled: str,
) -> Finding:
    return import_finding(
        name=name,
        message=f"imports @{path}, which is spelled {spelled}, so it loads only where the file system ignores case",
    )


def reaches_rules(
    target: Path,
    project: Path,
) -> bool:
    """Whether an import's target resolves to the project's AGENTS.md, however its path is spelled."""
    rules = resolved(project / RULES)

    return rules is not None and resolved(target) == rules


def bare_of(
    token: str,
) -> str:
    """The token with trailing emphasis markers, its fragment and trailing punctuation set aside, the form that says whether it is an import at all."""
    return EMPHASIS_END.sub(
        repl="",
        string=token,
    ).partition("#")[0].rstrip(GLUED)


def with_slashes(
    token: str,
) -> str:
    """The token with an escaped space read as a space and every other backslash turned to a slash, the path its writer meant."""
    spaced = token.replace(
        ESCAPED_SPACE,
        " ",
    )

    return spaced.replace(
        "\\",
        "/",
    )


def is_import(
    bare: str,
    project: Path,
) -> bool:
    """Whether a token, its trailing punctuation set aside, is an import and not a mention.

    It is one when it opens with `~`, `/`, `./` or `../`, when it names a
    markdown file, or when its target is there under any spelling of its
    case. Anything else that leads nowhere, a handle, a package or a
    decorator, is a mention.
    """
    if bare.startswith(("~", "/", "./", "../")) or bare.casefold().endswith(".md"):
        return True

    spelled = spelled_on_disk(
        base=project,
        path=bare,
    )

    return spelled is not None


def spelled_on_disk(
    base: Path,
    path: str,
) -> Optional[str]:
    """The path as the files spell it, or None when it leads to no file.

    It is walked part by part from `base`, and each part is looked up in
    its folder's listing, by its exact name first and then by its case.
    Asking the file system for the whole path would follow the machine's
    case rule, and the answer has to be the same on every machine. A
    path from the root of the disk never comes here, for the same
    reason.
    """
    current = base
    spelled: list[str] = []

    for part in path.split("/"):
        if part in ("", ".", ".."):
            spelled.append(part)
            current = current / part
            continue

        entry = spelling_of(
            folder=current,
            name=part,
        )
        if entry is None:
            return None

        spelled.append(entry)
        current = current / entry

    if not store.readable_file(current):
        return None

    return "/".join(spelled)


def hidden_regions(
    text: str,
) -> Blocks:
    """Where the code and the HTML comments sit, the blocks first and then the spans and comments inside what the blocks leave, with the offsets where a block starts."""
    blocks = block_regions(text)
    rest = blanked(
        text=text,
        regions=blocks.regions,
        kinds=(IN_CODE, IN_COMMENT),
    )
    spans = inline_regions(
        text=rest,
        starts=blocks.starts,
    )

    return Blocks(
        regions=blocks.regions + spans,
        starts=blocks.starts,
    )


def block_regions(
    text: str,
) -> Blocks:
    """Where the code blocks and the HTML comment blocks sit, read line by line in the order they come.

    A fence runs to the closer that matches it or to the end of the
    text, and a comment opened at the start of a line runs to its
    `-->`, so a fence inside a comment opens nothing. A fence opens on a
    list item's own line too, after its marker. A line indented four
    columns past its margin is code when no paragraph is open, which is
    at the start and after a blank line, a heading of either kind, a
    fence or a comment. The margin is where the text of the list item
    the line sits in starts, so a list item's paragraph indented four
    columns is read, and a heading or a fence left of that margin ends
    every list that is open. Each list item and each heading starts a
    block of its own, which the spans read later never reach into.
    """
    regions: list[Region] = []
    starts: list[int] = []
    items: list[int] = []
    fence: Optional[tuple[str, int]] = None
    in_comment = False
    after_break = True

    for match in LINE.finditer(text):
        line = match.group()
        start, end = match.span()

        if in_comment:
            close = line.find(COMMENT_CLOSE)
            if close == -1:
                regions.append((start, end, IN_COMMENT))
                continue

            regions.append((start, start + close + len(COMMENT_CLOSE), IN_COMMENT))
            in_comment = False
            after_break = True
            continue

        if fence is not None:
            regions.append((start, end, IN_CODE))
            closes = fence_closes(
                fence=fence,
                line=line,
            )
            if closes:
                fence = None
                after_break = True
            continue

        if not line.strip():
            after_break = True
            continue

        indent = indent_of(line)
        bare = line.lstrip()
        marker = LIST_MARKER.match(bare)
        heading = HEADING.match(bare) is not None
        underline = not after_break and SETEXT_UNDERLINE.match(bare) is not None
        opener = fence_opener(bare)
        interrupts = marker is not None or heading or underline or opener is not None
        if after_break or interrupts:
            while items and indent < items[-1]:
                items.pop()

        margin = items[-1] if items else 0
        if indent - margin >= CODE_INDENT:
            if after_break:
                regions.append((start, end, IN_CODE))
            continue

        if opener is not None:
            fence = (opener.group(1), margin)
            regions.append((start, end, IN_CODE))
            continue

        if bare.startswith(COMMENT_OPEN):
            opened = end - len(bare)
            close = bare.find(
                COMMENT_CLOSE,
                len(COMMENT_OPEN),
            )
            in_comment = close == -1
            closed = end if in_comment else opened + close + len(COMMENT_CLOSE)
            regions.append((opened, closed, IN_COMMENT))
            after_break = True
            continue

        if underline:
            after_break = True
            starts.append(end + 1)
            continue

        if heading:
            after_break = True
            starts.append(start)
            starts.append(end + 1)
            continue

        if marker is not None:
            item_margin = indent + len(marker.group(1)) + item_gap(marker.group(2))
            items.append(item_margin)
            starts.append(start)

            item_fence = fence_opener(bare[marker.end() :])
            if item_fence is not None:
                fence = (item_fence.group(1), item_margin)
                regions.append((start, end, IN_CODE))
                continue

        after_break = False

    return Blocks(
        regions=regions,
        starts=starts,
    )


def fence_opener(
    bare: str,
) -> Optional[re.Match]:
    """The fence the text opens with, or None, since a backtick fence's info string holds no backtick."""
    opener = FENCE_OPEN.match(bare)
    if opener is None:
        return None

    if opener.group(1).startswith("`") and "`" in opener.group(2):
        return None

    return opener


def fence_closes(
    fence: tuple[str, int],
    line: str,
) -> bool:
    """Whether the line closes the fence, a run of the same character at least as long, indented less than four columns past the fence's margin."""
    marker, margin = fence
    closer = FENCE_CLOSE.match(line.lstrip())
    if closer is None or indent_of(line) - margin >= CODE_INDENT:
        return False

    return closer.group(1)[0] == marker[0] and len(closer.group(1)) >= len(marker)


def item_gap(
    spacing: Optional[str],
) -> int:
    """How far past its marker a list item's text starts, one column when the marker ends the line or the gap would make its text code."""
    if not spacing or len(spacing) > CODE_INDENT:
        return 1

    return len(spacing)


def indent_of(
    line: str,
) -> int:
    """How many columns the line's leading whitespace spans, a tab reaching the next multiple of four."""
    columns = 0
    for character in line:
        if character == " ":
            columns += 1
        elif character == "\t":
            columns += CODE_INDENT - columns % CODE_INDENT
        else:
            break

    return columns


def inline_regions(
    text: str,
    starts: list[int],
) -> list[Region]:
    """Where the code spans and the HTML comments inside the blocks sit, read in the order they come.

    A run of backticks is closed by the next run as long and a comment
    by its `-->`, each inside its block, and whichever opens first holds
    what follows, so a backtick inside a comment opens no span. A block
    ends at a blank line or where `starts` says another begins. An
    opener with no closer in its block is plain text, and the reading
    moves on past it.
    """
    regions: list[Region] = []
    position = 0

    while True:
        opener = INLINE_OPEN.search(
            string=text,
            pos=position,
        )
        if opener is None:
            return regions

        closed = inline_end(
            text=text,
            opener=opener,
            starts=starts,
        )
        if closed is None:
            position = opener.end()
            continue

        kind = IN_COMMENT if opener.group() == COMMENT_OPEN else IN_CODE
        regions.append((opener.start(), closed, kind))
        position = closed


def inline_end(
    text: str,
    opener: re.Match,
    starts: list[int],
) -> Optional[int]:
    """Where the span or the comment the opener starts ends, or None when nothing closes it inside its block."""
    if opener.group() == COMMENT_OPEN:
        pattern = re.escape(COMMENT_CLOSE)
    else:
        pattern = f"(?<!`){opener.group()}(?!`)"

    closer = re.compile(pattern).search(
        string=text,
        pos=opener.end(),
    )
    if closer is None or PARAGRAPH_BREAK.search(text[opener.end() : closer.start()]):
        return None

    if any(opener.start() < start <= closer.start() for start in starts):
        return None

    return closer.end()


def blanked(
    text: str,
    regions: list[Region],
    kinds: tuple[str, ...],
) -> str:
    """The text with the regions of the given kinds turned to spaces, its line ends and so its offsets kept."""
    kept: list[str] = []
    cursor = 0

    for start, end, kind in sorted(regions):
        if kind not in kinds:
            continue

        kept.append(text[cursor:start])
        kept.append(
            NOT_A_LINE_END.sub(
                repl=" ",
                string=text[start:end],
            )
        )
        cursor = end

    kept.append(text[cursor:])

    return "".join(kept)


CHECKS: tuple[Callable[[store.Brain], list[Finding]], ...] = (
    check_leftovers,
    check_strays,
    check_frontmatter,
    check_names,
    check_kinds,
    check_fields,
    check_sections,
    check_steps,
    check_filenames,
    check_links,
    check_indexes,
    check_style,
    check_hygiene,
    check_layout,
    check_front_doors,
)
