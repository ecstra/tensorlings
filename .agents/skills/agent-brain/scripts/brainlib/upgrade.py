"""Bringing a project to the kit, a port when it has no brain and an upgrade when it has one.

The kit is a checkout of agent-setup, by default the one this tool runs
from. The project is the folder the command runs in. Both commands
install the kit's own files, keep everything the project wrote, carry
its section 0 into AGENTS.md, and end with doctor --fix. Between the two
sits the migration table, one step per layout change the kit has made,
each gated on the tree it changes, while the version stamp names the
release the project is on.
"""

import codecs
import json
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol, TextIO, Union
from urllib.parse import unquote

from . import doctor, frontmatter, schema, sections, store, style

FOLDER = ".agents"
OLD_FOLDER = ".agent"
VERSION = schema.VERSION_FILE

FRONT_DOORS = schema.FRONT_DOORS
OLD_LAYOUT_DOORS = ("CLAUDE.md", "AGENTS.md", "GEMINI.md")
ROOT_FILES = schema.ROOT_FILES
AGENT_FOLDERS = store.NEAR_ROOTS[1:]

# The hook configs are merged into what the project has, so its own hooks
# stay beside the kit's. Everything else the kit ships is copied whole.
MERGED = ("settings.json", "hooks.json")
NOT_SHIPPED = ("settings.local.json",)

# How many local origins the walk from a checkout to the repositories it
# comes from follows, two git calls each, more than any chain of local
# clones a person keeps.
ORIGIN_HOPS = 8

# The kit's code is never read by hand in a project, since the skill
# sheet and its reference say what every command does, so the port
# writes Claude Code the rules that refuse the read, which hold Edit
# and Write off the same files too, and those the upgrade replaces
# whole. The folder is named by both paths Claude Code reaches it
# through, the brain's and the skill link's, and the leading slash
# anchors each at the settings file's project, not at the filesystem
# root. A deny rule on Read is the kit's when it names that folder at
# the project, by either of those paths or the brain's old name, so a
# release that rewords a rule replaces the earlier one the way a kit
# hook group is replaced, while a rule for a copy kept elsewhere is the
# project's own and stays.
CLAUDE_SETTINGS = Path(".claude") / "settings.json"
READ_GUARDS = (
    f"Read(/{FOLDER}/{schema.SKILLS}/agent-brain/scripts/**)",
    f"Read(/{doctor.SKILL_LINKS.as_posix()}/agent-brain/scripts/**)",
)
KIT_RULE = re.compile(
    r"^Read\(/(?:"
    + "|".join(
        re.escape(folder)
        for folder in (
            f"{FOLDER}/{schema.SKILLS}",
            f"{OLD_FOLDER}/{schema.SKILLS}",
            doctor.SKILL_LINKS.as_posix(),
        )
    )
    + r")/agent-brain/scripts(?:/.*)?\)$"
)

# A hook group is the kit's when its command is one of two shapes with
# plain arguments after it: a call of a script under the kit's own skill
# folder, by either name of the brain, anchored at the project root the
# way each agent allows or not at all, which is what the kit shipped
# before the launcher, or python3 -c with a launcher text the kit has
# shipped, every version of it listed. A command that wraps such a
# call, or calls a copy kept elsewhere, is the project's own.
#
# The launcher is the one line every shipped hook command runs. It finds
# the hook script from the project folder the agent names, or by walking
# up from the session's own folder where none is named, hands it the
# event, and does nothing where there is no script, so a checkout
# without the kit never blocks a prompt or a write. Run through -c,
# Python puts the session's folder first on the import path, where a
# file named like a standard module would shadow it, so that entry goes
# before anything is imported.
LAUNCHER = (
    "import sys;"
    "'' in sys.path and sys.path.remove('');"
    "import os,pathlib,runpy;"
    "h='.agents/skills/agent-brain/scripts/hooks.py';"
    "c=pathlib.Path(os.environ.get('CLAUDE_PROJECT_DIR') or os.environ.get('GEMINI_PROJECT_DIR') or os.getcwd());"
    "p=next((d/h for d in [c,*c.parents] if (d/h).is_file()),None);"
    "p or sys.exit(0);"
    "sys.path.insert(0,str(p.parent));"
    "sys.argv[0]=str(p);"
    "runpy.run_path(str(p),run_name='__main__')"
)
# the anchors of the kit's commands before the launcher
ROOT_ANCHORS = (
    r'"\$CLAUDE_PROJECT_DIR"/',
    r"\$CLAUDE_PROJECT_DIR/",
    r'"\$GEMINI_PROJECT_DIR"/',
    r"\$GEMINI_PROJECT_DIR/",
)
# every launcher text the kit has shipped, the current one first, since a
# project ported under an earlier text upgrades by that text, so a change
# to the launcher appends the text it replaces here, a port from any
# commit being possible. The second is the first text, before the import
# path fix.
LAUNCHERS = (
    LAUNCHER,
    LAUNCHER.replace("import sys;'' in sys.path and sys.path.remove('');import os,pathlib,runpy;", "import os,pathlib,runpy,sys;"),
)
KIT_HOOK = re.compile(
    r"^python3 (?:" + "|".join(ROOT_ANCHORS) + r")?\.agents?/skills/agent-brain/scripts/[\w-]+\.py(?: [\w.-]+)*$"
    r'|^python3 -c "(?:' + "|".join(re.escape(text) for text in LAUNCHERS) + r')"(?: [\w.-]+)*$'
)
IGNORE_FILE = schema.IGNORE_FILE
IGNORE_HEADING = "# The agent kit's own ignores."
BOM = codecs.BOM_UTF8.decode("utf-8")
# a carriage return with or without a newline after it, the line ends
# text mode reads as a newline
OTHER_LINE_ENDS = re.compile(r"\r\n?")

# The project's own section, whatever it was called before, takes the
# heading the rulebook names. In the kit's AGENTS.md it ends at the
# comment line that says where the kit's words start, so nothing is
# told apart by the kit's wording, which changes between releases.
SECTION_ZERO_HEADING = "## 0. Project Rules"
KIT_MARK = "<!-- The kit's words start here. Section 0 goes above this line. -->"
MARK = re.compile(re.escape(KIT_MARK))
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
IMPORT_LINE = re.compile(r"^[ \t]*@\S+[ \t]*$", re.MULTILINE)

# Before 1.1.0 the kit's AGENTS.md carried no marker line, so in a file
# that kit wrote, section 0 ends where that release's first paragraph
# stood. The paragraph is history and never changes, which is what makes
# it safe to match, unlike the kit's current wording.
FIRST_KIT_WORDS = (
    "The rules for working in this repo are in @.agents/SYSTEM.md and every\n"
    "agent works by them. Claude Code and Gemini CLI pull that file into\n"
    "context on their own. Codex gets it from its session start hook. Anything\n"
    "else that reads this file as plain text opens it first.\n"
)
FIRST_KIT = re.compile(re.escape(FIRST_KIT_WORDS))


class UpgradeError(Exception):
    """The port or upgrade was refused, with the reason in the message."""


@dataclass
class Report:
    """What a port or upgrade did, one line each, in the order it happened.

    Given a stream, each line goes out the moment it is done, so a run
    that fails part way leaves the record of what it changed on screen.
    """

    out: Optional[TextIO] = None
    lines: list[str] = field(default_factory=list)
    findings: list[doctor.Finding] = field(default_factory=list)
    kits_own: list[doctor.Finding] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)

    def done(
        self,
        text: str,
    ) -> None:
        self.lines.append(text)
        if self.out is not None:
            print(text, file=self.out)

    def tail(self) -> str:
        """The doctor's answer and what is left to file, after the lines."""
        parts: list[str] = []

        if self.findings:
            parts.append("")
            parts.append("Doctor still finds:")
            parts += [f"  {finding.line()}" for finding in self.findings]
        else:
            parts.append("")
            parts.append("Doctor is clean.")

        if self.kits_own:
            parts.append("")
            parts.append("In the kit's own files, for the kit to put right, not counted here:")
            parts += [f"  {finding.line()}" for finding in self.kits_own]

        if self.candidates:
            parts.append("")
            parts.append("Left for the agent to file through brain new, never by hand:")
            parts += [f"  {candidate}" for candidate in self.candidates]

        return "\n".join(parts) + "\n"

    def render(self) -> str:
        return "\n".join(self.lines) + "\n" + self.tail()


@dataclass
class Kit:
    """A checkout of agent-setup, the source of everything the kit owns."""

    root: Path

    @property
    def brain(self) -> Path:
        return self.root / FOLDER

    def version(self) -> str:
        stamp = self.brain / VERSION
        if not store.readable_file(stamp):
            raise UpgradeError(f"{self.root} is not a kit checkout, it carries no {FOLDER}/{VERSION}")

        return read_version(stamp)

    def brain_files(self) -> list[Path]:
        """The kit's own files under .agents/, relative to it, the indexes left out since the tool builds those, and the tool's tests too.

        The skills are the checkout's own, every folder under its skills/
        that carries a SKILL.md, since the checkout may be a newer release
        than the tool running the install and ship a skill it never heard
        of.
        """
        found = [Path(name) for name in schema.KIT_FILES if name != schema.INDEX]
        found += sorted(path.relative_to(self.brain) for path in (self.brain / "spec").glob("*.md"))
        skills = store.skill_folders(self.brain)
        found += sorted(
            path.relative_to(self.brain)
            for path in (self.brain / schema.SKILLS).rglob("*")
            if ships(path) and path.relative_to(self.brain / schema.SKILLS).parts[0] in skills and "tests" not in path.relative_to(self.brain / schema.SKILLS).parts
        )
        found.append(Path(VERSION))

        return found

    def agent_files(self) -> list[Path]:
        """The hook configs and the chat style source under the agent folders, relative to the kit root.

        A symlink is the doctor's to make, and so are the chat style
        copies, from the source, so a copy the project made read only is
        replaced rather than written through. A personal settings file
        never ships.
        """
        found: list[Path] = []

        for folder in AGENT_FOLDERS:
            for path in sorted((self.root / folder).rglob("*")):
                relative = path.relative_to(self.root)
                if ships(path) and path.name not in NOT_SHIPPED and relative not in style.COPIES:
                    found.append(relative)

        return found

    def text(
        self,
        relative: Path,
    ) -> str:
        """A kit file the run composes with, read as text."""
        try:
            return (self.root / relative).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise UpgradeError(f"the kit's {relative} cannot be read: {error}") from error

    def data(
        self,
        relative: Path,
    ) -> tuple[bytes, int]:
        """A kit file copied whole, its bytes and its mode, whatever it holds."""
        path = self.root / relative
        try:
            return path.read_bytes(), os.stat(path).st_mode & 0o777
        except OSError as error:
            raise UpgradeError(f"the kit's {relative} cannot be read: {error}") from error


def ships(
    path: Path,
) -> bool:
    """Whether a file in the kit's tree is one the kit ships: a real file, not bytecode, not a dotfile."""
    if path.is_symlink() or not path.is_file():
        return False

    return "__pycache__" not in path.parts and path.suffix != ".pyc" and not path.name.startswith(".")


@dataclass
class Project:
    """The folder being brought to the kit."""

    root: Path

    @property
    def brain(self) -> Path:
        return self.root / FOLDER

    def state(self) -> str:
        """`none` with no brain at all, `old` with the .agent/ layout, `current` with .agents/."""
        if store.quietly(self.brain.is_dir):
            return "current"

        if store.quietly((self.root / OLD_FOLDER).is_dir):
            return "old"

        return "none"

    def version(self) -> str:
        stamp = self.brain / VERSION
        if not store.readable_file(stamp):
            return schema.NO_VERSION

        return read_version(stamp)


class Applies(Protocol):
    """The shape of a migration step, called by keyword like every function of the tool's own."""

    def __call__(
        self,
        project: Project,
        report: Report,
    ) -> None: ...


@dataclass
class Migration:
    """One layout change, applied while the tree calls for it and the kit is at least `to`.

    The tree, not the stamp, says whether a migration runs. A stamp can
    be missing or wrong, and a migration applied to a tree that does not
    need it rewrites what it should not.
    """

    to: str
    says: str
    needed: Callable[[Project], bool]
    apply: Applies


def read_version(
    stamp: Path,
) -> str:
    try:
        text = store.stamp_text(stamp)
    except (OSError, UnicodeDecodeError) as error:
        raise UpgradeError(f"{stamp.name} cannot be read: {error}") from error

    try:
        schema.as_tuple(text)
    except ValueError as error:
        raise UpgradeError(f"{stamp.name} says {error}") from error

    return text


def default_kit() -> Kit:
    """The kit this tool runs from, five folders above this file."""
    return Kit(root=Path(__file__).resolve().parents[5])


def port(
    kit: Kit,
    project: Project,
    out: Optional[TextIO] = None,
) -> Report:
    """Install the kit into a project that has no brain yet, and say what is left to file."""
    state = project.state()
    if state != "none":
        raise UpgradeError(f"{project.root} already carries a brain, run brain upgrade instead")

    prepared = prepare(kit=kit, project=project)
    report = Report(out=out)
    report.done(f"porting {project.root.resolve()}")
    refuse_when_dirty(kit=kit, project=project, paths=guarded_paths(kit=kit, project=project), report=report)
    install(kit=kit, project=project, prepared=prepared, report=report)
    report.candidates = candidates(project)
    report.findings, report.kits_own = check(project=project, kit=kit)

    return report


def upgrade(
    kit: Kit,
    project: Project,
    out: Optional[TextIO] = None,
) -> Report:
    """Bring a project that has a brain up to the kit, migrations first, then the kit's files."""
    state = project.state()
    if state == "none":
        raise UpgradeError(f"{project.root} carries no brain, run brain port instead")

    reason = is_the_kit(
        kit=kit,
        project=project,
    )
    if reason is not None:
        raise UpgradeError(reason)

    prepared = prepare(kit=kit, project=project)
    report = Report(out=out)
    report.done(f"upgrading {project.root.resolve()}")
    refuse_when_dirty(kit=kit, project=project, paths=guarded_paths(kit=kit, project=project), report=report)
    if schema.as_tuple(prepared.project_version) > schema.as_tuple(prepared.stamp):
        raise UpgradeError(f"the project is on {prepared.project_version} and this kit is {prepared.stamp}, use a kit at least that new")

    for migration in pending_migrations(project=project, want=prepared.stamp):
        migration.apply(project=project, report=report)
        report.done(f"migrated to {migration.to}: {migration.says}")

    install(kit=kit, project=project, prepared=prepared, report=report)
    report.findings, report.kits_own = check(project=project, kit=kit)

    return report


def named_root(
    text: str,
) -> Path:
    """The folder `--project` names, which has to be there."""
    root = Path(text)
    if not store.quietly(root.is_dir):
        raise UpgradeError(f"no folder at {root}, so --project names nothing")

    return root.resolve()


def port_root(
    start: Path,
) -> Path:
    """Where a port lands without `--project`: the top of the git repository around `start`, else `start` itself, never the home folder."""
    top = git(project=Project(root=start), arguments=["rev-parse", "--show-toplevel"])
    if top is None or top.returncode != 0:
        return never_home(root=start.resolve(), start=start)

    return never_home(root=Path(top.stdout.strip()).resolve(), start=start)


def upgrade_root(
    start: Path,
) -> Path:
    """Where an upgrade works without `--project`: the nearest folder at or above `start` that carries a brain, else `start` itself, never the home folder."""
    here = start.resolve()
    for candidate in [here, *here.parents]:
        if Project(root=candidate).state() != "none":
            return never_home(root=candidate, start=start)

    return never_home(root=here, start=start)


def never_home(
    root: Path,
    start: Path,
) -> Path:
    """The root found around `start`, refused when it is the home folder.

    A loose folder under the home, or a home that carries a brain or a
    repository of its own, would otherwise get the kit installed into
    the home itself and its own agent configs merged. The two are
    compared as files, not as spellings, so a case folding file system
    cannot slip the home past by another case.
    """
    if store.quietly(lambda: root.samefile(Path.home())):
        raise UpgradeError(f"the project folder found around {start} is your home folder, name the project with --project")

    return root


def pending_migrations(
    project: Project,
    want: str,
) -> list[Migration]:
    """The migrations the project's tree calls for, up to a kit stamped `want`, in order."""
    high = schema.as_tuple(want)

    return [migration for migration in MIGRATIONS if schema.as_tuple(migration.to) <= high and migration.needed(project)]


@dataclass
class Prepared:
    """What a run composes from the kit and the project's own configs, read and checked before any write, the project's stamp read once with them."""

    stamp: str
    project_version: str
    front_doors: dict[str, str]
    settings: str
    merged: list[Path]
    layout: str


def prepare(
    kit: Kit,
    project: Project,
) -> Prepared:
    """Read the kit's stamp and front doors and merge the hook configs, so a refusal comes before the first write."""
    project_version = project.version()
    stamped = schema.as_tuple(project_version)
    for name in store.skill_folders(kit.brain):
        skill = kit_skill_named(name)
        kits_own_file = skill_folder(
            brain=kit.brain,
            name=name,
        ) / schema.SKILL_FILE
        shipped = skill is not None and schema.ships(
            skill=skill,
            stamp=stamped,
        )
        for folder in (FOLDER, OLD_FOLDER):
            home = skill_folder(
                brain=project.root / folder,
                name=name,
            )
            if store.quietly(home.is_symlink):
                raise UpgradeError(f"{folder}/{schema.SKILLS}/{name} is a symlink, replace it with a real folder first")

            if shipped or not store.quietly(home.exists):
                continue

            kits_copy = store.same_bytes(
                one=home / schema.SKILL_FILE,
                other=kits_own_file,
            )
            if not kits_copy and skill is None:
                raise UpgradeError(
                    f"the kit ships a skill named {name} this tool does not know, and {folder}/{schema.SKILLS}/{name} differs "
                    "from it, so run the upgrade with the project's own tool if a kit wrote that folder, else move it aside first"
                )

            if not kits_copy:
                raise UpgradeError(
                    f"the kit now ships a skill named {name}, and {folder}/{schema.SKILLS}/{name} is the project's own, "
                    "move it aside first, or delete it if an interrupted upgrade left it"
                )

        linked = project.root / doctor.SKILL_LINKS / name
        own_folder = store.quietly(linked.is_dir) and not store.quietly(linked.is_symlink)
        if not shipped and own_folder and not store.same_bytes(one=linked / schema.SKILL_FILE, other=kits_own_file):
            raise UpgradeError(
                f"the kit now ships a skill named {name}, and {doctor.SKILL_LINKS / name} holds a folder of the project's own "
                "where the kit's skill link goes, move it aside first"
            )

        for definition in skill.definitions if skill else ():
            own = project.root / definition.path
            carried = schema.carries(
                since=definition.since,
                stamp=stamped,
            )
            if carried or not store.readable_file(path=own):
                continue

            kits_copy = store.same_bytes(
                one=own,
                other=kit.root / definition.path,
            )
            if not kits_copy:
                raise UpgradeError(
                    f"the kit now ships a sub agent definition at {definition.path} with the {name} skill, "
                    "and the file there is the project's own, move it aside first, or delete it if an interrupted upgrade left it"
                )

    stamp = kit.version()
    front_doors = {name: kit.text(Path(name)) for name in ROOT_FILES}
    if KIT_MARK not in front_doors["AGENTS.md"]:
        raise UpgradeError("the kit's AGENTS.md has no line saying where its words start, so section 0 has no place to end")

    settings = kit_config(kit=kit)
    merged: list[Path] = []

    for relative in kit.agent_files():
        target = project.root / relative
        if relative.name in MERGED and store.readable_file(target):
            merged_config(
                existing=read_as_is(target),
                kit=settings if relative == CLAUDE_SETTINGS else kit.text(relative),
                where=str(relative),
            )
            merged.append(relative)

    return Prepared(
        stamp=stamp,
        project_version=project_version,
        front_doors=front_doors,
        settings=settings,
        merged=merged,
        layout=project.state(),
    )


def install(
    kit: Kit,
    project: Project,
    prepared: Prepared,
    report: Report,
) -> None:
    """Copy the kit's own files in, leaving the project's alone, the version stamp last.

    The kit's texts were read and the hook configs merged once by
    `prepare` before anything was touched, so a refusal came first.
    Section 0 is read here and the configs are merged again here, from
    the files as the migration left them, pointed at the new folder name.
    """
    stamp = prepared.stamp
    front_doors = dict(prepared.front_doors)
    zero = section_zero_of(project=project, layout=prepared.layout)
    if zero is not None:
        front_doors["AGENTS.md"] = with_section_zero(text=front_doors["AGENTS.md"], zero=zero.text)
        report.done(zero.says)

    replaced: list[str] = []
    for name in ROOT_FILES:
        found = any_case(folder=project.root, name=name)
        settle_case(project=project, name=name, report=report)
        if put(project=project, relative=Path(name), text=front_doors[name], report=report):
            replaced.append(name if found is None else found.name)

    specs = kept_in_git(project=project, names=replaced)
    if specs:
        report.done("the old ones are still in git: " + ", ".join(f"git show {spec}" for spec in specs))

    shipped = [Path(FOLDER) / relative for relative in kit.brain_files() if relative.name != VERSION]
    written = sum(copy(kit=kit, project=project, relative=relative) for relative in shipped)
    keep = set(shipped)
    retired: list[str] = []
    for folder in kit_skill_folders(brain=project.brain, kit=kit):
        retired += retire(
            project=project,
            folder=folder,
            keep=keep,
        )
    report.done(f"the kit's {len(shipped)} files under {FOLDER}/, {written} written, {len(retired)} retired")
    for relative in retired:
        report.done(f"retired {relative}, the kit no longer ships it")

    for kind in schema.KINDS.values():
        (project.brain / kind.folder).mkdir(parents=True, exist_ok=True)

    for relative in kit.agent_files():
        if relative in prepared.merged:
            merged = merged_config(
                existing=read_as_is(project.root / relative),
                kit=prepared.settings if relative == CLAUDE_SETTINGS else kit.text(relative),
                where=str(relative),
            )
            put(project=project, relative=relative, text=merged, report=report)
            continue

        if relative == CLAUDE_SETTINGS:
            if put_config(
                project=project,
                relative=relative,
                text=prepared.settings,
                report=report,
            ):
                report.done(f"wrote {relative}")
            continue

        if copy(kit=kit, project=project, relative=relative):
            report.done(f"wrote {relative}")

    ignore(project=project, kit=kit, report=report)

    put(project=project, relative=Path(FOLDER) / VERSION, text=stamp + "\n", report=report, quiet=True)
    report.done(f"stamped {FOLDER}/{VERSION} {stamp}")


def check(
    project: Project,
    kit: Kit,
) -> tuple[list[doctor.Finding], list[doctor.Finding]]:
    """Doctor with --fix over the brain as installed: what it could not clear in the project's files, and what sits in the kit's own.

    A finding in one of the brain's files the kit ships, under one of its
    skill folders, or in a file of the agent folders the kit copies
    whole, is in a file the install replaces every time, so it is the
    kit's to put right, reported apart and never counted against the
    project. So is an import finding in a front door that the kit's own
    text of that door gives too. Each finding lands on one side or the
    other by its place in doctor's list, so two equal slips in
    AGENTS.md, one in the kit's words and one in section 0, are one for
    each.
    """
    found = doctor.run(brain=store.Brain(root=project.brain), fix=True)
    names = store.skill_folders(kit.brain)
    shipped = set(kit.brain_files())
    copied = {relative for relative in kit.agent_files() if relative.name not in MERGED} | set(style.COPIES)
    in_kits_words = {
        name: Counter(
            finding.message
            for finding in doctor.import_slips(
                name=name,
                text=kit.text(Path(name)),
                project=project.root,
            )
        )
        for name in FRONT_DOORS
    }
    projects: list[doctor.Finding] = []
    kits: list[doctor.Finding] = []
    for finding in found:
        if kits_own(
            finding=finding,
            names=names,
            shipped=shipped,
            copied=copied,
            in_kits_words=in_kits_words,
        ):
            kits.append(finding)
        else:
            projects.append(finding)

    return projects, kits


def kits_own(
    finding: doctor.Finding,
    names: list[str],
    shipped: set[Path],
    copied: set[Path],
    in_kits_words: dict[str, Counter[str]],
) -> bool:
    """Whether a doctor finding sits in what the install writes whole.

    That is one of the brain's files the kit ships, named relative to
    the brain in `shipped`, or anything else in one of the named skill
    folders, its path spelled from the brain or from the project root.
    It is a file of the agent folders in `copied`, named from the
    project root, which the install copies whole or doctor makes from
    the kit's chat style source, a merged config never. It is an import
    finding in a front door that the kit's own text of that door gives
    too, as the install writes CLAUDE.md and GEMINI.md as the kit's
    text and AGENTS.md below the marker as the kit's words, the part
    above them being the project's section 0. The kit's text gives each
    such finding so many times, and every one taken for the kit's comes
    off its count in `in_kits_words`, so the same finding once more is
    the project's, and so is one the kit's text never gives, a door
    that is missing, misspelled or cannot be read. An import finding is
    known by its check, so a project file of the same name inside the
    brain is never taken for a front door.
    """
    if finding.check == "import":
        left = in_kits_words.get(finding.path)
        if left is None or left[finding.message] < 1:
            return False

        left[finding.message] -= 1

        return True

    if Path(finding.path) in copied:
        return True

    parts = Path(finding.path).parts
    if parts and parts[0] in (FOLDER, OLD_FOLDER):
        parts = parts[1:]

    if parts and Path(*parts) in shipped:
        return True

    return len(parts) > 1 and parts[0] == schema.SKILLS and parts[1] in names


def candidates(
    project: Project,
) -> list[str]:
    """What an unported project holds that belongs in the brain, for the agent to file.

    Where git runs, every markdown file it tracks or sees outside the
    dot folders, the brain and the agent folders among them, the root
    files one by one and deeper ones counted by their top folder, with
    docs/ marked as the one the rules let a project keep. Outside git,
    the root markdown and the usual folder names.
    """
    kept = {name.lower() for name in (*ROOT_FILES, "README.md", "CODEOWNERS", "LICENSE.md", "CHANGELOG.md", "SECURITY.md")}
    listed = git(
        project=project,
        arguments=["ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "*.md"],
    )
    if listed is None or listed.returncode != 0:
        found = [
            path.name
            for path in sorted(project.root.glob("*.md"))
            if path.name.lower() not in kept and not path.name.startswith(".")
        ]
        found += [
            f"{folder}/"
            for folder in ("docs", "doc", "notes", "adr", "decisions")
            if store.quietly((project.root / folder).is_dir)
        ]

        return found

    files = [Path(entry) for entry in listed.stdout.split("\0") if entry]
    found = [str(path) for path in files if len(path.parts) == 1 and path.name.lower() not in kept and not path.name.startswith(".")]
    counts: dict[str, int] = {}
    for path in files:
        if len(path.parts) > 1 and not any(part.startswith(".") for part in path.parts):
            counts[path.parts[0]] = counts.get(path.parts[0], 0) + 1

    for folder, count in sorted(counts.items()):
        noun = "markdown file" if count == 1 else "markdown files"
        note = ", may stay per the rules, linked into the brain" if folder == "docs" else ""
        found.append(f"{folder}/ ({count} {noun}{note})")

    return found


def to_agents_folder(
    project: Project,
    report: Report,
) -> None:
    """The layout before 1.0.0 kept the brain in .agent/ and the rules in CLAUDE.md."""
    old = project.root / OLD_FOLDER
    if store.quietly(project.brain.exists):
        raise UpgradeError(f"both {OLD_FOLDER}/ and {FOLDER}/ exist, fold one into the other by hand first")

    targets = migration_targets(project)
    move(project=project, old=old, new=project.brain)
    report.done(f"renamed {OLD_FOLDER}/ to {FOLDER}/")

    changed = 0
    for path in targets:
        moved = project.brain / path.relative_to(old) if path.is_relative_to(old) else path
        if rewrite_folder_name(moved):
            changed += 1

    if changed:
        report.done(f"pointed {changed} files at {FOLDER}/ instead of {OLD_FOLDER}/")


def migration_targets(
    project: Project,
) -> list[Path]:
    """The files the folder migration points at the new name: every file naming .agent/ that git tracks, sees or ignores, symlinks and the kit's own skill folders left out.

    Outside git the root markdown, the ignore file and the brain's own
    markdown are read instead. The guard lists the same files, so the
    two never disagree about what the migration writes into, and a
    file git ignores is refused there rather than half migrated. Git
    is asked for names relative to the project whatever grep.fullName
    says.
    """
    found = git(
        project=project,
        arguments=[
            "grep",
            "-I",
            "-l",
            "-z",
            "--no-full-name",
            "--untracked",
            "--no-exclude-standard",
            "--fixed-strings",
            "-e",
            f"{OLD_FOLDER}/",
        ],
    )
    if found is not None and found.returncode in (0, 1):
        paths = [project.root / entry for entry in found.stdout.split("\0") if entry]
    else:
        brain = project.root / (OLD_FOLDER if project.state() == "old" else FOLDER)
        paths = [path for path in list_names(project.root) if path.suffix == ".md"]
        paths += [project.root / IGNORE_FILE, *brain_markdown(brain)]
        paths = [path for path in paths if names_old_folder(path)]

    return [
        path
        for path in paths
        if path.is_file() and not path.is_symlink() and not in_skill_folder(project=project, path=path)
    ]


def names_old_folder(
    path: Path,
) -> bool:
    try:
        return f"{OLD_FOLDER}/" in read_as_is(path)
    except UpgradeError:
        return False


def in_skill_folder(
    project: Project,
    path: Path,
) -> bool:
    """Whether the path sits under one of the kit's own skill folders, by either name of the brain."""
    parts = path.relative_to(project.root).parts

    return bool(parts) and parts[0] in (OLD_FOLDER, FOLDER) and kits_skill(parts[1:])


def kits_skill(
    parts: tuple[str, ...],
) -> bool:
    """Whether a path split under the brain leads into one of the kit's skill folders."""
    return len(parts) > 1 and parts[0] == schema.SKILLS and parts[1] in schema.KIT_SKILL_NAMES


def skill_folder(
    brain: Path,
    name: str,
) -> Path:
    """Where a skill of that name lives under a brain, whether or not it is there yet."""
    return brain / schema.SKILLS / name


def kit_skill_folders(
    brain: Path,
    kit: Kit,
) -> list[Path]:
    """The kit's skill folders under a brain, whether or not they are there yet, by the skills the kit checkout ships.

    The checkout decides, not the running tool's table, since the
    checkout may be a newer release with a skill the tool never heard
    of, and that folder is the kit's to write, retire under and guard
    like any other.
    """
    return [
        skill_folder(
            brain=brain,
            name=name,
        )
        for name in store.skill_folders(kit.brain)
    ]


def kit_skill_named(
    name: str,
) -> Optional[schema.KitSkill]:
    """The running tool's entry for a kit skill, or None for one the checkout ships that this tool never heard of."""
    return next((skill for skill in schema.KIT_SKILLS if skill.name == name), None)


def brain_markdown(
    brain: Path,
) -> list[Path]:
    """Every markdown file under the brain outside the kit's own skill folders, what the folder migration reads outside git."""
    if not store.quietly(brain.is_dir):
        return []

    return sorted(
        path
        for path in brain.rglob("*.md")
        if path.is_file() and not kits_skill(path.relative_to(brain).parts)
    )


def has_old_folder(
    project: Project,
) -> bool:
    return store.quietly((project.root / OLD_FOLDER).is_dir)


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        to="1.0.0",
        says=f"the brain lives in {FOLDER}/, the rules in the kit's own files, and section 0 in AGENTS.md",
        needed=has_old_folder,
        apply=to_agents_folder,
    ),
)


def rewrite_folder_name(
    path: Path,
) -> bool:
    """Point one text file at .agents/ where it named .agent/, and say whether it changed.

    The file's own line endings are kept, so the change is the folder
    name and nothing else.
    """
    if not store.readable_file(path):
        return False

    try:
        text = read_as_is(path)
    except UpgradeError:
        return False

    wanted = text.replace(f"{OLD_FOLDER}/", f"{FOLDER}/")
    if wanted == text:
        return False

    frontmatter.write_text(path=str(path), text=wanted)

    return True


def read_as_is(
    path: Path,
) -> str:
    """The file's text with its line endings as written, not folded to newlines."""
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            return handle.read()
    except (OSError, UnicodeDecodeError) as error:
        raise UpgradeError(f"{path.name} cannot be read: {error}") from error


def move(
    project: Project,
    old: Path,
    new: Path,
) -> None:
    """Rename a file or folder, through git when git tracks it, so the rename is recorded as one.

    A case only rename, `Agents.md` to `AGENTS.md`, goes through a
    passing name, since a case folding file system would otherwise call
    it no change at all.
    """
    moved = git(
        project=project,
        arguments=["mv", "-f", str(old.relative_to(project.root)), str(new.relative_to(project.root))],
    )
    if moved is not None and moved.returncode == 0:
        return

    if old.name.lower() == new.name.lower() and old.parent == new.parent:
        passing = old.parent / f".{new.name}.renaming"
        old.rename(passing)
        old = passing

    old.rename(new)


def is_the_kit(
    kit: Kit,
    project: Project,
) -> Optional[str]:
    """Why the project gets nothing installed, being the kit's own checkout, or None when it is not.

    The same folder under any spelling, or another checkout of the
    kit's repository, a worktree, a clone of the kit's origin, or a
    clone at most ORIGIN_HOPS local clones away. The kit is where the
    tool's code is the work, so it never gets the read guard and never
    has its own files written over.
    """
    if store.quietly(lambda: kit.root.samefile(project.root)):
        return "the kit is the project, there is nothing to install"

    if same_repository(
        kit=kit,
        project=project,
    ):
        return "the project is a worktree or a clone of the kit, there is nothing to install"

    return None


def same_repository(
    kit: Kit,
    project: Project,
) -> bool:
    """Whether the project is a checkout of the kit's repository, which it is when the two ancestries meet.

    Each folder has to be the top of its own repository first, since a
    kit vendored as plain files inside a project's repository, or a
    project folder inside the kit's, shares a `.git` and its origin
    without being a checkout of the kit.
    """
    if not is_a_top(kit.root) or not is_a_top(project.root):
        return False

    return not ancestry(kit.root).isdisjoint(ancestry(project.root))


def ancestry(
    folder: Path,
) -> set[Union[tuple[int, int], Path, str]]:
    """The repositories a checkout comes from, its own and each local origin's in turn, then the remote one it names.

    Each folder the walk stands in is recorded by its own identity
    before git is asked anything about it, so two clones of one folder
    meet there even when git cannot open it, and then by the identity
    of its git common dir, so a worktree and every spelling of the path
    read the same. A local origin, a file URL or a plain path, is
    resolved against the folder it was read in. One that is a folder
    is walked on from, at most ORIGIN_HOPS times and never into a
    folder already recorded, and one that is not there is recorded by
    its resolved path and ends the walk. An origin that is a remote URL
    is recorded as the repository it names and ends the walk too.
    """
    found: set[Union[tuple[int, int], Path, str]] = set()

    for _ in range(ORIGIN_HOPS + 1):
        place = identity(folder)
        if place is None or place in found:
            return found

        found.add(place)

        here = Project(root=folder)
        common = git(
            project=here,
            arguments=["rev-parse", "--git-common-dir"],
        )
        if not answered(common):
            return found

        repository = identity(folder / common.stdout.strip())
        if repository is not None:
            found.add(repository)

        origin = git(
            project=here,
            arguments=["remote", "get-url", "origin"],
        )
        if not answered(origin):
            return found

        url = origin.stdout.strip()
        checkout = local_origin(
            folder=folder,
            url=url,
        )
        if checkout is None:
            found.add(repository_named(url))
            return found

        if not store.quietly(checkout.is_dir):
            found.add(checkout)
            return found

        folder = checkout

    return found


def identity(
    path: Path,
) -> Optional[tuple[int, int]]:
    """The device and inode of a path, the same under every spelling of it, or None when nothing is there."""
    try:
        found = os.stat(path)
    except OSError:
        return None

    return found.st_dev, found.st_ino


def is_a_top(
    folder: Path,
) -> bool:
    """Whether the folder is the top of a git repository of its own."""
    top = git(
        project=Project(root=folder),
        arguments=["rev-parse", "--show-toplevel"],
    )

    return answered(top) and store.quietly(lambda: Path(top.stdout.strip()).samefile(folder))


def local_origin(
    folder: Path,
    url: str,
) -> Optional[Path]:
    """The folder a local origin names, resolved against the folder it was read in with its `.git` taken off, or None for a remote URL.

    Git's own rule tells them apart: a file URL is local, any other
    scheme is remote, and text with no scheme is a local path unless a
    colon comes before its first slash, which makes it host:path, a
    drive letter aside. A file URL keeps its percent escapes in the
    config, `my%20kit` for `my kit`, so they are decoded.
    """
    if url.startswith("file://"):
        named = unquote(url[len("file://") :])
    elif "://" in url:
        return None
    else:
        colon = url.find(":")
        slash = url.find("/")
        host_and_path = colon != -1 and not 0 <= slash < colon
        if host_and_path and not Path(url).drive:
            return None

        named = url

    path = Path(os.path.realpath(folder / named))

    return path.parent if path.name == ".git" else path


def repository_named(
    url: str,
) -> str:
    """A git URL as host and path, lowercased, without the scheme, the user, the port, a trailing slash or `.git`, so two spellings of one origin read the same."""
    rest = url.strip()
    if "://" in rest:
        rest = rest.split(
            sep="://",
            maxsplit=1,
        )[1]
    elif ":" in rest and not rest.startswith("/"):
        rest = rest.replace(
            ":",
            "/",
            1,
        )

    host, _, path = rest.partition("/")
    host = host.rsplit(
        sep="@",
        maxsplit=1,
    )[-1]
    host = host.split(
        sep=":",
        maxsplit=1,
    )[0]
    path = path.rstrip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]

    return f"{host}/{path}".lower()


def answered(
    result: Optional["subprocess.CompletedProcess[str]"],
) -> bool:
    """Whether git ran and succeeded."""
    return result is not None and result.returncode == 0


def git(
    project: Project,
    arguments: list[str],
) -> Optional["subprocess.CompletedProcess[str]"]:
    """Run git in the project, or None when git is not there to run."""
    try:
        return subprocess.run(
            ["git", "-C", str(project.root), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None


@dataclass
class Zero:
    """The project's own rules as they go under the kit's title, and the report line saying where they came from."""

    text: str
    says: str


def section_zero_of(
    project: Project,
    layout: str,
) -> Optional[Zero]:
    """The project's own rules, from the first front door that carries a `## 0.` section with words in it.

    `layout` is the project's state before any migration. On the old
    layout CLAUDE.md held the rules and AGENTS.md copied it, so CLAUDE.md
    is read first there. Otherwise AGENTS.md is, since that is where the
    kit keeps section 0. On a port, with no such section anywhere, a
    front door that says something of its own goes whole under the
    heading, for the agent to read against the rulebook, because the
    kit's copy is about to replace it. Never on an upgrade, where a front
    door without a section is the kit's own earlier text.
    """
    texts: dict[str, str] = {}
    for name in OLD_LAYOUT_DOORS if layout == "old" else FRONT_DOORS:
        path = any_case(folder=project.root, name=name)
        if path is None:
            continue

        try:
            texts[path.name] = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            continue

    for name, text in texts.items():
        zero = section_zero_in(text)
        if zero is not None:
            return Zero(text=zero, says=f"kept section 0 from {name}")

    if layout != "none":
        return None

    for name, text in texts.items():
        words = own_words(text)
        if words:
            return Zero(
                text=f"{SECTION_ZERO_HEADING}\n\n{words}\n",
                says=f"carried {name} whole into section 0, read it against the rulebook and keep only what the kit does not cover",
            )

    return None


def section_zero_in(
    text: str,
) -> Optional[str]:
    """The `## 0.` section of a file under the rulebook's heading, or None when there is none or it says nothing.

    In an AGENTS.md the kit wrote, the section is everything down to
    the kit's marker line, headings included, so the file gives back the
    section it was given whatever the kit's own words say now. In one
    the first kit wrote, before the marker, it ends where that kit's
    words stood. Anywhere else it runs to the next `#` or `##` heading
    outside code. Code never ends it, so a `## ` or a quoted marker
    inside a fenced block stays in it.
    """
    heading = next((match for match in sections.heading_matches(text) if match.group(1).startswith("0.")), None)
    if heading is None:
        return None

    end = first_outside_code(text=text, pattern=MARK, start=heading.end())
    if end == -1:
        end = sections.heading_after(body=text, start=heading.end(), depth=2)
        words = first_outside_code(text=text, pattern=FIRST_KIT, start=heading.end(), end=end)
        if words != -1:
            end = words

    body = text[heading.end() : end].strip()
    if not body:
        return None

    return f"{SECTION_ZERO_HEADING}\n\n{body}\n"


def first_outside_code(
    text: str,
    pattern: "re.Pattern[str]",
    start: int,
    end: Optional[int] = None,
) -> int:
    """Where the pattern first matches outside code between `start` and `end`, or -1 when it does not."""
    spans = sections.code_spans(text)
    for match in pattern.finditer(text, start, len(text) if end is None else end):
        if not sections.in_code(spans=spans, position=match.start()):
            return match.start()

    return -1


def own_words(
    text: str,
) -> str:
    """What a front door says on its own: the text before the kit's marker, without the title, the comments and the `@path` import lines.

    Code keeps its comments and its `@` lines, a decorator for one.
    """
    mark = first_outside_code(text=text, pattern=MARK, start=0)
    own = text if mark == -1 else text[:mark]
    for pattern in (COMMENT, IMPORT_LINE):
        own = without(text=own, pattern=pattern)

    lines = own.strip().split("\n")
    if lines and lines[0].startswith("# "):
        lines = lines[1:]

    return "\n".join(lines).strip()


def without(
    text: str,
    pattern: "re.Pattern[str]",
) -> str:
    """The text without the pattern's matches that lie outside code.

    A match that starts or ends inside code is no match, and the search
    goes on one character past its start, so a comment opened in prose
    and closed inside a fence stays, and a lone opener inside a fence
    hides nothing after it.
    """
    spans = sections.code_spans(text)
    kept: list[str] = []
    last = 0
    cursor = 0
    while True:
        match = pattern.search(text, cursor)
        if match is None:
            break

        starts_in_code = sections.in_code(spans=spans, position=match.start())
        ends_in_code = sections.in_code(spans=spans, position=match.end() - 1)
        if starts_in_code or ends_in_code:
            cursor = match.start() + 1
            continue

        kept.append(text[last : match.start()])
        last = match.end()
        cursor = match.end()
    kept.append(text[last:])

    return "".join(kept)


def kept_in_git(
    project: Project,
    names: list[str],
) -> list[str]:
    """Where git's last commit holds each of these root files, as `HEAD:path` specs from the repository's top, the ones it does not hold left out."""
    prefix = git(project=project, arguments=["rev-parse", "--show-prefix"])
    if prefix is None or prefix.returncode != 0:
        return []

    specs: list[str] = []
    for name in names:
        spec = f"HEAD:{prefix.stdout.strip()}{name}"
        held = git(project=project, arguments=["rev-parse", "--verify", "--quiet", spec])
        if held is not None and held.returncode == 0:
            specs.append(spec)

    return specs


def with_section_zero(
    text: str,
    zero: str,
) -> str:
    """The kit's AGENTS.md with the project's section 0 between the title and the kit's marker line."""
    title, rest = text.split(KIT_MARK, 1)

    return f"{title.rstrip()}\n\n{zero.rstrip()}\n\n{KIT_MARK}{rest}"


def merged_config(
    existing: str,
    kit: str,
    where: str,
) -> str:
    """The project's hook config with the kit's hooks in it, its own hooks kept.

    The kit's earlier groups go first, under every event they sit in,
    so a release that renames a command or moves an event leaves no
    double, and then the kit's current groups are added under theirs.
    An event left with no group stays as an empty list, which runs
    nothing, so a list the project wrote empty on purpose is kept too.
    An event whose value is not a list is left alone unless the kit
    defines it. The kit's deny rules go in the same way as its hook
    groups, its earlier ones replaced and the project's own kept. Any
    other key the kit sets is added only where the project has none.
    The file keeps its indent and its own characters.
    """
    try:
        have = json.loads(existing)
        want = json.loads(kit)
    except json.JSONDecodeError as error:
        raise UpgradeError(f"{where} is not valid JSON: {error}") from error

    if not isinstance(have, dict) or not isinstance(want, dict):
        raise UpgradeError(f"{where} is not a JSON object")

    hooks = have.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise UpgradeError(f"hooks in {where} is not a JSON object")

    for event, mine in hooks.items():
        if not isinstance(mine, list):
            if event in want.get("hooks", {}):
                raise UpgradeError(f"hooks.{event} in {where} is not a JSON list")

            continue

        hooks[event] = [group for group in mine if not is_kit_group(group)]

    for event, groups in want.get("hooks", {}).items():
        hooks[event] = hooks.get(event, []) + groups

    if isinstance(
        want.get("permissions"),
        dict,
    ):
        merge_guards(
            have=have,
            guards=want["permissions"].get(
                "deny",
                [],
            ),
            where=where,
        )

    for key, value in want.items():
        if key == "hooks":
            continue

        if isinstance(
            value,
            dict,
        ):
            if key in have and not isinstance(
                have[key],
                dict,
            ):
                raise UpgradeError(f"{key} in {where} is not a JSON object")

            for inner, inner_value in value.items():
                have.setdefault(
                    key,
                    {},
                ).setdefault(
                    inner,
                    inner_value,
                )
            continue

        have.setdefault(key, value)

    return json.dumps(have, indent=indent_of(existing), ensure_ascii=False) + "\n"


def merge_guards(
    have: dict,
    guards: list,
    where: str,
) -> None:
    """The kit's deny rules into the project's list, the kit's earlier spellings dropped and the project's own rules kept in their order."""
    permissions = have.setdefault(
        "permissions",
        {},
    )
    if not isinstance(
        permissions,
        dict,
    ):
        raise UpgradeError(f"permissions in {where} is not a JSON object")

    deny = permissions.setdefault(
        "deny",
        [],
    )
    if not isinstance(
        deny,
        list,
    ):
        raise UpgradeError(f"permissions.deny in {where} is not a JSON list")

    permissions["deny"] = [rule for rule in deny if not is_kit_rule(rule)] + guards


def is_kit_rule(
    rule: object,
) -> bool:
    if not isinstance(
        rule,
        str,
    ):
        return False

    return KIT_RULE.match(rule) is not None


def kit_config(
    kit: Kit,
) -> str:
    """The kit's Claude Code settings as they go into a project, with the read guards under permissions.deny.

    The kit's own checkout never gets them, since the run refuses to
    install the kit onto itself, and that checkout is where the tool's
    code is the work.
    """
    text = kit.text(CLAUDE_SETTINGS)
    try:
        settings = json.loads(text)
    except json.JSONDecodeError as error:
        raise UpgradeError(f"the kit's {CLAUDE_SETTINGS} is not valid JSON: {error}") from error

    if not isinstance(
        settings,
        dict,
    ):
        raise UpgradeError(f"the kit's {CLAUDE_SETTINGS} is not a JSON object")

    settings["permissions"] = {"deny": list(READ_GUARDS)}

    return json.dumps(
        obj=settings,
        indent=indent_of(text),
        ensure_ascii=False,
    ) + "\n"


def put_config(
    project: Project,
    relative: Path,
    text: str,
    report: Report,
) -> bool:
    """Write a composed kit config into the project, and say whether anything was written."""
    target = project.root / relative
    if store.readable_file(target) and says(
        path=target,
        text=text,
    ):
        return False

    put(
        project=project,
        relative=relative,
        text=text,
        report=report,
        quiet=True,
    )

    return True


def indent_of(
    text: str,
) -> str:
    """The indent a JSON file uses, read off its first indented line whatever its line breaks, two spaces when it has none."""
    match = re.search(r"(?:^|[\r\n])([ \t]+)\S", text)

    return match.group(1) if match else "  "


def is_kit_group(
    group: object,
) -> bool:
    if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
        return False

    return any(KIT_HOOK.match(str(hook.get("command", ""))) for hook in group["hooks"] if isinstance(hook, dict))


def ignore(
    project: Project,
    kit: Kit,
    report: Report,
) -> None:
    """The kit's ignore block at the top of the project's .gitignore, the project's own lines below it.

    Git reads the file top down and the last match wins, so the block
    goes first and everything the project wrote outranks it: an
    exception of the project's, a file it ignores by name, a spelling
    of its own, all stand as written, and the kit's lines reach only
    what the project says nothing about. The block holds the kit's
    lines in the kit's order under one heading, and is rebuilt at the
    top on a later run wherever it sat. Every other line stays in the
    order the project wrote it, a line the project put inside an
    earlier block included, which stays where that block was, so no
    two of the project's own lines ever swap places. A file already in
    that shape is left alone.
    """
    wanted = [line for line in kit.text(Path(IGNORE_FILE)).splitlines() if line.strip() and not line.startswith("#")]
    target = project.root / IGNORE_FILE
    existing = read_as_is(target) if store.readable_file(target) else ""
    ending = "\r\n" if "\r\n" in existing else "\n"
    mark = BOM if existing.startswith(BOM) else ""
    lines = [line[:-1] if line.endswith("\r") else line for line in existing[len(mark) :].split("\n")]

    rest: list[str] = []
    in_block = False
    for line in lines:
        if line.rstrip() == IGNORE_HEADING:
            in_block = True
            continue

        if not line.strip():
            in_block = False

        if not (in_block and line in wanted):
            rest.append(line)

    spaced: list[str] = []
    for line in rest:
        if line.strip() or (spaced and spaced[-1].strip()):
            spaced.append(line)
    while spaced and not spaced[-1].strip():
        spaced.pop()

    body = [IGNORE_HEADING, *wanted]
    if spaced:
        body += ["", *spaced]

    text = mark + ending.join(body) + ending
    if text == existing:
        return

    frontmatter.write_text(path=str(target), text=text)
    report.done(f"set the kit's ignore block at the top of {IGNORE_FILE}, the project's own lines below it win")


def guarded_paths(
    kit: Kit,
    project: Project,
) -> list[str]:
    """Every project path a port or upgrade writes over or removes, relative to the project.

    The front doors and CONTRIBUTING.md are replaced whole, so are the
    kit's files under the brain and the style files in the agent
    folders, the kit's skill folders are cleared and rewritten, and the
    folder migration rewrites every file that names the old folder. On the old
    layout the brain is still .agent/, and git names its files by that
    spelling, so the list uses it too. The ignore file and the hook
    configs keep what they hold, so an edit there is safe and not
    listed, unless the migration is about to rewrite them.
    """
    paths = [*ROOT_FILES]
    paths += [str(relative) for relative in kit.agent_files() if relative.name not in MERGED]
    paths += [str(relative) for relative in style.COPIES]

    brain = project.root / (OLD_FOLDER if project.state() == "old" else FOLDER)
    paths += [str(brain.relative_to(project.root) / relative) for relative in kit.brain_files()]
    for folder in kit_skill_folders(brain=brain, kit=kit):
        paths += files_under(
            root=project.root,
            folder=folder,
            taken=retire_takes,
        )

    if pending_migrations(project=project, want=kit.version()):
        paths += [str(path.relative_to(project.root)) for path in migration_targets(project)]

    return paths


def files_under(
    root: Path,
    folder: Path,
    taken: Callable[[Path], bool],
) -> list[str]:
    """Every file below the folder that the run would take, relative to the root, nothing when the folder is not there.

    The predicate is the same one the taking step uses, so the guard
    and the writer never disagree about what is at stake.
    """
    if not store.quietly(folder.is_dir):
        return []

    return [str(path.relative_to(root)) for path in sorted(folder.rglob("*")) if taken(path)]


def retire_takes(
    path: Path,
) -> bool:
    """What retire removes under a kit skill folder file by file: a shipped shape file or a symlink, never bytecode or a dotfile.

    A folder of nothing but bytecode goes whole with retire's own rule,
    since bytecode is never listed, guarded or worth keeping.
    """
    return ships(path) or path.is_symlink()


def holds_only_bytecode(
    folder: Path,
) -> bool:
    """Whether the folder holds nothing but compiled bytecode, so removing it whole loses nothing anyone wrote."""
    return all(child.suffix == ".pyc" for child in folder.rglob("*") if not child.is_dir())


def refuse_when_dirty(
    kit: Kit,
    project: Project,
    paths: list[str],
    report: Report,
) -> None:
    """Refuse while a file about to be written over or removed holds uncommitted work.

    Git is asked for the whole working tree once, ignored files
    included since a removed one is lost all the same (`--ignored` with
    no mode, which every git reads and which names the files inside an
    ignored folder when every untracked file is asked for), and the answer
    is matched against the paths without regard to case, so a front
    door spelled `Agents.md` is seen. Git names paths from the
    repository's root, so the project's place below it is put in front
    of each path first. A repository whose git fails is refused too,
    since nothing would be guarded. A project outside git gets no
    guard, and the report says so.
    """
    if not inside_repository(project):
        report.done("not a git repository, so nothing is saved before a file is replaced")
        return

    prefix = git(project=project, arguments=["rev-parse", "--show-prefix"])
    if prefix is None:
        report.done("git is not on this machine, so nothing is saved before a file is replaced")
        return

    result = git(project=project, arguments=["status", "--porcelain", "--untracked-files=all", "--ignored", "-z"])
    if prefix.returncode != 0 or result is None or result.returncode != 0:
        words = (prefix.stderr if prefix.returncode != 0 else (result.stderr if result else "")).strip()
        raise UpgradeError(f"git could not read the working tree, so nothing is guarded: {words}")

    raw = prefix.stdout.strip()
    base = folded(raw)
    wanted = {base + folded(path): path for path in paths}
    dirty: list[str] = []
    ignored: list[str] = []
    for status, entry in changed_paths(result.stdout):
        path = wanted.get(folded(entry))
        if path is None or (status in ("??", "!!") and same_as_kit(kit=kit, project=project, relative=path)):
            continue

        shown = entry[len(raw) :] if folded(entry[: len(raw)]) == base else path
        (ignored if status == "!!" else dirty).append(shown)

    reasons: list[str] = []
    if dirty:
        reasons.append("commit, stash or move these first, they would be written over or removed: " + ", ".join(sorted(dirty)))

    if ignored:
        reasons.append("git ignores these and they would be written over or removed, move them out of the way first: " + ", ".join(sorted(ignored)))

    if reasons:
        raise UpgradeError(". ".join(reasons))


def inside_repository(
    project: Project,
) -> bool:
    """Whether a `.git` sits at the project's root or above it, answered without git so no locale is in the way."""
    root = project.root.resolve()

    return any(store.quietly((folder / ".git").exists) for folder in [root, *root.parents])


def folded(
    path: str,
) -> str:
    """A path as git names it and as this tool names it, made comparable: one slash, one case, one Unicode form."""
    return unicodedata.normalize("NFC", path.replace("\\", "/")).lower()


def same_as_kit(
    kit: Kit,
    project: Project,
    relative: str,
) -> bool:
    """Whether a file git does not hold already has exactly what the run would write, so nothing is at stake.

    A port followed by an upgrade before a commit leaves every kit file
    untracked and identical, and a project that ignores an agent folder
    leaves the kit's files there ignored and identical. Only a file that
    maps to one the kit copies whole can be judged this way, and the
    version stamp, which the run never copies but writes as the kit's
    stamp trimmed, so it is judged against that text.
    """
    if Path(relative) == Path(FOLDER) / VERSION:
        return says(
            path=project.root / relative,
            text=kit.version() + "\n",
        )

    source = kit_source(relative)
    if source is None:
        return False

    return same_text(
        one=project.root / relative,
        other=kit.root / source,
    )


def same_text(
    one: Path,
    other: Path,
) -> bool:
    """Whether two text files say the same, their line ends aside, false when either cannot be read.

    Every file the kit ships is text, and the root files, the version
    stamp and the chat style copies are written in text mode, so on
    Windows the port leaves them with CRLF while the kit's checkout may
    hold either, and bytes would tell the port's own copy apart from
    the kit's.
    """
    try:
        return one.read_text(encoding="utf-8") == other.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False


def kit_source(
    relative: str,
) -> Optional[Path]:
    """The kit file a project path is a copy of, or None when the run composes it.

    The front doors count, since without a section 0 they are the kit's
    text verbatim, and with one the bytes differ and the file is
    guarded as the project's own.
    """
    parts = Path(relative).parts
    if not parts:
        return None

    if parts[0] == FOLDER or parts[0] in AGENT_FOLDERS or relative in ROOT_FILES:
        return Path(relative)

    return None


def changed_paths(
    porcelain: str,
) -> list[tuple[str, str]]:
    """The status and path of each entry in a NUL separated porcelain listing, a rename's old name included."""
    found: list[tuple[str, str]] = []
    tokens = porcelain.split("\0")
    index = 0

    while index < len(tokens):
        entry = tokens[index]
        index += 1
        if len(entry) < 4:
            continue

        status = entry[:2]
        found.append((status, entry[3:]))
        if entry[0] in "RC" and index < len(tokens):
            found.append((status, tokens[index]))
            index += 1

    return found


def settle_case(
    project: Project,
    name: str,
    report: Report,
) -> None:
    """A front door spelled in another case, `Agents.md` for one, takes the kit's spelling first."""
    found = any_case(folder=project.root, name=name)
    if found is None or found.name == name:
        return

    move(project=project, old=found, new=project.root / name)
    report.done(f"renamed {found.name} to {name}")


def any_case(
    folder: Path,
    name: str,
) -> Optional[Path]:
    """The file in the folder whose name matches in any case, the exact spelling first."""
    listed = list_names(folder)
    if folder / name in listed:
        return folder / name

    for path in listed:
        if path.name.lower() == name.lower():
            return path

    return None


def list_names(
    folder: Path,
) -> list[Path]:
    try:
        return sorted(path for path in folder.iterdir() if path.is_file())
    except OSError:
        return []


def put(
    project: Project,
    relative: Path,
    text: str,
    report: Report,
    quiet: bool = False,
) -> bool:
    """Write one file whole, and say whether it was new or replaced. True when a file that said something else was written over.

    A target that already says the text, its line ends aside, is left
    alone, so a run that changes nothing touches nothing.
    """
    target = project.root / relative
    was_there = store.readable_file(target)
    if was_there and says(
        path=target,
        text=text,
    ):
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    frontmatter.write_text(path=str(target), text=text)

    if not quiet:
        report.done(f"{'replaced' if was_there else 'created'} {relative}")

    return was_there


def says(
    path: Path,
    text: str,
) -> bool:
    """Whether the file already says this text, its line ends aside, false when it cannot be read.

    The run writes its texts in text mode, so on Windows a file it wrote
    holds CRLF where the text holds a newline, and bytes would call the
    file changed on every run.
    """
    try:
        said = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False

    return said == OTHER_LINE_ENDS.sub(
        repl="\n",
        string=text,
    )


def holds(
    path: Path,
    data: bytes,
) -> bool:
    """Whether the file already holds exactly these bytes, false when it cannot be read."""
    try:
        return path.read_bytes() == data
    except OSError:
        return False


def copy(
    kit: Kit,
    project: Project,
    relative: Path,
) -> bool:
    """Copy one kit file whole, bytes and mode, and say whether anything was written.

    A target that already holds the same bytes is left alone, so a run
    that changes nothing touches nothing.
    """
    data, mode = kit.data(relative)
    target = project.root / relative
    if store.readable_file(target) and holds(path=target, data=data):
        return False

    write_bytes(path=target, data=data, mode=mode)

    return True


def write_bytes(
    path: Path,
    data: bytes,
    mode: int,
) -> None:
    """Write a file whole or not at all, through a temporary file beside it, with the mode given.

    The temporary file takes the same shape as the tool's other writes,
    so one left beside its target by an interrupted run is found and
    removed by doctor.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, passing = tempfile.mkstemp(prefix=f".{path.name}.", suffix=frontmatter.TEMP_SUFFIX, dir=str(path.parent))
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())

        os.chmod(passing, mode)
        os.replace(passing, path)
    except BaseException:
        try:
            os.unlink(passing)
        except OSError:
            pass

        raise


def retire(
    project: Project,
    folder: Path,
    keep: set[Path],
) -> list[str]:
    """Remove the files under one of the kit's skill folders that the kit no longer ships, and the folders that emptied.

    The kit's new files are written before this runs, so the tool is
    never half gone. The folder is the kit's own, so what the kit does
    not ship goes, named in the report, and the guard has already
    refused if any of it held uncommitted work. A dotfile is left. A
    bytecode folder holding nothing else goes whole and unnamed, since
    nobody wrote it and the guard never lists it.
    """
    if not store.quietly(folder.is_dir):
        return []

    retired: list[str] = []
    for path in sorted(folder.rglob("*"), reverse=True):
        relative = path.relative_to(project.root)
        if retire_takes(path) and relative not in keep:
            path.unlink()
            retired.append(str(relative))
        elif path.name == "__pycache__" and path.is_dir() and not path.is_symlink() and holds_only_bytecode(path):
            shutil.rmtree(path)
        elif path.is_dir() and not path.is_symlink() and not any(path.iterdir()):
            path.rmdir()

    return retired
