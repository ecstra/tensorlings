"""Finding the brain, reading what is in it, and naming what goes in next.

Every command starts here. Ids come from the files themselves, so the
next one is the highest that exists plus one. A gap left by a file that
moved or was deleted stays a gap, because the id it held is written into
commits, ledgers and links that nobody can rewrite.
"""

import os
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable, Optional

from . import frontmatter, schema

# what the store notes for a folder it could not list, beside the files it could not read
UNLISTED = "cannot be listed"

# the hosts Windows serves a WSL distro's files under, casefolded
WSL_HOSTS = ("wsl.localhost", "wsl$")

# where WSL mounts each Windows drive, one folder per letter
WSL_DRIVES = PurePosixPath("/mnt")


class StoreError(Exception):
    """The brain is missing, or something in it cannot be read."""


class Ambiguous(StoreError):
    """A key that names more than one file, with the paths to choose from in the message."""


@dataclass
class Entry:
    """One file in the brain, with its frontmatter already parsed."""

    path: Path
    root: Path
    kind: schema.Kind
    document: frontmatter.Document

    @property
    def relative(self) -> str:
        return str(self.path.relative_to(self.root))

    @property
    def name(self) -> str:
        return self.document.get("name") or self.path.stem

    @property
    def identifier(self) -> Optional[str]:
        match = schema.FILENAME_WITH_ID.match(self.path.name)
        if match is None:
            return None

        return f"{match.group(1)}-{match.group(2)}"

    @property
    def slug(self) -> str:
        """The filename with its id taken off, which is what a title becomes."""
        match = schema.FILENAME_WITH_ID.match(self.path.name)
        if match is not None:
            return match.group(3)

        if self.kind.name == "review":
            return self.path.parent.name

        return self.path.stem

    @property
    def number(self) -> Optional[int]:
        match = schema.FILENAME_WITH_ID.match(self.path.name)
        if match is None:
            return None

        return int(match.group(2))

    def field(
        self,
        key: str,
    ) -> Optional[str]:
        return self.document.get(key)


@dataclass
class Brain:
    """The `.agents/` directory and everything filed inside it.

    Each folder is scanned once per command and the result kept, since a
    doctor run asks about the same files hundreds of times. Anything that
    writes calls `refresh` so the next question reads what is on disk.
    """

    root: Path

    def __post_init__(self) -> None:
        self.loaded: dict[str, list[Entry]] = {}
        self.unreadable: dict[Path, str] = {}
        self.keyed: Optional[dict[str, list[Entry]]] = None
        self.kit_read: Optional[list[tuple[Path, frontmatter.Document]]] = None

    def refresh(self) -> None:
        self.loaded = {}
        self.unreadable = {}
        self.keyed = None
        self.kit_read = None

    def folder(
        self,
        kind: schema.Kind,
    ) -> Path:
        return self.root / kind.folder

    def entries(
        self,
        kind: Optional[schema.Kind] = None,
    ) -> list[Entry]:
        kinds = [kind] if kind is not None else list(schema.KINDS.values())
        found: list[Entry] = []

        for one in kinds:
            found.extend(self.entries_of(one))

        return found

    def entries_of(
        self,
        kind: schema.Kind,
    ) -> list[Entry]:
        if kind.name not in self.loaded:
            self.loaded[kind.name] = self.scan(kind)

        return list(self.loaded[kind.name])

    def scan(
        self,
        kind: schema.Kind,
    ) -> list[Entry]:
        """Read one folder. A file that cannot be read or parsed is noted, not fatal, and so is a folder that cannot be listed."""
        folder = self.folder(kind)
        if not folder.is_dir():
            return []

        found: list[Entry] = []

        for path in self.markdown_under(folder):
            if hidden(path.relative_to(folder)) or path.name == schema.INDEX:
                continue

            if kind.name == "review" and (path.name != schema.LEDGER or path.parent.parent != folder):
                continue

            try:
                found.append(self.read(path=path, kind=kind))
            except StoreError as error:
                self.unreadable[path] = str(error)

        return found

    def markdown_under(
        self,
        folder: Path,
    ) -> list[Path]:
        """Every path named like a markdown file under the folder, sorted, with a folder the walk could not list noted as unreadable rather than skipped without a word.

        A folder named like a file is listed too, so the read notes it
        the way it always has, and the suffix is read the way the
        platform's glob reads it, without regard to case on Windows, so
        the store and the doctor's own glob name the same files. A
        dotted folder is never entered, since the brain never lists
        what is under one, and a folder gone between the listing and
        the descent, a scratch folder another write just removed, is
        nothing to note.
        """
        trouble: list[OSError] = []

        def note(error: OSError) -> None:
            if not isinstance(error, FileNotFoundError):
                trouble.append(error)

        paths: list[Path] = []
        for root, folders, names in os.walk(folder, onerror=note):
            folders[:] = [name for name in folders if not schema.is_hidden(name)]
            paths += [Path(root) / name for name in folders + names if os.path.normcase(name).endswith(".md")]

        for error in trouble:
            self.unreadable[Path(error.filename)] = UNLISTED

        return sorted(paths)

    def read(
        self,
        path: Path,
        kind: schema.Kind,
    ) -> Entry:
        try:
            document = frontmatter.read(str(path))
        except frontmatter.FrontmatterError as error:
            raise StoreError(f"{near(str(path))}: {error}") from error

        return Entry(
            path=path,
            root=self.root,
            kind=kind,
            document=document,
        )

    def known(self) -> set[Path]:
        """Every path the store lists, the entries and the files it could not read."""
        listed = {entry.path for entry in self.entries()}

        return listed | set(self.unreadable)

    def kit_files(self) -> list[Path]:
        """The kit's own files that exist here, the rulebook excepted."""
        candidates = [self.root / name for name in schema.KIT_FILES if name not in schema.NO_FRONTMATTER]
        candidates += [path for path in sorted((self.root / "spec").glob("*.md")) if path.name != schema.INDEX and not hidden(Path(path.name))]
        candidates += [self.root / folder / schema.INDEX for folder in schema.INDEXED_FOLDERS]

        return [path for path in candidates if readable_file(path)]

    def next_number(
        self,
        kind: schema.Kind,
    ) -> int:
        if not kind.has_ids:
            raise StoreError(f"{kind.name} files carry no id")

        numbers = [entry.number for entry in self.entries_of(kind)]
        numbers += [number_in(filename=path.name, kind=kind) for path in self.unreadable if self.folder(kind) in path.parents]
        taken = [number for number in numbers if number is not None]
        if not taken:
            return 1

        return max(taken) + 1

    def next_id(
        self,
        kind: schema.Kind,
    ) -> str:
        return schema.format_id(
            prefix=kind.id_prefix,
            number=self.next_number(kind),
        )

    def find(
        self,
        key: str,
    ) -> Optional[Entry]:
        """Resolve a key to one entry, or to nothing when no file answers to it.

        A key is an id, a name, or a path to the file with slashes or
        backslashes. The path may start in the brain, in the project's
        folder, at the top of the git repository or at the root of the
        disk. Everything up to the last brain folder in it is dropped
        before the lookup, once what stood in front of that folder is
        known to lead to this project, and a path into another
        project's brain is found nowhere.

        A key that fits more than one file, a stem two kinds share or
        the ledger.md every review is, is refused with the paths named,
        rather than answered with whichever came first.
        """
        parts: list[str] = key.replace(
            "\\",
            "/",
        ).split("/")
        brain_folders: list[int] = [index for index, part in enumerate(parts[:-1]) if part == self.root.name]
        if brain_folders:
            brain_folder: int = brain_folders[-1]
            in_front: str = "".join(f"{part}/" for part in parts[:brain_folder])
            if in_front and not self.is_this_project(in_front):
                return None

            parts = parts[brain_folder + 1 :]

        found = self.by_key().get(
            "/".join(parts),
            [],
        )
        if len(found) > 1:
            paths = ", ".join(entry.relative for entry in found)
            raise Ambiguous(f"'{key}' names {len(found)} files, use the path: {paths}")

        return found[0] if found else None

    def is_this_project(
        self,
        folder: str,
    ) -> bool:
        """Whether the folder a path names in front of a brain folder is this brain's project.

        A leading ~ is the home folder. It is this project when it is the
        same folder on disk, given absolute or read from the current
        folder. A path that is not absolute here is also taken when its
        folders are the last ones of the project's own path, which is
        how git prints a path from a top above the project.

        A Windows path, with a drive letter or a share in front, on a
        system that does not read it as one, names a folder inside WSL
        as `wsl_parts` reads it. When that folder is on disk it is this
        project when it is the same folder, so a drive mount that opens
        folders in any case is asked rather than second guessed. When it
        is not, it is this project only when it spells the project's
        whole path exactly. Outside WSL it is never a folder here.
        """
        project: Path = self.root.parent
        expanded: str = os.path.expanduser(folder)
        try:
            same_folder: bool = os.path.samefile(
                f1=Path.cwd() / expanded,
                f2=project,
            )
        except (OSError, ValueError):
            # a folder that is not on disk, or cannot be read, is not this project
            same_folder = False

        if same_folder:
            return True

        windows = PureWindowsPath(expanded)
        # POSIX reads a share, //wsl.localhost/Ubuntu/..., as absolute, so a Windows path is settled before the refusal below
        if windows.drive and windows.root and not Path(expanded).drive:
            inside_wsl: Optional[tuple[str, ...]] = wsl_parts(windows)
            if inside_wsl is None:
                return False

            try:
                return os.path.samefile(
                    f1=Path(*inside_wsl),
                    f2=project,
                )
            except (OSError, ValueError):
                # the folder is not on disk, or cannot be read, so only its spelling can say
                return inside_wsl == project.parts

        if Path(expanded).is_absolute():
            return False

        folders: tuple[str, ...] = PurePosixPath(expanded).parts

        return folders == project.parts[-len(folders) :]

    def by_key(self) -> dict[str, list[Entry]]:
        """Every entry under each key it answers to, built once per scan.

        A link check asks `find` once per link, so the answer is a
        lookup rather than a walk of every entry each time.
        """
        if self.keyed is None:
            keyed: dict[str, list[Entry]] = {}
            for entry in self.entries():
                # `find` turns a key's backslashes to slashes, so on Windows only this spelling of the path can match it
                with_slashes: str = entry.path.relative_to(entry.root).as_posix()
                keys = {entry.identifier, entry.name, entry.relative, with_slashes, entry.path.stem} - {None}
                # a review's file is its folder's ledger, so the folder's name answers for it as a plan's stem does
                if entry.kind.name == "review":
                    keys.add(entry.path.parent.name)

                for key in keys:
                    keyed.setdefault(key, []).append(entry)
            self.keyed = keyed

        return self.keyed

    def kit_documents(self) -> list[tuple[Path, frontmatter.Document]]:
        """The kit's own files that carry a header the tool reads, read once per scan.

        Each caller gets its own copy of every document, so a change one
        makes in memory never shows up in another's read.
        """
        if self.kit_read is None:
            found: list[tuple[Path, frontmatter.Document]] = []
            for path in self.kit_files():
                try:
                    found.append((path, frontmatter.read(str(path))))
                except frontmatter.FrontmatterError:
                    continue
            self.kit_read = found

        return [
            (path, replace(document, lines=list(document.lines)))
            for path, document in self.kit_read
        ]

    def with_slug(
        self,
        kind: schema.Kind,
        slug: str,
    ) -> Optional[Entry]:
        """The entry already filed under this title, whatever id it carries."""
        for entry in self.entries_of(kind):
            if entry.slug == slug:
                return entry

        return None

    def path_for(
        self,
        kind: schema.Kind,
        title: str,
        category: str = "",
    ) -> Path:
        slug = schema.slugify(title)
        if not slug:
            raise StoreError(f"'{title}' leaves nothing to name a file with")

        folder = self.folder(kind)
        if category:
            folder = folder / category_folder(category)

        if kind.name == "review":
            return folder / slug / schema.LEDGER

        if kind.has_ids:
            return folder / f"{self.next_id(kind)}-{slug}.md"

        return folder / f"{slug}.md"


def wsl_parts(
    windows: PureWindowsPath,
) -> Optional[tuple[str, ...]]:
    """The folders a Windows path names inside the WSL distro the tool runs in, or None when it names none.

    The distro is the one WSL_DISTRO_NAME names, and outside WSL the
    variable is unset, so no Windows path names a folder there. A drive
    letter path is that letter's folder under /mnt, the letter lowered.
    A share is the distro's root only when its host is wsl.localhost or
    wsl$ and the share is the distro, both in any case.
    """
    distro: str = os.environ.get(
        "WSL_DISTRO_NAME",
        "",
    )
    if distro == "":
        return None

    below_drive: tuple[str, ...] = windows.parts[1:]
    if not windows.drive.startswith("\\\\"):
        return (*WSL_DRIVES.parts, windows.drive[0].lower(), *below_drive)

    host, _, share = windows.drive[2:].partition("\\")
    if host.casefold() not in WSL_HOSTS or share.casefold() != distro.casefold():
        return None

    return ("/", *below_drive)


def hidden(
    relative: Path,
) -> bool:
    """Whether any part of the path starts with a dot, per schema.is_hidden."""
    return schema.is_hidden("/".join(relative.parts))


def number_in(
    filename: str,
    kind: schema.Kind,
) -> Optional[int]:
    """The id number a filename of this kind carries, readable or not.

    An id is taken the moment a file holds it in its name, so a file the
    store could not read still keeps its number out of circulation.
    """
    match = schema.FILENAME_WITH_ID.match(filename)
    if match is None or match.group(1) != kind.id_prefix:
        return None

    return int(match.group(2))


def quietly(
    question: Callable[[], bool],
) -> bool:
    """A path question such as `path.is_file`, answered False instead of raised.

    On the 3.9 floor `is_file`, `is_dir`, `is_symlink` and `exists` raise
    when the folder above cannot be searched, where newer interpreters
    answer False. Every gate above a read or a decision asks here, so
    both floors degrade the same way.
    """
    try:
        return question()
    except OSError:
        return False


def readable_file(
    path: Path,
) -> bool:
    """True when `path` is a file the process can see."""
    return quietly(path.is_file)


def same_bytes(
    one: Path,
    other: Path,
) -> bool:
    """Whether the two files hold exactly the same bytes, false when either cannot be read.

    The one answer to "is this file the kit's copy", asked by doctor of
    a skill folder and by the upgrade of a file it would replace.
    """
    try:
        return one.read_bytes() == other.read_bytes()
    except OSError:
        return False


def stamp_text(
    path: Path,
) -> str:
    """A version stamp's text, trimmed, a byte order mark dropped, the one reading every user of a stamp goes through.

    Raises OSError when the file cannot be read and UnicodeDecodeError
    when it is not UTF-8, for each caller to turn into its own answer.
    """
    return path.read_text(encoding="utf-8-sig").strip()


def stamp_of(
    brain: Path,
) -> Optional[tuple[int, ...]]:
    """The kit release the brain is stamped with, as numbers, or None when the stamp is missing or not a version."""
    stamp = brain / schema.VERSION_FILE
    if not readable_file(stamp):
        return None

    try:
        return schema.as_tuple(stamp_text(stamp))
    except (OSError, UnicodeDecodeError, ValueError):
        return None


def kit_skills_shipped(
    brain: Path,
) -> list[str]:
    """The kit skills a brain should carry by its stamp, those the stamped release shipped, with no stamp to read taken as the oldest release."""
    stamped = stamp_of(brain) or schema.as_tuple(schema.NO_VERSION)

    return [skill.name for skill in schema.KIT_SKILLS if schema.ships(skill=skill, stamp=stamped)]


def skill_folders(
    brain: Path,
) -> list[str]:
    """The folders under skills/ that carry a SKILL.md and a slug for a name, in name order.

    Claude Code loads a skill only under a slug, lowercase words joined
    by hyphens, so a folder named otherwise, a dotted scratch folder
    included, is nobody's skill and never gets a link or a spine line.
    Raises OSError when the folder is there but cannot be listed, so a
    caller that checks the skills can say so rather than see none.
    """
    folder = brain / schema.SKILLS
    if not quietly(folder.is_dir):
        return []

    return sorted(
        path.name
        for path in folder.iterdir()
        if schema.is_slug(path.name) and readable_file(path / schema.SKILL_FILE)
    )


def project_skills(
    brain: Path,
    shipped: list[str],
) -> list[str]:
    """The skill folders the project keeps beside the kit's shipped ones, in name order.

    The one split of skills/ into the kit's and the project's, read by
    doctor for the links and by the spine for its lines, so the two
    never disagree. Raises OSError when the folder cannot be listed,
    the way `skill_folders` does.
    """
    return [name for name in skill_folders(brain) if name not in shipped]


def category_folder(
    category: str,
) -> Path:
    """The folder a category names, one or more slugs deep, never outside the kind's folder."""
    parts = category.split("/")
    if not all(schema.is_slug(part) for part in parts):
        raise StoreError(f"'{category}' is not a category, which is one or more slugs such as systems/auth")

    return Path(*parts)


def explain(
    error: BaseException,
) -> str:
    """The error as one plain line, an OS error without the machine's path.

    A KeyError prints its key in quotes, which is the key that was
    looked up and not found, so it reads as that.
    """
    if isinstance(error, OSError) and error.filename:
        reason = error.strerror or type(error).__name__

        return f"{reason}: {near(os.fsdecode(error.filename))}"

    if isinstance(error, KeyError):
        key = error.args[0] if error.args else "a key"

        return f"{key} is missing"

    return str(error)


# The folders a message may name a path from, so it reads the same on
# every machine: the brain and the three agents' own folders.
NEAR_ROOTS = (".agents", ".claude", ".codex", ".gemini")


def near(
    path: str,
) -> str:
    """A path from the brain or an agent folder down, or just the filename elsewhere.

    The last such folder on the path wins, since a checkout can itself
    live under a folder named .claude, as Claude Code's own projects do.
    """
    parts = Path(path).parts
    positions = [position for position, part in enumerate(parts) if part in NEAR_ROOTS]
    if positions:
        return str(Path(*parts[positions[-1] :]))

    return Path(path).name


def find_brain(
    start: Optional[Path] = None,
) -> Brain:
    """Walk up from `start` until the `.agents/` directory turns up."""
    try:
        here = (start or Path.cwd()).resolve()
    except (OSError, RuntimeError, ValueError) as error:
        raise StoreError(f"the working directory cannot be resolved: {error}") from error

    for candidate in [here, *here.parents]:
        if candidate.name == ".agents" and readable_file(candidate / "INDEX.md"):
            return Brain(root=candidate)

        inside = candidate / ".agents"
        if readable_file(inside / "INDEX.md"):
            return Brain(root=inside)

    raise StoreError(
        "no .agents/ directory here or above, so there is no brain to work on"
    )
