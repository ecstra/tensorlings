"""Reading and writing a file's frontmatter without touching its body.

The body is what a person wrote. Every function here hands it back
exactly as it came in, which is why nothing reflows, strips or re-indents
it. Frontmatter lines are kept as raw text too, so a field nobody edited
is written back character for character.

The header is the kit's own shape, not general YAML: one key per line
at the margin, `key: value`, each key once, nothing blank or indented
between the fences. A value is plain text taken as written, a colon
inside it is fine and nothing is quoted. A list sits on its key's line
as `[a, b]`. A header written any other way is refused, with the line
and the fix named, rather than read as something it is not.
"""

import errno
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from typing import Optional

FENCE = "---"

# What a write in flight is called, beside its target: the target's own
# name between a dot and the eight characters mkstemp draws, whatever
# the target, a brain file, the owner's word, VERSION or a hook config,
# since every write goes through the one writer. Doctor knows the exact
# shape and sweeps it where the tool writes in every run, so one left by
# an interrupted run there is found. A person's own dotfile matches only
# when its own suffix is eight such characters, so at the top of the
# project doctor looks only beside the kit's own files.
TEMP_SUFFIX = ".tmp"
TEMP_SHAPE = re.compile(r"^\.(.+)\.[a-z0-9_]{8}\.tmp$")

# What a field's key looks like: lowercase, digits and hyphens.
KEY = re.compile(r"^[a-z][a-z0-9-]*$")


class FrontmatterError(Exception):
    """A file that cannot be read as frontmatter plus body."""


@dataclass
class Document:
    """One brain file, split at the closing fence.

    `lines` holds the frontmatter as it was written, without the fences.
    `body` is everything after the closing fence line, including the
    newline that follows it.
    """

    lines: list[str]
    body: str

    def get(
        self,
        key: str,
    ) -> Optional[str]:
        for line in self.lines:
            name, value = split_field(line)
            if name == key:
                return value

        return None

    def set(
        self,
        key: str,
        value: str,
    ) -> None:
        """Set one field, on one line. A value that would not fit on a line is refused."""
        key, value = key.strip(), value.strip()
        if "\n" in key or "\r" in key or "\n" in value or "\r" in value:
            raise FrontmatterError(f"{key} is one line, and this value has a line break in it")

        for index, line in enumerate(self.lines):
            name, _ = split_field(line)
            if name == key:
                self.lines[index] = f"{key}: {value}"
                return

        self.lines.append(f"{key}: {value}")

    def drop(
        self,
        key: str,
    ) -> None:
        """Take one field out of the header, and do nothing when it is not there."""
        self.lines = [line for line in self.lines if split_field(line)[0] != key.strip()]

    def fields(self) -> dict[str, str]:
        found: dict[str, str] = {}

        for line in self.lines:
            name, value = split_field(line)
            if name is not None:
                found[name] = value or ""

        return found

    def get_list(
        self,
        key: str,
    ) -> list[str]:
        value = self.get(key)
        if value is None:
            return []

        return parse_list(value)

    def set_list(
        self,
        key: str,
        values: list[str],
    ) -> None:
        self.set(
            key=key,
            value=render_list(values),
        )


def split_field(
    line: str,
) -> tuple[Optional[str], Optional[str]]:
    """Split one frontmatter line into its name and its raw value.

    A field is a key of lowercase letters, digits and hyphens at the
    margin, a colon, then a space and the value or nothing. Anything
    else, an indented line, a capitalised or spaced key, a colon with no
    space after it, is not a field. The parser refuses such a line in a
    header, and the header guard uses the same answer to tell a field
    line from prose.
    """
    if ":" not in line or line[:1] in (" ", "\t"):
        return None, None

    name, _, value = line.partition(":")
    if KEY.match(name) is None or (value and not value.startswith(" ")):
        return None, None

    return name, value.strip()


def parse_list(
    value: str,
) -> list[str]:
    inner = value.strip()
    if not inner.startswith("[") or not inner.endswith("]"):
        return [inner] if inner else []

    inner = inner[1:-1].strip()
    if not inner:
        return []

    return [item.strip() for item in inner.split(",") if item.strip()]


def render_list(
    values: list[str],
) -> str:
    return "[" + ", ".join(values) + "]"


def parse(
    text: str,
) -> Document:
    """Split `text` into frontmatter lines and the body after them."""
    if not text.startswith(FENCE + "\n"):
        raise FrontmatterError("the file does not open with a --- fence")

    rest = text[len(FENCE) + 1 :]
    closing = find_closing_fence(rest)
    if closing is None:
        raise FrontmatterError("the frontmatter fence is never closed")

    head, body = rest[:closing], rest[closing:]
    lines = head.split("\n")[:-1] if head else []
    check_lines(lines)

    return Document(
        lines=lines,
        body=body[len(FENCE) + 1 :],
    )


def check_lines(
    lines: list[str],
) -> None:
    """Refuse a header the line model cannot hold, saying what to write instead.

    The header is one key per line, each key once. Two lines with one
    key would leave `get` and `fields` answering differently, and a
    block list under a key would read as nothing, so the file is refused
    before either can happen.
    """
    seen: set[str] = set()

    for number, line in enumerate(lines, start=2):
        if "\n" in line or "\r" in line:
            raise FrontmatterError(f"line {number} of the header has a line break in it")

        name, _ = split_field(line)
        if name is None:
            raise FrontmatterError(line_trouble(line=line, number=number))

        if name in seen:
            raise FrontmatterError(f"line {number} of the header repeats {name}, one key per line and each key once")

        seen.add(name)


def line_trouble(
    line: str,
    number: int,
) -> str:
    """Why a header line is not `key: value`, in words that say what to write."""
    if not line.strip():
        return f"line {number} of the header is blank, the kit writes one key per line with nothing between"

    if line.strip().startswith("- "):
        return f"line {number} of the header is a block list item, the kit writes a list on its key's line as [a, b]"

    if line[:1] in (" ", "\t"):
        return f"line {number} of the header is indented, the kit writes one key per line at the margin"

    return f"line {number} of the header is not key: value"


def find_closing_fence(
    rest: str,
) -> Optional[int]:
    """Offset of the closing fence line, or None when there is none.

    The body of a brain file often carries `---` as a rule between
    sections, so only a fence reached before any body content counts.
    """
    offset = 0

    for line in rest.split("\n"):
        if line == FENCE:
            return offset

        offset += len(line) + 1

    return None


def render(
    document: Document,
) -> str:
    """The file's text, refused when its header could not be read back.

    The same rules the parser holds are held here, so a document built
    with a line the parser would refuse never reaches the disk.
    """
    check_lines(document.lines)
    head = "\n".join(document.lines)
    if head:
        head += "\n"

    return f"{FENCE}\n{head}{FENCE}\n{document.body}"


def load(
    path: str,
) -> str:
    """The file's text, refused as frontmatter when it cannot be read as such.

    A file that is not UTF-8, a dangling symlink, a folder named like a
    file or a file the process may not open all become one refusal, so
    a caller that notes bad files notes these too instead of dying.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except UnicodeDecodeError as error:
        raise FrontmatterError("the file is not UTF-8 text") from error
    except OSError as error:
        raise FrontmatterError(f"the file cannot be read: {error.strerror or error}") from error


def read(
    path: str,
) -> Document:
    return parse(load(path))


def write(
    path: str,
    document: Document,
) -> None:
    """Write the document as a real file, whole or not at all."""
    write_text(path=path, text=render(document))


def target_of(
    name: str,
) -> Optional[str]:
    """The file a write in flight was for, read off its name, or None for a name of any other shape."""
    match = TEMP_SHAPE.match(name)

    return None if match is None else match.group(1)


def write_text(
    path: str,
    text: str,
) -> None:
    """Write text as a real file, whole or not at all.

    The text goes to a temporary file beside the target and is moved
    into place in one step, so a failure part way leaves the old file
    as it was. The brain holds real files: a symlink standing in the
    file's place is replaced rather than written through, while a
    folder somebody linked into the brain is theirs, and a file filed
    in it lives where they put the folder. A file or folder the process
    may not write is refused before anything is touched.
    """
    folder = os.path.dirname(path) or "."
    real_file = os.path.isfile(path) and not os.path.islink(path)
    if not os.access(folder, os.W_OK) or (real_file and not os.access(path, os.W_OK)):
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), path)

    mode = stat.S_IMODE(os.stat(path).st_mode) if real_file else default_mode()
    handle, temporary = tempfile.mkstemp(dir=folder, prefix=f".{os.path.basename(path)}.", suffix=TEMP_SUFFIX)

    try:
        with os.fdopen(handle, "w", encoding="utf-8") as opened:
            opened.write(text)
            opened.flush()
            os.fsync(opened.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        settle(folder)
    except BaseException:
        if os.path.lexists(temporary):
            os.unlink(temporary)
        raise


def settle(
    folder: str,
) -> None:
    """Flush the folder's own record of the move, where the platform allows it.

    The text is flushed to disk before the move, and the folder after
    it, so a crash right then leaves either the old file or the new one
    and never a shorter one. Windows cannot open a folder this way and
    gets no worse than before.
    """
    try:
        handle = os.open(folder, os.O_RDONLY)
    except OSError:
        return

    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)


def default_mode() -> int:
    """The mode a new file gets under the current umask, which mkstemp does not give."""
    current = os.umask(0)
    os.umask(current)

    return 0o666 & ~current
