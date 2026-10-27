"""Filing a new entry and changing one that is already filed.

Every field a kind carries is checked here before anything reaches disk,
so a file cannot arrive in the brain already breaking its spec. The
prose comes from the caller and is written through as it came, the
Steps section aside, whose numbered list becomes the table.
"""

from datetime import date
from pathlib import Path
from typing import Optional

from . import frontmatter, links, schema, sections, steps, store, task


class WriteError(Exception):
    """The change was refused, with the reason in the message."""


class Partial(WriteError):
    """The entry was written, and a step elsewhere that points at it could not be, with the reason in the message, so a caller never takes the raise for nothing written."""


def refresh_after(
    brain: store.Brain,
    entry: store.Entry,
    stamp: str,
) -> None:
    """Refresh every step that points at the entry just written, and say so as a partial write when the brain's folder refuses the second write."""
    try:
        steps.refresh_pointing(brain=brain, key=entry.relative, stamp=stamp)
    except OSError as error:
        raise Partial(f"{entry.name} was written, and a step that points at it could not be refreshed, {store.explain(error)}, which brain doctor reports until the folder can be written") from error


def today() -> str:
    return date.today().isoformat()


def create(
    brain: store.Brain,
    kind: schema.Kind,
    title: str,
    description: str,
    values: Optional[dict[str, str]] = None,
    body: str = "",
    category: str = "",
    links: Optional[list[str]] = None,
    stamp: str = "",
    in_place_of: Optional[set[str]] = None,
) -> store.Entry:
    """File a new entry, its steps checked against the keys of the entry it is filed in place of, since a refile hands that entry's pointers on to it."""
    stamp = stamp or today()

    slug = schema.slugify(title)
    if kind.name == "review" and not schema.is_review_folder(slug):
        raise WriteError(f"a review folder is named <yyyy-mm>-<branch-slug>, and '{slug}' is not")

    taken = brain.with_slug(
        kind=kind,
        slug=slug,
    )
    if taken is not None:
        raise WriteError(f"{taken.name} already carries that title")

    path = brain.path_for(
        kind=kind,
        title=title,
        category=category,
    )
    if path.exists():
        raise WriteError(f"{path.name} is already filed")

    name = entry_name(path=path, kind=kind)
    holder = holder_of(brain=brain, name=name)
    if holder is not None:
        raise WriteError(f"{name} is already the name of {holder}, and a name is unique across the brain")

    body = body if body.strip() else ""
    if body and sections.missing(body=body, kind=kind):
        wanted = ", ".join(sections.missing(body=body, kind=kind))
        raise WriteError(f"the body is missing the {wanted} heading")

    fields = opening_fields(
        kind=kind,
        name=name,
        description=description,
        links=path_links(brain=brain, references=links or []),
        stamp=stamp,
    )
    fields.update(defaults(kind=kind, stamp=stamp))
    fields.update(values or {})

    check(kind=kind, fields=fields)
    text = framed(body) if body else sections.skeleton(kind=kind, title=title)
    if kind.name in steps.STEPPED_KINDS:
        text = steps.tabled(brain=brain, body=text, by=fields.get("by") or "agent", kind=kind, own=in_place_of)
    refuse_gaps(sections.lifecycle_gaps(kind=kind, fields=fields, body=text) + steps.closing_gaps(brain=brain, kind=kind, fields=fields, body=text))

    document = frontmatter.Document(
        lines=render_lines(kind=kind, fields=fields),
        body=text,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter.write(path=str(path), document=document)
    brain.refresh()

    return brain.read(path=path, kind=kind)


def update(
    brain: store.Brain,
    key: str,
    changes: dict[str, str],
    stamp: str = "",
) -> store.Entry:
    entry = located(brain=brain, key=key)

    stamp = stamp or today()
    changes = dict(changes)
    if "links" in changes:
        changes["links"] = frontmatter.render_list(
            settled_links(
                brain=brain,
                wanted=frontmatter.parse_list(changes["links"]),
                existing=entry.document.get_list("links"),
            )
        )

    fields = entry.document.fields()
    fields.update(changes)

    # every change is judged against the lifecycle rules, not only one that
    # moves the status, so a done todo cannot lose its date, and every rule
    # the change falls short of is named at once
    check(kind=entry.kind, fields=fields)
    gaps = sections.lifecycle_gaps(kind=entry.kind, fields=fields, body=entry.document.body)
    moving = changes.get("status", entry.field("status")) != entry.field("status")
    if moving:
        gaps += steps.gaps(brain=brain, entry=entry, fields=fields)
        gaps += steps.reopening_gaps(brain=brain, entry=entry, fields=fields)

    if "became" in changes:
        gaps += steps.became_gaps(brain=brain, entry=entry, fields=fields)

    refuse_gaps(gaps)

    # a table moves as its readers would see it, so the file settles before any status move is written, a close from dropped as much as from open
    if moving and entry.kind.name in steps.STEPPED_KINDS:
        steps.settle(brain=brain, entry=entry)

    was_refiled = entry.field("status") == "refiled"
    for name, value in changes.items():
        entry.document.set(key=name, value=value)

    # an entry that leaves refiled behind was refiled as nothing any more
    if was_refiled and changes.get("status", "refiled") != "refiled" and "became" not in changes:
        entry.document.drop("became")

    entry.document.set(key="updated", value=stamp)
    frontmatter.write(path=str(entry.path), document=entry.document)
    brain.refresh()

    if "status" in changes and entry.kind.name in steps.POINTED_KINDS:
        refresh_after(brain=brain, entry=entry, stamp=stamp)

    return brain.read(path=entry.path, kind=entry.kind)


def retitle(
    brain: store.Brain,
    key: str,
    title: str,
    stamp: str = "",
) -> store.Entry:
    """Rename a file for a new title. The id stays, every link follows, and so does the chain where it holds the name."""
    entry = located(brain=brain, key=key)
    old_stem = entry.path.stem

    if entry.kind.name == "review":
        raise WriteError("a review keeps the name of its branch")

    slug = schema.slugify(title)
    if not slug:
        raise WriteError(f"'{title}' leaves nothing to name a file with")

    taken = brain.with_slug(kind=entry.kind, slug=slug)
    if taken is not None and taken.path != entry.path:
        raise WriteError(f"{taken.name} already carries that title")

    stem = f"{entry.identifier}-{slug}" if entry.identifier else slug
    target = entry.path.with_name(f"{stem}.md")
    if occupied(target=target, by=entry.path):
        raise WriteError(f"{target.relative_to(brain.root)} is already on disk, the brain does not write over it")

    holder = holder_of(brain=brain, name=stem)
    if holder is not None and holder != entry.relative:
        raise WriteError(f"{stem} is already the name of {holder}, and a name is unique across the brain")

    stamp = stamp or today()
    old_name = entry.name
    old_keys = steps.own_keys(entry)
    pointer = steps.pointer_of(entry)
    # read before the rename, since a file with no id is pointed at by its name, which the rename takes off it
    two_files = entry.kind.name in steps.POINTED_KINDS and steps.doubled(brain=brain, key=pointer)

    before = frontmatter.load(str(entry.path))
    moved = links.move_links(
        brain=brain,
        old=entry.relative,
        new=str(target.relative_to(brain.root)),
        stamp=stamp,
    )

    entry.document.set(key="name", value=stem)
    entry.document.set(key="updated", value=stamp)
    try:
        frontmatter.write(path=str(entry.path), document=entry.document)
        entry.path.rename(target)
    except OSError as error:
        undo_retitle(path=entry.path, before=before, moved=moved, error=error)
        raise
    finally:
        brain.refresh()

    # the writes that follow the rename go to other files, so each runs whatever the one before it did, and the failures are said together at the end, the first one chained on
    failures: list[str] = []
    first_failure: Optional[Exception] = None

    # a step cell that named the entry by hand follows it once the rename is done, written as the id, since a key move is no mark and the old name would strand, and a rename that fails leaves every cell as it was, while two files carry the id, only the cells that name this file move, to its new name, and a cell holding the doubled id stays, since it could mean either file, and a file with no id, which doctor reports, is pointed at by its name, so the cells follow it to the new one, unless two files carry that name too, when every cell could mean either and stays
    if entry.kind.name in steps.POINTED_KINDS and not (two_files and entry.identifier is None):
        by_name = two_files or entry.identifier is None
        try:
            steps.repoint(brain=brain, keys={old_name} if two_files else old_keys, to=stem if by_name else pointer, stamp=stamp)
        except OSError as error:
            first_failure = first_failure or error
            failures.append(f"a step that named it could not follow, {store.explain(error)}, which brain doctor reports until the folder can be written")

    # a became field written by hand may name the file by its name or its path rather than its id, so it follows the rename too, written as the id, or as the new name while two files carry the id or the file has none, and when two files with no id share the name only a field naming this file by its path moves, since the name could mean either
    if entry.kind.name in ("todo", "bug"):
        naming = {entry.relative} if two_files and entry.identifier is None else {old_name, old_stem, entry.relative}
        try:
            move_became(brain=brain, old=naming, to=stem if two_files or entry.identifier is None else entry.identifier, stamp=stamp)
        except OSError as error:
            first_failure = first_failure or error
            failures.append(f"a became field that named it could not follow, {store.explain(error)}, which brain doctor reports until the folder can be written")

    # the chain stores a plan by its file, and a todo or a bug with no id too, so the stem it holds follows
    if entry.kind.name == "plan" or entry.identifier is None:
        try:
            task.renamed(brain=brain, kind=entry.kind.name, old=old_stem, new=stem)
        except (OSError, task.TaskError) as error:
            first_failure = first_failure or error
            failures.append(f"the chain could not follow the name, {store.explain(error)}, so it says {task.WORDS[entry.kind.name]} {old_stem}, not in this checkout, until {task.FILE} can be written again and brain task off drops it, link by link, for brain task on {stem} to start it")

    if failures:
        raise Partial(f"{stem} was written, and {', and '.join(failures)}") from first_failure

    return brain.read(path=target, kind=entry.kind)


def move_became(
    brain: store.Brain,
    old: set[str],
    to: str,
    stamp: str,
) -> None:
    """Write every became field that names the entry by one of its old keys as the key given, in the refiled todos and bugs that carry one."""
    first_failure: Optional[OSError] = None
    for kind_name in ("todo", "bug"):
        for holder in brain.entries_of(schema.kind_named(kind_name)):
            became = holder.field("became") or ""
            if became not in old or became == to:
                continue

            holder.document.set(key="became", value=to)
            try:
                steps.save(brain=brain, entry=holder, stamp=stamp)
            except OSError as error:
                first_failure = first_failure or error

    if first_failure is not None:
        raise first_failure


def undo_retitle(
    path: Path,
    before: str,
    moved: list[links.Planned],
    error: OSError,
) -> None:
    """Put the file and every link that followed it back, or say which would not go.

    The file goes back only if it changed. A write refused up front
    left it as it was, and a refused file cannot take the restore either.
    What could not go back is named beside the failure that started it.
    """
    stuck: list[Path] = []

    try:
        if frontmatter.load(str(path)) != before:
            frontmatter.write_text(path=str(path), text=before)
    except (OSError, frontmatter.FrontmatterError):
        stuck.append(path)

    stuck += links.put_back_all(moved)
    if stuck:
        raise OSError(f"{store.explain(error)}, and {links.names_of(stuck)} could not be put back") from error


def located(
    brain: store.Brain,
    key: str,
) -> store.Entry:
    """The one entry a key names, refused in the writer's own words when none or several do."""
    try:
        entry = brain.find(key)
    except store.Ambiguous as clash:
        raise WriteError(str(clash)) from clash

    if entry is None:
        raise WriteError(f"nothing in the brain answers to '{key}'")

    return entry


def holder_of(
    brain: store.Brain,
    name: str,
) -> Optional[str]:
    """The brain relative path of whatever already carries this name, entry or kit file, or None.

    A name is unique across the whole brain, the kit's own files
    included, so a doc may not be called conventions while the spec is.
    A name two files already share is refused with both named.
    """
    try:
        entry = brain.find(name)
    except store.Ambiguous as clash:
        raise WriteError(str(clash)) from clash

    if entry is not None:
        return entry.relative

    for path, document in links.kit_documents(brain):
        if document.get("name") == name:
            return str(path.relative_to(brain.root))

    return None


def occupied(
    target: Path,
    by: Path,
) -> bool:
    """Whether something other than the file being renamed already sits at `target`.

    On a file system that folds case, a rename that changes only the
    case sees itself at the target, and that is not an occupant.
    """
    if store.quietly(target.is_symlink):
        return True

    if not store.quietly(target.exists):
        return False

    try:
        return not target.samefile(by)
    except OSError:
        return True


def convert(
    brain: store.Brain,
    key: str,
    kind: schema.Kind,
    description: str,
    values: Optional[dict[str, str]] = None,
    body: str = "",
    stamp: str = "",
) -> store.Entry:
    """File a todo as a bug, or the reverse, keeping the id trail in links.

    The old file stays, marked refiled, and the two point at each other.
    An id is never reused, so the old one is not deleted. The closing
    goes through `update`, so the same rules hold as for any status
    change, and a closing that is refused takes the new file with it.
    """
    source = located(brain=brain, key=key)

    if source.kind.name == kind.name:
        raise WriteError(f"{source.name} is already a {kind.name}")

    if source.kind.name not in ("todo", "bug") or kind.name not in ("todo", "bug"):
        raise WriteError("only a todo and a bug convert into each other")

    stamp = stamp or today()
    created = create(
        brain=brain,
        kind=kind,
        title=source.slug.replace("-", " "),
        description=description,
        values=values,
        body=body,
        links=[source.relative],
        stamp=stamp,
        in_place_of=steps.own_keys(source),
    )

    source = located(brain=brain, key=key)

    trail = [link for link in source.document.get_list("links") if link != created.relative]
    closing = {"status": "refiled", "links": frontmatter.render_list([*trail, created.relative]), "became": created.identifier or created.relative}
    if source.kind.name == "todo":
        closing["done"] = stamp

    try:
        update(brain=brain, key=key, changes=closing, stamp=stamp)
    except Partial as partial:
        raise Partial(f"{created.relative} was filed and {partial}") from partial
    except (WriteError, OSError):
        created.path.unlink()
        brain.refresh()
        raise

    return created


def framed(
    body: str,
) -> str:
    """The prose as given, with a blank line before it and a newline after.

    The text itself is untouched. This only settles where the body sits
    against the closing fence, which the caller should not have to think
    about.
    """
    opening = "" if body.startswith("\n") else "\n"
    closing = "" if body.endswith("\n") else "\n"

    return f"{opening}{body}{closing}"


def entry_name(
    path: Path,
    kind: schema.Kind,
) -> str:
    if kind.name == "review":
        return path.parent.name

    return path.stem


def settled_links(
    brain: store.Brain,
    wanted: list[str],
    existing: list[str],
) -> list[str]:
    """The links a change leaves, a new one in path form or refused, an old one as it was.

    A link the file already carried is not this change's to answer for.
    If it is stale, doctor reports it, and a change that only adds a
    link beside it, a refile for one, is not blocked by it.
    """
    found: list[str] = []
    reached: list[str] = []

    for link in wanted:
        settled = [link] if link in existing else path_links(brain=brain, references=[link])
        for candidate in settled:
            target = links.canonical(brain=brain, reference=candidate) or candidate
            if target in reached:
                continue

            reached.append(target)
            found.append(candidate)

    return found


def path_links(
    brain: store.Brain,
    references: list[str],
) -> list[str]:
    """Every link in path form, each once, refused when one reaches nothing in the brain."""
    found: list[str] = []

    for reference in references:
        path = links.canonical(brain=brain, reference=reference)
        if path is None:
            raise WriteError(f"link '{reference}' reaches nothing inside .agents/")

        if path not in found:
            found.append(path)

    return found


def opening_fields(
    kind: schema.Kind,
    name: str,
    description: str,
    links: list[str],
    stamp: str,
) -> dict[str, str]:
    return {
        "name": name,
        "kind": kind.kind_value,
        "description": description,
        "updated": stamp,
        "links": frontmatter.render_list(links),
    }


def defaults(
    kind: schema.Kind,
    stamp: str,
) -> dict[str, str]:
    """The fields a new file of this kind can fill in for itself."""
    starting = {
        "bug": {"status": "open", "reported": stamp},
        "todo": {"status": "open", "born": stamp},
        "decision": {"status": "standing", "date": stamp},
        "plan": {"status": "open", "born": stamp},
        "review": {"status": "open"},
    }

    return dict(starting.get(kind.name, {}))


def render_lines(
    kind: schema.Kind,
    fields: dict[str, str],
) -> list[str]:
    """Frontmatter lines in the order the spec lists them, each value as written, trimmed."""
    return [
        f"{item.name}: {fields[item.name].strip()}"
        for item in schema.fields_for(kind)
        if item.name in fields
    ]


def check(
    kind: schema.Kind,
    fields: dict[str, str],
) -> None:
    """Every field against the spec, the common ones and the kind's own."""
    check_against(kind=kind, fields=fields, items=schema.fields_for(kind))


def check_own(
    kind: schema.Kind,
    fields: dict[str, str],
) -> None:
    """The kind's own fields only, for a caller that checks the common ones itself."""
    check_against(kind=kind, fields=fields, items=kind.fields)


def check_against(
    kind: schema.Kind,
    fields: dict[str, str],
    items: tuple[schema.Field, ...],
) -> None:
    known = {item.name for item in schema.fields_for(kind)}

    for name in fields:
        if name not in known:
            raise WriteError(f"{kind.name} files carry no '{name}' field")

    for item in items:
        value = fields.get(item.name, "").strip()
        if "\n" in value or "\r" in value:
            raise WriteError(f"{item.name} is one line, and this one has a line break in it")

        for slip in sections.style_slips(value):
            raise WriteError(f"{item.name} {slip}")

        if item.required and not value:
            raise WriteError(f"a {kind.name} needs {item.name} filled in")

        if not value:
            continue

        if item.values and value not in item.values:
            allowed = " or ".join(item.values)
            raise WriteError(f"{item.name} is {allowed}, not '{value}'")

        if item.is_date and not schema.is_date(value):
            raise WriteError(f"{item.name} reads YYYY-MM-DD, not '{value}'")


def refuse_gaps(
    gaps: list[str],
) -> None:
    """Refuse with every rule the file would fall short of, named together, or pass when none."""
    if gaps:
        raise WriteError(", and ".join(gaps))
