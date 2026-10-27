"""The brain tool. Files, changes, counts and checks everything in .agents/.

Run from anywhere inside a project that carries the kit:

    python3 .agents/skills/agent-brain/scripts/brain.py status

Prose for a new file comes in on standard input, so a heredoc in single
quotes passes it through without the shell touching it. Exit codes: 0
when it did the thing, 1 when it refused and said why, 2 when doctor
still has findings.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

from brainlib import autoack, clock, doctor, frontmatter, index, ledger, plans, reads, schema, steps, store, task, upgrade, writes

REFUSED = 1
FINDINGS_REMAIN = 2

FIELD_FLAGS = ("status", "severity", "area", "by", "type", "branch", "date")


def main(
    argv: Optional[list[str]] = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return REFUSED

    # a step's text is read as UTF-8 whatever the locale says, so a character the output codec lacks is written as its escape rather than ending the command in a traceback
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")

    # an argument the shell handed over in bytes that are not UTF-8 carries escapes no file takes, so it is refused up front the way standard input is
    for word in sys.argv[1:] if argv is None else argv:
        try:
            word.encode("utf-8")
        except UnicodeEncodeError:
            print(f"an argument is not UTF-8, '{word.encode('utf-8', 'backslashreplace').decode('ascii')}'", file=sys.stderr)
            return REFUSED

    try:
        # port and upgrade run before a brain exists, and the flag is theirs alone, since a default a parent parser sets too would win over theirs on Python 3.9.8 and 3.10.0
        if not getattr(args, "needs_brain", True):
            return args.handler(brain=None, args=args)

        brain = store.find_brain(Path.cwd())
        return args.handler(brain=brain, args=args)
    except writes.Partial as partial:
        # the entry was written, so the command did its work and exits as done, the indexes follow it where they can, and the second write that failed is named on standard error
        try:
            index.write(brain)
        except OSError as error:
            print(store.explain(error), file=sys.stderr)
        print(store.explain(partial), file=sys.stderr)
        return 0
    except (
        store.StoreError,
        writes.WriteError,
        ledger.LedgerError,
        plans.PlanError,
        steps.StepsError,
        task.TaskError,
        upgrade.UpgradeError,
        autoack.AckError,
        frontmatter.FrontmatterError,
        schema.UnknownKind,
        OSError,
        KeyError,
    ) as error:
        print(store.explain(error), file=sys.stderr)
        return REFUSED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brain",
        description="file, change, count and check what is in .agents/",
    )
    commands = parser.add_subparsers(dest="command")

    add_version(commands)
    add_status(commands)
    add_ack(commands)
    add_task(commands)
    add_list(commands)
    add_show(commands)
    add_path(commands)
    add_links(commands)
    add_new(commands)
    add_set(commands)
    add_refile(commands)
    add_index(commands)
    add_doctor(commands)
    add_steps(commands)
    add_ledger(commands)
    add_port(commands)
    add_upgrade(commands)

    return parser


def add_version(
    commands: argparse._SubParsersAction,
) -> None:
    command = commands.add_parser("version", help="the kit version this project carries")
    command.set_defaults(handler=run_version)


def run_version(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    print(reads.version(brain))
    return 0


def add_ack(
    commands: argparse._SubParsersAction,
) -> None:
    command = commands.add_parser("ack", help="the owner's standing word to proceed, on with an end, off, or where it stands")
    command.add_argument("state", nargs="?", choices=("on", "off"), help="on or off, nothing to see where it stands")
    command.add_argument("--for", dest="lasting", help="how long, such as 8h, 90m or 2d")
    command.add_argument("--until", help="a moment, such as 22:00, tomorrow 22:00 or 2026-09-24 22:00")
    command.add_argument("--until-done", dest="task", help="an entry the word ends with, a todo, a bug or a review, once it is done or gone, a bug on a branch still counting as open, and one the brain cannot read right now holding the word off")
    command.set_defaults(handler=run_ack)


def run_ack(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    an_end = args.lasting is not None or args.until is not None or args.task is not None
    if args.state == "off" and an_end:
        print("off takes no end, give it alone", file=sys.stderr)
        return REFUSED

    if args.state is None and an_end:
        print("give on to set the word, its end goes with on", file=sys.stderr)
        return REFUSED

    if args.state == "off":
        autoack.turn_off(brain=brain)

    if args.state == "on":
        autoack.turn_on(
            brain=brain,
            hours=autoack.parse_duration(args.lasting) if args.lasting is not None else None,
            until_text=args.until,
            task=args.task,
        )

    print(clock.line())
    print(autoack.describe(brain=brain))
    return 0


def add_task(
    commands: argparse._SubParsersAction,
) -> None:
    command = commands.add_parser("task", help="the current work: on an entry, its step done, skipped or pending again, off, or where it stands")
    actions = command.add_subparsers(dest="action")

    starting = actions.add_parser("on", help="start work on a plan, a todo, a bug or a review")
    starting.add_argument("key", help="the entry, by id, name or path")
    starting.add_argument("--replace", action="store_true", help="in place of the chain that stands, which is otherwise kept")

    finishing = actions.add_parser("done", help="mark the step the work stands on done, and stand on the next")
    finishing.set_defaults(status="done")

    skipping = actions.add_parser("skip", help="mark the step the work stands on skipped, and stand on the next")
    skipping.set_defaults(status="skipped")

    reopening = actions.add_parser("step", help="mark a step pending again, and stand on the first pending step")
    reopening.add_argument("number", type=int, help="the step's number in the entry's table")

    stopping = actions.add_parser("off", help="drop the innermost link, or the whole chain")
    stopping.add_argument("--all", dest="everything", action="store_true", help="the whole chain, and the file with it")

    # the one handler picks the action itself, since a default the sub actions set too would lose to it on Python 3.9.8 and 3.10.0
    command.set_defaults(handler=run_task)


def run_task(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    handlers = {
        None: run_task_show,
        "on": run_task_on,
        "done": run_task_mark,
        "skip": run_task_mark,
        "step": run_task_step,
        "off": run_task_off,
    }

    return handlers[args.action](brain=brain, args=args)


def run_task_show(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    print(task.describe(brain=brain))
    return 0


def run_task_on(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    task.turn_on(
        brain=brain,
        key=args.key,
        replace=args.replace,
    )
    print(task.describe(brain=brain))
    return 0


def run_task_mark(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    task.mark(brain=brain, status=args.status)
    index.write(brain)
    print(task.describe(brain=brain))
    return 0


def run_task_step(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    task.reopen(brain=brain, number=args.number)
    index.write(brain)
    print(task.describe(brain=brain))
    return 0


def run_task_off(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    task.turn_off(brain=brain, everything=args.everything)
    print(task.describe(brain=brain))
    return 0


def add_status(
    commands: argparse._SubParsersAction,
) -> None:
    command = commands.add_parser("status", help="the whole picture in a few lines")
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=run_status)


def run_status(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    data = reads.status(brain=brain)
    if args.json:
        print(json.dumps(data, indent=2))
        return 0

    print(clock.line())
    print(autoack.describe(brain=brain))
    print(task.describe(brain=brain))
    print(reads.render_status(data))
    return 0


def add_list(
    commands: argparse._SubParsersAction,
) -> None:
    command = commands.add_parser("list", help="one line per entry of a kind, narrowed by field")
    command.add_argument("kind", choices=sorted(schema.KINDS))
    for flag in FIELD_FLAGS:
        command.add_argument(f"--{flag}")
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=run_list)


def run_list(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    kind = schema.kind_named(args.kind)
    filters = {flag: getattr(args, flag) for flag in FIELD_FLAGS if getattr(args, flag)}
    found = reads.rows(brain=brain, kind=kind, filters=filters)
    print(json.dumps(found, indent=2) if args.json else reads.render_rows(kind=kind, found=found))
    return 0


def add_show(
    commands: argparse._SubParsersAction,
) -> None:
    command = commands.add_parser("show", help="print one entry, by id, name or path")
    command.add_argument("key")
    command.add_argument("--meta", action="store_true", help="the frontmatter only")
    command.set_defaults(handler=run_show)


def run_show(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    text = reads.show(brain=brain, key=args.key, meta_only=args.meta)
    if text is None:
        return missing(args.key)

    print(text, end="" if text.endswith("\n") else "\n")
    return 0


def add_path(
    commands: argparse._SubParsersAction,
) -> None:
    command = commands.add_parser("path", help="the file behind an id or a name")
    command.add_argument("key")
    command.set_defaults(handler=run_path)


def run_path(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    path = reads.path_of(brain=brain, key=args.key)
    if path is None:
        return missing(args.key)

    print(path)
    return 0


def add_links(
    commands: argparse._SubParsersAction,
) -> None:
    command = commands.add_parser("links", help="what an entry points at, and what points back")
    command.add_argument("key")
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=run_links)


def run_links(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    data = reads.links_of(brain=brain, key=args.key)
    if data is None:
        return missing(args.key)

    print(json.dumps(data, indent=2) if args.json else reads.render_links(data))
    return 0


def add_new(
    commands: argparse._SubParsersAction,
) -> None:
    command = commands.add_parser("new", help="file a new entry, prose on standard input")
    command.add_argument("kind", choices=sorted(schema.KINDS))
    command.add_argument("--title", required=True)
    command.add_argument("--description", required=True, help="the one line the index shows")
    command.add_argument("--category", default="", help="for a doc, its category folder")
    command.add_argument("--link", action="append", default=[], help="a related file, repeatable")
    for flag in FIELD_FLAGS:
        command.add_argument(f"--{flag}")
    command.set_defaults(handler=run_new)


def run_new(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    kind = schema.kind_named(args.kind)
    if kind.name == "plan":
        entry = plans.create(
            brain=brain,
            title=args.title,
            description=args.description,
            body=stdin_text(),
            values=field_values(args),
            category=args.category,
            links=args.link,
        )
        index.write(brain)
        print(entry.relative)
        return 0

    entry = writes.create(
        brain=brain,
        kind=kind,
        title=args.title,
        description=args.description,
        values=field_values(args),
        body=stdin_text(),
        category=args.category,
        links=args.link,
    )
    index.write(brain)
    print(entry.relative)
    return 0


def add_set(
    commands: argparse._SubParsersAction,
) -> None:
    command = commands.add_parser("set", help="change fields on an entry, as field=value")
    command.add_argument("key")
    command.add_argument("changes", nargs="+", metavar="field=value")
    command.set_defaults(handler=run_set)


def run_set(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    entry = writes.update(
        brain=brain,
        key=args.key,
        changes=pairs(args.changes),
    )
    index.write(brain)
    print(entry.relative)
    return 0


def add_refile(
    commands: argparse._SubParsersAction,
) -> None:
    command = commands.add_parser("refile", help="retitle an entry, or turn a todo into a bug")
    command.add_argument("key")
    command.add_argument("--title", help="the new title, the id stays")
    command.add_argument("--kind", choices=("todo", "bug"), help="the kind it becomes")
    command.add_argument("--description", default="", help="with --kind, the new one line")
    for flag in FIELD_FLAGS:
        command.add_argument(f"--{flag}")
    command.set_defaults(handler=run_refile)


def run_refile(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    if bool(args.title) == bool(args.kind):
        print("refile takes --title or --kind, one of the two", file=sys.stderr)
        return REFUSED

    if args.title:
        entry = writes.retitle(brain=brain, key=args.key, title=args.title)
    else:
        entry = writes.convert(
            brain=brain,
            key=args.key,
            kind=schema.kind_named(args.kind),
            description=args.description,
            values=field_values(args),
            body=stdin_text(),
        )

    index.write(brain)
    print(entry.relative)
    return 0


def add_index(
    commands: argparse._SubParsersAction,
) -> None:
    command = commands.add_parser("index", help="rebuild every INDEX.md from its folder")
    command.set_defaults(handler=run_index)


def run_index(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    for path in index.write(brain):
        print(path.relative_to(brain.root))
    return 0


def add_doctor(
    commands: argparse._SubParsersAction,
) -> None:
    command = commands.add_parser("doctor", help="check the whole brain, fix what is mechanical")
    command.add_argument("--fix", action="store_true")
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=run_doctor)


def run_doctor(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    found = doctor.run(brain=brain, fix=args.fix)
    if args.json:
        print(json.dumps([finding.as_dict() for finding in found], indent=2))
    else:
        for finding in found:
            print(finding.line())
        print("clean" if not found else f"{len(found)} findings")

    return FINDINGS_REMAIN if found else 0


def add_steps(
    commands: argparse._SubParsersAction,
) -> None:
    command = commands.add_parser("steps", help="the steps of a plan, a todo or a bug: show them, add some, or mark a plain one")
    command.add_argument("key", help="the entry, by id, name or path")
    actions = command.add_subparsers(dest="action")

    adding = actions.add_parser("add", help="append steps, one per line of standard input, or the one given")
    adding.add_argument("step", nargs="?", help="one step, or - to read one per line from standard input")
    adding.add_argument("--by", default="agent", choices=steps.BY, help="who set the steps, agent unless said otherwise")

    marking = actions.add_parser("step", help="mark a step that points at nothing, or one whose pointer reaches nothing the brain can read")
    marking.add_argument("number", type=int, help="the step's number in the entry's table")
    marking.add_argument("status", choices=steps.STEP_STATUSES, help="the mark to give it")

    # the one handler picks the action itself, for the same reason as the task command's
    command.set_defaults(handler=run_steps)


def run_steps(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    handlers = {
        None: run_steps_show,
        "add": run_steps_add,
        "step": run_steps_mark,
    }

    return handlers[args.action](brain=brain, args=args)


def run_steps_show(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    print(steps.view(brain=brain, key=args.key))
    return 0


def run_steps_add(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    lines = [args.step] if args.step and args.step != "-" else [line.strip() for line in stdin_text().split("\n") if line.strip()]
    if not lines:
        print("no steps given, one per line on standard input, or one after add", file=sys.stderr)
        return REFUSED

    numbers = steps.add(brain=brain, key=args.key, lines=lines, by=args.by)
    index.write(brain)
    print("\n".join(str(number) for number in numbers))
    return 0


def run_steps_mark(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    steps.mark(brain=brain, key=args.key, number=args.number, status=args.status)
    index.write(brain)
    return 0


def add_ledger(
    commands: argparse._SubParsersAction,
) -> None:
    command = commands.add_parser("ledger", help="review ledgers: open, batch, add, set, report, close")
    actions = command.add_subparsers(dest="action")

    opening = actions.add_parser("open", help="start the ledger for a branch")
    opening.add_argument("branch")
    opening.add_argument("--summary", required=True, help="one line, or - to read it from standard input")
    opening.add_argument("--level", default=ledger.DEFAULT_LEVEL, choices=ledger.LEVELS, help="how wide the finding pass goes")
    opening.set_defaults(handler=run_ledger_open)

    batch = actions.add_parser("batch", help="add a batch, or set one's status, ack or theme")
    batch.add_argument("key")
    batch.add_argument("number", nargs="?", type=int)
    batch.add_argument("--theme")
    batch.add_argument("--status")
    batch.add_argument("--ack")
    batch.set_defaults(handler=run_ledger_batch)

    adding = actions.add_parser("add", help="record a finding")
    adding.add_argument("key")
    adding.add_argument("claim", help="one line, or - to read one claim per line from standard input")
    adding.add_argument("--severity", required=True, choices=schema.SEVERITIES)
    adding.add_argument("--found-by", default="review", choices=ledger.FOUND_BY)
    adding.add_argument("--batch", type=int, default=0)
    adding.set_defaults(handler=run_ledger_add)

    setting = actions.add_parser("set", help="change any field of one finding or several")
    setting.add_argument("key")
    setting.add_argument("words", nargs="+", metavar="word", help="the findings first, like F-03, then the changes, like status=fixed")
    setting.set_defaults(handler=run_ledger_set)

    reporting = actions.add_parser("report", help="the ledger's state in a few lines, and what comes next")
    reporting.add_argument("key")
    reporting.set_defaults(handler=run_ledger_report)

    closing = actions.add_parser("close", help="close the ledger once every finding landed")
    closing.add_argument("key")
    closing.set_defaults(handler=run_ledger_close)


def run_ledger_open(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    summary = (stdin_text() if args.summary == "-" else args.summary).strip()
    if not summary:
        print("the summary is empty", file=sys.stderr)
        return REFUSED

    entry = ledger.open_ledger(
        brain=brain,
        branch=args.branch,
        summary=summary,
        level=args.level,
    )
    index.write(brain)
    print(entry.relative)

    kept_off = task.push_review(brain=brain, name=entry.name)
    if kept_off is not None:
        print(f"the review is not the current work, {kept_off}", file=sys.stderr)

    return 0


def run_ledger_batch(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    if args.number is None:
        if args.status or args.ack:
            print("give the batch's number to set its status or ack", file=sys.stderr)
            return REFUSED

        if not args.theme:
            print("a new batch needs --theme", file=sys.stderr)
            return REFUSED

        print(ledger.add_batch(brain=brain, key=args.key, theme=args.theme))
        return 0

    ledger.set_batch(
        brain=brain,
        key=args.key,
        number=args.number,
        status=args.status or "",
        ack=args.ack or "",
        theme=args.theme or "",
    )
    return 0


def run_ledger_add(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    """File one claim from the command line, or one per line of standard input, so the shell never touches the words."""
    claims = [args.claim]
    if args.claim == "-":
        claims = [line.strip() for line in stdin_text().split("\n") if line.strip()]
        if not claims:
            print("no claims on standard input, one per line", file=sys.stderr)
            return REFUSED

    filed = ledger.add_findings(
        brain=brain,
        key=args.key,
        claims=claims,
        severity=args.severity,
        found_by=args.found_by,
        batch=args.batch,
    )
    print("\n".join(filed))

    return 0


def run_ledger_set(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    """Change every finding named, the findings like F-03 first and the field=value changes after them."""
    key_is_a_finding = ledger.FINDING_ID.fullmatch(string=args.key.upper()) is not None
    if key_is_a_finding:
        raise writes.WriteError(f"'{args.key}' names a finding, give the ledger's key before the findings")

    identifiers, changes = findings_and_changes(words=args.words)
    ledger.set_findings(
        brain=brain,
        key=args.key,
        identifiers=identifiers,
        changes=changes,
    )
    return 0


def findings_and_changes(
    words: list[str],
) -> tuple[list[str], dict[str, str]]:
    """The findings a `ledger set` names and the changes after them, refused at the first word out of its place.

    Every finding comes before the first change and every field is given
    once, so no word meant for one finding lands on another. An id in
    lower case is refused as that rather than read as a change.
    """
    identifiers: list[str] = []
    changes: dict[str, str] = {}

    for word in words:
        is_finding = ledger.FINDING_ID.fullmatch(string=word) is not None
        if not is_finding and ledger.FINDING_ID.fullmatch(string=word.upper()):
            raise writes.WriteError(f"'{word}' names a finding in lower case, write it {word.upper()}")

        if is_finding and changes:
            raise writes.WriteError(f"'{word}' comes after a change, name every finding before the changes")

        if is_finding:
            identifiers.append(word)
            continue

        if not identifiers:
            raise writes.WriteError(f"'{word}' comes before any finding, name the findings first, like F-03 status=fixed")

        change = pairs(items=[word])
        field = next(iter(change))
        if field in changes:
            raise writes.WriteError(f"'{word}' sets {field} a second time, give each field once")

        changes.update(change)

    return identifiers, changes


def run_ledger_report(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    state = ledger.report(
        brain=brain,
        key=args.key,
    )
    print(state)
    return 0


def run_ledger_close(
    brain: store.Brain,
    args: argparse.Namespace,
) -> int:
    ledger.close(brain=brain, key=args.key)
    index.write(brain)
    return 0


def field_values(
    args: argparse.Namespace,
) -> dict[str, str]:
    return {flag: getattr(args, flag) for flag in FIELD_FLAGS if getattr(args, flag, None)}


def pairs(
    items: list[str],
) -> dict[str, str]:
    found: dict[str, str] = {}

    for item in items:
        if "=" not in item:
            raise writes.WriteError(f"'{item}' is not field=value")

        field, _, value = item.partition("=")
        found[field.strip()] = value.strip()

    return found


def stdin_text() -> str:
    """The prose on standard input, or nothing when there is no input at all.

    A closed standard input leaves `sys.stdin` as None, and a terminal
    means nobody piped anything, so both read as an empty body. The
    bytes are read as UTF-8 whatever the locale says, and a byte that
    is not UTF-8 is a refusal rather than a traceback. Line breaks come
    out as one newline whatever the pipe wrote, the way the text layer
    reads a pipe on Windows, since the bytes are read under it.
    """
    if sys.stdin is None or sys.stdin.isatty():
        return ""

    try:
        text = sys.stdin.buffer.read().decode("utf-8")
    except UnicodeDecodeError as bad:
        raise writes.WriteError(f"standard input is not UTF-8 at byte {bad.start}") from bad

    return text.replace("\r\n", "\n").replace("\r", "\n")


def missing(
    key: str,
) -> int:
    print(f"nothing in the brain answers to '{key}'", file=sys.stderr)
    return REFUSED


def add_port(
    commands: argparse._SubParsersAction,
) -> None:
    command = commands.add_parser("port", help="install the kit into a project that has no brain yet")
    add_kit_arguments(command)
    command.set_defaults(handler=run_port, needs_brain=False)


def add_upgrade(
    commands: argparse._SubParsersAction,
) -> None:
    command = commands.add_parser("upgrade", help="bring a project's brain up to this kit, migrations included")
    add_kit_arguments(command)
    command.set_defaults(handler=run_upgrade, needs_brain=False)


def add_kit_arguments(
    command: argparse.ArgumentParser,
) -> None:
    command.add_argument("--kit", help="a checkout of agent-setup, by default the one this tool runs from")
    command.add_argument("--project", help="the project's root folder, by default the git root around the current one for a port and the nearest brain above it for an upgrade")


def run_port(
    brain: Optional[store.Brain],
    args: argparse.Namespace,
) -> int:
    report = upgrade.port(kit=kit_of(args), project=project_of(args=args, found_by=upgrade.port_root), out=sys.stdout)
    print(report.tail(), end="")

    return FINDINGS_REMAIN if report.findings else 0


def run_upgrade(
    brain: Optional[store.Brain],
    args: argparse.Namespace,
) -> int:
    report = upgrade.upgrade(kit=kit_of(args), project=project_of(args=args, found_by=upgrade.upgrade_root), out=sys.stdout)
    print(report.tail(), end="")

    return FINDINGS_REMAIN if report.findings else 0


def kit_of(
    args: argparse.Namespace,
) -> upgrade.Kit:
    if args.kit:
        return upgrade.Kit(root=Path(args.kit).resolve())

    return upgrade.default_kit()


def project_of(
    args: argparse.Namespace,
    found_by: Callable[[Path], Path],
) -> upgrade.Project:
    """The project named with --project, which has to exist, or the one `found_by` finds around the current folder."""
    if args.project:
        return upgrade.Project(root=upgrade.named_root(args.project))

    return upgrade.Project(root=found_by(Path.cwd()))


if __name__ == "__main__":
    sys.exit(main())
