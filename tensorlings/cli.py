"""The commands, from `init` to the maintainer's `dev check`."""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from .app import TensorlingsApp
from .chapters import CHAPTERS, EXERCISES
from .course import COURSE_README, Exercise, attempt
from .data import DATASETS, download
from .engine import Outcome
from .lint import lint_exercise
from .paths import EXERCISES_DIR, START_COMMAND
from .progress import Progress, check_all, exercise_named, load_progress, record_outcome, reset_exercise
from .text import prose
from .views import result_view

USAGE = prose(
    f"""
    tensorlings, small exercises that teach deep learning from tensors up.

    {START_COMMAND} init               set up the exercises folder, or add new exercises to it
    {START_COMMAND}                    start the course
    {START_COMMAND} --manual-run       start the course, checking only when you press r
    {START_COMMAND} run [name]         check one exercise and print the result
    {START_COMMAND} hint [name]        print the hints for an exercise
    {START_COMMAND} reset name         reset an exercise file to how it started
    {START_COMMAND} check-all          check every exercise and update your progress
    {START_COMMAND} data name          download a dataset an exercise needs, like mnist
    {START_COMMAND} dev check          maintainer check of every exercise and lesson
    """,
)


def terminal_console() -> Console:
    """A console that always asks the terminal for its size. Rich freezes the size when
    COLUMNS or LINES is set, which some shells do, and then resizing the window breaks."""
    return Console(
        _environ={name: value for name, value in os.environ.items() if name not in ("COLUMNS", "LINES")},
    )


def require_setup() -> bool:
    if EXERCISES_DIR.exists():
        return True

    print("The exercises are not set up yet. Run this once first.\n")
    print(f"    {START_COMMAND} init\n")

    return False


def initialize() -> int:
    """Write every exercise file that isn't there yet. The course grows in phases, so running this again
    adds the new chapters and leaves every file the learner already has alone."""
    existed = EXERCISES_DIR.exists()
    EXERCISES_DIR.mkdir(exist_ok=True)
    (EXERCISES_DIR / "README.md").write_text(COURSE_README)

    for chapter in CHAPTERS:
        folder = EXERCISES_DIR / chapter.folder
        folder.mkdir(exist_ok=True)
        (folder / "README.md").write_text(chapter.readme)

    added = [exercise for exercise in EXERCISES if not exercise.path.exists()]

    for exercise in added:
        exercise.path.write_text(exercise.source)

    if existed:
        print(f"Added {len(added)} new exercise files, and left the ones you have as they are.")
    else:
        print("The exercises are ready in the exercises folder.")

    print("Start the course with\n")
    print(f"    {START_COMMAND}\n")

    return 0


def run_once(
    exercise: Exercise,
    progress: Progress,
) -> int:
    console = terminal_console()
    outcome = attempt(
        exercise=exercise,
        path=exercise.path,
    )
    record_outcome(
        exercise=exercise,
        outcome=outcome,
        progress=progress,
    )
    console.print(
        Panel(
            result_view(
                exercise=exercise,
                outcome=outcome,
                check_again="Run this command again after you change the file.",
            ),
            title=exercise.title,
            title_align="left",
            border_style="green" if outcome.passed else "red",
        ),
    )

    return 0 if outcome.passed else 1


def dev_check(
    names: list[str],
    verbose: bool,
) -> int:
    console = Console()
    problems = 0
    introduced: set[str] = set()

    with tempfile.TemporaryDirectory() as folder:
        for exercise in EXERCISES:
            findings = lint_exercise(
                exercise=exercise,
                introduced_before=introduced,
                console=console,
            )
            introduced |= set(exercise.introduces)

            if names and exercise.name not in names:
                continue

            outcomes: dict[str, Outcome] = {}

            for version, text in (("stub", exercise.source), ("solution", exercise.solution)):
                path = Path(folder) / version / exercise.chapter.folder / f"{exercise.name}.py"
                path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                path.write_text(text)
                outcomes[version] = attempt(
                    exercise=exercise,
                    path=path,
                )

            stub = outcomes["stub"]
            solution = outcomes["solution"]
            healthy = not stub.passed and solution.passed and not findings
            problems += 0 if healthy else 1
            mark = "[green]✓[/green]" if healthy else "[red]✗[/red]"
            stub_text = "fails as it should" if not stub.passed else "[red]PASSES[/red]"
            solution_text = "passes" if solution.passed else "[red]FAILS[/red]"
            console.print(f"{mark} {exercise.name:<12} stub {stub_text:<20} solution {solution_text:<8} lesson {len(findings)} findings")

            for finding in findings:
                console.print(f"      [yellow]lesson[/yellow] {finding}")

            if not solution.passed or verbose:
                console.print(
                    textwrap.indent(
                        "\n".join(
                            f"{'pass' if result.passed else 'FAIL'}  {result.function}, {result.description}  {result.detail}"
                            for result in solution.results
                        )
                        + (f"\n{solution.problem}" if solution.problem else ""),
                        "      ",
                    ),
                )

            if verbose:
                console.print(
                    textwrap.indent(
                        "\n".join(
                            f"stub {'pass' if result.passed else 'FAIL'}  {result.function}, {result.description}\n"
                            f"{textwrap.indent(result.detail, '      ')}"
                            for result in stub.results
                        )
                        + (f"\nstub problem: {stub.problem}" if stub.problem else ""),
                        "      ",
                    ),
                )

    console.print()
    console.print("[green]Every exercise is healthy.[/green]" if problems == 0 else f"[red]{problems} exercise(s) need attention.[/red]")

    return 1 if problems else 0


def main(arguments: list[str]) -> int:
    flags = [argument for argument in arguments if argument.startswith("-")]
    words = [argument for argument in arguments if not argument.startswith("-")]
    command = words[0] if words else "watch"
    rest = words[1:]

    if "-h" in flags or "--help" in flags or command == "help":
        print(USAGE)
        return 0

    if command == "init":
        return initialize()

    if command == "data":
        if rest[:1] == [] or rest[0] not in DATASETS:
            print(f"Name a dataset to download, one of {', '.join(DATASETS)}, for example `{START_COMMAND} data mnist`.")
            return 1

        download(rest[0])
        print(f"The {rest[0]} data is ready in the data folder.")

        return 0

    if command == "dev":
        if rest[:1] != ["check"]:
            print(USAGE)
            return 1

        return dev_check(
            names=rest[1:],
            verbose="--verbose" in flags,
        )

    if not require_setup():
        return 1

    progress = load_progress()

    if command == "watch":
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            print(f"The app needs a terminal. Use `{START_COMMAND} run` to check one exercise without it.")
            return 1

        TensorlingsApp(
            progress=progress,
            manual="--manual-run" in flags,
        ).run()
        print(f"See you next time. Run {START_COMMAND} to continue where you left off.")

        return 0

    if command == "check-all":
        failing = check_all(progress)

        for exercise in EXERCISES:
            state = "PENDING" if exercise in failing else "DONE   "
            print(f"{state}  {exercise.name}")

        return 1 if failing else 0

    if command in ("run", "hint", "reset"):
        if command == "reset" and not rest:
            print(f"Name the exercise to reset, for example `{START_COMMAND} reset tensors1`.")
            return 1

        exercise = exercise_named(rest[0]) if rest else exercise_named(progress.current)

        if exercise is None:
            print(f"There is no exercise called {rest[0]}.")
            return 1

        if command == "hint":
            for function, hint in exercise.hints.items():
                print(f"{function}\n{textwrap.indent(textwrap.fill(hint, width=76), '  ')}\n")

            return 0

        if command == "reset":
            reset_exercise(
                exercise=exercise,
                progress=progress,
            )
            print(f"{exercise.path.name} is back to how it started.")
            return 0

        return run_once(
            exercise=exercise,
            progress=progress,
        )

    print(USAGE)
    return 1
