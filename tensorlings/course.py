"""The exercise model, one attempt at an exercise, and the course's own pages."""

from __future__ import annotations

import contextlib
import importlib
import io
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .engine import TIME_LIMIT_SECONDS, Checker, Outcome, Problem, load_module
from .paths import EXERCISES_DIR, SOLUTIONS_DIR
from .text import prose, source_file


@dataclass(frozen=True)
class Chapter:
    folder: str
    title: str
    readme: str


@dataclass(frozen=True)
class Exercise:
    name: str
    chapter: Chapter
    title: str
    lesson: str
    source: str
    solution: str
    # One hint per function or name, shown for the ones that don't pass yet.
    hints: dict[str, str]
    introduces: tuple[str, ...]
    check: Callable[[Checker], None]
    # How long one test may run. Exercises that train on real data need longer than the usual few seconds.
    time_limit: float = TIME_LIMIT_SECONDS
    # Modules imported before the learner's file runs, so a slow first import doesn't count against the time limit.
    imports_first: tuple[str, ...] = ()

    @property
    def path(self) -> Path:
        return EXERCISES_DIR / self.chapter.folder / f"{self.name}.py"

    @property
    def solution_path(self) -> Path:
        return SOLUTIONS_DIR / self.chapter.folder / f"{self.name}.py"


# Two checks never run the learner's code at once. Each one swaps out
# sys.stdout to catch what the code prints, and overlapping swaps would mix up.
CHECK_LOCK = threading.Lock()

# The first optimizer torch.optim makes imports torch._dynamo, which takes 10 to 20 seconds and, run under
# the time limit's trace, can leave sympy half imported. Every exercise that makes one imports it first.
OPTIMIZER_IMPORTS: tuple[str, ...] = ("torch._dynamo",)


def attempt(
    exercise: Exercise,
    path: Path,
) -> Outcome:
    printed = io.StringIO()

    with CHECK_LOCK, contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
        for module_name in exercise.imports_first:
            importlib.import_module(module_name)

        try:
            module, crashes = load_module(
                path=path,
                name=exercise.name,
                seconds=exercise.time_limit,
            )
        except Problem as problem:
            return Outcome(
                problem=str(problem),
                results=[],
                printed=printed.getvalue(),
            )

        checker = Checker(
            module=module,
            path=path,
            crashes=crashes,
            seconds=exercise.time_limit,
        )

        try:
            exercise.check(checker)
        except Problem as problem:
            return Outcome(
                problem=str(problem),
                results=checker.results,
                printed=printed.getvalue(),
            )

    unchecked = checker.unchecked_crashes()

    return Outcome(
        problem=f"{path.name} crashed while loading.\n\n{unchecked[0].text}" if unchecked else None,
        results=checker.results,
        printed=printed.getvalue(),
    )


# Every quiz goes in this one folder, as in rustlings, wherever in the course it falls.
QUIZZES = Chapter(
    folder="quizzes",
    title="Quizzes",
    readme=source_file(
        """
        # Quizzes

        A quiz comes after a group of chapters and uses their tools together,
        with nothing new. quiz1 follows the tensor and matrix multiplication
        chapters.
        """,
    ),
)

COURSE_README = source_file(
    """
    # tensorlings exercises

    Start the course from the project folder.

    ```bash
    .venv/bin/python run.py
    ```

    Each chapter folder has a README with further reading. You can skip them.
    Every lesson you need shows in the app next to the exercise.
    """,
)

WELCOME = prose(
    """
    tensorlings teaches deep learning from the ground up. You start with
    tensors, the grids of numbers every model is made of, and work up to
    building transformers and designing modules of your own.

    **How it works**

    - Each exercise is one Python file in the `exercises` folder.
    - This screen shows the lesson. Keep your editor open beside it.
    - Fill in the code under each `TODO` and save. The app checks your code every time you save.
    - When every function passes, press `n` for the next exercise.

    **Keys**

    - `1`, `2` and `3` show the lesson, the hints and the result. After each save the result shows by itself.
    - The hints tab has one hint for each function that doesn't pass yet. `h` opens it too.
    - The arrow keys and the mouse wheel scroll. `l` lists every exercise, `x` resets the file, and `q` quits.
    - To copy, hold `option` while you drag over the text, then press `cmd+c`. In macOS Terminal hold `fn` instead.
    """,
)

FINISH = prose(
    """
    You finished every exercise that exists so far.

    The course grows in phases, from tensors to transformers to modules you
    design yourself. New exercises show up in the list as soon as they are
    added. Run the app again to continue.
    """,
)
