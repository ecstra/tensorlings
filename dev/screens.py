"""Drive the tensorlings app headless with Textual's Pilot and save screenshots.

    .venv/bin/python dev/screens.py COURSE_DIR OUT_DIR WIDTHxHEIGHT STEP...

COURSE_DIR holds a copy of run.py with its exercises set up by `run.py init`.
Use a scratch copy, since the steps edit exercise files and save progress.
Each STEP is one of these.

    snap:NAME               save OUT_DIR/NAME.svg
    press:KEY               press a key, such as h, tab, down or enter
    wait:SECONDS            give checks and redraws time to finish
    size:WIDTHxHEIGHT       resize the terminal
    edit:PATH|OLD|NEW       replace text in a file under COURSE_DIR, \\n for a newline

On macOS, turn a screenshot into a PNG to look at with
`qlmanage -t -s 1600 -o OUT_DIR OUT_DIR/NAME.svg`.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# Loading run.py would otherwise leave a __pycache__ folder in the course copy.
sys.dont_write_bytecode = True


def load_course(course: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name="run",
        location=course / "run.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run"] = module
    spec.loader.exec_module(module)

    return module


def parse_size(text: str) -> tuple[int, int]:
    width, height = text.split("x")

    return int(width), int(height)


def edit_file(
    course: Path,
    argument: str,
) -> None:
    relative_path, old, new = argument.replace("\\n", "\n").split("|")
    path = course / relative_path
    text = path.read_text()

    if old not in text:
        raise SystemExit(f"edit could not find {old!r} in {path}")

    path.write_text(text.replace(old, new))


async def drive(
    course: Path,
    out: Path,
    size: tuple[int, int],
    steps: list[str],
) -> None:
    run = load_course(course)
    app = run.TensorlingsApp(
        progress=run.load_progress(),
        manual=False,
    )

    async with app.run_test(size=size, notifications=True) as pilot:
        await pilot.pause(1.5)

        for step in steps:
            action, _, argument = step.partition(":")

            if action == "snap":
                app.save_screenshot(
                    filename=f"{argument}.svg",
                    path=str(out),
                )
                print(out / f"{argument}.svg")
            elif action == "press":
                await pilot.press(argument)
                await pilot.pause(0.3)
            elif action == "wait":
                await pilot.pause(float(argument))
            elif action == "size":
                await pilot.resize_terminal(*parse_size(argument))
                await pilot.pause(0.5)
            elif action == "edit":
                edit_file(
                    course=course,
                    argument=argument,
                )
            else:
                raise SystemExit(f"unknown step {step}")


def main(arguments: list[str]) -> None:
    course = Path(arguments[0]).resolve()
    out = Path(arguments[1]).resolve()
    out.mkdir(
        parents=True,
        exist_ok=True,
    )
    asyncio.run(
        drive(
            course=course,
            out=out,
            size=parse_size(arguments[2]),
            steps=arguments[3:],
        ),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
