"""Drive the tensorlings app in a pseudo terminal and print each screen as text.

The app draws a full-screen display, so plain output capture only shows escape
codes. This feeds the output to pyte, a terminal emulator, and prints what a
person would see.

    uv run --no-project --with pyte==0.8.2 python dev/drive_screen.py COURSE_DIR COLUMNS ROWS -- STEP...

COURSE_DIR holds a copy of run.py with its exercises set up by `run.py init`.
Use a scratch copy, since the steps edit exercise files. Each STEP is
"seconds:action", and the action is one of these.

    SNAP                  print the screen, with rows in reverse video marked R
    ENTER, UP, DOWN, TAB  press that key
    RESIZE:COLSxROWS      resize the terminal
    EDIT:path|old|new     replace text in a file under COURSE_DIR, \\n for a newline
    anything else         typed as keys

APP_ARGS in the environment passes extra arguments to run.py, such as --manual-run.
"""

import fcntl
import os
import pty
import select
import signal
import struct
import sys
import termios
import time
from pathlib import Path

import pyte

PYTHON = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
KEYS = {
    "ENTER": b"\r",
    "UP": b"\x1b[A",
    "DOWN": b"\x1b[B",
    "TAB": b"\t",
}


def set_window_size(
    terminal: int,
    columns: int,
    rows: int,
) -> None:
    fcntl.ioctl(
        terminal,
        termios.TIOCSWINSZ,
        struct.pack("HHHH", rows, columns, 0, 0),
    )


def pump(
    terminal: int,
    stream: pyte.ByteStream,
    seconds: float,
) -> None:
    """Feed everything the app writes for the given time into the emulator."""
    deadline = time.time() + seconds

    while time.time() < deadline:
        ready, _, _ = select.select([terminal], [], [], 0.05)

        if not ready:
            continue

        try:
            chunk = os.read(terminal, 65536)
        except OSError:
            return

        if not chunk:
            return

        stream.feed(chunk)


def print_screen(
    screen: pyte.Screen,
    number: int,
) -> None:
    print(f"=============== screen {number} ===============")

    for row, line in enumerate(screen.display):
        reversed_row = any(screen.buffer[row][column].reverse for column in range(screen.columns))
        print(("R " if reversed_row else "  ") + line.rstrip())


def edit_file(
    course: Path,
    action: str,
) -> None:
    relative_path, old, new = action.removeprefix("EDIT:").replace("\\n", "\n").split("|")
    path = course / relative_path
    text = path.read_text()

    if old not in text:
        raise SystemExit(f"EDIT could not find {old!r} in {path}")

    path.write_text(text.replace(old, new))


def main(arguments: list[str]) -> None:
    course = Path(arguments[0]).resolve()
    columns = int(arguments[1])
    rows = int(arguments[2])
    steps = [step.split(":", 1) for step in arguments[arguments.index("--") + 1 :]]

    screen = pyte.Screen(columns, rows)
    stream = pyte.ByteStream(screen)
    child, terminal = pty.fork()

    if child == 0:
        os.chdir(course)
        os.environ["TERM"] = "xterm-256color"
        os.execv(PYTHON, [str(PYTHON), "-I", "run.py", *os.environ.get("APP_ARGS", "").split()])

    set_window_size(
        terminal=terminal,
        columns=columns,
        rows=rows,
    )
    snapshots = 0

    for seconds, action in steps:
        pump(
            terminal=terminal,
            stream=stream,
            seconds=float(seconds),
        )

        if action == "SNAP":
            snapshots += 1
            print_screen(
                screen=screen,
                number=snapshots,
            )
        elif action.startswith("RESIZE:"):
            new_columns, new_rows = (int(size) for size in action.removeprefix("RESIZE:").split("x"))
            screen.resize(new_rows, new_columns)
            set_window_size(
                terminal=terminal,
                columns=new_columns,
                rows=new_rows,
            )
            os.kill(child, signal.SIGWINCH)
        elif action.startswith("EDIT:"):
            edit_file(
                course=course,
                action=action,
            )
        else:
            os.write(terminal, KEYS.get(action, action.encode()))

    pump(
        terminal=terminal,
        stream=stream,
        seconds=1,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
