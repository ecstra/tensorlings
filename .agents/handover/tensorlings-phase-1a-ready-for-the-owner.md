---
name: tensorlings-phase-1a-ready-for-the-owner
kind: doc
description: the app and the first 6 exercises are built and checked, waiting for the owner to try them
updated: 2026-10-27
links: [handover/tensorlings-before-phase-1a.md]
---

# tensorlings phase 1a ready for the owner

Written 2026-10-27, after phase 1a was built. It follows [the handover before phase 1a](tensorlings-before-phase-1a.md).

## Where it stands

- `run.py` is the tensorlings course, with the full-screen app and 6 exercises, tensors1 to tensors5 in `01_tensors` and matmul1 in `02_matmul`. The owner asked for about 5, and the beginner review split broadcasting into two exercises to keep one idea each.
- The old AIlings files are gone from the repo. `exercises/` does not exist until the owner runs `.venv/bin/python run.py init`, as rustlings does.
- `rich` 15.0.0 is installed in `.venv` and pinned with torch 2.14.1 in `requirements.txt`.
- TODO-003 has every step done but stays open, since its Done when needs the owner to try the course. Plan step 2 stays pending until then.

## Verified against written

- Verified: `run.py dev check` reports all 6 healthy, every stub failing, every solution passing, and a clean lesson lint. Every output the lessons show was run in PyTorch 2.14.1 and matches. Common wrong answers were run through the checks and give plain messages. The app was driven in a pseudo terminal through welcome, failing, hint, pass, next, list with search and filter, reset with confirmation, scrolling, manual run, a window resize, the finish line, and the plain commands, at 100 by 34 and 80 by 24.
- Two rounds of beginner reader review by a sub agent found real errors, among them a false claim about `.max(dim=...)` and division called a reduction, and every valid finding was fixed.
- Not yet verified: the owner's own terminal, editor and reading of the lessons.

## Next step

Wait for the owner to try phase 1a. Apply what they ask for. Once they approve, close TODO-003, mark plan step 2 and 3 done, and build phase 1b, the rest of parts 1 and 2, under the owner's standing ack, with the same dev check, pseudo terminal run and beginner review before each hand over.

## Traps

- Two owner decisions are still open, building autograd by hand in part 1 and downloading about 200 MB of public datasets for later projects. Ask before the autograd chapter and before any download.
- `rich` freezes the console size when COLUMNS or LINES is set. `terminal_console` in `run.py` strips them, so keep every console for the app going through it.
- A tall result panel is cut at the bottom, so `result_body` puts the hint and the first failure first. Keep that order when adding output.
- Any new technical term in a lesson must be added to `JARGON` in `run.py` and introduced by an exercise's `introduces`, or the lint fails.
- The pseudo terminal test driver and wrong answer fixtures live only in a session scratch folder. Rebuild them from the course design doc's verification section if needed.
