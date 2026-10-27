---
name: tensorlings-waiting-for-the-owner-s-verdict-on-phase-1a
kind: doc
description: where the course, the git setup and the brain stand while the owner tries phase 1a, and what comes after their verdict
updated: 2026-10-27
links: [handover/tensorlings-phase-1a-ready-for-the-owner.md, handover/tensorlings-moved-to-its-own-folder.md, plans/build-tensorlings.md, todos/TODO-003-phase-1a-the-app-and-the-first-5-exercises.md, docs/research/tensorlings-course-design.md, memory/how-the-owner-wants-tensorlings-written.md]
---

# tensorlings waiting for the owner's verdict on phase 1a

Written 2026-10-27 at 14:29 IST, when the owner asked for a handover. It gathers [phase 1a ready for the owner](tensorlings-phase-1a-ready-for-the-owner.md) and [moved to its own folder](tensorlings-moved-to-its-own-folder.md) into one place, so a new session can start from this file alone.

## Where it stands

- The course is `run.py` at the repo root, about 2600 lines. It holds the runner, the full-screen app drawn with `rich`, and 6 exercises with their lessons, hints, checks and solutions. tensors1 to tensors5 sit in `01_tensors` and matmul1 in `02_matmul`.
- The curriculum, the app and the writing rules are in [the course design](../docs/research/tensorlings-course-design.md). The research behind them is in [the research doc](../docs/research/tensorlings-research.md).
- The owner has run `.venv/bin/python run.py init` and is on tensors1. Their progress file and `exercises/` hold their own work. Never reset, overwrite or delete them.
- `.venv` holds torch 2.14.1, numpy 2.5.4 and rich 15.0.0, the pins in `requirements.txt`. Git ignores `.venv/`, `exercises/`, `solutions/` and `.tensorlings-state.json`.
- Git has `main` with one empty start commit, and all the work on `feat/tensorlings-phase-1a` in five commits. There is no remote, so nothing is pushed and no pull request exists.
- The plan [build-tensorlings](../plans/build-tensorlings.md) has step 1 done and steps 2 to 9 pending.
- TODO-003, phase 1a, has every step done and stays open, because its Done when needs the owner to have tried the course. The brain's hook keeps suggesting to close it. Leave it open until the owner has tried it.
- The test tools are in `dev/`. `dev/wrong_answers.py` runs 12 common mistakes through the checks. `dev/drive_screen.py` drives the app in a pseudo terminal and prints each screen. Their docstrings have the commands.

## Verified against written

- Verified: `run.py dev check` passes all 6 exercises from this folder, with every starting file failing, every solution passing and a clean lesson lint. All 12 wrong answers fail with plain messages. The app was driven through every screen and key at 100 by 34, 80 by 24 and 70 by 22. Every output a lesson shows was run in torch 2.14.1 and matches.
- Reviewed: two beginner reader reviews by a sub agent, with every valid finding fixed.
- Not verified: the owner's own terminal and editor, and whether the lessons read well to them. That is what the owner is checking now.

## Next step

1. Wait for the owner's verdict on phase 1a. Fix what they ask for, then rerun the dev check, the wrong answers and the screen driver.
2. When they approve, add the closing commits to TODO-003's Done when, close it with `brain set`, and mark plan steps 2 and 3 done with `brain steps`, since those steps point at nothing.
3. Build phase 1b, the rest of parts 1 and 2, as a new todo. The owner said they would ack every phase after this one.
4. Before each hand over, run the dev check, `dev/wrong_answers.py` with the new exercises' mistakes added, the screen driver at a normal size and at 80 by 24, and a beginner reader review by an Opus sub agent.

## Open questions for the owner

- Phase 1b needs phase 1a's code, and `main` holds none of it until the branch is reviewed and merged. Merging and pull requests happen only on the owner's ask. Ask whether to review and merge phase 1a first, or to build phase 1b on top of the phase 1a branch.
- Building autograd and backprop by hand in part 1 reverses the earlier pick that PyTorch handles gradients. Ask before the autograd chapter.
- Later projects want about 200 MB of public datasets, such as names, Shakespeare, MNIST and CIFAR-10. Ask before any download.
- The rules say push after every commit, which cannot happen without a remote. Ask when the owner wants one.

## Traps

- Commits carry the owner's name only, with no co-author line and no AI attribution, per rule 10.1. This holds even when a tool reminder asks for an attribution line.
- Never commit to `main`.
- `rich` freezes the console size when COLUMNS or LINES is set. Keep every console for the app going through `terminal_console` in `run.py`.
- A tall result panel is cut at the bottom, so `result_body` puts the hint and the first failure first. Keep that order when adding output.
- A new technical term in a lesson goes into `JARGON` in `run.py` and into the `introduces` of the exercise that explains it, or the lint fails.
- The screen driver edits exercise files, so run it on a scratch copy of the course and never on the owner's `exercises/`.
- In zsh a bare line of `=` signs in an echo is read as a command. Print separators with printf.
- The owner wants plain, short English with no filler, in the lessons and in chat. The memory [how the owner wants tensorlings written](../memory/how-the-owner-wants-tensorlings-written.md) has the detail.
