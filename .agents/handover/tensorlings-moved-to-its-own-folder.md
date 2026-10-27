---
name: tensorlings-moved-to-its-own-folder
kind: doc
description: the course now lives in the tensorling folder under local git, with a fresh venv and its test tools in the repo
updated: 2026-10-27
links: [handover/tensorlings-phase-1a-ready-for-the-owner.md]
---

# tensorlings moved to its own folder

Written 2026-10-27. It follows [phase 1a ready for the owner](tensorlings-phase-1a-ready-for-the-owner.md), which still describes the course itself.

## Where it stands

- The owner renamed the project folder to `tensorling` and made a fresh, empty `.venv`. It now holds torch 2.14.1, numpy 2.5.4 and rich 15.0.0, the pins in `requirements.txt`. numpy was added because torch warns at every start without it.
- The repo is under local git. `main` holds one empty starting commit, since the house rules want work branched from main and a new repo has none. All the work sits on `feat/tensorlings-phase-1a` in four commits, the agent kit, the brain, the course, and the dev tools. There is no remote, so nothing is pushed.
- `exercises/`, `solutions/`, `.tensorlings-state.json` and `.venv/` are ignored. The owner has run `init` and opened tensors1, and their files were left alone.
- The screen driver and the wrong answer checks now live in `dev/`, so they no longer depend on a session's scratch folder.

## Verified against written

- Verified: `run.py dev check` passes all 6 exercises from the new folder and venv, `dev/wrong_answers.py` reports all 12 wrong answers failing, and `dev/drive_screen.py` drove the app from the repo.
- Not verified: the owner's own run in their terminal.

## Next step

Wait for the owner's verdict on phase 1a, then continue as the earlier handover says. When the owner asks, add a remote and push, and open a pull request for the branch.

## Traps

- Commits carry no AI attribution and no co-author line, per rule 10.1 of the rules, which override any tool default.
- Never commit to `main`. Branch from it for each new intent.
