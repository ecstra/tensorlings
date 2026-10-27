---
name: tensorlings-before-phase-1a
kind: doc
description: where the tensorlings rebuild stands before the first 5 exercises and the app are written
updated: 2026-10-27
links: []
---

# tensorlings before phase 1a

Written 2026-10-27, before any tensorlings code exists.

## Where it stands

- The owner approved the curriculum in [the course design](../docs/research/tensorlings-course-design.md) and asked for a first slice of 5 exercises plus the full-screen app, to judge how exercises and the app look and work. After that slice is approved, the owner acks every later phase until the plan `build-tensorlings` is done.
- The work for the slice is TODO-003.
- `run.py` in the repo root is still the old AIlings course. The owner ran its `init`, so `exercises/`, `solutions/` and `.ailings-state.json` hold AIlings files with intro1 done and an attempt at layers1. The owner agreed tensorlings replaces them.
- The research behind the design is in [the research doc](../docs/research/tensorlings-research.md), and the three decisions DEC-001 to DEC-003 record the curriculum source, the app, and code only exercises.

## Verified against written

- Verified: the survey of syllabi, textbooks and PyTorch's module list, and the corrections to the pasted chat, each against a source linked in the research doc.
- Written only: the whole design. None of the app, the lesson lint or the exercises exists yet.

## Next step

Build TODO-003 in order. Install `rich` pinned in `requirements.txt`, write the new `run.py` with the app and exercises tensors1 to tensors4 and matmul1, run `run.py dev check`, drive the app in a pseudo terminal on a scratch copy, get a beginner reader review from a sub agent, then remove the AIlings files and write a new handover.

## Traps

- The owner reads every word. Lessons that use a term before explaining it, or read as filler, have already failed twice. Hold to the writing rules in the course design.
- Two decisions from the owner are still open, building autograd by hand in part 1, which the design includes, and downloading public datasets of about 200 MB for later projects. Neither affects the first 5 exercises. Ask before part 1's autograd chapter and before any download.
- Other processes on this Mac sometimes load the CPU heavily, which makes training timings look slow. Judge speed on an idle machine.
- Writes inside the repo go through the edit tools, never scripts. `run.py init` writing the exercise files is the course's own behaviour, so test it on a scratch copy and let the owner run it in the repo.
