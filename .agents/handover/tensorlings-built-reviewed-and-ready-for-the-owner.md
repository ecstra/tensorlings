---
name: tensorlings-built-reviewed-and-ready-for-the-owner
kind: doc
description: all 14 sections and 370 exercises pass the full check after the final deep review, and the owner's folder is set up from nothing
updated: 2026-11-05
links: []
---

## Where it stands

The branch is feat/tensorlings-phase-6. The course has 14 sections, 370 exercises and 17 quizzes, and it is ready for the owner to start.

The final deep review, TODO-020, is complete. Every chapter was reviewed in 26 slices, and each review checked:
- facts against primary sources
- every lesson number rerun
- wrong and right answers tried against every check
- term order and beginner clarity
- the owner's rule that every demonstration shows its method working

Every finding was fixed, or refuted with its reason, and committed slice by slice. On top of the slices:
- Every quiz from quiz1 to quiz17 now combines two or three earlier moves.
- A new lint rule flags any hint that quotes a solution line, and the lint finds none across the course.
- Every timing claim was remeasured on 2026-11-05 at a load of 3 to 5 with threads pinned.

The owner's folder is set up from nothing. The old exercises, solutions and progress are gone, and `run.py init` wrote all 370 exercise files fresh.

## Verified against written

- `run.py dev check` on all 370 exercises reported 368 healthy and 2 with one sentence over 25 words. Both sentences were split and rechecked healthy.
- `dev/wrong_answers.py` over the whole course reported all 2,724 wrong answers failing.
- `dev/screens.py` drove the app headless. The welcome screen, an exercise and its hints tab rendered with the progress bar at 0 of 370.
- The reference checkpoints were in `checkpoints/` for those runs. They now sit in `dev/reference_checkpoints/`, which git ignores, so the owner trains their own. Until then, the checks that read a checkpoint fail with a message naming the `run.py train` command to run. A dev check with the checkpoints moved out lists 20 of them: dna2, dna3, adder2, gpt3, gpt4, topics2, topics3, digits2, digits3, finetune1, lora1, sft2, post2, post3, quant3, fast1, fast2, fast3, reader3 and paper2. Copy a file back into `checkpoints/` to run its check as maintainer.

## Next step

The owner starts the course with `.venv/bin/python run.py` from the repo root.

## Open questions for the owner

- Merging the stacked phase branches into main is the owner's call.
- So is adding a remote. Nothing has been pushed, since the repo has none.

## Traps

- Another session on this Mac often pushes the load average to 30 to 100. Measure timings only with threads pinned, and say the load.
- A wait loop written as `until ! pgrep -f "name"` never ends, because pgrep matches the loop's own command line. Wait on a process id instead.
- When several agents share dev/wrong_answers.py, commit each one's cases alone by filtering the diff to its exercise names.
