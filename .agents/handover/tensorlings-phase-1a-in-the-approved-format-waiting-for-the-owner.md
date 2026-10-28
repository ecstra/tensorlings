---
name: tensorlings-phase-1a-in-the-approved-format-waiting-for-the-owner
kind: doc
description: all seven phase 1a exercises use explain then try, fix the error and try first, checked by name, and wait for the owner to try them
updated: 2026-10-28
links: [handover/tensorlings-phase-1a-reworked-waiting-for-the-owner.md, todos/TODO-009-convert-the-rest-of-phase-1a-to-explain-then-try.md, decisions/DEC-013-exercises-explain-then-try-fix-an-error-or-try-first.md]
---

# tensorlings phase 1a in the approved format, waiting for the owner

Written 2026-10-28 at 17:27 IST. It follows [the earlier handover](tensorlings-phase-1a-reworked-waiting-for-the-owner.md), which describes a format since replaced.

## Where it stands

- Phase 1a is seven exercises in `run.py`, tensors1, indexing1, math1, broadcasting1 and reductions1 in `01_tensors`, matmul1 in `02_matmul`, and quiz1 in `quizzes`. Each uses the methods in [DEC-013](../decisions/DEC-013-exercises-explain-then-try-fix-an-error-or-try-first.md), a short working example above each TODO, a line that crashes on purpose where the error teaches, and a try first TODO opening indexing1, reductions1 and matmul1.
- The files have no print lines and no tests. The checker runs them one statement at a time and checks the names they set, and the result lists every TODO with its value or its first problem.
- The owner approved tensors1 and indexing1 and finished both. The other five were written to their folder on 2026-10-28, and their progress points at math1.
- [TODO-009](../todos/TODO-009-convert-the-rest-of-phase-1a-to-explain-then-try.md) has every step done and stays open until the owner tries the five, and so do TODO-006, TODO-005 and TODO-003 above it.

## Verified against written

- Verified: `run.py dev check` passes all seven with a clean lesson lint, all 38 cases in `dev/wrong_answers.py` fail with plain messages, every lesson output and file comment value was run in torch 2.14.1, and `dev/screens.py` screenshots at 100 by 34 and 80 by 24 look right.
- Reviewed: one beginner reader review by an Opus sub agent, with every finding fixed, including contradicting hints about dimension 1 and a quiz header that claimed every step built on the one before.
- Not verified: how math1 to quiz1 read to the owner.

## Next step

1. Wait for the owner's verdict on math1 to quiz1 and fix what they ask for.
2. When they call phase 1a fine, close TODO-009, TODO-006, TODO-005 and TODO-003 with their closing commits in Done when, and mark plan steps 2 and 3 done.
3. Phase 1b is the rest of Sections 1 and 2 in the same format. Section 1's end of section project is not named yet, so propose one to the owner first.

## Open questions for the owner

- Whether to build autograd and backprop by hand in Section 1.
- Whether downloading about 200 MB of public datasets is fine.
- Whether to review and merge phase 1a before phase 1b, and whether to add a remote.

## Traps

- Every TODO repeats the move its example just showed or changes one thing. Never ask for several new moves at once.
- Hints must agree across exercises. Say dim=1 and what it leaves, never "rows run along" a dimension.
- Nothing in the app is gated, and the result lists every TODO together. The owner tried and dropped a lesson lock and a one problem at a time result.
- When the owner names a tool, read how it really works before building.
