---
name: tensorlings-phase-1a-reworked-waiting-for-the-owner
kind: doc
description: seven exercises, one Section 1 topic each plus quiz1, built and checked, waiting for the owner to try them
updated: 2026-10-28
links: [handover/tensorlings-waiting-for-the-owner-s-verdict-on-phase-1a.md, todos/TODO-006-six-exercises-one-topic-each-with-a-closing-quiz.md, decisions/DEC-012-one-topic-per-exercise-with-real-model-operations-and-quizzes.md]
---

# tensorlings phase 1a reworked, waiting for the owner

Written 2026-10-28 at 14:39 IST. It follows [the earlier handover](tensorlings-waiting-for-the-owner-s-verdict-on-phase-1a.md), much of which is out of date. The app is on Textual now, not rich, and the exercises changed twice since.

## Where it stands

- `run.py` holds seven exercises. tensors1 to tensors5 teach tensors and shapes, indexing and slicing, elementwise math, broadcasting and reductions. matmul1 teaches matrix multiplication and ends with a layer computed the way `torch.nn.functional.linear` does it. quiz1 in `exercises/quizzes/` combines them into the forward pass of a small sentence classifier, with nothing new.
- Every task is an operation real models run, per [DEC-012](../decisions/DEC-012-one-topic-per-exercise-with-real-model-operations-and-quizzes.md). The makemore bigram model is gone. It was never in the agreed curriculum, see [the curriculum memory](../memory/the-curriculum-changes-only-on-the-owner-s-word.md).
- The owner calls the 13 curriculum parts sections. The docs, the plan's prose and the memories say section. The plan's step rows still say parts, since the brain tool never rewords a step.
- The owner's `exercises/` folder and progress were reset to the new versions. They hold no work of the owner's.
- [TODO-006](../todos/TODO-006-six-exercises-one-topic-each-with-a-closing-quiz.md) has every step done and stays open until the owner tries it, and so do TODO-005 and TODO-003 above it.

## Verified against written

- Verified: `run.py dev check` passes all seven, with every stub failing, every solution passing and a clean lesson lint. All 35 cases in `dev/wrong_answers.py` fail. Every output a lesson shows was run in torch 2.14.1. `dev/screens.py` screenshots at 100 by 34 and 80 by 24 look right, the quiz lesson included.
- Reviewed: one beginner reader review by an Opus sub agent. Every finding was fixed, including hints and lesson examples that gave answers away and 3D cases the quiz needed.
- Not verified: how the lessons read to the owner.

## Next step

1. Wait for the owner's verdict on the seven exercises and fix what they ask for.
2. When they approve, close TODO-006, TODO-005 and TODO-003 with their closing commits named in Done when, and mark plan steps 2 and 3 done.
3. Phase 1b is the rest of Sections 1 and 2. Section 1's end of section project is not named yet, so propose one to the owner before building it.

## Open questions for the owner

- Whether to build autograd and backprop by hand in Section 1, which reverses an earlier pick that PyTorch handles gradients.
- Whether downloading about 200 MB of public datasets is fine.
- Whether to review and merge phase 1a before phase 1b, and whether to add a remote.

## Traps

- Never change the curriculum, a section's topics, its project or the order without the owner's clear yes.
- A lesson may show a tool on its own example, but never the task's line with the names changed. The lint only catches word for word copies, so read the hints and examples against each solution too.
- A lesson must teach the 3D form of a tool before a quiz needs it.
- Lessons say "a later chapter", never "Section 3", since the app shows chapters only.
