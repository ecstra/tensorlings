---
name: tensorlings-final-deep-review-paused-with-slices-23-to-26-partly-open
kind: doc
description: slices 1 to 22 are reviewed, fixed and committed, and what is left of the deep review is a short list of research redesigns, quiz reworks and the course wide passes
updated: 2026-11-04
links: []
---

## Where it stands

The branch is feat/tensorlings-phase-6, with 370 exercises in 14 sections. Everything below is committed unless named as open.

The final deep review, TODO-020, ran one fresh reviewer per slice of chapters with a shared brief, then had the same agent fix its own findings. Each slice was committed only after its dev check and wrong answers passed on a staged copy of the tree.

- Slices 1 to 26 are reviewed, fixed and committed, apart from the open items below.
- In the evening's run:
  - Every quiz from quiz1 to quiz17 now combines earlier moves.
  - A lint rule now flags any hint that quotes a solution line, and every flagged hint was rewritten, so the lint finds none across the course.
  - bridge2 now shows stage one keeping more of the GPT's Shakespeare on every seed, at LLaVA-1.5's rate.
  - physics3 decays its rate, so no seed stalls.
  - quiz17's train_step combines micro batches with clipping.
  - combine2 gains a case where adding wins.
  - combine3's zero start wins with a wide branch.
  - scaling3's rate was tuned per point.
- Every agent from this session can be resumed by message with its context and scratch work intact. The slice 23 to 25 agents hold the detail behind the open items.

## Open, in the order to take them

1. **series3**. Its text now states the measured numbers plainly, but neither of PatchTST's two choices clearly wins at the 104 week lookback. The lead is a longer history for fewer tokens: 64 patches over 148 weeks beat 105 week tokens over 104 weeks on all 3 seeds at 24 weeks, and on 2 of 3 at 36. Run 48 and 60 weeks, finish the seed matched week token runs, then rebuild the table through `measure_choices(seeds=range(3), lookback=148)`. 148 weeks is the course's choice, not the paper's, so don't credit it to PatchTST.
2. **table3**. Attention over the column tokens loses to the MLP on complete rows. An agent was trying the FT-Transformer paper's levers at the stop (the quantile transform, d 192 with ReGLU, lr 1e-4 with early stopping), so read its report first.
3. **combine4 and combine5**. combine4's attention and convolution across blocks are in the file with `measure_across`. Its run writes across.json in the slice 25 scratch folder, so add the row and a paragraph from it. combine5's fourth problem, counting the bumps of the kind a query names, needs conditioning, but joining the query on works as well as FiLM or a gate. Make its check accept all three, and steady the gate's training first, since it failed one seed in three.
4. **Small follow ups**.
   - bridge2's Shakespeare mixing run is coded but not run. Stage one raising the question scores is still not shown, and stage two at LLaVA-1.5's LoRA rate is the untested idea.
   - scaling3's tuned exponent is 0.62, so a grid reaching 4e-2 for the narrowest widths would rule out the grid's edge.
   - scaling3's solution can pass its 120 second limit on its first sweep line under heavy load, so consider a longer limit.
   - paper2 runs on the GPU, which doesn't repeat exactly. A CPU run would make its tables repeat, at about twice the time.
5. **The quiet timing pass**, step 28 of TODO-020. Still to remeasure quietly with threads pinned are profile1's 1 and 10 thread sentence, lstm3, adder2, gpt3, bpe3, topics1, word2vec2 and reader2's "about 2 minutes". fast3's lesson sits right at the length limit.
6. **The close-out**, TODO-017 step 8. Run the full dev check and every wrong answer on a quiet machine, run the screen driver, and set up the owner's folder from nothing. The owner's exercises/ folder holds stale files for several changed exercises, so remove exercises/, solutions/ and .tensorlings-state.json before `run.py init`. Then close TODO-017, TODO-018, TODO-019 and TODO-020.

## Traps

- Another session on this Mac often pushes the load average to 30 to 100, so seed runs crawl and timings swing. Commit timing tables only from pinned runs on a quiet machine.
- Several agents edit dev/wrong_answers.py and shared files at once. Commit each slice's wrong answers alone by filtering the diff to its exercise names and applying the result to the index, and split shared files such as engine.py by hunk.
- The staged copy of the tree needs links to the repo's data and checkpoints folders before its dev check can load datasets and checkpoints.
- quiz13_module builds its module before it seeds, so the quiz's modules aren't really seeded. The checks stay consistent, since the learner and the reference get the same module.
