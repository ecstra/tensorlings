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

- Slices 1 to 22 are reviewed, fixed and committed.
- Slices 23 to 26 are reviewed and committed with their quick fixes. Each has research redesigns or reworks still open (below).
- The check on quiz1 to quiz8 is done and its rework is open.
- Every agent from this session can be resumed by message with its context and scratch work intact, as this morning's were. The slice 23 to 26 reviewers and the quiz check agent hold the detail behind items 1 to 7.

## Open, in the order to take them

1. **bridge2**. Stage one never pays off on the caption task: stage two alone scores as well, and both runs lose most of the GPT's Shakespeare. Try LLaVA's ablation setting first, with plentiful captions for stage one and a small fixed set of question examples for stage two, where an aligned projector should raise the final score. Failing that, mix Shakespeare batches into stage two as LLaVA-1.5 mixed ShareGPT text, and measure what that keeps. Also move stage one's rate to 1e-3 as LLaVA-1.5 did for its MLP projector, then rerun bridge2, bridge3 and reader2.
2. **series3**. Patches lose to one token a week by about 3% at three horizons, and mixing the columns beats keeping them apart at 48 and 60 weeks. The best lead is PatchTST/64's own flu setting, a lookback of 148 weeks in 64 patches, which scored 1.87 to 1.98 at 24 weeks on 3 seeds against 2.02 for week tokens. It needs the other horizons and seed matched week token runs. The lesson still says patches give "the same error".
3. **table3**. Attention over the column tokens loses to the MLP on complete rows, and the missing column win holds only against an MLP that never trained with holes. Untested levers from the FT-Transformer paper are the quantile transform on every input, d 192 with ReGLU and no norm before the first block, and lr 1e-4 with early stopping at patience 16.
4. **scaling3**. The IsoFLOP sweep's best size grows as compute to 0.61, not Chinchilla's 0.5, and tokens per parameter fall from 263 to 115. A peak rate of 1e-2 helped every width up to 24 on the first three budgets. A proper retune needs 6e-3 and the two top budgets, about 25 to 40 minutes a sweep. The lesson already ties its numbers to scaling2's 20 and gives Porian and colleagues' causes.
5. **quiz17 and physics3**. quiz17's train_step repeats gpt's train_step, so combine it with ddp2's micro batch accumulation. physics3's RelativeForces stalls at 66% on seed 1 while absolute pairs reach 78% to 88%. Try lr 1e-3 for all four modules and remeasure both tables.
6. **quiz1 to quiz8**. 13 of their 32 parts repeat one earlier TODO: quiz2's partials_of_exp_product, quiz3's standardize_both, quiz4's batchnorm_mlp and eval_accuracy, quiz6's adamw_for, set_rate and start_and_baseline, quiz7's image_size_through and conv_bn_relu, and all four of quiz8's parts. Rework each to combine two or three earlier moves, as quiz9 to quiz17 were. Fix the quiz tables that cite the wrong exercise too (quiz2's partials, quiz6's set_rate and clipped_step). quiz10's rms_pre_norm_block also repeats quiz5's rms_pre_norm.
7. **Slice 25's research items**, from combining and papers. combine3's signal column shows Kaiming ending lowest (0.0041 to 0.0048 against the zero start's 0.0054 to 0.0059), and no setup tried made the zero start win there, though ControlNet's ablation claims it at scale. combine5, the chapter's capstone, never asks for a combine1 to combine4 piece: add a problem that needs conditioning, such as counting the bumps of the kind a query names. combine4 never says how to choose the across piece: add a row with attention as the across block. combine2's "add when signals are evidence for the same answer" has no case that shows it. paper2 runs on the GPU, which doesn't repeat exactly, so a CPU run would make its tables repeat at about twice the time.
8. **The course wide passes**, steps 28 and 29 of TODO-020. Remeasure every timing table on a quiet machine with threads pinned, including profile1, kvcache2, fast1, fast3's three sizes, lstm3, adder2, gpt3, bpe3, topics1 and word2vec2, and drop the busy machine notes. Add a lint rule that flags a hint quoting a solution line word for word, then rewrite every hint it finds (about 71 exercises).
9. **The close-out**, TODO-017 step 8. Run the full dev check and every wrong answer on a quiet machine, run the screen driver, and set up the owner's folder from nothing. The owner's exercises/ folder holds stale files for several changed exercises, so remove exercises/, solutions/ and .tensorlings-state.json before `run.py init`. Then close TODO-017, TODO-018, TODO-019 and TODO-020.

## Traps

- Another session on this Mac often pushes the load average to 30 to 100, so seed runs crawl and timings swing. Commit timing tables only from pinned runs on a quiet machine.
- Several agents edit dev/wrong_answers.py and shared files at once. Commit each slice's wrong answers alone by filtering the diff to its exercise names and applying the result to the index, and split shared files such as engine.py by hunk.
- The staged copy of the tree needs links to the repo's data and checkpoints folders before its dev check can load datasets and checkpoints.
- quiz13_module builds its module before it seeds, so the quiz's modules aren't really seeded. The checks stay consistent, since the learner and the reference get the same module.
