---
name: tensorlings-final-deep-review-paused-after-slice-5
kind: doc
description: every section and quiz is built and committed, the deep review has closed slices 1 to 5, and slices 6 to 10 are paused mid fix or mid review
updated: 2026-11-03
links: []
---

## Where it stands

The branch is feat/tensorlings-phase-6. All 14 sections are built, 370 exercises with all 17 quizzes, and everything below is committed unless named as stashed.

Done on 2026-11-03:
- The Section 13 and 14 reviews, every finding fixed and committed.
- Every caveat redesign, the last being post training (DPO beats SFT on self sampled judged pairs). TODO-018 has steps 1 to 6 done.
- quiz16 (Section 13, in outputs.py) and quiz17 (the whole course, in interpretability.py).
- The final deep review, TODO-020, slices 1 to 5 reviewed, fixed and committed. DEC-022 records the residual3 split.

Paused at 17:10:
- Slices 6, 7 and 9 were reviewed, and their fixers were part way through. Their uncommitted edits sit in a git stash named "deep review slices 6 7 9 partial fixes". The approved fix lists are below.
- Slices 8 and 10 were mid review. What they found so far is below.
- Slices 11 to 26 are not started.

## The reviewer brief

Each slice gets a fresh general purpose agent with this brief, then the same agent fixes its own findings on the main agent's word. It reviews one slice and edits nothing until told to fix. For every exercise and quiz it checks:
- facts against primary sources with WebFetch, readme links included
- every lesson number rerun through the module's measure functions, all seeds when under about 10 minutes
- the owner's rule that every demonstration shows its method working, with a researched and measured setup where one underdelivers
- wrong answers that should fail and valid alternatives that should pass
- term order, so nothing is used before it is taught
- beginner clarity of TODOs and terms
- whether each layer leaves the learner knowing when to use it, shown by a measured comparison

It pins threads to 4, runs one experiment at a time, and keeps scripts in a scratch folder. The report lists numbered findings with What, Evidence and Fix, then what was checked and fine, what was not done, and sources. Fixes go through the Edit tool only, each new table comes from a measure function, and each wrong answer found goes into dev/wrong_answers.py. The main agent commits per slice after rerunning the slice's dev check and wrong answers on a staged copy of the tree.

## Approved fixes waiting in the stash

Slice 6, 23_names to 26_schedules:
- adamw2 leads with the two weight table (Adam with L2 shrinks weights by gradient size, AdamW by the same fraction). Try a rate by strength grid for decoupling's separability, and drop the generalization reading unless shown.
- warmup1 rebuilt on 32 post norm blocks, 3 seeds, warmup 100 to 200 steps, framed around the rate that only trains with warmup, plus a pre norm row.
- momentum2 gains the noisy valley table, where momentum's lead shrinks as gradient noise grows.
- adam2 shows three seed ranges for the 10 step columns, and the deep model row in place of the 2000 step columns.
- Checks: eps inside the root in rmsprop1 and adam1, names3's eval and no_grad, warmup1's `<=`, momentum2's tie, adamw1's unused parameter, schedule1's step order, names1's dtype and names2's norm.
- Minor: rmsprop1's settling count, momentum1's ball picture cut, momentum2's rate grid named, two failure messages, measure functions for names3, momentum2 and adam2, schedule1's step per epoch note.

Slice 6 at the stop: everything is in except four runs. Its slice dev check was healthy at 16:57 with every wrong answer failing.
- adamw2 still shows its old decay table and "as Loshchilov and Hutter found", while adamw1 already says "adamw2 measures that". Run `measure_decay_grid` and rewrite adamw2's decay part from it, or drop that part.
- warmup1's table is in (0.0007 with 500 warmup steps beats the best plain rate on every seed, 2.249 to 2.255 against 2.255 to 2.266), but the final `measure_warmup` rerun never ran.
- momentum2's noise rows at 100 and 300 still need their 20 seed run, and adam2's deep row needs seed 2.
- `measure_names` was written but never run as a whole.

Slice 7, 27_clipping to 29_broken_training:
- broken2's check compares parameters, so a wrong clip value fails. debug5 accepts restoring `model.train()`.
- broken2 gets a table of what each piece buys. Clipping is shown by a loss spike rows, and weight decay is dropped unless a setting shows it helping, since it measured identical without it.
- clipping2 gets clipped rows, the share of steps clipped, and the cap rule near the 90th percentile of logged norms. It also gets `measure_spike` at +10 and +100.
- broken1 says what the broken run really does (the loss climbs to tens of thousands) and fixes its hint's example pair.
- debug5 gets a measured table, debug3's norm spread wording, debug2's "loss of the last step", debug4's shuffle check, attributions to CS231n and Karpathy's tweet, debug1's wording, and clipping1 names `clip_grad_value_`.

Slice 7 at the stop: the code for every finding but two is in clipping.py, debugging.py and broken_training.py, and the lessons pass the lint. Nothing has had a dev check yet.
- broken2's new recipe is Adam at 0.003, 200 warmup steps, cosine down to a tenth, batches of 64, and a global norm clip at 1.3, the 90th percentile of its norms. Weight decay is dropped, along with the `make_optimizer` TODO, since it measured the same with and without it, even on a wide model. Each piece is measured: cosine is worth 0.05 to 0.06, and clipping saves 0.025 to 0.030 after a loss spike.
- Not done: debug5's check that accepts restoring `model.train()`, and debug5's lesson table. `measure_slips` has run, and both slips lead after 10 steps and fall far behind by 1000.
- dev/wrong_answers.py will error on broken2 until its `make_optimizer` case and its clip at 1.0 case are updated. It also needs the new cases for clipping at 0.5 or 1.0.
- broken2's curriculum row becomes "the fixed run tuned with Adam, warmup, cosine decay and clipping at the 90th percentile of its norms, below a validation loss target, with a table of what each piece buys".

Slice 9, 34_upsampling to 38_soft_design:
- dna3's TODO says "AT", not "AG". dna2 points at conv4, not conv3.
- upsample1's align_corners sentence is rewritten (ask for (n - 1) * s + 1 outputs with size=), and bilinear's TODO says to keep the default corners.
- unet2's iou check gets an input where a 0.5 cut on the logits fails, and a `measure_unet` backs its table with three seed ranges (U-Net 90% to 92% of edge pixels against 82% for no skips).
- dna1 teaches `.transpose`, and transposed1 gets the stamp count example lines.
- Minor wording: augment2's generator, dna2's 10 seconds, the augmentation readme link, unet1's paper detail, IoU before the table, augment2's colour line, augment1's example, transfer2's ULMFiT and flips, dna3's 32 detectors, and quiz7's pointers.

Slice 9 at the stop: only upsampling.py is touched. upsample1, transposed1 and unet1 are done. unet2 has the new check input and `measure_unet`, but the wrong answer case, a module run of `measure_unet` and the new table are not done. Nothing in dna.py, augmentation.py or transfer.py is done yet.

## Slice 10's review, not yet fixed

Slice 10, 39_rnn to 42_bidirectional, was reviewed up to the stop. Its findings, none fixed yet:
1. bptt1 and bptt3 teach that tanh RNNs don't explode. With a vector state they do: a tanh `W_hh` at gain 3 gives gradients 6e6 to 3e8 at 59 steps back, and Pascanu clipped tanh RNNs. Limit bptt1 to the one number case and add a tanh gain 3 row to bptt3.
2. bptt3's default start rows stall from plain SGD at 0.01 with tiny gradients, not from fading. With Adam and clipping the default ties the identity start on votes. Replace them with lstm2's marked value task at length 40, where the default sits at chance (0.333) and the clipped identity start reaches 0.003 to 0.026 on 5 seeds.
3. rnn3 says the RNN keeps a count "whatever the length". It falls to 67% to 83% at length 101. Add the window's 65.2% there and claim only that it beats the window at every length tried. The readme should say "beats a fixed window once votes outgrow it".
4. lstm2 says a bias of 1 made the LSTM "nearly match" the GRU. Jozefowicz found it beat the GRU on almost every task.
5. lstm2 gives the LSTM a tuned forget bias but the GRU nothing. Measure the GRU with its update gate bias at 3 (chrono style). lstm3 should name the bias it recommends, since `make_recurrent`'s default of 1 fails at length 40.
6. bptt3's `identity_start_` needs `torch.eye` and "identity matrix", which no lesson teaches. `rnn.hidden_size` is never shown either.
7. The lstm3 case "make_recurrent never opens the forget gate" in dev/wrong_answers.py leaves a syntax error. Replace the deleted line with `pass`.
8. gating1 builds gates with no measured comparison, and its opening calls the gate the first input computed weight though derive1, lstm1 and gru1 came first.
9. lstm1 and gru1 switch to math notation, and gru1 reuses the letter c.
10. Most of the slice's tables have no measure function, and bptt3's first table is a little off in its last digits.

Not reviewed: the checks of gru1 and bidir1, bidir2's numbers, lstm2's last four cells and lstm3's equal size table and timing.

## Next step

1. Pop the stash, check each slice's files import, and finish slices 6, 7 and 9 with new fixer agents, using the lists above.
2. Rerun or finish the reviews of slices 8 and 10.
3. Run slices 11 to 26 the same way, about five agents at a time.
4. Then TODO-017's close-out: the full dev check and every wrong answer on a quiet machine, the screen driver, setting up the owner's folder from nothing, and the handovers. Close TODO-017, TODO-018 and TODO-019.

## Open questions for the owner

None new. The branch merges and a remote are still open from before.

## Traps

- Another session on this Mac uses about 10 cores. Load swings from 7 to 80, so timings in lessons are measured at low load or stated as ranges.
- Several agents edit dev/wrong_answers.py at once. Commit each slice's cases alone by filtering the diff by exercise name and applying it to the index.
- engine.py is shared by every exercise. Split its hunks by slice before staging.
- The staged tree needs a `data` link to the repo's data folder for exercises that load datasets.
