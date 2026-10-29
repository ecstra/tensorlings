---
name: tensorlings-phase-3-paused-with-sections-5-and-part-of-6-built
kind: doc
description: 158 exercises, CIFAR project trained and checked, Section 6 and 47_attention partly built, paused at the owner's 17:00 on 2026-10-29
updated: 2026-10-29
links: []
---

## Where it stands

Phase 3 is part built on `feat/tensorlings-phase-3`, and everything is committed. The owner paused work at 17:00 on 2026-10-29 and continues the next day. The course has 158 exercises.

- Section 5 is done: 30_convolution to 36_transfer, quiz7, and the CIFAR project, cifar1 to 3.
- The DEC-017 additions so far: soft1 and soft2 in 16_activations, fft1 in 30_convolution, 38_soft_design with derive1 and derive2, stft1 in 44_causal_conv.
- Section 6 so far: rnn1 to 3, lstm1, gru1, gating1, bidir1, causal1, dilated1, wavenet1, stft1, ssm1 to 3.
- Section 7 so far: attention1 and attention2.

CIFAR-10 downloads with `run.py data cifar10` and is read straight out of its tar archive. `run.py train cifar2` runs the file's `main()` on the best device and saves `checkpoints/cifar2.pt`. The solution's own run reached 85.6% on validation in 11.3 minutes on MPS under heavy load, and 85.3% on test.

## Verified against written

- Every new exercise passed `run.py dev check` with no lesson findings when it was written, cifar2 and cifar3 against the real checkpoint. The full dev check was not run at the pause.
- The wrong answers of every new exercise fail as they should, except cifar2's 8 and cifar3's 7, which are written but not yet run.
- Four forks built chapters 39 to 45 and 47 in parallel, and the main agent checked each exercise they reported before committing it.

## Next step

Keep on with TODO-014. What is left, with the numbers already measured.

- 40_bptt: bptt1 to 3. Measured with a default `nn.RNN` of hidden size 32, the gradient is 1.03 at the last step, 1.2e-4 at 20 steps back and 3e-11 at 50. Truncated backprop on copying a value from 3 steps earlier ends with a loss of 0.957 at chunks of 2, about 1e-4 at chunks of 4 to 20, 0.001 at 100, and 0.503 with the full 1,000 steps. A ReLU RNN started at the identity on votes of 51, with SGD at 0.01, reaches 61.8% without clipping and a largest gradient norm of 2,211, and 81.6% clipped at 1.0. Started at 1.05 times the identity it reaches 50.6% with a norm of 141 million, and 79.5% clipped.
- 41_gated_rnns: lstm2 and lstm3. Recalling the first step after 50, hidden 16, Adam at 0.01, a plain RNN got the sign task right 100% of the time while the LSTM and the GRU got 50%. On the real value task the errors were 0.21, 0.32 and 0.32, where guessing gives 0.33. With the forget gate's bias started at 3, the LSTM reached 0.0037 in 300 steps, but that run took 22 seconds. lstm2 needs a task where the gap shows inside a check's time, and its lesson must say plainly what the forget bias does.
- 42_bidirectional: bidir2.
- 43_seq2seq: encdec1, encdec2, greedy1, beam1. Copying digits with hidden size 32 and 1,500 steps gave token accuracy 0.988 and exact match 0.94 at length 5, 0.876 and 0.23 at 10, 0.494 and 0 at 20, 0.233 and 0 at 40.
- quiz8 and the adder project, 46_adder.
- 47_attention: attention3 to 6. For attention6, with 16 ids and 400 steps, attention recalled every pair at 2, 4, 8 and 16 pairs, while a GRU scored 0.534, 0.313, 0.214 and 0.164. At 4,000 steps the GRU still scored 0.336 at 4 pairs and 0.178 at 16.
- Then 48_linear_attention on, induction1, dynamic1, quiz9, quiz10 and the GPT project.
- Then step 11 to 13 of TODO-014: run every wrong answer, beginner reviews, the full dev check and the screens.

## Open questions for the owner

- When to merge the stacked branches into main, and whether to add a remote to push to.

## Traps

- "token" is in the lint's jargon list, and no lesson introduces it yet. The adder project or 47_attention has to bold it before any lesson uses it.
- Forks that edit `tensorlings/chapters/__init__.py` at once can leave a half written module registered for a minute, which stops every dev check from loading.
- A CIFAR split takes about 3 seconds to read out of the archive, and checking 5,000 images on the CPU about 22 seconds under load. The CIFAR exercises set time limits of 60 to 180.
- The Further reading links of 44_causal_conv were not fetched to check them.
