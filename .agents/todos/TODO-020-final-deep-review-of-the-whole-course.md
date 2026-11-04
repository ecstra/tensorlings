---
name: TODO-020-final-deep-review-of-the-whole-course
kind: todo
description: every lesson fact checked against its sources, every number rerun, every check tried with wrong and right answers, fixed before the owner starts
updated: 2026-11-04
links: []
status: open
by: owner
area: course
born: 2026-11-03
---

## What

The owner asked for one last deep review with research before they start, "so I don't mislearn". It covers all 370 exercises in 26 slices of whole chapters. Each slice gets a fresh reviewer with one shared brief, which checks facts against primary sources, reruns the numbers, tries wrong and right answers on the checks, and looks at term order, beginner clarity, the owner's rule that every demonstration shows its method working, and whether each layer leaves the learner knowing when to use it. The reviewer then fixes its own findings, and each fix is committed on its own.

## Steps

| # | Step | Points at | By | Status |
|---|---|---|---|---|
| 1 | Review and fix 01_tensors to 06_tensor_backprop, with quiz1 and quiz2 | | owner | done |
| 2 | Review and fix 07_micrograd to 12_likelihood | | owner | done |
| 3 | Review and fix 13_generalization to 16_activations, with quiz3 | | owner | done |
| 4 | Review and fix 17_mlp to 19_initialization | | owner | done |
| 5 | Review and fix 20_dropout to 22_residuals, with quiz4 and quiz5 | | owner | done |
| 6 | Review and fix 23_names to 26_schedules | | owner | done |
| 7 | Review and fix 27_clipping to 29_broken_training, with quiz6 | | owner | done |
| 8 | Review and fix 30_convolution to 33_classic_nets | | owner | done |
| 9 | Review and fix 34_upsampling to 38_soft_design, with quiz7 | | owner | done |
| 10 | Review and fix 39_rnn to 42_bidirectional | | owner | done |
| 11 | Review and fix 43_seq2seq to 46_adder, with quiz8 | | owner | done |
| 12 | Review and fix 47_attention to 50_multihead | | owner | done |
| 13 | Review and fix 51_cross_attention to 55_modern_llms | | owner | done |
| 14 | Review and fix 56_moe to 59_gpt, with quiz9 and quiz10 | | owner | done |
| 15 | Review and fix 60_word2vec to 64_paper_topics, with quiz11 | | owner | done |
| 16 | Review and fix 65_sampling to 68_flows | | owner | done |
| 17 | Review and fix 69_diffusion to 71_digit_diffusion, with quiz12 | | owner | done |
| 18 | Review and fix 72_sets to 75_shortest_paths, with quiz13 | | owner | done |
| 19 | Review and fix 76_finetuning to 80_ppo | | owner | done |
| 20 | Review and fix 81_reward_models to 83_post_training, with quiz14 | | owner | pending |
| 21 | Review and fix 84_cost to 89_quantization | | owner | done |
| 22 | Review and fix 90_parallel to 93_fast_gpt, with quiz15 | | owner | done |
| 23 | Review and fix 94_tables to 99_shared_space | | owner | pending |
| 24 | Review and fix 100_fusion to 103_signal_reader, with quiz16 | | owner | pending |
| 25 | Review and fix 104_combining to 106_papers | | owner | pending |
| 26 | Review and fix 107_interpretability to 110_own_module, with quiz17 | | owner | pending |
| 27 | Check quiz1 to quiz8 combine two or three earlier moves per part, since quiz9 to quiz11 repeated earlier TODOs word for word | | agent | done |
| 28 | Remeasure the timing tables on a quiet machine with threads pinned, profile1 and kvcache2 first, and drop their busy machine notes | | agent | pending |
| 29 | Add a lint rule that flags a hint quoting a solution line word for word, and rewrite every hint it finds across the course | | agent | pending |

## Done when

Every slice is reviewed and every finding is fixed or refuted with its reason, each fix committed. Then the full dev check is healthy and every wrong answer fails on a quiet machine.
