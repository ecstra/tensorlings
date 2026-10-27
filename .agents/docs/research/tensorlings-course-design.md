---
name: tensorlings-course-design
kind: doc
description: the tensorlings curriculum, how each block is taught, the combination rules, the design method, the exercise format, the app and the writing rules
updated: 2026-10-27
links: []
---

# tensorlings course design

tensorlings is a rustlings style course that takes the owner from tensors to designing new deep learning modules. One file, `run.py`, holds the runner, the full-screen app and every exercise, lesson, hint and solution. `run.py init` writes the exercise files into `exercises/`. The research behind every choice here is in [the research doc](tensorlings-research.md), and the build order is the plan `build-tensorlings`.

## The goal

By the end the owner can explain any common layer four ways, what it computes, why it exists, when to use it and how to build it from plain tensors. The owner can combine layers into larger models such as a transformer, and design new modules for their own problems, such as part of an echo canceller or a small quant model.

## The curriculum

About 300 small exercises in 13 parts. Each part ends with a project.

| Part | Content |
|---|---|
| 1 Foundations | tensors, shapes, indexing, broadcasting, reductions, matrix multiply, derivatives, the chain rule, a scalar autograd engine by hand, backprop through tensors by hand |
| 2 Learning | linear and softmax regression, losses as probability, gradient descent, minibatches, train, validation and test splits, overfitting, weight decay |
| 3 Building blocks | Linear, the common activations and why each, embeddings, MLPs, initialization, dropout, normalization, residuals. Project, a name generator |
| 4 Optimization | SGD, momentum, RMSProp, Adam, AdamW, schedules, warmup, gradient clipping, the debugging recipe |
| 5 Convolutional nets | convolution derived from Linear, channels, stride, dilation, pooling, LeNet to ResNet, upsampling, U-Net, augmentation, transfer learning. Project, a small image classifier |
| 6 Sequences | RNN, backprop through time, LSTM, GRU, bidirectional models, encoder and decoder, beam search, causal and dilated convolution, state space models |
| 7 Attention and transformers | attention derived from averaging, masks, multi head, cross attention, positional encodings with RoPE, encoder and decoder models, GPT-2 to Llama, mixture of experts, vision transformers, byte pair encoding. Project, a GPT on real text |
| 8 Representation learning | word2vec, contrastive learning, masked modeling, autoencoders |
| 9 Generative models | sampling, VAE, GAN, normalizing flows, diffusion, VQ-VAE |
| 10 Symmetry and graphs | Deep Sets, message passing, equivariance, structure chosen from the data's symmetries |
| 11 Post training and RL | fine tuning, LoRA, SFT, policy gradients, PPO, reward models, DPO |
| 12 Systems and inference | cost of each block, profiling, mixed precision, FlashAttention's tiling idea, KV cache, quantization, data parallel training, scaling laws, data cleaning |
| 13 Your own modules | when and why to combine blocks, the design method, reproducing a paper, interpretability tools. Projects, a toy echo canceller, a toy quant forecaster with leakage traps, and a module of the owner's choice |

## How every block is taught

Each block gets five small exercises.

1. What it computes, the formula in plain words, a worked example with real numbers, and the shapes in and out.
2. Build it from plain tensors, then match PyTorch's own version.
3. Why it exists, a toy task where the model fails without it and works with it.
4. When to use it, a rule of thumb, then choosing the right block for problems the owner has not seen.
5. From the paper, implementing it from the paper's formula where there is one.

## When and why to combine blocks

Each rule gets exercises where the wrong combination fails and the right one works.

- A nonlinearity goes between two Linears, because two Linears in a row collapse into one.
- The nonlinearity follows its job. ReLU or GELU for hidden features, sigmoid for a gate, tanh for a bounded value, softmax to choose among options.
- Discrete inputs enter through an embedding, and a Linear head with softmax gives the output scores.
- Mixing across positions alternates with mixing within each position.
- Project before comparing and merge after, which is why attention has Q, K, V and output layers.
- Pool after a weight shared layer to turn "the same response anywhere" into one answer.
- Normalize before a sublayer and add a residual around it.
- A new branch starts as "do nothing", with its last layer at zero.
- Gate when one signal should control another.
- FiLM when a side input should change behaviour.
- Top k routing for more capacity at the same compute, with a balancing loss.

## The design method

1. Build a toy task that isolates one skill.
2. Train the simplest baseline and diagnose it. Did it fail to fit, or fit and fail to generalize?
3. Write the exact algorithm you would code by hand.
4. Swap each hard step for a differentiable one.
5. Check gradient health before training. Does it start as the identity, and can gradients pass it?
6. Test the property directly by shuffling, shifting, rotating or lengthening the input.

## Exercise format

- One file per exercise, `exercises/<chapter>/<name>.py`, like `exercises/01_tensors/tensors1.py`.
- The file holds the code, a `TODO` at every place to change, and a closing comment that lists what the check tests.
- Every exercise is code to write or fix. There are no prediction only exercises.
- Each check is a list of named tests. A failing test says in plain words what came back and what was expected.
- After a pass the reference solution is written to `solutions/<chapter>/<name>.py`.

## The app

- `run.py` draws a full-screen app with the `rich` package in the terminal's alternate screen, so it redraws in place and never scrolls.
- The screen has a header with the exercise and progress, a lesson panel that scrolls with the arrow keys, a hint panel that `h` opens and closes, a result panel with each test, and a key bar.
- The header takes two lines and a blank one under them, the course name, chapter and progress first, then the exercise and its file. The owner found a tighter header cramped. Spacing and color separate the parts, never a dot, which the owner disliked.
- The hint has its own panel between the lesson and the result, because it is help and not a result.
- Keys are `n` next, `h` to show or hide the hint, `l` list, `x` reset, `q` quit, `r` run again, and the arrows to scroll.
- The list screen works like rustlings, with DONE and PENDING, the current exercise marked, and keys to move, continue at, reset, search and filter.
- Commands are `init`, `run`, `hint`, `reset`, `check-all`, `--manual-run` and `dev check`.
- The result panel puts the most important thing first, because a tall result is cut off at the bottom. The done message comes first, then the tests. Only the first failing test shows its details, the way a compiler shows its first error. Functions that still return nothing share one note instead of one message each. See `result_body` in `run.py`.
- Common PyTorch errors get a plain words explanation above PyTorch's own message, from the table `PLAIN_ERRORS` in `run.py`. A new error that confuses learners gets a row there.
- Anything the learner's code prints is caught and shown in the result panel, since printing straight to the screen would break the drawing.
- Arrow keys are read one escape sequence at a time in `KeyReader.read_escape_sequence`, so a held arrow key scrolls instead of reading as escape.

## Writing rules

- Plain English, mostly under 20 words per sentence, one idea per sentence.
- Every term explained where it first appears, never relying on a later lesson.
- Every lesson opens with a concrete example using real numbers, then names the general idea.
- One idea per exercise, and a lesson fits on one screen.
- Every lesson says in one sentence why the thing exists.

## Verification

- `run.py dev check` runs every exercise twice. The starting file must fail and the solution must pass.
- A lesson lint in the dev check flags sentences over 25 words, filler words, dashes and semicolons, a missing "Why it exists" or "Your task" paragraph, a lesson taller than 48 lines at width 96, and any term from `JARGON` in `run.py` used before an exercise lists it in `introduces`. A new technical term gets added to `JARGON` the first time a lesson needs it.
- A separate reviewer agent reads every lesson as a complete beginner before the owner sees it.
- `dev/wrong_answers.py` runs common wrong answers through the checks. Every one must fail with a message a beginner can act on, and the script exits with 1 if one passes. New exercises add their likely mistakes to its `CASES`.
- `dev/drive_screen.py` drives the app in a pseudo terminal on a scratch copy of the course and prints each screen through the `pyte` terminal emulator. It runs before each hand over, at a normal size and at 80 by 24. Its docstring has the command.

## Limits on this machine

CUDA kernels, Triton and multi GPU training cannot run on the owner's Mac. Those lessons teach the idea in plain PyTorch on the CPU or the Mac's GPU and say where real hardware is needed.
