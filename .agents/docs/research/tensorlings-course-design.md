---
name: tensorlings-course-design
kind: doc
description: the tensorlings curriculum, how each block is taught, the combination rules, the design method, the exercise format, the app and the writing rules
updated: 2026-11-10
links: []
---

# tensorlings course design

tensorlings is a rustlings style course that takes the owner from tensors to designing new deep learning modules. `run.py` starts it, and the `tensorlings/` package holds the rest, per [DEC-015](../../decisions/DEC-015-the-course-lives-in-a-package-with-one-module-per-chapter.md). Its docstring names each module's job, and every chapter, with its lessons, hints, checks and solutions, has its own module in `tensorlings/chapters/`. `run.py init` writes the exercise files into `exercises/`. The research behind every choice here is in [the research doc](tensorlings-research.md), and the build order is the plan `build-tensorlings`.

## The goal

By the end the owner can explain any common layer four ways, what it computes, why it exists, when to use it and how to build it from plain tensors. The owner can combine layers into larger models such as a transformer, and design new modules for their own problems, such as a learned physics simulator or a small quant model.

## The curriculum

About 370 small exercises in 14 sections. Each section ends with a project that uses what it taught, and the capstones are the projects of Section 14. The table names the projects the owner set, and [the detailed curriculum](tensorlings-detailed-curriculum.md) lists every chapter, exercise and project, including the ones the agent picked per [DEC-014](../../decisions/DEC-014-projects-for-the-sections-the-table-left-open.md).

| Section | Content |
|---|---|
| 1 Foundations | tensors, shapes, indexing, broadcasting, reductions, matrix multiply, derivatives, the chain rule, a scalar autograd engine by hand, backprop through tensors by hand |
| 2 Learning | linear and softmax regression, losses as probability, gradient descent, minibatches, train, validation and test splits, overfitting, weight decay |
| 3 Building blocks | Linear, the common activations and why each, embeddings, MLPs, initialization, dropout, normalization, residuals. Project, a name generator |
| 4 Optimization | SGD, momentum, RMSProp, Adam, AdamW, schedules, warmup, gradient clipping, the debugging recipe |
| 5 Convolutional nets | convolution derived from Linear, channels, stride, dilation, pooling, LeNet to ResNet, upsampling, U-Net, augmentation, transfer learning. Project, a DNA motif finder |
| 6 Sequences | RNN, backprop through time, LSTM, GRU, bidirectional models, encoder and decoder, beam search, causal and dilated convolution, state space models |
| 7 Attention and transformers | attention derived from averaging, masks, multi head, cross attention, positional encodings with RoPE, encoder and decoder models, GPT-2 to Llama, mixture of experts, vision transformers, byte pair encoding. Project, a GPT on real text |
| 8 Representation learning | word2vec, contrastive learning, masked modeling, autoencoders |
| 9 Generative models | sampling, VAE, GAN, normalizing flows, diffusion, VQ-VAE |
| 10 Symmetry and graphs | Deep Sets, message passing, equivariance, structure chosen from the data's symmetries |
| 11 Post training and RL | fine tuning, LoRA, SFT, policy gradients, PPO, reward models, DPO |
| 12 Systems and inference | cost of each block, profiling, mixed precision, FlashAttention's tiling idea, KV cache, quantization, data parallel training, scaling laws, data cleaning |
| 13 Modalities | each kind of input on its own, tables, time series, audio and video, then kinds joined in one transformer and in your GPT, a shared space, fusion and outputs. Project, your GPT reading signals |
| 14 Your own modules | when and why to combine blocks, the design method, reproducing a paper, interpretability tools. Projects, a learned particle simulator, a toy quant forecaster with leakage traps, and a module of the owner's choice |

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

- One file per exercise, `exercises/<chapter>/<name>.py`, like `exercises/01_tensors/tensors1.py`. Quizzes go in `exercises/quizzes/`, as in rustlings.
- The file holds the code and a `TODO` at every place to change, with no print lines and no tests.
- Every exercise is code to write or fix. There are no prediction only exercises.
- The checker runs the file one top level statement at a time, so a line that crashes on purpose hides nothing after it. A crash fails only the names its statement sets, and a crash no check reports fails the file. A statement that runs past the time limit counts the same way, so a slow top level line fails its own names and the function rows still run. See `load_module` in `tensorlings/engine.py`.
- Each check is a list of tests, and every test belongs to one name the file sets, through `Checker.check_name`, one function, through `Checker.test`, or one method of a class the file defines, through `Checker.test_method`, whose row reads like `Value.__mul__`. A failing test says in plain words what came back and what was expected. A passing name shows its value and shape in the result, in place of print lines.
- The inputs a check calls a function on are small whole numbers, from `small_tensor` in `tensorlings/engine.py`, so a failing call stays short enough to read. Training data, functions and random generators are wrapped in `Described`, which the result shows by a short text like `x of shape (8, 2)` or `a generator seeded with 0`, and `seeded` makes such a generator. `check_on_inputs` hands the learner's function and the reference each a fresh copy of every tensor and generator.
- Each name or function an exercise checks has one hint, per [DEC-010](../../decisions/DEC-010-one-hint-per-function.md).
- A check tests whatever its exercise produces, not only names. A function is called on several inputs, and a later exercise can build the learner's class and compare it with PyTorch's, or run their training loop and check that the loss goes down. A new kind of check gets added to `Checker` when an exercise first needs it.
- A test that runs past `TIME_LIMIT_SECONDS` fails with a note about loops that never end, so a learner's endless loop can't stall the app. See `run_with_time_limit` in `tensorlings/engine.py`. An exercise that trains on real data sets a longer `time_limit` of its own, as the exoplanet project does. The projects of Sections 3 and 4 train their small names model inside the check too, for at most 10,000 steps, which takes from 10 seconds to a minute or two on a busy laptop CPU.
- A project whose training runs for minutes, dna2, adder2, gpt3, topics1, digits2 and post1 to post3 so far, trains outside the app. `run.py train <name>` runs the file's `main()` on `best_device()` per DEC-016, which saves `checkpoints/<name>.pt` with `save_checkpoint`. The check reads it back on the CPU with `load_checkpoint`, and a missing checkpoint fails with the command to run. See `tensorlings/training.py`.
- A check that needs to know how a function used another one, like whether `exact_match` ran `greedy` in eval mode without autograd, swaps a scripted stand in into the learner's module globals for the call and puts the original back after. See `check_exact_match` in `tensorlings/chapters/adder.py`.
- Checks run on the CPU, so a training function that forgets `.to(device)` would pass and then crash in `main()` on a GPU. `expect_on_device` in `tensorlings/engine.py` runs such a function with the device set to `meta` and a `DeviceRecorder` stand in for the model, and fails on any batch or target left on the CPU.
- A function whose reference gives a plain number takes a plain number only, and a tensor fails with a pointer to `.item()`. A reference that gives a tensor still accepts a plain number or a tensor of one entry. See `single_entry` in `tensorlings/engine.py`.
- A `...` left in a TODO makes the test say the function is not written yet, and names the line when the line that failed holds the `...`.
- An exercise that needs a dataset loads it with a loader from `tensorlings/data.py`, like `load_mnist`, which the learner's file imports. `run.py data <name>` downloads it once into `data/`, checked against its checksum, and a file run before that crashes with the command to run. The download goes through `curl` when the machine has it, since Python from python.org on a Mac ships without certificates.
- A check never passes a tensor to `.backward()`. Its first call in a process imports sympy, which takes about 4 seconds untraced and runs past the time limit under the trace, leaving sympy half imported until the app restarts. A reference gradient multiplies the result by the gradient handed down and sums it, which gives the same gradient with a plain `.backward()`. See `autograd_backward` in `tensorlings/chapters/tensor_backprop.py`.
- The first optimizer `torch.optim` makes in a process imports `torch._dynamo`, which pulls in sympy and takes 10 to 20 seconds. That can't be avoided, so every exercise that makes an optimizer sets `imports_first=OPTIMIZER_IMPORTS`, and `attempt` imports those modules before the clock starts. The first check of such an exercise is slow once per process, and never breaks. In the same way, `prepare` lists work the course's own code does before the clock starts, such as digits3 training its judge, so a check's time limit only ever measures the learner's code. A test that runs only reference code passes `when_late`, so running out of time on a busy machine says so instead of pointing at the learner's loop.
- A wrong tensor with more than `SHOWN_ENTRIES` entries is described by the entries that differ, so a big tensor never prints whole.
- After a pass the reference solution is written to `solutions/<chapter>/<name>.py`.

## The app

- `run.py` draws a full-screen app with Textual, per [DEC-004](../../decisions/DEC-004-the-app-is-built-on-textual.md). See `TensorlingsApp` and `WatchScreen` in `tensorlings/app.py`.
- From the top the screen shows one labeled line with the exercise file and the progress, three tabs, Lesson, Hint and Result, and Textual's footer of keys.
- Each tab gets the whole screen and scrolls with the arrow keys and the mouse wheel. `1`, `2` and `3` switch tabs and `tab` cycles them. The owner picked tabs over stacked boxes, which left each box too small. The Result tab's label shows how many rows pass, so the state shows from any tab.
- After each save the app switches to the Result tab by itself, as rustlings shows output on save. Opening an exercise shows the lesson.
- On a narrow terminal the progress bar shrinks, then goes, so the file path stays whole. See `status_line` in `tensorlings/views.py`.
- The owner tried several tops before this one. Two bare lines felt cramped, a dot between the parts looked bad, and spaced words with no labels meant nothing. So every piece on the screen has a label or sits in a titled box.
- The result is an aligned table with one row per thing the checks are about, a name or a function, saying what a passing name holds, passes, not written yet, or how many of its checks fail. Under it, each failing row's first failing check is explained the way a compiler shows its first error. It grows with the TODOs and not with the checks, and passing rows stay in view, since the owner could not find a function that already worked when only failures were listed. Showing only the first problem, as rustlings does, was tried and dropped as confusing. See `result_view` in `tensorlings/views.py`.
- The Hint tab shows the hint for each function that doesn't pass yet. `2` or `h` opens it, and nothing is locked.
- What the learner's code prints is caught and shown in a magenta box under the results, since printing straight to the screen would break the drawing.
- Common PyTorch errors get a plain words explanation above PyTorch's own message, from the table `PLAIN_ERRORS` in `tensorlings/engine.py`. A new error that confuses learners gets a row there.
- Checks run in a thread worker so the screen never freezes, one at a time under `CHECK_LOCK`, and only the latest check of a burst of saves is shown. The app polls the file's time twice a second.
- The list screen works like rustlings, with DONE and PENDING, the current exercise marked, and keys to continue at, reset, search and filter. Resetting the open file asks first in a small box.
- The app keeps the mouse, so tabs and list rows are clickable, per [DEC-009](../../decisions/DEC-009-the-app-keeps-the-mouse-and-option-drag-copies.md). Cmd+C never reaches a terminal app on a Mac, since VS Code and Terminal keep it, and it only copies the terminal's own selection. So the learner holds Option while dragging, which `.vscode/settings.json` allows through `terminal.integrated.macOptionClickForcesSelection`, then presses cmd+c. In macOS Terminal it is Fn. A plain drag makes Textual's own selection, which ctrl+c copies through `pbcopy`.
- A new term in a lesson is bold in the Markdown and shows in blue with an underline, since the owner found plain bold barely visible. See `#lesson-text` in `APP_CSS` in `tensorlings/app.py`.
- The tab bar never takes focus. A focused tab bar paints the active tab as a solid block, which flashed on every click.
- Commands are `init`, `run`, `hint`, `reset`, `check-all`, `data`, `--manual-run` and `dev check`.
- New chapters show up in the list as soon as they are added. Running `run.py init` again writes the files a learner's `exercises/` folder is missing and leaves every file they have alone.

## Writing rules

From [the research](tensorlings-research.md#research-on-writing-that-sounds-human).

- Write for someone who can program and knows no deep learning. Skip what a programmer knows, like tuples, ints and floats, and explain every deep learning term in plain words where it first appears.
- Write the way a clear textbook explains things to a friend who programs, in full natural sentences that make sense on the first read. Connecting words like because, so and which means show how one fact leads to the next.
- Use the standard word for a thing, like derivative, standard deviation, increases or changes, and explain it where it first appears. Never swap it for a made up phrase. In November 2026 the owner could not read the TODO "Return how far f(x) moves when x grows by 0.001, divided by 0.001". It now says to compute `f(x + 0.001) - f(x)`, the change in the output when x increases by 0.001, and divide it by 0.001.
- When a formula or a line of code says something most clearly, write it, like `(f(x + h) - f(x)) / h`, and say in words what it means.
- A formula in a lesson's prose, a TODO, a hint or a check message carries brackets wherever its grouping could be misread. That means around a power that sits next to another operator, like `2 * (sigma ** 2)` or `-(distance ** 2) / temperature`, and around the first division or `@` in a chain, like `((a - 1) / rate) * b` or `(q @ k.T) / sqrt(d)`. The owner got stuck on `5 * x ** 4 - 2 * x + 7` with no brackets. math1 and derivatives2 state Python's order of operations, and derivatives2 keeps its plain forms because it teaches that order. Real Python in code blocks and given code stays as Python writes it.
- No story examples and no comparisons. The owner rejected shops, apples and "picture a grid". Examples are plain tensors, tasks are plain operations like making a grid or dropping a column, and a short paragraph says where models use the idea.
- Show code and its output first, then give the name. One new idea per paragraph.
- No bold labels on paragraphs, no clauses bolted onto the end of a sentence, no tidy closing summary.
- Every term explained where it first appears, never relying on a later lesson.
- Give the why as a case the learner will meet, inside the lesson's flow.
- Sentences mostly under 20 words, but never cut so short that they lose the words that make them clear. Read it aloud, and rewrite what you would not say to a friend.

## Tasks

From [the research](tensorlings-research.md#research-on-tasks-that-make-the-learner-think) and [DEC-013](../../decisions/DEC-013-exercises-explain-then-try-fix-an-error-or-try-first.md).

- Each exercise teaches one topic of its section, as rustlings does.
- Explain then try. A short working example, a plain line with its result in a comment, sits right above each TODO, and the TODO repeats that move or changes one thing. A task never asks for several new moves at once. See `MATH1` in `tensorlings/chapters/tensors.py`.
- A TODO that computes something from data is a function, so the checker can call it on inputs the file never shows and a typed in answer fails. It is empty when the move is one line, and partly written with one or two lines left as `...` when it has steps, as rustlings leaves a line to finish or fix. A TODO is a variable only when it builds one specific value, as in tensors1. See `check_on_inputs` in `tensorlings/engine.py`.
- Fix the error. A line crashes on purpose where the error teaches the idea, such as a list of lists with uneven rows.
- Try first. A topic opens with a task the learner attempts with plain Python before reading the lesson, and the lesson then shows the tensor way. Nothing is locked, since the owner did not want the lesson unlocked by a save. See `INDEXING1` in `tensorlings/chapters/tensors.py`.
- Phase 1a has one exercise per topic, tensors1, indexing1, math1, broadcasting1, reductions1 and matmul1, and the owner approved the format on tensors1 and indexing1. indexing1, reductions1 and matmul1 open with a try first TODO. From phase 1b on, a topic gets as many exercises as it needs, as [the detailed curriculum](tensorlings-detailed-curriculum.md) lists them.
- A quiz closes a group of chapters, as in rustlings. It has no examples and no new tools, and each of its functions combines two or three earlier moves, as a rustlings quiz combines earlier concepts. Some are empty and some are partly written. `quiz1` closes the tensor and matrix multiplication chapters.
- The lesson teaches every tool its tasks need, on its own small example. A task never needs a function or argument that no lesson showed.
- No line of a solution appears in the lesson or the hints as the task writes it.
- The file gives the scaffolding and leaves the tensor work as `TODO`s, as rustlings does.
- The TODO says the operation in plain programming words and standard terms, like "convert the list of lists into a tensor of floats", with the formula when that is clearest, and shows an example call with its result. A deep learning word in a TODO is explained right there, and the lint fails a starting file that uses one no lesson or TODO has explained. A TODO carries no why section, since the owner found one on every task to be noise.
- The tests include the case that catches the likely mistake, such as a square batch where rows and columns mix up.

## Verification

- `run.py dev check` runs every exercise twice. The starting file must fail and the solution must pass.
- A lesson lint in the dev check flags sentences over 25 words, filler words, dashes and semicolons, a paragraph that opens with a bold label, a lesson taller than 80 lines at width 96, any term from `JARGON` in `tensorlings/lint.py` used before an exercise lists it in `introduces`, and any solution body that shows up word for word in the lesson or the hints. A new technical term gets added to `JARGON` the first time a lesson needs it.
- A separate reviewer agent reads every lesson as a complete beginner before the owner sees it.
- `dev/wrong_answers.py` runs common wrong answers through the checks. Every one must fail with a message a beginner can act on, and the script exits with 1 if one passes. New exercises add their likely mistakes to its `CASES`.
- `dev/screens.py` drives the app headless with Textual's Pilot on a scratch copy of the course and saves SVG screenshots, which `qlmanage` turns into PNGs to look at. It runs before each hand over, at a normal size and at 80 by 24. Its docstring has the command.

## Limits on this machine

CUDA kernels, Triton and multi GPU training cannot run on the owner's Mac. Those lessons teach the idea in plain PyTorch on the CPU or the Mac's GPU and say where real hardware is needed.
