---
name: DEC-016-training-picks-the-best-device-and-exercises-stay-on-the-cpu
kind: decision
description: projects that train for real run on CUDA, MPS or the CPU, whichever the machine has, while short exercises and every check run on the CPU
updated: 2026-10-29
links: []
date: 2026-10-29
area: course
status: standing
---

## Decision

Every exercise that trains long enough to gain from a GPU starts with one line that picks the best device the machine has, CUDA, MPS or another accelerator, and the CPU when there is none.

```python
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
```

Short exercises and every check run on the CPU. The lesson that first trains for real teaches the line, with the measurements below. The owner asked on 2026-10-29 that PyTorch use "mps, cuda or cpu ... whatever is the best automatically on the running device".

## Alternatives

Moving everything to the accelerator. On the owner's Mac with PyTorch 2.14.1, a 512 by 784 times 784 by 10 multiply ran 200 times in 0.016 seconds on the CPU and 0.67 seconds on MPS, while a 4096 by 4096 square ran 5 times in 2.05 seconds on the CPU and 0.39 on MPS. Small work is about 40 times slower on MPS, and MPS cannot hold float64, which the gradient checks use. Writing the CUDA, MPS and CPU tests by hand was the other option, but `torch.accelerator` already covers every backend.

## Consequences

A check never assumes a device and moves what it compares to the CPU. A model trained on MPS loads on the CPU for its check through `map_location`. Code in the exercises never asks for float64 on the training device.
