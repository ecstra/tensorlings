---
name: how-the-owner-wants-tensorlings-written
kind: memory
description: plain English with every term explained, code exercises only, nothing that assumes forgotten basics
updated: 2026-10-28
links: []
type: feedback
---

The owner wants every lesson in plain, natural English, with each term explained where it first appears, a concrete example with real numbers before any general idea, and nothing that assumes they remember deep learning basics. They rejected lessons that read as confusing or as filler, prediction only exercises, and lessons hidden in README files. In the second round they said the lessons "sound like AI" and that a task "just gave the answer and asked me to type it". They also want the screen to label every piece, never a dot between parts, and they cannot make the window taller, so everything must scroll. Later they rejected story examples like a shop selling apples and pears, and vague comparisons like "picture a grid", and said to assume the learner knows programming. Examples use the data deep learning code handles, such as a batch of examples by features, token ids, biases and weights, stated directly. In the third round they said lessons overreached, since a task needed `.numel()` that the lesson never taught, and that tasks felt like leetcode. They asked for "proper rustling or zero to hero style", with every tool a task needs taught first. Reading that as one running model through Section 1 went past what they asked, per [the curriculum memory](the-curriculum-changes-only-on-the-owner-s-word.md).

This matters because two versions of the course failed on exactly these points, and the owner wants to go through the whole course without asking for rewrites.

To apply it, hold every lesson to the writing rules and the task rules in the course design doc, run the lesson lint and a beginner reader review before anything reaches the owner, and keep every exercise as code.
