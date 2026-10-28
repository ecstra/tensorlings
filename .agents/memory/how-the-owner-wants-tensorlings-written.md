---
name: how-the-owner-wants-tensorlings-written
kind: memory
description: plain English with every term explained, code exercises only, nothing that assumes forgotten basics
updated: 2026-10-28
links: []
type: feedback
---

The owner wants every lesson in plain, natural English, with each term explained where it first appears, a concrete example with real numbers before any general idea, and nothing that assumes they remember deep learning basics. They rejected lessons that read as confusing or as filler, prediction only exercises, and lessons hidden in README files. In the second round they said the lessons "sound like AI" and that a task "just gave the answer and asked me to type it". They also want the screen to label every piece, never a dot between parts, and they cannot make the window taller, so everything must scroll. Later they rejected story examples like a shop selling apples and pears, and vague comparisons like "picture a grid", and said to assume the learner knows programming. In the third round they said lessons overreached, since a task needed `.numel()` that the lesson never taught, and that tasks felt like leetcode. They asked for "proper rustling or zero to hero style", with every tool a task needs taught first. Reading that as one running model through Section 1 went past what they asked, per [the curriculum memory](the-curriculum-changes-only-on-the-owner-s-word.md). In the fourth round they could not follow a TODO that said "Turn a list of examples into a batch", and asked why it couldn't just say "convert the given list to tensor", and why the task was there at all. So each TODO says the operation in plain programming words and shows an example call with its result. A why on every TODO was too much, they asked "Just... why?" of a line about GPT-2's size, so a TODO carries no why section. They then said to keep the real deep learning words, like batch, bias, token or padding, and explain each one in the TODO where it first appears, never assuming the learner knows the terminology. Swapping the real words for made up plain ones, like filler for padding, is not what they want.

This matters because two versions of the course failed on exactly these points, and the owner wants to go through the whole course without asking for rewrites.

To apply it, hold every lesson to the writing rules and the task rules in the course design doc, run the lesson lint and a beginner reader review before anything reaches the owner, and keep every exercise as code.
