---
name: DEC-013-exercises-explain-then-try-fix-an-error-or-try-first
kind: decision
description: each task repeats the move its example just showed, fixes a line that crashes on purpose, or is tried before the lesson opens, checked by name with no print lines
updated: 2026-10-28
links: [decisions/DEC-012-one-topic-per-exercise-with-real-model-operations-and-quizzes.md]
date: 2026-10-28
area: course
status: standing
---

## Decision

Exercises use three methods the owner picked from the research. Explain then try puts a short working example right above each TODO, and the TODO repeats that move on new numbers or changes one thing. Fix the error gives a line that crashes on purpose, and the plain words error in the result is the lesson. Try first opens a topic with a task the learner attempts with plain Python before reading the lesson, which stays readable throughout. The file has no print lines and no tests. Examples are plain lines with their result in a comment. A TODO that computes something from data is a function, empty when the move is one line and partly written with one or two lines left as `...` when it has steps, and the checker calls it on inputs the file never shows. A TODO is a variable only when it builds one specific value, as in tensors1. The checker runs the file one top level statement at a time, and the result lists every TODO with its value or its first problem. From DEC-012 three rules stay. Each exercise teaches one topic, a quiz combines earlier tools with nothing new, and a lesson teaches every tool its tasks need.

## Alternatives

DEC-012 made every task a real model operation written as a function from a spec, like splitting off a label column. The owner called it leetcode, since each task combined several moves no example had shown together. Copying rustlings' visible test module was rejected as copying the form. Predict, run, investigate, change and make, Parsons problems and building along with a real model were weighed and set aside. Locking a try first lesson until the first saved change was built and dropped, since the owner did not like unlocking the lesson. Showing only the first problem, as rustlings shows its first error, was built and dropped too, since the owner found one TODO at a time confusing and wanted every TODO shown together. Checking every TODO as a variable was dropped once the owner typed an answer in and passed. Making every example and TODO a function was dropped too, since files became lists of tiny functions and none was partly written.

## Consequences

tensors1 and indexing1 use the methods as a trial, and the other exercises keep the function style until the owner's verdict. A crash in one statement fails only the names that statement sets, and a crash no check reports fails the file. Hints are keyed by the name or function they help with. The owner asked for this on 2026-10-28.
