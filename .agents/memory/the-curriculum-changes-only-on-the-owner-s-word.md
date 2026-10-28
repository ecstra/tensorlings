---
name: the-curriculum-changes-only-on-the-owner-s-word
kind: memory
description: the 13 section curriculum was agreed on 2026-10-27 and an agent never changes a section, its topics, its project or the order without asking
updated: 2026-10-28
links: []
type: feedback
---

The 13 section curriculum in the [course design](../docs/research/tensorlings-course-design.md) is the one the owner agreed to on 2026-10-27, when they answered it with "Phase 1 is fine but take like 5 exercises", and confirmed on 2026-10-28 with "this 13 is what we agreed to, correct". The owner calls them sections, not parts. Before that they rejected a 6 section plan of about 160 exercises as biased toward a pasted chat, which [DEC-001](../decisions/DEC-001-curriculum-from-a-survey-of-expert-courses.md) records. Their very first ask, in the AIlings days, was "use a toy problem to teach me AI like rustlings do".

This matters because commit 8b18d20 rewrote Section 1 so the first six exercises build Karpathy's makemore bigram model, and changed Section 3's project to match, taking the reply "you remember the original goal I have right?" as approval. The owner said "We never agreed on changing the high level curriculum", and commit 01cbd5e put the table back.

To apply it, put any change to the curriculum table, a section's topics, a section's project or the order of sections to the owner as a question first, and wait for a clear yes. A teaching choice that changes what a section covers, such as a running example that pulls in another section's topics, counts as a curriculum change. A vague or annoyed reply is not a yes.
