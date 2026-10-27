# AGENTS.md

<!-- The kit's words start here. Section 0 goes above this line. -->

The rules for working in this repo are in @.agents/SYSTEM.md and every
agent works by them. Claude Code and Gemini CLI pull that file into
context on their own. Codex gets it from its session start hook. Anything
else that reads this file as plain text opens it first.

Rules that belong to this project alone go in a `## 0. Project Rules`
section directly under the title above, ahead of the comment line.
Everything from that line on is the kit's, and the upgrade keeps what
sits above it.
