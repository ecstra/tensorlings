"""The hook blocks a brain file written by hand, and reports after a write."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import hooks
from brainlib import autoack, doctor, index, plans, schema, steps, task, writes
from tests import fixtures


class HookCase(fixtures.BrainCase):
    def setUp(self) -> None:
        super().setUp()
        self.project = self.root.parent
        index.write(brain=self.brain, stamp="2026-09-16")
        doctor.run(brain=self.brain, fix=True)

    def run_hook(
        self,
        event: str,
        payload: dict,
        agent: str = "claude",
        with_rules: bool = False,
    ) -> tuple[int, str, str]:
        payload = {"cwd": str(self.project), **payload}
        out, err = io.StringIO(), io.StringIO()
        argv = [event, "--agent", agent] + (["--with-rules"] if with_rules else [])

        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with mock.patch("sys.stdin", io.StringIO(json.dumps(payload))):
                code = hooks.main(argv)

        return code, out.getvalue(), err.getvalue()


class SessionStart(HookCase):
    def test_claude_gets_the_status_as_added_context(self) -> None:
        code, out, _ = self.run_hook(event="session-start", payload={})

        self.assertEqual(code, 0)
        payload: dict = json.loads(out)
        self.assertIn("kit 1.0.0", payload["hookSpecificOutput"]["additionalContext"])

    def test_codex_gets_plain_text_with_the_rules_and_the_chat_style(self) -> None:
        fixtures.write(path=self.project / ".codex" / "plain-and-short.md", text="# Plain & Short\n\nAnswer first.\n")

        code, out, _ = self.run_hook(event="session-start", payload={}, agent="codex", with_rules=True)

        self.assertEqual(code, 0)
        self.assertIn("kit 1.0.0", out)
        self.assertIn("# System Rules", out)
        self.assertIn("from .codex/plain-and-short.md:\n\n# Plain & Short", out)

    def test_codex_plain_text_goes_out_as_utf8_whatever_the_locale_says(self) -> None:
        fixtures.write(path=self.root / "SYSTEM.md", text="# System Rules\n\n## 1. Coding\n\n- \"Add validation\" → tests first\n")
        raw = json.dumps({"cwd": str(self.project)}).encode("utf-8")
        sink = io.BytesIO()
        out = io.TextIOWrapper(sink, encoding="cp1252")

        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            with mock.patch("sys.stdin", io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8")):
                code = hooks.main(["session-start", "--agent", "codex", "--with-rules"])
            out.flush()

        self.assertEqual(code, 0)
        self.assertIn("→ tests first", sink.getvalue().decode("utf-8"))

    def test_rules_the_process_cannot_open_are_said_so_and_the_rest_arrives(self) -> None:
        (self.root / "SYSTEM.md").chmod(0)

        code, out, _ = self.run_hook(event="session-start", payload={}, agent="codex", with_rules=True)

        self.assertEqual(code, 0)
        self.assertIn("kit 1.0.0", out)
        self.assertIn("(.agents/SYSTEM.md cannot be read: Permission denied)", out)

    def test_rules_that_are_not_utf8_still_reach_the_agent(self) -> None:
        (self.root / "SYSTEM.md").write_bytes(b"# System Rules\n\ncaf\xe9\n")

        code, out, _ = self.run_hook(event="session-start", payload={}, agent="codex", with_rules=True)

        self.assertEqual(code, 0)
        self.assertIn("# System Rules", out)

    def test_outside_a_project_it_stays_quiet(self) -> None:
        with tempfile.TemporaryDirectory() as elsewhere:
            code, out, _ = self.run_hook(event="session-start", payload={"cwd": elsewhere})

        self.assertEqual((code, out), (0, ""))

    def test_the_session_opens_with_the_clock_and_the_standing_word(self) -> None:
        code, out, _ = self.run_hook(event="session-start", payload={})

        context: str = json.loads(out)["hookSpecificOutput"]["additionalContext"]
        self.assertEqual(code, 0)
        self.assertTrue(context.startswith("now: "))
        self.assertIn("\nauto ack off\ntask none\n\nThe brain, counted from the files right now:", context)

    def test_the_session_start_and_the_prompt_carry_the_word_and_the_work_while_they_hold(self) -> None:
        autoack.turn_on(brain=self.brain, hours=2)
        task.turn_on(brain=self.brain, key="TODO-001")
        standing = r"auto ack on, [^\n]+ left, until [^\n]+\ntask TODO-001"

        code, out, _ = self.run_hook(event="session-start", payload={})
        context: str = json.loads(out)["hookSpecificOutput"]["additionalContext"]
        self.assertEqual(code, 0)
        self.assertRegex(context, r"^now: [^\n]+\n" + standing + r"\n\nThe brain, counted")

        code, out, _ = self.run_hook(event="prompt", payload={})
        context = json.loads(out)["hookSpecificOutput"]["additionalContext"]
        self.assertEqual(code, 0)
        self.assertRegex(context, r"^now: [^\n]+\n" + standing + r"$")

        code, out, _ = self.run_hook(event="prompt", payload={}, agent="codex")
        self.assertEqual(code, 0)
        self.assertRegex(out, r"^now: [^\n]+\n" + standing + r"\n$")


class PromptAndTick(HookCase):
    def test_each_prompt_carries_the_clock_and_the_word_for_every_agent(self) -> None:
        for agent, event_name in (("claude", "UserPromptSubmit"), ("gemini", "BeforeAgent")):
            with self.subTest(agent=agent):
                code, out, _ = self.run_hook(event="prompt", payload={}, agent=agent)

                payload: dict = json.loads(out)["hookSpecificOutput"]
                self.assertEqual((code, payload["hookEventName"]), (0, event_name))
                self.assertRegex(payload["additionalContext"], r"^now: \w+ \d{4}-\d{2}-\d{2} \d{2}:\d{2} .+\nauto ack off\ntask none$")

        code, out, _ = self.run_hook(event="prompt", payload={}, agent="codex")

        self.assertEqual(code, 0)
        self.assertRegex(out, r"^now: .+\nauto ack off\ntask none\n$")

    def test_the_tick_carries_the_clock_the_word_and_the_work_on_or_off(self) -> None:
        for agent, event_name in (("claude", "PostToolBatch"), ("codex", "PostToolUse"), ("gemini", "AfterTool")):
            with self.subTest(agent=agent):
                code, out, _ = self.run_hook(event="tick", payload={}, agent=agent)

                payload: dict = json.loads(out)["hookSpecificOutput"]
                self.assertEqual((code, payload["hookEventName"]), (0, event_name))
                self.assertRegex(payload["additionalContext"], r"^now: [^\n]+\nauto ack off\ntask none$")

        autoack.turn_on(brain=self.brain, hours=2)
        plans.create(brain=self.brain, title="Phase 7", description="the rest", body="## Why\n\nWhy.\n\n## Steps\n\n1. TODO-001, the export.\n\n## Done when\n\nDone.\n")
        task.turn_on(brain=self.brain, key="phase-7")
        task.turn_on(brain=self.brain, key="TODO-001")

        code, out, _ = self.run_hook(event="tick", payload={})

        context: str = json.loads(out)["hookSpecificOutput"]["additionalContext"]
        self.assertEqual(code, 0)
        self.assertRegex(context, r"^now: [^\n]+\nauto ack on, [^\n]+ left, until [^\n]+\nplan phase-7, step 1 of 1, the export\.\ntask TODO-001$")

        writes.update(brain=self.brain, key="TODO-001", changes={"status": "done", "done": "2026-09-24"})

        code, out, _ = self.run_hook(event="tick", payload={})

        context = json.loads(out)["hookSpecificOutput"]["additionalContext"]
        self.assertEqual(code, 0)
        self.assertRegex(context, r"\nplan phase-7, every step settled, close it with brain set phase-7 status=done done=\d{4}-\d{2}-\d{2}$")
        self.assertEqual(len(json.loads((self.root / task.FILE).read_text(encoding="utf-8"))["chain"]), 2)

    def test_the_event_is_read_as_utf8_whatever_the_locale_says(self) -> None:
        raw = json.dumps({"cwd": str(self.project), "prompt": "Á hola"}, ensure_ascii=False).encode("utf-8")
        out = io.StringIO()

        with contextlib.redirect_stdout(out):
            with mock.patch("sys.stdin", io.TextIOWrapper(io.BytesIO(raw), encoding="cp1252")):
                code = hooks.main(["prompt", "--agent", "claude"])

        self.assertEqual(code, 0)
        self.assertRegex(json.loads(out.getvalue())["hookSpecificOutput"]["additionalContext"], r"^now: .+\nauto ack off\ntask none$")


class PreWrite(HookCase):
    def test_a_new_brain_file_by_hand_is_blocked(self) -> None:
        target = self.root / "bugs" / "BUG-009-by-hand.md"

        code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"file_path": str(target)}})

        self.assertEqual(code, 2)
        self.assertIn("brain new", err)

    def test_an_edit_to_an_existing_file_passes(self) -> None:
        target = self.root / "bugs" / "BUG-002-login-loop.md"

        code, _, _ = self.run_hook(event="pre-write", payload={"tool_input": {"file_path": str(target)}})

        self.assertEqual(code, 0)

    def test_the_owners_word_is_never_written_by_hand(self) -> None:
        word = '{"since": "2026-09-23T12:00+05:30", "until": "2999-12-31T00:00+00:00", "task": null}\n'

        for tool_input in (
            {"file_path": str(self.root / autoack.FILE), "content": word},
            {"file_path": str(self.root / autoack.FILE), "old_string": "2026", "new_string": "2999"},
        ):
            with self.subTest(tool_input=tool_input):
                code, _, err = self.run_hook(event="pre-write", payload={"tool_input": tool_input})

                self.assertEqual(code, 2)
                self.assertIn(".ack is the owner's standing word, set with brain ack on and never written by hand", err)

        self.assertFalse((self.root / autoack.FILE).exists())

        adding = "*** Begin Patch\n*** Add File: .agents/.ack\n+" + word + "*** End Patch\n"
        deleting = "*** Begin Patch\n*** Delete File: .agents/.ack\n*** End Patch\n"
        self.assertEqual(self.run_hook(event="pre-write", payload={"tool_input": {"patch": adding}}, agent="codex")[0], 2)
        self.assertEqual(self.run_hook(event="pre-write", payload={"tool_input": {"patch": deleting}}, agent="codex")[0], 0)

        code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"file_path": str(self.root / ".ACK"), "content": word}})
        self.assertEqual(code, 2)
        self.assertIn(".ack is the owner's standing word", err)

    def test_the_current_work_is_never_written_by_hand(self) -> None:
        chain = '{"chain": [{"kind": "todo", "key": "TODO-001", "step": null}], "since": "2026-09-24T09:00+05:30"}\n'

        for tool_input in (
            {"file_path": str(self.root / task.FILE), "content": chain},
            {"file_path": str(self.root / ".TASK"), "content": chain},
            {"file_path": str(self.root / task.FILE), "old_string": "TODO-001", "new_string": "BUG-002"},
        ):
            with self.subTest(tool_input=tool_input):
                code, _, err = self.run_hook(event="pre-write", payload={"tool_input": tool_input})

                self.assertEqual(code, 2)
                self.assertIn(".task is the current work, set with brain task on and never written by hand", err)

        self.assertFalse((self.root / task.FILE).exists())

        deleting = "*** Begin Patch\n*** Delete File: .agents/.task\n*** End Patch\n"
        self.assertEqual(self.run_hook(event="pre-write", payload={"tool_input": {"patch": deleting}}, agent="codex")[0], 0)

    def test_the_suffix_is_read_the_way_the_store_reads_it(self) -> None:
        loud = self.root / "bugs" / "BUG-009-by-hand.MD"
        payload = {"tool_input": {"file_path": str(loud), "content": "---\nname: x\n---\n"}}

        self.assertEqual(self.run_hook(event="pre-write", payload=payload)[0], 0)

        with mock.patch.object(os.path, "normcase", str.lower):
            code, _, err = self.run_hook(event="pre-write", payload=payload)

        self.assertEqual(code, 2)
        self.assertIn("brain new", err)


class TheStepsTableIsTheTools(HookCase):
    def setUp(self) -> None:
        super().setUp()
        self.plan = plans.create(brain=self.brain, title="Kit", description="kit", body="## Why\n\nWhy.\n\n## Steps\n\n1. TODO-001, the export.\n2. The docs.\n\n## Done when\n\nDone.\n", stamp="2026-09-16")
        self.text = self.plan.path.read_text(encoding="utf-8")

    def pre_write(
        self,
        **tool_input: str,
    ) -> tuple[int, str, str]:
        return self.run_hook(event="pre-write", payload={"tool_input": tool_input})

    def test_a_row_marked_reworded_or_deleted_by_hand_is_blocked(self) -> None:
        for old, new, why in (
            ("| 2 | The docs. | | owner | pending |", "| 2 | The docs. | | owner | done |", "step 2 is marked done with brain task done or brain steps kit step 2 done"),
            ("| 2 | The docs. | | owner | pending |", "| 2 | The docs. | | owner | skipped |", "step 2 is marked skipped with brain task skip or brain steps kit step 2 skipped"),
            ("| 1 | the export. | TODO-001 | owner | pending |", "| 1 | the export. | TODO-001 | owner | done |", "step 1 points at TODO-001 and takes its mark from that entry"),
            ("| 2 | The docs. | | owner | pending |", "| 2 | The docs, all of them. | | owner | pending |", "step 2 is worded as it was filed"),
            ("| 2 | The docs. | | owner | pending |\n", "", "step 2 is never deleted, a step that turned out wrong is skipped"),
            ("| 1 | the export. | TODO-001 | owner | pending |", "| 1 | the export. | | owner | pending |", "step 1 points where it was filed to point"),
            ("| 2 | The docs. | | owner | pending |", "| 2 | The docs. | | agent | pending |", "step 2 is set by whom it was filed by"),
        ):
            with self.subTest(new=new):
                code, _, err = self.pre_write(file_path=str(self.plan.path), old_string=old, new_string=new)

                self.assertEqual(code, 2)
                self.assertIn(f"plans/kit.md: the steps table is the tool's, {why}, and the prose around it is yours to edit", err)

    def test_the_hint_for_a_changed_mark_follows_the_closed_table_first_and_a_dead_pointer_is_marked_by_hand(self) -> None:
        (self.root / "todos" / "TODO-001-export-to-csv.md").unlink()
        self.brain.refresh()
        code, _, err = self.pre_write(file_path=str(self.plan.path), old_string="| 1 | the export. | TODO-001 | owner | pending |", new_string="| 1 | the export. | TODO-001 | owner | done |")

        self.assertEqual(code, 2)
        self.assertIn("step 1 is marked done with brain task done or brain steps kit step 1 done", err)

        steps.mark(brain=self.brain, key="kit", number=2, status="done")
        code, _, err = self.pre_write(file_path=str(self.plan.path), old_string="| 2 | The docs. | | owner | done |", new_string="| 2 | The docs. | | owner | pending |")

        self.assertEqual(code, 2)
        self.assertIn("step 2 is marked pending with brain task step 2 or brain steps kit step 2 pending", err)

        fixtures.write(path=self.plan.path, text=self.text.replace("status: open", "status: done\ndone: 2026-09-16").replace("| owner | pending |", "| owner | done |"))
        self.brain.refresh()
        code, _, err = self.pre_write(file_path=str(self.plan.path), old_string="| 1 | the export. | TODO-001 | owner | done |", new_string="| 1 | the export. | TODO-001 | owner | pending |")

        self.assertEqual(code, 2)
        self.assertIn("plans/kit.md: the steps table is the tool's, kit is done, and a closed table is its record, and the prose around it is yours to edit", err)

    def test_a_row_appended_by_hand_is_held_to_what_add_would_hold_it_to(self) -> None:
        for row, why in (
            ("| 3 | The readme. | | owner | done |", "step 3 is added pending, and marked once it is done with brain task done or brain steps kit step 3 done"),
            ("| 3 | The readme. | | owner | skipped |", "step 3 is added pending, and marked once it is skipped with brain task skip or brain steps kit step 3 skipped"),
            ("| 3 | the loop. | BUG-002 | owner | done |", "step 3 is added pending, and takes its mark from BUG-002"),
            ("| 3 | The readme. | TODO-099 | owner | pending |", "step 3 points at 'TODO-099', which is not a todo, a bug or a review the brain knows"),
            ("| 3 | TODO-003, the theme. | | owner | pending |", "step 3 opens with TODO-003 in its Step cell, and a row points through its Points at cell"),
            ("| 3 | TODO-099 the theme. | | owner | pending |", "step 3 points at 'TODO-099', which is not a todo, a bug or a review the brain knows"),
        ):
            with self.subTest(row=row):
                code, _, err = self.pre_write(file_path=str(self.plan.path), old_string="| 2 | The docs. | | owner | pending |\n", new_string=f"| 2 | The docs. | | owner | pending |\n{row}\n")

                self.assertEqual(code, 2)
                self.assertIn(f"plans/kit.md: the steps table is the tool's, {why}, and the prose around it is yours to edit", err)

        writes.update(brain=self.brain, key="TODO-001", changes={"status": "done", "done": "2026-09-16"}, stamp="2026-09-16")
        steps.mark(brain=self.brain, key="kit", number=2, status="done")
        writes.update(brain=self.brain, key="kit", changes={"status": "done", "done": "2026-09-16"}, stamp="2026-09-16")
        code, _, err = self.pre_write(file_path=str(self.plan.path), old_string="| 2 | The docs. | | owner | done |\n", new_string="| 2 | The docs. | | owner | done |\n| 3 | Shipped it too. | | owner | pending |\n")

        self.assertEqual(code, 2)
        self.assertIn("plans/kit.md: the steps table is the tool's, kit is done, and a closed table is its record, so no step is added to it, and the prose around it is yours to edit", err)

    def test_a_row_appended_by_hand_may_not_point_at_its_own_entry_or_close_a_loop(self) -> None:
        steps.add(brain=self.brain, key="TODO-001", lines=["Read the table."], by="owner")
        steps.add(brain=self.brain, key="BUG-002", lines=["TODO-001, the export first."], by="owner")
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"

        for row, why in (
            ("| 2 | Myself. | TODO-001 | owner | pending |", "step 2 points at TODO-001-export-to-csv itself, and a step points at other work"),
            ("| 2 | The loop. | BUG-002 | owner | pending |", "step 2 points at BUG-002, whose steps point back here, and the two would wait on each other for good"),
        ):
            with self.subTest(row=row):
                code, _, err = self.pre_write(file_path=str(todo), old_string="| 1 | Read the table. | | owner | pending |\n", new_string=f"| 1 | Read the table. | | owner | pending |\n{row}\n")

                self.assertEqual(code, 2)
                self.assertIn(f"todos/TODO-001-export-to-csv.md: the steps table is the tool's, {why}, and the prose around it is yours to edit", err)

    def test_one_patch_may_not_add_the_two_halves_of_a_loop_to_two_files(self) -> None:
        steps.add(brain=self.brain, key="TODO-001", lines=["Read the table."], by="owner")
        steps.add(brain=self.brain, key="BUG-002", lines=["Find it."], by="owner")
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"
        bug = self.root / "bugs" / "BUG-002-login-loop.md"
        patch = (
            f"*** Begin Patch\n*** Update File: {todo}\n@@\n-| 1 | Read the table. | | owner | pending |\n+| 1 | Read the table. | | owner | pending |\n+| 2 | Bug first. | BUG-002 | owner | pending |\n"
            f"*** Update File: {bug}\n@@\n-| 1 | Find it. | | owner | pending |\n+| 1 | Find it. | | owner | pending |\n+| 2 | Todo first. | TODO-001 | owner | pending |\n*** End Patch\n"
        )

        code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"patch": patch}}, agent="codex")

        self.assertEqual(code, 2)
        self.assertIn("todos/TODO-001-export-to-csv.md: the steps table is the tool's, step 2 points at BUG-002, whose steps lead back here once the patch lands, and the entries would wait on each other for good, and the prose around it is yours to edit", err)

    def test_one_patch_may_not_close_a_loop_through_a_third_file_or_one_already_on_disk(self) -> None:
        other = writes.create(brain=self.brain, kind=schema.kind_named("todo"), title="Another todo", description="another", values={"area": "reports", "by": "owner"}, body="## What\n\nWhat.\n\n## Done when\n\nDone.\n")
        steps.add(brain=self.brain, key="TODO-001", lines=["Read the table."], by="owner")
        steps.add(brain=self.brain, key="BUG-002", lines=["Find it."], by="owner")
        steps.add(brain=self.brain, key=other.identifier, lines=["Start."], by="owner")
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"
        bug = self.root / "bugs" / "BUG-002-login-loop.md"
        ring = (
            f"*** Begin Patch\n*** Update File: {bug}\n@@\n-| 1 | Find it. | | owner | pending |\n+| 1 | Find it. | | owner | pending |\n+| 2 | Todo first. | TODO-001 | owner | pending |\n"
            f"*** Update File: {todo}\n@@\n-| 1 | Read the table. | | owner | pending |\n+| 1 | Read the table. | | owner | pending |\n+| 2 | The other. | {other.identifier} | owner | pending |\n"
            f"*** Update File: {other.path}\n@@\n-| 1 | Start. | | owner | pending |\n+| 1 | Start. | | owner | pending |\n+| 2 | The bug. | BUG-002 | owner | pending |\n*** End Patch\n"
        )

        code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"patch": ring}}, agent="codex")

        self.assertEqual(code, 2)
        self.assertIn("whose steps lead back here once the patch lands", err)

        steps.add(brain=self.brain, key=other.identifier, lines=["BUG-002, the bug."], by="owner")
        chain = (
            f"*** Begin Patch\n*** Update File: {bug}\n@@\n-| 1 | Find it. | | owner | pending |\n+| 1 | Find it. | | owner | pending |\n+| 2 | Todo first. | TODO-001 | owner | pending |\n"
            f"*** Update File: {todo}\n@@\n-| 1 | Read the table. | | owner | pending |\n+| 1 | Read the table. | | owner | pending |\n+| 2 | The other. | {other.identifier} | owner | pending |\n*** End Patch\n"
        )

        code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"patch": chain}}, agent="codex")

        self.assertEqual(code, 2)
        self.assertIn("whose steps lead back here once the patch lands", err)

        writes.update(brain=self.brain, key=other.identifier, changes={"status": "dropped", "done": "2026-09-16"}, stamp="2026-09-16")
        code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"patch": chain}}, agent="codex")

        self.assertEqual((code, err), (0, ""))

    def test_a_file_the_hook_cannot_read_is_taken_as_its_repair_and_the_rest_of_the_patch_is_still_judged(self) -> None:
        doc = self.root / "docs" / "notes.md"
        doc.write_bytes(b"---\nname: notes\nkind: doc\ndescription: caf\xe9\nupdated: 2026-09-16\nlinks: []\n---\n\n# Notes\n")
        bug = self.root / "bugs" / "BUG-002-login-loop.md"

        self.assertEqual(self.pre_write(file_path=str(doc), old_string="Notes", new_string="Notes, clean")[0], 0)

        patch = f"*** Begin Patch\n*** Update File: {doc}\n@@\n-# Notes\n+# Notes, clean\n*** Update File: {bug}\n@@\n-status: open\n+status: fixed\n*** End Patch\n"
        code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"patch": patch}}, agent="codex")

        self.assertEqual(code, 2)
        self.assertIn("the header block changes through brain set", err)

    def test_the_files_own_keys_are_read_whatever_case_the_write_spells_the_path_in(self) -> None:
        self.assertEqual(steps.own_keys(hooks.entry_at(brain=self.brain, inside="todos/todo-001-export-to-csv.md")), {"TODO-001", "TODO-001-export-to-csv"})

        steps.add(brain=self.brain, key="TODO-001", lines=["Read the table."], by="owner")
        lower = self.root / "todos" / "todo-001-export-to-csv.md"
        if not lower.exists():
            self.skipTest("needs a disk that ignores case")

        code, _, err = self.pre_write(file_path=str(lower), old_string="| 1 | Read the table. | | owner | pending |\n", new_string="| 1 | Read the table. | | owner | pending |\n| 2 | Me. | TODO-001 | owner | pending |\n")

        self.assertEqual(code, 2)
        self.assertIn("points at TODO-001-export-to-csv itself", err)

        code, _, err = self.pre_write(file_path=str(lower), old_string="| 1 | Read the table. | | owner | pending |\n", new_string="| 1 | Read the table. | | owner | done |\n")

        self.assertEqual(code, 2)
        self.assertIn("brain steps TODO-001-export-to-csv step 1 done", err)

    def test_a_plan_whose_slug_starts_like_an_id_carries_no_id_in_the_hook_either(self) -> None:
        plan = plans.create(brain=self.brain, title="Todo 001 follow up", description="the follow up", body="## Why\n\nWhy.\n\n## Steps\n\n1. Read the table.\n\n## Done when\n\nDone.\n")
        self.brain.refresh()

        self.assertEqual(plan.relative, "plans/todo-001-follow-up.md")
        self.assertEqual(steps.own_keys(hooks.entry_at(brain=self.brain, inside=plan.relative)), {"todo-001-follow-up"})

        code, _, err = self.pre_write(file_path=str(plan.path), old_string="| 1 | Read the table. | | owner | pending |\n", new_string="| 1 | Read the table. | | owner | pending |\n| 2 | Finish the export. | TODO-001 | owner | pending |\n")

        self.assertEqual((code, err), (0, ""))

    def test_a_line_written_to_a_list_already_on_disk_is_held_to_what_filing_holds_it_to(self) -> None:
        steps.add(brain=self.brain, key="BUG-002", lines=["TODO-001, the export first."], by="owner")
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"
        fixtures.write(path=todo, text=todo.read_text(encoding="utf-8").replace("## Done when", "## Steps\n\n1. Read the table.\n\n## Done when"))
        self.brain.refresh()

        for line, why in (
            ("2. TODO-001, redo it.", "step 2 points at TODO-001 itself, and a step points at other work"),
            ("2. BUG-002, fix first.", "step 2 points at BUG-002, whose steps point back here, and the two would wait on each other for good"),
        ):
            with self.subTest(line=line):
                code, _, err = self.pre_write(file_path=str(todo), old_string="1. Read the table.\n", new_string=f"1. Read the table.\n{line}\n")

                self.assertEqual(code, 2)
                self.assertIn(f"todos/TODO-001-export-to-csv.md: the steps table is the tool's, {why}, and the prose around it is yours to edit", err)

        self.assertEqual(self.pre_write(file_path=str(todo), old_string="1. Read the table.\n", new_string="1. Read the table.\n2. TODO-003, the theme.\n")[0], 0)
        self.assertEqual(self.pre_write(file_path=str(todo), old_string="What.", new_string="What, at length.")[0], 0)

    def test_a_list_written_under_steps_on_a_closed_entry_is_refused_as_a_row_would_be(self) -> None:
        other = self.root / "todos" / "TODO-003-dark-mode.md"

        code, _, err = self.pre_write(file_path=str(other), old_string="## Done when", new_string="## Steps\n\n1. Did x.\n2. Did y.\n\n## Done when")

        self.assertEqual(code, 2)
        self.assertIn("todos/TODO-003-dark-mode.md: the steps table is the tool's, TODO-003-dark-mode is done, and a closed table is its record, so no step is added to it, and the prose around it is yours to edit", err)

        fixtures.write(path=other, text=other.read_text(encoding="utf-8").replace("## Done when", "## Steps\n\n1. Did x.\n\n## Done when"))
        self.brain.refresh()

        self.assertEqual(self.pre_write(file_path=str(other), old_string="## What\n\na dark theme", new_string="## What\n\na darker theme")[0:3:2], (0, ""))

    def test_a_patch_across_files_is_judged_while_another_entry_holds_a_table_that_does_not_read(self) -> None:
        fixtures.write(path=self.plan.path, text=self.plan.path.read_text(encoding="utf-8").replace("| 2 | The docs. | | owner | pending |", "| 2 | The docs. | | owner | finished |"))
        bug = self.root / "bugs" / "BUG-002-login-loop.md"
        fixtures.write(path=bug, text=bug.read_text(encoding="utf-8").replace("## Evidence", "## Steps\n\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n| 1 | the theme. | TODO-003 | agent | pending |\n\n## Evidence"))
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"
        other = self.root / "todos" / "TODO-003-dark-mode.md"
        fixtures.write(path=other, text=other.read_text(encoding="utf-8").replace("status: done", "status: open"))
        self.brain.refresh()

        patch = (
            f"*** Begin Patch\n*** Update File: {todo}\n@@\n-## Done when\n+## Steps\n+\n+1. BUG-002, the loop first.\n+\n+## Done when\n"
            f"*** Update File: {other}\n@@\n-## Done when\n+## Steps\n+\n+1. TODO-001, the export first.\n+\n+## Done when\n*** End Patch\n"
        )
        code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"patch": patch}}, agent="codex")

        self.assertEqual(code, 2)
        self.assertIn("whose steps lead back here once the patch lands", err)

        patch = (
            f"*** Begin Patch\n*** Update File: {todo}\n@@\n-## Done when\n+## Steps\n+\n+1. BUG-002, the loop first.\n+\n+## Done when\n"
            f"*** Update File: {other}\n@@\n-a dark theme\n+a darker theme\n*** End Patch\n"
        )

        self.assertEqual(self.run_hook(event="pre-write", payload={"tool_input": {"patch": patch}}, agent="codex")[0:3:2], (0, ""))

    def test_the_walk_across_a_patch_ends_on_a_loop_that_sits_outside_the_written_files(self) -> None:
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"
        other = self.root / "todos" / "TODO-003-dark-mode.md"
        fixtures.write(path=todo, text=todo.read_text(encoding="utf-8").replace("## Done when", "## Steps\n\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n| 1 | the theme. | TODO-003 | owner | pending |\n\n## Done when"))
        fixtures.write(path=other, text=other.read_text(encoding="utf-8").replace("status: done", "status: open").replace("## Done when", "## Steps\n\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n| 1 | the export. | TODO-001 | owner | pending |\n\n## Done when"))
        self.brain.refresh()
        bug = self.root / "bugs" / "BUG-002-login-loop.md"

        patch = (
            f"*** Begin Patch\n*** Update File: {self.plan.path}\n@@\n-| 2 | The docs. | | owner | pending |\n+| 2 | The docs. | | owner | pending |\n+| 3 | After the export. | TODO-001 | owner | pending |\n"
            f"*** Update File: {bug}\n@@\n-A log line.\n+Two log lines.\n*** End Patch\n"
        )
        code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"patch": patch}}, agent="codex")

        self.assertEqual((code, err), (0, ""))

    def test_a_list_that_leads_back_through_another_list_is_refused_in_one_patch_and_across_two_writes(self) -> None:
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"
        other = self.root / "todos" / "TODO-003-dark-mode.md"
        fixtures.write(path=other, text=other.read_text(encoding="utf-8").replace("status: done", "status: open"))
        self.brain.refresh()

        patch = (
            f"*** Begin Patch\n*** Update File: {todo}\n@@\n-## Done when\n+## Steps\n+\n+1. TODO-003, the theme first.\n+\n+## Done when\n"
            f"*** Update File: {other}\n@@\n-## Done when\n+## Steps\n+\n+1. TODO-001, the export first.\n+\n+## Done when\n*** End Patch\n"
        )
        code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"patch": patch}}, agent="codex")

        self.assertEqual(code, 2)
        self.assertIn("todos/TODO-001-export-to-csv.md: the steps table is the tool's, step 1 points at TODO-003, whose steps lead back here once the patch lands, and the entries would wait on each other for good, and the prose around it is yours to edit", err)

        self.assertEqual(self.pre_write(file_path=str(todo), old_string="## Done when", new_string="## Steps\n\n1. TODO-003, the theme first.\n\n## Done when")[0], 0)
        fixtures.write(path=todo, text=todo.read_text(encoding="utf-8").replace("## Done when", "## Steps\n\n1. TODO-003, the theme first.\n\n## Done when"))
        self.brain.refresh()

        code, _, err = self.pre_write(file_path=str(other), old_string="## Done when", new_string="## Steps\n\n1. TODO-001, the export first.\n\n## Done when")

        self.assertEqual(code, 2)
        self.assertIn("todos/TODO-003-dark-mode.md: the steps table is the tool's, step 1 points at TODO-001, whose steps point back here, and the two would wait on each other for good, and the prose around it is yours to edit", err)

    def test_a_list_written_under_a_new_steps_heading_is_held_to_what_filing_holds_it_to(self) -> None:
        steps.add(brain=self.brain, key="BUG-002", lines=["TODO-001, the export first."], by="owner")
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"

        for listing, why in (
            ("1. BUG-002, fix first.", "step 1 points at BUG-002, whose steps point back here, and the two would wait on each other for good"),
            ("1. TODO-099, the ghost.", "step 1 points at 'TODO-099', which is not a todo, a bug or a review the brain knows"),
            ("- check the logs first", "the Steps section holds a line that is not a numbered step, one per line with its wrapped lines indented under it: '- check the logs first'"),
        ):
            with self.subTest(listing=listing):
                code, _, err = self.pre_write(file_path=str(todo), old_string="## Done when", new_string=f"## Steps\n\n{listing}\n\n## Done when")

                self.assertEqual(code, 2)
                self.assertIn(f"todos/TODO-001-export-to-csv.md: the steps table is the tool's, {why}, and the prose around it is yours to edit", err)

        self.assertEqual(self.pre_write(file_path=str(todo), old_string="## Done when", new_string="## Steps\n\n1. Read the table.\n2. TODO-003, the theme.\n\n## Done when")[0], 0)

    def test_a_row_appended_by_hand_at_an_entry_whose_id_two_files_carry_is_refused_as_add_refuses_it(self) -> None:
        alpha = self.root / "todos" / "TODO-001-export-to-csv.md"
        fixtures.write(path=self.root / "todos" / "TODO-001-other-thing.md", text=alpha.read_text(encoding="utf-8").replace("TODO-001-export-to-csv", "TODO-001-other-thing"))
        self.brain.refresh()

        code, _, err = self.pre_write(file_path=str(self.plan.path), old_string="| 2 | The docs. | | owner | pending |\n", new_string="| 2 | The docs. | | owner | pending |\n| 3 | The other. | TODO-001-other-thing | owner | pending |\n")

        self.assertEqual(code, 2)
        self.assertIn("plans/kit.md: the steps table is the tool's, step 3 points at 'TODO-001-other-thing', and its id TODO-001 two files carry, so it would reach nothing, and the prose around it is yours to edit", err)

    def test_an_empty_step_cell_is_filled_in_by_hand_and_a_filled_one_is_not_reworded(self) -> None:
        fixtures.write(path=self.plan.path, text=self.plan.path.read_text(encoding="utf-8").replace("| 2 | The docs. | | owner | pending |", "| 2 |  | | owner | pending |"))
        self.brain.refresh()

        self.assertEqual(self.pre_write(file_path=str(self.plan.path), old_string="| 2 |  | | owner | pending |\n", new_string="| 2 | The docs. | | owner | pending |\n")[0], 0)

        for filled, why in (
            ("TODO-001 then fix", "step 2 opens with TODO-001 in its Step cell, and a row points through its Points at cell"),
            ("TODO-099, the ghost", "step 2 points at 'TODO-099', which is not a todo, a bug or a review the brain knows"),
        ):
            with self.subTest(filled=filled):
                code, _, err = self.pre_write(file_path=str(self.plan.path), old_string="| 2 |  | | owner | pending |\n", new_string=f"| 2 | {filled} | | owner | pending |\n")

                self.assertEqual(code, 2)
                self.assertIn(why, err)

        code, _, err = self.pre_write(file_path=str(self.plan.path), old_string="| 1 | the export. | TODO-001 | owner | pending |\n", new_string="| 1 | the export, at length. | TODO-001 | owner | pending |\n")

        self.assertEqual(code, 2)
        self.assertIn("step 1 is worded as it was filed", err)

    def test_a_row_appended_by_hand_needs_text(self) -> None:
        code, _, err = self.pre_write(file_path=str(self.plan.path), old_string="| 2 | The docs. | | owner | pending |\n", new_string="| 2 | The docs. | | owner | pending |\n| 3 |  |  | owner | pending |\n")

        self.assertEqual(code, 2)
        self.assertIn("plans/kit.md: the steps table is the tool's, step 3 has no text, and the prose around it is yours to edit", err)

    def test_two_files_carrying_one_id_stay_two_nodes_of_the_graph_a_patch_is_judged_over(self) -> None:
        alpha = self.root / "todos" / "TODO-001-export-to-csv.md"
        beta = self.root / "todos" / "TODO-001-other-thing.md"
        fixtures.write(path=beta, text=alpha.read_text(encoding="utf-8").replace("TODO-001-export-to-csv", "TODO-001-other-thing"))
        other = writes.create(brain=self.brain, kind=schema.kind_named("todo"), title="Another todo", description="another", values={"area": "reports", "by": "owner"}, body="## What\n\nWhat.\n\n## Done when\n\nDone.\n")
        fixtures.write(path=other.path, text=other.path.read_text(encoding="utf-8").replace("## Done when", "## Steps\n\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n| 1 | the other. | TODO-001-other-thing | owner | pending |\n\n## Done when"))
        steps.add(brain=self.brain, key="todos/TODO-001-export-to-csv.md", lines=["Read the table."], by="owner")
        self.brain.refresh()
        bug = self.root / "bugs" / "BUG-002-login-loop.md"
        patch = (
            f"*** Begin Patch\n*** Update File: {alpha}\n@@\n-| 1 | Read the table. | | owner | pending |\n+| 1 | Read the table. | | owner | pending |\n+| 2 | After the other. | {other.identifier} | owner | pending |\n"
            f"*** Update File: {bug}\n@@\n-A log line.\n+Two log lines.\n*** End Patch\n"
        )

        code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"patch": patch}}, agent="codex")

        self.assertEqual((code, err), (0, ""))

    def test_a_row_appended_or_prose_around_the_table_passes(self) -> None:
        self.assertEqual(self.pre_write(file_path=str(self.plan.path), old_string="| 2 | The docs. | | owner | pending |\n", new_string="| 2 | The docs. | | owner | pending |\n| 3 | The readme. | | owner | pending |\n")[0], 0)
        self.assertEqual(self.pre_write(file_path=str(self.plan.path), old_string="| 2 | The docs. | | owner | pending |\n", new_string="| 2 | The docs. | | owner | pending |\n| 3 | the loop. | BUG-002 | agent | pending |\n")[0], 0)
        self.assertEqual(self.pre_write(file_path=str(self.plan.path), old_string="Why.", new_string="Why, at length.")[0], 0)

    def test_a_list_written_under_a_new_steps_heading_passes_and_a_table_taken_away_does_not(self) -> None:
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"
        code, _, _ = self.pre_write(file_path=str(todo), old_string="## Done when", new_string="## Steps\n\n1. Read the table.\n2. Write it.\n\n## Done when")

        self.assertEqual(code, 0)

        table = "| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n| 1 | the export. | TODO-001 | owner | pending |\n| 2 | The docs. | | owner | pending |\n"
        code, _, err = self.pre_write(file_path=str(self.plan.path), old_string=table, new_string="1. the export.\n2. The docs.\n")

        self.assertEqual(code, 2)
        self.assertIn("plans/kit.md: the steps table is the tool's, its rows are never taken away, and the prose around it is yours to edit", err)

    def test_a_plans_empty_table_and_its_steps_section_are_never_taken_away(self) -> None:
        bare = plans.create(brain=self.brain, title="Bare", description="bare", body="## Why\n\nWhy.\n\n## Done when\n\nDone.\n", stamp="2026-09-16")
        self.brain.refresh()
        empty = "## Steps\n\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n"
        self.assertIn(empty, bare.path.read_text(encoding="utf-8"))

        for gone in ("## Steps\n\n", ""):
            with self.subTest(gone=gone):
                code, _, err = self.pre_write(file_path=str(bare.path), old_string=empty, new_string=gone)

                self.assertEqual(code, 2)
                self.assertIn("plans/bare.md: the steps table is the tool's, and a plan's table is never taken away, empty or not, and the prose around it is yours to edit", err)

        self.assertEqual(self.pre_write(file_path=str(bare.path), old_string="Why.", new_string="Why, at length.")[0], 0)

        for path in (bare.path, self.plan.path):
            with self.subTest(path=path.name):
                loose = f"*** Begin Patch\n*** Update File: {path}\n@@\n-| # | Step | Points at | By | Status | \n-|---|---|---|---|---|\n*** End Patch\n"
                code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"patch": loose}}, agent="codex")

                self.assertEqual(code, 2)
                self.assertIn(f"plans/{path.name}: a hunk the hook cannot place touches the steps table with '| # | Step | Points at | By | Status | ', and the steps table is the tool's", err)

        bug = self.root / "bugs" / "BUG-002-login-loop.md"
        fixtures.write(path=bug, text=bug.read_text(encoding="utf-8").replace("A log line.", "A log line.  \n\n| # | when | what |\n|---|---|---|\n| 1 | 09:00 | a loop |"))
        self.brain.refresh()
        loose = f"*** Begin Patch\n*** Update File: {bug}\n@@\n-A log line.\n-\n-| # | when | what |\n+| # | when | what happened |\n*** End Patch\n"

        self.assertEqual(self.run_hook(event="pre-write", payload={"tool_input": {"patch": loose}}, agent="codex")[0], 0)

    def test_a_write_that_leaves_the_table_unreadable_is_blocked(self) -> None:
        code, _, err = self.pre_write(file_path=str(self.plan.path), old_string="| 2 | The docs. | | owner | pending |", new_string="| 2 | The docs.")

        self.assertEqual(code, 2)
        self.assertIn("plans/kit.md: after that write the steps table would not read, a line of the steps table is not a whole row", err)

    def test_a_table_that_does_not_read_now_is_left_to_doctor(self) -> None:
        fixtures.write(path=self.plan.path, text=self.text.replace("| 2 | The docs. | | owner | pending |", "| 2 | The docs."))

        self.assertEqual(self.pre_write(file_path=str(self.plan.path), old_string="Why.", new_string="Why, at length.")[0], 0)

    def test_a_codex_patch_on_a_row_is_judged_placed_or_not(self) -> None:
        patch = f"*** Begin Patch\n*** Update File: {self.plan.path}\n@@\n-| 2 | The docs. | | owner | pending |\n+| 2 | The docs. | | owner | done |\n*** End Patch\n"
        code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"patch": patch}}, agent="codex")

        self.assertEqual(code, 2)
        self.assertIn("step 2 is marked done with brain task done or brain steps kit step 2 done", err)

        for removed in ("| 2 | The docs. | | owner | pending | ", " | 2 | The docs. | | owner | pending |"):
            with self.subTest(removed=removed):
                loose = f"*** Begin Patch\n*** Update File: {self.plan.path}\n@@\n-{removed}\n+| 2 | The docs. | | owner | done |\n*** End Patch\n"
                code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"patch": loose}}, agent="codex")

                self.assertEqual(code, 2)
                self.assertIn(f"plans/kit.md: a hunk the hook cannot place touches the steps table with '{removed}', and the steps table is the tool's", err)

    def test_a_codex_hunk_the_hook_cannot_place_that_carries_a_pointed_step_or_the_steps_heading_is_refused(self) -> None:
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"
        fixtures.write(path=todo, text=todo.read_text(encoding="utf-8").replace("## Done when", "## Steps\n\n1. Read the table.\n\n## Done when"))
        other = self.root / "todos" / "TODO-003-dark-mode.md"
        self.brain.refresh()

        for path, removed, added, carried in (
            (todo, "1. Read the table.   ", "2. TODO-001, again.", "2. TODO-001, again."),
            (todo, "1. Read the table.   ", "2. bug-002: the loop.", "2. bug-002: the loop."),
            (other, "## Done when   ", "## Steps\n+\n+1. pick colours\n+\n+## Done when", "## Steps"),
        ):
            with self.subTest(added=added):
                loose = f"*** Begin Patch\n*** Update File: {path}\n@@\n-{removed}\n+{added}\n*** End Patch\n"
                code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"patch": loose}}, agent="codex")

                self.assertEqual(code, 2)
                self.assertIn(f"a hunk the hook cannot place carries '{carried}', which reads as a step or the Steps heading and could land in the list the tool turns into the table, and the steps table is the tool's", err)

        loose = f"*** Begin Patch\n*** Update File: {todo}\n@@\n-What.   \n+What, at length.\n+\n+1. First we read, then we write.\n*** End Patch\n"
        self.assertEqual(self.run_hook(event="pre-write", payload={"tool_input": {"patch": loose}}, agent="codex")[0], 0)

        loose = f"*** Begin Patch\n*** Update File: {other}\n@@\n-## Done when   \n+2. ship\n+\n+## Done when\n*** End Patch\n"
        code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"patch": loose}}, agent="codex")

        self.assertEqual(code, 2)
        self.assertIn("todos/TODO-003-dark-mode.md: a hunk the hook cannot place carries '2. ship', which reads as a step, and TODO-003-dark-mode is done, a closed table being its record, so no step is added to it", err)


class TheHeaderIsTheTools(HookCase):
    def setUp(self) -> None:
        super().setUp()
        self.target = self.root / "bugs" / "BUG-002-login-loop.md"
        self.text = self.target.read_text(encoding="utf-8")

    def pre_write(
        self,
        agent: str = "claude",
        **tool_input: str,
    ) -> tuple[int, str, str]:
        return self.run_hook(event="pre-write", payload={"tool_input": tool_input}, agent=agent)

    def test_a_whole_write_that_changes_a_header_line_is_blocked(self) -> None:
        code, _, err = self.pre_write(file_path=str(self.target), content=self.text.replace("status: open", "status: fixed"))

        self.assertEqual(code, 2)
        self.assertIn("bugs/BUG-002-login-loop.md: the header block changes through brain set or brain refile", err)

    def test_a_whole_write_that_only_changes_the_prose_or_the_date_passes(self) -> None:
        content = self.text.replace("A log line.", "Two log lines.").replace("updated: 2026-09-16", "updated: 2026-10-01")

        code, _, _ = self.pre_write(file_path=str(self.target), content=content)

        self.assertEqual(code, 0)

    def test_a_whole_write_with_windows_line_endings_is_judged_by_its_header_not_its_endings(self) -> None:
        same = self.text.replace("\n", "\r\n")
        changed = same.replace("status: open", "status: fixed")

        self.assertEqual(self.pre_write(file_path=str(self.target), content=same)[0], 0)
        self.assertEqual(self.pre_write(file_path=str(self.target), content=changed)[0], 2)
        self.assertEqual(self.pre_write(file_path=str(self.target), old_string="A log line.\r\n", new_string="Two.\r\n")[0], 0)

    def test_a_whole_write_that_drops_the_header_is_blocked(self) -> None:
        code, _, err = self.pre_write(file_path=str(self.target), content="# Just prose now\n")

        self.assertEqual(code, 2)
        self.assertIn("after that write the header would not read, the file does not open with a --- fence", err)

    def test_an_edit_that_replaces_a_header_line_is_blocked(self) -> None:
        code, _, err = self.pre_write(file_path=str(self.target), old_string="severity: bug", new_string="severity: nit")

        self.assertEqual(code, 2)
        self.assertIn("the header block changes through brain set", err)

    def test_an_edit_in_the_prose_or_on_the_date_passes(self) -> None:
        for old, new in (("A log line.", "Two log lines."), ("updated: 2026-09-16", "updated: 2026-10-01"), ("2026-09-16", "2026-10-01")):
            with self.subTest(old_string=old):
                code, _, _ = self.pre_write(file_path=str(self.target), old_string=old, new_string=new)

                self.assertEqual(code, 0)

    def test_an_edit_that_means_every_occurrence_is_judged_that_way_whatever_the_tool_calls_it(self) -> None:
        self.target.write_text(self.text.replace("reported: 2026-09-02", "reported: 2026-09-16"), encoding="utf-8")

        for flag in ({"replace_all": True}, {"allow_multiple": True}, {"expected_replacements": 2}):
            with self.subTest(flag=flag):
                payload = {"tool_input": {"file_path": str(self.target), "old_string": "2026-09-16", "new_string": "2026-10-01", **flag}}

                code, _, _ = self.run_hook(event="pre-write", payload=payload)

                self.assertEqual(code, 2)

    def test_an_edit_that_turns_the_date_line_into_something_else_is_blocked(self) -> None:
        code, _, _ = self.pre_write(file_path=str(self.target), old_string="updated: 2026-09-16", new_string="x")

        self.assertEqual(code, 2)

    def test_an_edit_anchored_on_the_date_cannot_smuggle_a_header_line_in(self) -> None:
        code, _, err = self.pre_write(
            file_path=str(self.target),
            old_string="updated: 2026-09-16",
            new_string="updated: 2026-09-17\nstatus: fixed",
        )

        self.assertEqual(code, 2)
        self.assertIn("the header block changes through brain set", err)

    def test_an_empty_old_string_is_judged_by_what_it_would_leave(self) -> None:
        code, _, _ = self.pre_write(file_path=str(self.target), old_string="", new_string="status: fixed\n")

        self.assertEqual(code, 2)

    def test_a_codex_patch_that_takes_out_a_body_line_that_reads_like_a_header_line_passes(self) -> None:
        self.target.write_text(self.text + "\n```text\nstatus: open\n```\n", encoding="utf-8")
        patch = f"*** Begin Patch\n*** Update File: {self.target}\n@@\n ```text\n-status: open\n+status: closed\n ```\n*** End Patch\n"

        code, _, _ = self.pre_write(agent="codex", patch=patch)

        self.assertEqual(code, 0)

    def test_a_codex_patch_that_adds_a_header_line_is_blocked(self) -> None:
        patch = f"*** Begin Patch\n*** Update File: {self.target}\n@@\n area: auth\n+parent: BUG-001\n reported: 2026-09-02\n*** End Patch\n"

        code, _, err = self.pre_write(agent="codex", patch=patch)

        self.assertEqual(code, 2)
        self.assertIn("the header block changes through brain set", err)

    def test_a_bare_blank_line_in_a_hunk_is_blank_context_not_a_way_around(self) -> None:
        patch = f"*** Begin Patch\n*** Update File: {self.target}\n@@\n links: []\n-status: open\n+status: fixed\n\n\n*** End Patch\n"

        code, _, err = self.pre_write(agent="codex", patch=patch)

        self.assertEqual(code, 2)
        self.assertIn("the header block changes through brain set", err)

    def test_a_codex_patch_whose_only_hunk_does_not_fit_is_not_judged(self) -> None:
        patch = f"*** Begin Patch\n*** Update File: {self.target}\n@@\n-no such line\n+something else\n*** End Patch\n"

        code, _, _ = self.pre_write(agent="codex", patch=patch)

        self.assertEqual(code, 0)

    def test_one_loose_hunk_does_not_disarm_the_guard_for_the_others(self) -> None:
        patch = f"*** Begin Patch\n*** Update File: {self.target}\n@@\n-status: open\n+status: fixed\n@@\n-no such line\n+nor this\n*** End Patch\n"

        code, _, err = self.pre_write(agent="codex", patch=patch)

        self.assertEqual(code, 2)
        self.assertIn("the header block changes through brain set", err)

    def test_a_hunk_the_hook_cannot_place_may_not_touch_a_header_line_by_any_sign(self) -> None:
        shapes = (
            ("+status: fixed", "status: fixed"),
            ("+---", "---"),
            (" status: open \n-severity: bug", "status: open "),
            (" links: []\n-reported: 2026-09-02\n+reported: 2026-09-03", "links: []"),
            ("-no such context\n+parent: BUG-001", "parent: BUG-001"),
        )

        for shape, named in shapes:
            with self.subTest(shape=shape):
                patch = f"*** Begin Patch\n*** Update File: {self.target}\n@@\n{shape}\n*** End Patch\n"

                code, _, err = self.pre_write(agent="codex", patch=patch)

                self.assertEqual(code, 2)
                self.assertIn(f"a hunk the hook cannot place touches the header block with '{named}'", err)

    def test_a_hunk_that_takes_a_fence_out_is_placed_and_refused_for_losing_the_header(self) -> None:
        opening = "----"
        closing = " reported: 2026-09-02\n----\n \n ## Problem"

        for shape in (opening, closing):
            with self.subTest(shape=shape):
                patch = f"*** Begin Patch\n*** Update File: {self.target}\n@@\n{shape}\n*** End Patch\n"

                code, _, err = self.pre_write(agent="codex", patch=patch)

                self.assertEqual(code, 2)
                self.assertIn("after that write the header would not read", err)

    def test_a_hunk_the_hook_cannot_place_that_only_touches_prose_is_let_be(self) -> None:
        for shape in ("+Just a sentence with no colon.", "-no such line\n+Note: see below", "+updated: 2026-10-01"):
            with self.subTest(shape=shape):
                patch = f"*** Begin Patch\n*** Update File: {self.target}\n@@\n{shape}\n*** End Patch\n"

                code, _, _ = self.pre_write(agent="codex", patch=patch)

                self.assertEqual(code, 0)

    def test_hunks_are_placed_one_after_another_as_the_patch_applies_them(self) -> None:
        patch = (
            f"*** Begin Patch\n*** Update File: {self.target}\n"
            "@@\n-updated: 2026-09-16\n+updated: 2026-10-01\n"
            "@@\n updated: 2026-09-16\n links: []\n-status: open\n+status: fixed\n"
            "*** End Patch\n"
        )

        code, _, err = self.pre_write(agent="codex", patch=patch)

        self.assertEqual(code, 2)
        self.assertIn("a hunk the hook cannot place touches the header block", err)

    def test_an_edit_across_the_closing_fence_is_blocked(self) -> None:
        code, _, _ = self.pre_write(file_path=str(self.target), old_string="---\n\n## Problem", new_string="## Problem")

        self.assertEqual(code, 2)

    def test_a_codex_patch_that_takes_a_header_line_out_is_blocked(self) -> None:
        patch = f"*** Begin Patch\n*** Update File: {self.target}\n@@\n-status: open\n+status: fixed\n*** End Patch\n"

        code, _, err = self.pre_write(agent="codex", patch=patch)

        self.assertEqual(code, 2)
        self.assertIn("the header block changes through brain set", err)

    def test_a_codex_patch_that_only_changes_the_prose_passes(self) -> None:
        patch = f"*** Begin Patch\n*** Update File: {self.target}\n@@\n-A log line.\n+Two log lines.\n*** End Patch\n"

        code, _, _ = self.pre_write(agent="codex", patch=patch)

        self.assertEqual(code, 0)

    def test_an_existing_index_is_still_the_tools(self) -> None:
        code, _, err = self.pre_write(file_path=str(self.root / "bugs" / "INDEX.md"), old_string="Bugs Index", new_string="Mine")

        self.assertEqual(code, 2)
        self.assertIn("run brain index", err)

    def test_a_file_whose_header_does_not_read_takes_a_repair_and_nothing_else(self) -> None:
        broken = self.text.replace("name: BUG-002-login-loop\n", "name: BUG-002-login-loop\n\n")
        self.target.write_text(broken, encoding="utf-8")

        repaired = self.pre_write(file_path=str(self.target), content=self.text.replace("status: open", "status: fixed"))
        worse = self.pre_write(file_path=str(self.target), content="# Just prose now\n")

        self.assertEqual(repaired[0], 0)
        self.assertEqual(worse[0], 2)
        self.assertIn("does not read as a header now, line 3 of the header is blank", worse[2])
        self.assertIn("and that write would not make it one, the file does not open with a --- fence", worse[2])

    def test_a_write_that_leaves_a_broken_header_as_it_was_says_the_reason_once(self) -> None:
        self.target.write_text(self.text.replace("kind: bug\n", "kind: bug\nname: twice\n"), encoding="utf-8")

        code, _, err = self.pre_write(file_path=str(self.target), old_string="A log line.", new_string="Two.")

        self.assertEqual(code, 2)
        self.assertIn("and that write would leave it so", err)
        self.assertEqual(err.count("repeats name"), 1)

    def test_a_loose_hunk_is_judged_by_its_shape_even_when_the_header_does_not_read(self) -> None:
        self.target.write_text(self.text.replace("name: BUG-002-login-loop\n", "name: BUG-002-login-loop\n\n"), encoding="utf-8")
        patch = (
            f"*** Begin Patch\n*** Update File: {self.target}\n"
            "@@\n name: BUG-002-login-loop\n-\n kind: bug\n"
            "@@\n-no such line\n+status: done\n"
            "*** End Patch\n"
        )

        code, _, err = self.pre_write(agent="codex", patch=patch)

        self.assertEqual(code, 2)
        self.assertIn("a hunk the hook cannot place touches the header block with 'status: done'", err)

    def test_a_kit_files_header_is_a_persons_to_edit(self) -> None:
        spec = self.root / "spec" / "todos.md"
        fixtures.write(path=spec, text="---\nname: spec-todos\nkind: doc\ndescription: old\nupdated: 2026-09-16\nlinks: []\n---\n\n# Todos\n")

        code, _, _ = self.pre_write(file_path=str(spec), old_string="description: old", new_string="description: new")

        self.assertEqual(code, 0)

    def test_a_new_spec_or_skill_file_passes(self) -> None:
        for target in (self.root / "skills" / "agent-brain" / "references" / "new.md", self.root / "spec" / "new.md", self.root / "README.md"):
            with self.subTest(target=str(target.relative_to(self.root))):
                code, _, _ = self.run_hook(event="pre-write", payload={"tool_input": {"file_path": str(target)}})

                self.assertEqual(code, 0)

    def test_a_dotted_path_is_blocked_with_the_reason(self) -> None:
        for target in (self.root / "docs" / ".draft.md", self.root / "memory" / ".putback.xyz" / "note.md"):
            with self.subTest(target=str(target.relative_to(self.root))):
                code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"file_path": str(target)}})

                self.assertEqual(code, 2)
                self.assertIn("would be a dotted path", err)

    def test_a_nested_spec_file_is_blocked_with_the_reason(self) -> None:
        target = self.root / "spec" / "reviews" / "batches.md"

        code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"file_path": str(target)}})

        self.assertEqual(code, 2)
        self.assertIn("would nest inside spec/", err)

    def test_a_readme_inside_a_tracker_is_a_brain_file_and_is_blocked(self) -> None:
        target = self.root / "docs" / "README.md"

        code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"file_path": str(target)}})

        self.assertEqual(code, 2)
        self.assertIn("docs/README.md would be a new brain file", err)

    def test_a_file_outside_the_brain_passes(self) -> None:
        target = self.project / "lib" / "main.dart"

        code, _, _ = self.run_hook(event="pre-write", payload={"tool_input": {"file_path": str(target)}})

        self.assertEqual(code, 0)

    def test_an_index_written_by_hand_is_blocked(self) -> None:
        target = self.root / "bugs" / "INDEX.md"
        target.unlink()

        code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"file_path": str(target)}})

        self.assertEqual(code, 2)
        self.assertIn("brain index", err)

    def test_an_index_where_none_belongs_is_blocked_with_that_reason(self) -> None:
        for target in (self.root / "handover" / "INDEX.md", self.root / "spec" / "INDEX.md"):
            with self.subTest(target=str(target.relative_to(self.root))):
                code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"file_path": str(target)}})

                self.assertEqual(code, 2)
                self.assertIn("an index where none belongs", err)

    def test_an_index_inside_a_skill_is_the_skills_own_business(self) -> None:
        target = self.root / "skills" / "my-skill" / "INDEX.md"

        code, _, _ = self.run_hook(event="pre-write", payload={"tool_input": {"file_path": str(target)}})

        self.assertEqual(code, 0)

    def test_a_relative_path_is_taken_against_the_events_directory(self) -> None:
        payload = {"cwd": str(self.root / "todos"), "tool_input": {"file_path": "TODO-009-by-hand.md"}}

        code, _, err = self.run_hook(event="pre-write", payload=payload)

        self.assertEqual(code, 2)
        self.assertIn("todos/TODO-009-by-hand.md", err)

    def test_a_relative_path_from_a_subfolder_of_the_project_is_still_blocked(self) -> None:
        payload = {"cwd": str(self.project / "lib"), "tool_input": {"file_path": "../.agents/bugs/BUG-009-by-hand.md"}}

        code, _, err = self.run_hook(event="pre-write", payload=payload)

        self.assertEqual(code, 2)
        self.assertIn("bugs/BUG-009-by-hand.md", err)

    def test_a_project_reached_through_a_symlink_is_still_guarded(self) -> None:
        alias = self.project / "alias"
        alias.symlink_to(self.project)
        payload = {"cwd": str(alias), "tool_input": {"file_path": str(alias / ".agents" / "bugs" / "BUG-009-by-hand.md")}}

        code, _, err = self.run_hook(event="pre-write", payload=payload)
        relative = self.run_hook(event="pre-write", payload={"cwd": str(alias), "tool_input": {"file_path": ".agents/bugs/BUG-009-by-hand.md"}})

        self.assertEqual(code, 2)
        self.assertIn("bugs/BUG-009-by-hand.md", err)
        self.assertEqual(relative[0], 2)

    def test_a_folder_the_process_cannot_search_still_blocks_a_hand_written_file(self) -> None:
        docs = self.root / "docs"
        docs.chmod(0)

        try:
            code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"file_path": str(docs / "by-hand.md")}})
        finally:
            docs.chmod(0o700)

        self.assertEqual(code, 2)
        self.assertIn("docs/by-hand.md", err)

    def test_a_symlink_loop_in_the_written_path_does_not_raise(self) -> None:
        (self.root / "docs" / "loop-a").symlink_to("loop-b")
        (self.root / "docs" / "loop-b").symlink_to("loop-a")
        target = self.root / "docs" / "loop-a" / "note.md"

        code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"file_path": str(target)}})

        self.assertEqual(code, 2)
        self.assertIn("docs/loop-a/note.md", err)

    def test_a_path_with_a_null_byte_ends_in_one_line_not_a_traceback(self) -> None:
        target = str(self.root / "bugs" / ("BUG-009" + chr(0) + ".md"))

        code, out, err = self.run_hook(event="pre-write", payload={"tool_input": {"file_path": target}})

        self.assertIn(code, (1, 2))
        self.assertEqual(out, "")
        self.assertEqual(len(err.strip().splitlines()), 1)

    def test_a_codex_patch_that_adds_a_brain_file_is_blocked(self) -> None:
        patch = f"*** Begin Patch\n*** Add File: {self.root / 'todos' / 'TODO-009-x.md'}\n+hello\n*** End Patch\n"

        code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"patch": patch}}, agent="codex")

        self.assertEqual(code, 2)
        self.assertIn("TODO-009-x.md", err)

    def test_a_codex_patch_that_moves_a_brain_file_anywhere_is_blocked(self) -> None:
        old = self.root / "bugs" / "BUG-002-login-loop.md"
        destinations = (old.with_name("BUG-002-renamed.md"), old.with_name("BUG-002-login-loop.txt"), self.project / "notes" / "login.md")

        for destination in destinations:
            with self.subTest(destination=destination.name):
                patch = f"*** Begin Patch\n*** Update File: {old}\n*** Move to: {destination}\n@@\n-A log line.\n+Two.\n*** End Patch\n"

                code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"patch": patch}}, agent="codex")

                self.assertEqual(code, 2)
                self.assertIn("bugs/BUG-002-login-loop.md is renamed with brain refile, not by hand", err)

    def test_prose_that_quotes_a_patch_is_not_taken_for_one(self) -> None:
        spec = self.root / "spec" / "hooks.md"
        content = "---\nname: spec-hooks\n---\n\n```text\n*** Begin Patch\n*** Add File: .agents/todos/TODO-101-example.md\n*** End Patch\n```\n"

        code, _, _ = self.run_hook(event="pre-write", payload={"tool_input": {"file_path": str(spec), "content": content}})

        self.assertEqual(code, 0)

    def test_codex_is_read_from_its_patch_and_nothing_else(self) -> None:
        target = self.root / "bugs" / "BUG-009-by-hand.md"

        code, _, _ = self.run_hook(event="pre-write", payload={"tool_input": {"file_path": str(target)}}, agent="codex")

        self.assertEqual(code, 0)

    def test_a_codex_patch_that_moves_a_file_outside_the_brain_passes(self) -> None:
        old = self.project / "lib" / "main.dart"
        fixtures.write(path=old, text="void main() {}\n")
        patch = f"*** Begin Patch\n*** Update File: {old}\n*** Move to: {old.with_name('app.dart')}\n*** End Patch\n"

        code, _, _ = self.run_hook(event="pre-write", payload={"tool_input": {"patch": patch}}, agent="codex")

        self.assertEqual(code, 0)

    def test_a_codex_patch_names_every_file_and_an_update_carries_its_hunks(self) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Update File: bugs/BUG-002-login-loop.md\n*** Move to: bugs/BUG-002-renamed.md\n@@\n-severity: bug\n+severity: nit\n context\n@@ ## Evidence\n-A log line.\n+Two.\n"
            "*** Delete File: todos/TODO-003-dark-mode.md\n"
            "*** Add File: docs/new.md\n+hello\n"
            "*** End Patch\n"
        )

        found = hooks.patch_touches(base=self.root, patch=patch)

        self.assertEqual([touch.path.name for touch in found], ["BUG-002-login-loop.md", "BUG-002-renamed.md", "TODO-003-dark-mode.md", "new.md"])
        self.assertEqual(found[0].hunks, (("-severity: bug", "+severity: nit", " context"), ("-A log line.", "+Two.")))
        self.assertEqual(found[1].moved_from, found[0].path)
        self.assertEqual((found[1].hunks, found[2].hunks, found[3].hunks), ((), (), ()))
        self.assertEqual(found[3].content, "hello\n")

    def test_a_codex_add_keeps_a_bare_blank_line_as_a_blank_line(self) -> None:
        patch = f"*** Begin Patch\n*** Add File: {self.root / 'docs' / 'new.md'}\n+---\n+name: new\n+---\n\n+# New\n*** End Patch\n"

        found = hooks.patch_touches(base=self.root, patch=patch)

        self.assertEqual(found[0].content, "---\nname: new\n---\n\n# New\n")

    def test_a_codex_add_over_an_existing_brain_file_is_judged_by_the_text_it_brings(self) -> None:
        target = self.root / "bugs" / "BUG-002-login-loop.md"
        text = target.read_text(encoding="utf-8").replace("status: open", "status: fixed")
        patch = f"*** Begin Patch\n*** Add File: {target}\n" + "".join(f"+{line}\n" for line in text.splitlines()) + "*** End Patch\n"

        code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"patch": patch}}, agent="codex")

        self.assertEqual(code, 2)
        self.assertIn("the header block changes through brain set", err)

    def test_windows_line_endings_do_not_hide_the_file_from_the_hook(self) -> None:
        target = self.root / "todos" / "TODO-009-x.md"
        patch = f"*** Begin Patch\r\n*** Add File: {target}\r\n+hello\r\n*** End Patch\r\n"

        by_patch = self.run_hook(event="pre-write", payload={"tool_input": {"patch": patch}}, agent="codex")
        by_path = self.run_hook(event="pre-write", payload={"tool_input": {"file_path": f"{target}\r"}})

        self.assertEqual((by_patch[0], by_path[0]), (2, 2))
        self.assertIn("todos/TODO-009-x.md would be", by_patch[2])
        self.assertIn("todos/TODO-009-x.md would be", by_path[2])


class ABrokenBrain(HookCase):
    def test_a_folder_named_like_a_file_is_noted_and_the_hook_still_runs(self) -> None:
        (self.root / "docs" / "notes.md").mkdir()

        code, out, _ = self.run_hook(event="session-start", payload={})

        self.assertEqual(code, 0)
        self.assertIn("unreadable docs/notes.md", json.loads(out)["hookSpecificOutput"]["additionalContext"])

    def test_a_brain_the_hook_cannot_read_fails_visibly_not_with_a_traceback(self) -> None:
        target = self.root / "bugs" / "BUG-002-login-loop.md"
        failing = PermissionError(13, "Permission denied", str(self.root / "docs" / "notes.md"))

        with mock.patch("brainlib.doctor.run", side_effect=failing):
            starting = self.run_hook(event="session-start", payload={})
            after = self.run_hook(event="post-write", payload={"tool_input": {"file_path": str(target)}})

        for code, out, err in (starting, after):
            self.assertEqual((code, out), (1, ""))
            self.assertEqual(err.strip(), "the brain hook could not run: Permission denied: .agents/docs/notes.md")


class AnEventTheScriptDoesNotKnow(HookCase):
    def test_it_is_the_error_every_agent_shows_and_never_the_block(self) -> None:
        out, err = io.StringIO(), io.StringIO()

        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with mock.patch("sys.stdin", io.StringIO(json.dumps({"cwd": str(self.project)}))):
                with self.assertRaises(SystemExit) as caught:
                    hooks.main(["compact", "--agent", "claude"])

        self.assertEqual(caught.exception.code, 1)
        self.assertEqual(out.getvalue(), "")
        self.assertTrue(err.getvalue().startswith("the brain hook could not run: argument event: invalid choice: 'compact'"), err.getvalue())


class NoInput(HookCase):
    def test_a_closed_standard_input_is_no_event_not_a_crash(self) -> None:
        out, err = io.StringIO(), io.StringIO()

        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with mock.patch("sys.stdin", None):
                codes = [hooks.main([event, "--agent", "claude"]) for event in ("session-start", "pre-write", "post-write")]

        self.assertEqual(codes, [0, 0, 0])
        self.assertEqual(err.getvalue(), "")


class PostWrite(HookCase):
    def test_a_write_that_breaks_the_brain_is_reported(self) -> None:
        target = self.root / "bugs" / "BUG-002-login-loop.md"
        target.write_text(target.read_text(encoding="utf-8").replace("severity: bug", "severity: huge"), encoding="utf-8")

        code, out, _ = self.run_hook(event="post-write", payload={"tool_input": {"file_path": str(target)}})

        self.assertEqual(code, 0)
        context: str = json.loads(out)["hookSpecificOutput"]["additionalContext"]
        self.assertIn("severity is breaks or bug or debt or nit, not 'huge'", context)

    def test_a_clean_write_says_nothing(self) -> None:
        target = self.root / "bugs" / "BUG-002-login-loop.md"

        code, out, _ = self.run_hook(event="post-write", payload={"tool_input": {"file_path": str(target)}})

        self.assertEqual((code, out), (0, ""))

    def test_gemini_gets_an_empty_object_when_there_is_nothing_to_say(self) -> None:
        target = self.root / "bugs" / "BUG-002-login-loop.md"

        after = self.run_hook(event="post-write", payload={"tool_input": {"file_path": str(target)}}, agent="gemini")
        before = self.run_hook(event="pre-write", payload={"tool_input": {"file_path": str(target)}}, agent="gemini")

        self.assertEqual((after[0], after[1].strip()), (0, "{}"))
        self.assertEqual((before[0], before[1].strip()), (0, "{}"))

    def test_the_reply_names_the_event_the_way_each_agent_calls_it(self) -> None:
        target = self.root / "bugs" / "BUG-002-login-loop.md"
        target.write_text(target.read_text(encoding="utf-8").replace("severity: bug", "severity: huge"), encoding="utf-8")

        for agent, event in (("claude", "PostToolUse"), ("gemini", "AfterTool")):
            with self.subTest(agent=agent):
                _, out, _ = self.run_hook(event="post-write", payload={"tool_input": {"file_path": str(target)}}, agent=agent)

                self.assertEqual(json.loads(out)["hookSpecificOutput"]["hookEventName"], event)

        _, out, _ = self.run_hook(event="session-start", payload={}, agent="gemini")
        self.assertEqual(json.loads(out)["hookSpecificOutput"]["hookEventName"], "SessionStart")

    def test_a_codex_write_is_read_from_the_command_field_its_hook_reference_names(self) -> None:
        target = self.root / "bugs" / "BUG-009-by-hand.md"
        patch = f"*** Begin Patch\n*** Add File: {target}\n+hello\n*** End Patch\n"

        code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"command": patch}}, agent="codex")

        self.assertEqual(code, 2)
        self.assertIn("bugs/BUG-009-by-hand.md would be", err)

    def test_a_codex_patch_handed_over_inside_a_list_is_read_too(self) -> None:
        target = self.root / "bugs" / "BUG-009-by-hand.md"
        patch = f"*** Begin Patch\n*** Add File: {target}\n+hello\n*** End Patch\n"

        code, _, err = self.run_hook(event="pre-write", payload={"tool_input": {"command": ["apply_patch", patch]}}, agent="codex")

        self.assertEqual(code, 2)
        self.assertIn("bugs/BUG-009-by-hand.md would be", err)

    def test_a_codex_patch_that_updates_or_deletes_a_brain_file_runs_doctor_after(self) -> None:
        target = self.root / "bugs" / "BUG-002-login-loop.md"
        target.write_text(target.read_text(encoding="utf-8").replace("severity: bug", "severity: huge"), encoding="utf-8")
        updating = f"*** Begin Patch\n*** Update File: {target}\n@@\n-severity: bug\n+severity: huge\n*** End Patch\n"
        deleting = f"*** Begin Patch\n*** Delete File: {target}\n*** End Patch\n"

        after_update = self.run_hook(event="post-write", payload={"tool_input": {"patch": updating}}, agent="codex")
        after_delete = self.run_hook(event="post-write", payload={"tool_input": {"patch": deleting}}, agent="codex")

        for code, out, _ in (after_update, after_delete):
            self.assertEqual(code, 0)
            self.assertIn("severity is breaks or bug or debt or nit, not 'huge'", out)

    def test_a_write_elsewhere_is_ignored(self) -> None:
        code, out, _ = self.run_hook(event="post-write", payload={"tool_input": {"file_path": str(self.project / "lib" / "x.dart")}})

        self.assertEqual((code, out), (0, ""))


if __name__ == "__main__":
    unittest.main()
