"""The command line refuses cleanly, exits the way a gate expects, and files prose."""

import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import brain
from brainlib import doctor, index, store, task
from tests import fixtures


class Feed(io.TextIOWrapper):
    """Standard input as the tool sees it: bytes under the text like a pipe, and a terminal when there is nothing to read."""

    def __init__(
        self,
        text: str = "",
        raw: bytes = b"",
    ) -> None:
        super().__init__(io.BytesIO(raw or text.encode("utf-8")), encoding="utf-8")
        self.fed = bool(raw or text)

    def isatty(self) -> bool:
        return not self.fed


class CliCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = fixtures.build(Path(self.tmp.name))
        kit = store.Brain(root=self.root)
        index.write(brain=kit, stamp="2026-09-16")
        doctor.run(brain=kit, fix=True)
        self.before = os.getcwd()
        os.chdir(self.root.parent)

    def tearDown(self) -> None:
        os.chdir(self.before)
        self.tmp.cleanup()

    def run_cli(
        self,
        *argv: str,
        stdin: str = "",
        raw: bytes = b"",
    ) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()

        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with mock.patch("sys.stdin", Feed(text=stdin, raw=raw)):
                code = brain.main(list(argv))

        return code, out.getvalue(), err.getvalue()


class Reading(CliCase):
    def test_status_exits_clean(self) -> None:
        code, out, _ = self.run_cli("status")

        self.assertEqual(code, 0)
        self.assertIn("doctor clean", out)

    def test_status_opens_with_the_clock_the_word_and_the_work_and_the_json_form_carries_the_work_alone(self) -> None:
        code, out, _ = self.run_cli("status")

        self.assertEqual(code, 0)
        self.assertRegex(out, r"^now: .+\nauto ack off\ntask none\n")

        code, out, _ = self.run_cli("status", "--json")

        self.assertEqual(code, 0)
        self.assertNotIn("auto ack", out)
        self.assertNotIn("now:", out)
        self.assertIn("counts", json.loads(out))
        self.assertEqual(json.loads(out)["task"], [])

        self.run_cli("ack", "on", "--for", "2h")
        self.run_cli("task", "on", "BUG-002")
        code, out, _ = self.run_cli("status")

        self.assertEqual(code, 0)
        self.assertRegex(out, r"^now: .+\nauto ack on, (2 h|1 h 59 min) left, until [^\n]+\nbug BUG-002\n")

        code, out, _ = self.run_cli("status", "--json")
        self.assertEqual(json.loads(out)["task"], [{"kind": "bug", "key": "BUG-002"}])
        self.run_cli("ack", "off")

        (self.root / task.FILE).write_text("not json", encoding="utf-8")
        code, out, _ = self.run_cli("status", "--json")
        self.assertEqual((code, json.loads(out)["task"]), (0, None))
        code, out, _ = self.run_cli("status")
        self.assertIn("\ntask none, .task does not read as a chain and brain task off --all clears it\n", out)

    def test_the_task_help_names_every_action_and_every_argument(self) -> None:
        for argv, wanted in (
            (("--help",), "pending again"),
            (("task", "step", "--help"), "the step's number"),
            (("task", "off", "--help"), "the whole chain"),
            (("steps", "TODO-001", "step", "--help"), "the step's number"),
            (("steps", "TODO-001", "step", "--help"), "the mark to give it"),
        ):
            with self.subTest(argv=argv):
                out = io.StringIO()
                with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as leaving:
                    brain.main(list(argv))
                self.assertEqual(leaving.exception.code, 0)
                self.assertIn(wanted, out.getvalue())

    def test_the_ledger_help_says_set_changes_any_field_of_a_finding(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(new_target=out):
            with self.assertRaises(SystemExit) as leaving:
                brain.main(
                    argv=["ledger", "--help"],
                )

        self.assertEqual(leaving.exception.code, 0)
        self.assertIn("change any field of one finding or several", out.getvalue())

    def test_list_narrows_and_lines_up(self) -> None:
        code, out, _ = self.run_cli("list", "bug", "--status", "open")

        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "BUG-002  open    bug       auth  the sign in screen loops")

    def test_a_key_that_fits_two_files_is_a_refusal_naming_both(self) -> None:
        fixtures.write(
            path=self.root / "reviews" / "2026-10-fix-login" / "ledger.md",
            text=(self.root / "reviews" / "2026-09-feat-billing" / "ledger.md").read_text(encoding="utf-8").replace("2026-09-feat-billing", "2026-10-fix-login"),
        )

        code, _, err = self.run_cli("show", "ledger")

        self.assertEqual(code, 1)
        self.assertIn("'ledger' names 2 files, use the path: reviews/2026-09-feat-billing/ledger.md, reviews/2026-10-fix-login/ledger.md", err)

    def test_an_unknown_key_is_a_refusal(self) -> None:
        code, _, err = self.run_cli("show", "BUG-404")

        self.assertEqual(code, 1)
        self.assertIn("BUG-404", err)

    def test_a_file_that_is_not_utf8_is_a_finding_not_a_traceback(self) -> None:
        (self.root / "bugs" / "BUG-005-latin.md").write_bytes(b"---\nname: latin\n---\n\ncaf\xe9\n")

        code, out, _ = self.run_cli("doctor")

        self.assertEqual(code, 2)
        self.assertIn("bugs/BUG-005-latin.md: frontmatter: the file is not UTF-8 text", out)

    def test_a_folder_named_like_a_file_is_noted_and_the_rest_still_answers(self) -> None:
        (self.root / "docs" / "notes.md").mkdir()

        code, out, _ = self.run_cli("status")

        self.assertEqual(code, 0)
        self.assertIn("unreadable docs/notes.md", out)
        self.assertIn("bugs       1 open", out)

    def test_an_os_error_is_a_plain_refusal_without_the_machines_path(self) -> None:
        failing = PermissionError(13, "Permission denied", str(self.root / "docs" / "notes.md"))

        with mock.patch("brainlib.reads.status", side_effect=failing):
            code, _, err = self.run_cli("status")

        self.assertEqual(code, 1)
        self.assertEqual(err.strip(), "Permission denied: .agents/docs/notes.md")

    def test_doctor_as_json_says_whether_each_finding_is_fixable(self) -> None:
        target = self.root / "memory" / "stripe-retries-twice.md"
        target.write_text(target.read_text(encoding="utf-8").replace("updated: 2026-09-16\n", ""), encoding="utf-8")

        code, out, _ = self.run_cli("doctor", "--json")

        self.assertEqual(code, 2)
        self.assertIn(
            {"path": "memory/stripe-retries-twice.md", "check": "frontmatter", "message": "no updated date", "fixable": True},
            json.loads(out),
        )

    def test_doctor_exits_two_while_findings_remain(self) -> None:
        target = self.root / "docs" / "systems" / "login.md"
        target.write_text(target.read_text(encoding="utf-8") + "\nA dash — here.\n", encoding="utf-8")

        code, out, _ = self.run_cli("doctor")

        self.assertEqual(code, 2)
        self.assertIn("carries a dash the house style bans", out)
        self.assertIn("1 findings", out)


class Porting(CliCase):
    def test_a_project_path_that_is_not_there_is_refused(self) -> None:
        code, _, err = self.run_cli("port", "--project", str(Path(self.tmp.name) / "nope"))

        self.assertEqual(code, 1)
        self.assertIn("so --project names nothing", err)

    def test_an_upgrade_finds_the_brain_above_the_current_folder(self) -> None:
        os.chdir(self.root / "docs")

        code, out, _ = self.run_cli("upgrade", "--kit", str(Path(brain.__file__).resolve().parents[4]))

        self.assertEqual(out.splitlines()[0], f"upgrading {self.root.parent.resolve()}")
        self.assertIn(code, (0, 2))


class Tasking(CliCase):
    PROSE = "## Why\n\nThe rest of phase 7.\n\n## Steps\n\n1. TODO-001, the export.\n2. BUG-002, the login loop.\n\n## Done when\n\nAll of it.\n"

    def test_the_work_goes_on_marks_its_steps_and_goes_off_link_by_link(self) -> None:
        code, out, _ = self.run_cli("task")
        self.assertEqual((code, out), (0, "task none\n"))

        self.run_cli("new", "plan", "--title", "Kit phase 7", "--description", "the rest of phase 7", stdin=self.PROSE)
        self.run_cli("steps", "kit-phase-7", "add", "The docs.")

        code, out, err = self.run_cli("task", "on", "kit-phase-7")
        self.assertEqual((code, out, err), (0, "plan kit-phase-7, step 1 of 3, the export.\n", ""))

        code, out, err = self.run_cli("task", "on", "BUG-002")
        self.assertEqual((code, out), (brain.REFUSED, ""))
        self.assertEqual(err.strip(), "the work is on kit-phase-7, step 1 of 3, the export., finish it or drop it with brain task off first")

        code, _, err = self.run_cli("task", "done")
        self.assertEqual((code, err.strip()), (brain.REFUSED, "step 1 points at TODO-001, so its mark follows that entry, and goes done once TODO-001 is done"))

        self.run_cli("set", "TODO-001", "status=done", "done=2026-09-24")
        code, out, _ = self.run_cli("task", "on", "BUG-002")
        self.assertEqual((code, out), (0, "plan kit-phase-7, step 2 of 3, the login loop.\nbug BUG-002\n"))

        code, out, _ = self.run_cli("task", "off")
        self.assertEqual((code, out), (0, "plan kit-phase-7, step 2 of 3, the login loop.\n"))

        self.run_cli("set", "BUG-002", "status=wontfix")
        code, out, _ = self.run_cli("task", "done")
        self.assertEqual(code, 0)
        self.assertRegex(out, r"^plan kit-phase-7, every step settled, close it with brain set kit-phase-7 status=done done=\d{4}-\d{2}-\d{2}\n$")
        self.assertIn("| 3 | The docs. | | agent | done |", (self.root / "plans" / "kit-phase-7.md").read_text(encoding="utf-8"))

        code, out, _ = self.run_cli("task", "step", "3")
        self.assertEqual((code, out), (0, "plan kit-phase-7, step 3 of 3, The docs.\n"))

        code, out, _ = self.run_cli("task", "skip")
        self.assertEqual(code, 0)
        self.assertRegex(out, r"^plan kit-phase-7, every step settled, close it with brain set kit-phase-7 status=done done=\d{4}-\d{2}-\d{2}\n$")

        code, out, _ = self.run_cli("task", "on", "TODO-001", "--replace")
        self.assertEqual((code, out), (brain.REFUSED, ""))

        code, out, _ = self.run_cli("task", "off", "--all")
        self.assertEqual((code, out), (0, "task none\n"))
        self.assertFalse((self.root / ".task").exists())

    def test_a_mark_with_nothing_to_mark_and_a_closed_entry_are_refused_in_words(self) -> None:
        code, out, err = self.run_cli("task", "on", "TODO-003")
        self.assertEqual((code, out), (brain.REFUSED, ""))
        self.assertEqual(err.strip(), "TODO-003-dark-mode is done, and the work is on something open")

        code, out, err = self.run_cli("task", "done")
        self.assertEqual((code, out), (brain.REFUSED, ""))
        self.assertEqual(err.strip(), "no work is on, start it with brain task on")

        self.run_cli("task", "on", "TODO-001")
        code, out, err = self.run_cli("task", "done")
        self.assertEqual((code, out), (brain.REFUSED, ""))
        self.assertEqual(err.strip(), "TODO-001-export-to-csv has no steps, add some with brain steps TODO-001 add")

        code, out, err = self.run_cli("task", "step", "2")
        self.assertEqual((code, out), (brain.REFUSED, ""))
        self.assertEqual(err.strip(), "TODO-001-export-to-csv has no steps, add some with brain steps TODO-001 add")

    def test_a_ledger_that_opens_goes_under_the_work_and_says_when_it_stays_off(self) -> None:
        self.run_cli("task", "on", "TODO-001")

        code, out, err = self.run_cli("ledger", "open", "fix/login", "--summary", "The login branch.")
        key = out.strip().split("/")[1]
        self.assertEqual((code, err), (0, ""))

        code, out, _ = self.run_cli("task")
        self.assertEqual((code, out), (0, f"task TODO-001\nreview {key}, no batches yet\n"))

        code, out, err = self.run_cli("ledger", "open", "fix/other", "--summary", "The other branch.")
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("reviews/"))
        self.assertEqual(err.strip(), f"the review is not the current work, the work is on {key}, no batches yet, finish it or drop it with brain task off first")


class Planning(CliCase):
    PROSE = "## Why\n\nThe rest of phase 7.\n\n## Steps\n\n1. TODO-001, auto ack.\n2. The owner's own run.\n\n## Done when\n\nAll of it.\n"

    @unittest.skipUnless(os.name == "posix" and os.geteuid() != 0, "needs a folder the process cannot write")
    def test_a_write_that_lands_before_a_second_write_fails_exits_as_done_with_the_failure_on_standard_error(self) -> None:
        self.run_cli("new", "plan", "--title", "Kit phase 7", "--description", "the rest of phase 7", stdin="## Why\n\nWhy.\n\n## Steps\n\n1. TODO-001, the export.\n\n## Done when\n\nDone.\n")
        self.run_cli("new", "plan", "--title", "Named", "--description", "names the todo", stdin="## Why\n\nWhy.\n\n## Steps\n\n1. TODO-001, by name.\n\n## Done when\n\nDone.\n")
        named = self.root / "plans" / "named.md"
        named.write_text(named.read_text(encoding="utf-8").replace("| TODO-001 |", "| TODO-001-export-to-csv |"), encoding="utf-8")
        (self.root / "todos" / "INDEX.md").write_text("stale\n", encoding="utf-8")
        (self.root / "plans" / "INDEX.md").write_text("stale\n", encoding="utf-8")
        folder = self.root / "plans"
        mode = folder.stat().st_mode
        os.chmod(folder, 0o555)
        try:
            code, out, err = self.run_cli("refile", "TODO-001", "--title", "Export it")

            self.assertEqual((code, out), (0, ""))
            self.assertIn("plans/INDEX.md", err.splitlines()[0])
            self.assertTrue(err.splitlines()[1].startswith("TODO-001-export-it was written, and a step that named it could not follow"), err)
            todos_index = (self.root / "todos" / "INDEX.md").read_text(encoding="utf-8")
            self.assertIn("(TODO-001-export-it.md)", todos_index)
            self.assertNotIn("stale", todos_index)

            code, out, err = self.run_cli("set", "TODO-001", "status=done", "done=2026-09-25")

            self.assertEqual((code, out), (0, ""))
            self.assertIn("plans/INDEX.md", err.splitlines()[0])
            self.assertTrue(err.splitlines()[1].startswith("TODO-001-export-it was written, and a step that points at it could not be refreshed"), err)
            self.assertIn("status: done", (self.root / "todos" / "TODO-001-export-it.md").read_text(encoding="utf-8"))
            self.assertIn("Nothing open.", (self.root / "todos" / "INDEX.md").read_text(encoding="utf-8"))
        finally:
            os.chmod(folder, mode)

        self.assertEqual((self.root / "plans" / "INDEX.md").read_text(encoding="utf-8"), "stale\n")

    @unittest.skipUnless(os.name == "posix" and os.geteuid() != 0, "needs a folder the process cannot write")
    def test_a_partial_write_rebuilds_the_indexes_after_the_one_it_cannot_write(self) -> None:
        self.run_cli("steps", "BUG-002", "add", "TODO-001, the export first.")
        (self.root / "bugs" / "INDEX.md").write_text("stale\n", encoding="utf-8")
        folder = self.root / "bugs"
        mode = folder.stat().st_mode
        os.chmod(folder, 0o555)
        try:
            code, out, err = self.run_cli("set", "TODO-001", "status=done", "done=2026-09-25")
        finally:
            os.chmod(folder, mode)

        self.assertEqual((code, out), (0, ""))
        self.assertIn("bugs/INDEX.md", err.splitlines()[0])
        self.assertTrue(err.splitlines()[1].startswith("TODO-001-export-to-csv was written, and a step that points at it could not be refreshed"), err)
        self.assertIn("Nothing open.", (self.root / "todos" / "INDEX.md").read_text(encoding="utf-8"))
        self.assertEqual((self.root / "bugs" / "INDEX.md").read_text(encoding="utf-8"), "stale\n")

    def test_a_plan_is_filed_from_its_prose_and_shown_with_its_marks(self) -> None:
        code, out, err = self.run_cli("new", "plan", "--title", "Kit phase 7", "--description", "the rest of phase 7", stdin=self.PROSE)

        self.assertEqual((code, out.strip(), err), (0, "plans/kit-phase-7.md", ""))
        self.assertIn("- [kit-phase-7](kit-phase-7.md), the rest of phase 7", (self.root / "plans" / "INDEX.md").read_text(encoding="utf-8"))

        code, out, _ = self.run_cli("steps", "kit-phase-7")
        self.assertEqual(code, 0)
        self.assertEqual(out, "kit-phase-7, open, 0 of 2 steps settled\n1. auto ack., TODO-001, pending\n2. The owner's own run., pending\n")

    def test_a_plan_files_with_its_links_and_category_and_refuses_a_field_it_does_not_carry(self) -> None:
        code, out, err = self.run_cli("new", "plan", "--title", "Kit phase 8", "--description", "the next", "--link", "todos/TODO-001-export-to-csv.md", "--category", "kits", "--by", "agent", stdin=self.PROSE)

        self.assertEqual((code, out.strip(), err), (0, "plans/kits/kit-phase-8.md", ""))
        text = (self.root / "plans" / "kits" / "kit-phase-8.md").read_text(encoding="utf-8")
        self.assertIn("todos/TODO-001-export-to-csv.md", text)
        self.assertIn("by: agent", text)

        code, out, err = self.run_cli("new", "plan", "--title", "Kit phase 9", "--description", "the next", "--severity", "bug", stdin=self.PROSE)

        self.assertEqual((code, out), (brain.REFUSED, ""))
        self.assertEqual(err.strip(), "plan files carry no 'severity' field")

    def test_no_sub_action_sets_a_default_its_parent_sets_too(self) -> None:
        # Python 3.9.8 and 3.10.0 let a parent parser's default win over a sub action's for the same name, so the two never share one
        parsers = [brain.build_parser()]
        while parsers:
            parent = parsers.pop()
            for action in parent._actions:
                if not isinstance(action, argparse._SubParsersAction):
                    continue

                for name, child in action.choices.items():
                    with self.subTest(command=name):
                        self.assertEqual(set(parent._defaults) & set(child._defaults), set())
                    parsers.append(child)

    def test_a_step_with_a_character_the_output_codec_lacks_is_printed_and_never_a_traceback(self) -> None:
        self.run_cli("new", "plan", "--title", "Kit phase 7", "--description", "the rest", stdin=self.PROSE)
        self.run_cli("steps", "kit-phase-7", "add", "Ship it → main.")
        out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")

        with contextlib.redirect_stdout(out):
            code = brain.main(["steps", "kit-phase-7"])
        out.flush()

        self.assertEqual(code, 0)
        self.assertIn(b"Ship it \\u2192 main.", out.buffer.getvalue())

    def test_steps_are_added_from_standard_input_or_the_line_given_and_a_plain_one_is_marked(self) -> None:
        self.run_cli("new", "plan", "--title", "Kit phase 7", "--description", "the rest", stdin=self.PROSE)

        code, out, _ = self.run_cli("steps", "kit-phase-7", "add", "-", "--by", "owner", stdin="BUG-002, the login loop\n\nthe six nits\n")
        self.assertEqual((code, out), (0, "3\n4\n"))

        code, out, _ = self.run_cli("steps", "kit-phase-7", "add", "one more")
        self.assertEqual((code, out), (0, "5\n"))

        code, _, _ = self.run_cli("steps", "kit-phase-7", "step", "2", "done")
        self.assertEqual(code, 0)

        _, out, _ = self.run_cli("steps", "kit-phase-7")
        self.assertIn("2. The owner's own run., done", out)
        self.assertIn("3. the login loop, BUG-002, pending", out)
        self.assertIn("5. one more, pending", out)

    def test_a_todo_takes_steps_the_same_way_listed_on_filing_or_added_after(self) -> None:
        prose = "## What\n\nThe export.\n\n## Steps\n\n1. Read the table.\n2. Write the file.\n\n## Done when\n\nIt opens.\n"
        code, out, _ = self.run_cli("new", "todo", "--title", "Export again", "--description", "the export", "--area", "reports", "--by", "owner", stdin=prose)
        self.assertEqual((code, out.strip()), (0, "todos/TODO-004-export-again.md"))
        self.assertIn("| 1 | Read the table. | | owner | pending |\n| 2 | Write the file. | | owner | pending |", (self.root / "todos" / "TODO-004-export-again.md").read_text(encoding="utf-8"))

        code, out, _ = self.run_cli("steps", "BUG-002", "add", "Find the loop.")
        self.assertEqual((code, out), (0, "1\n"))

        code, out, _ = self.run_cli("steps", "BUG-002")
        self.assertEqual((code, out), (0, "BUG-002-login-loop, open, 0 of 1 steps settled\n1. Find the loop., pending\n"))

    def test_a_pointed_step_marked_by_hand_and_an_empty_add_are_refused_in_words(self) -> None:
        self.run_cli("new", "plan", "--title", "Kit phase 7", "--description", "the rest", stdin=self.PROSE)

        code, out, err = self.run_cli("steps", "kit-phase-7", "step", "1", "done")
        self.assertEqual((code, out), (brain.REFUSED, ""))
        self.assertEqual(err.strip(), "step 1 points at TODO-001, so its mark follows that entry, and goes done once TODO-001 is done")

        code, out, err = self.run_cli("steps", "kit-phase-7", "add", "-", stdin="\n")
        self.assertEqual((code, out), (brain.REFUSED, ""))
        self.assertEqual(err.strip(), "no steps given, one per line on standard input, or one after add")

        code, out, err = self.run_cli("steps", "login", "add", "x")
        self.assertEqual((code, out), (brain.REFUSED, ""))
        self.assertEqual(err.strip(), "no plan, todo or bug answers to 'login'")

    def test_a_plan_with_no_prose_is_refused(self) -> None:
        code, out, err = self.run_cli("new", "plan", "--title", "Kit phase 7", "--description", "the rest")

        self.assertEqual((code, out), (brain.REFUSED, ""))
        self.assertEqual(err.strip(), "a plan needs its prose on standard input, the Why and the Done when at least")


class Writing(CliCase):
    def test_a_batch_change_without_its_number_is_refused(self) -> None:
        code, out, _ = self.run_cli("ledger", "open", "fix/login", "--summary", "The login branch.")
        self.assertEqual(code, 0)
        key = out.strip().split("/")[1]

        code, out, err = self.run_cli("ledger", "batch", key, "--status", "done")

        self.assertEqual((code, out), (brain.REFUSED, ""))
        self.assertEqual(err.strip(), "give the batch's number to set its status or ack")

        code, out, _ = self.run_cli("ledger", "batch", key, "--theme", "correctness")

        self.assertEqual((code, out.strip()), (0, "1"))

    def test_ledger_add_takes_claims_from_standard_input_as_written(self) -> None:
        _, out, _ = self.run_cli("ledger", "open", "fix/login", "--summary", "The login branch.")
        key = out.strip().split("/")[1]
        self.run_cli("ledger", "batch", key, "--theme", "correctness")

        code, out, _ = self.run_cli("ledger", "add", key, "-", "--severity", "bug", "--batch", "1", stdin="one `two`\n\n  $three four  \r\n")

        self.assertEqual((code, out.split()), (0, ["F-01", "F-02"]))
        text = (self.root / "reviews" / key / "ledger.md").read_text(encoding="utf-8")
        self.assertIn("### F-01: one `two`\n", text)
        self.assertIn("### F-02: $three four\n", text)
        self.assertEqual(text.count("- batch: 1\n"), 2)

        code, out, err = self.run_cli("ledger", "add", key, "-", "--severity", "bug")

        self.assertEqual((code, out), (brain.REFUSED, ""))
        self.assertEqual(err.strip(), "no claims on standard input, one per line")

    def test_ledger_set_changes_every_finding_named_before_the_changes_and_refuses_a_word_out_of_place(self) -> None:
        _, out, _ = self.run_cli("ledger", "open", "fix/login", "--summary", "The login branch.")
        key = out.strip().split("/")[1]
        self.run_cli("ledger", "add", key, "-", "--severity", "bug", stdin="one\ntwo\nthree\n")
        path = self.root / "reviews" / key / "ledger.md"

        code, out, err = self.run_cli(
            "ledger",
            "set",
            key,
            "F-01",
            "F-03",
            "status=fixed",
            "fix=abc1234",
        )

        self.assertEqual((code, out, err), (0, "", ""))
        text = path.read_text(encoding="utf-8")
        self.assertIn("### F-01: one\n\n- severity: bug\n- found-by: review\n- batch: \n- status: fixed\n- fix: abc1234\n", text)
        self.assertIn("### F-02: two\n\n- severity: bug\n- found-by: review\n- batch: \n- status: open\n- fix:\n", text)
        self.assertIn("### F-03: three\n\n- severity: bug\n- found-by: review\n- batch: \n- status: fixed\n- fix: abc1234\n", text)

        for words, said in (
            (("F-02",), "nothing to set, give a change like status=fixed"),
            (("status=wontfix", "reason=no"), "'status=wontfix' comes before any finding, name the findings first, like F-03 status=fixed"),
            (("F-02", "wontfix"), "'wontfix' is not field=value"),
            (("F-02", "F-09", "status=wontfix", "reason=no"), f"{key} has no finding F-09"),
            (("F-02", "status=wontfix", "reason=duplicates", "F-03"), "'F-03' comes after a change, name every finding before the changes"),
            (("F-01", "status=fixed", "fix=aaa1111", "F-02", "status=fixed", "fix=bbb2222"), "'F-02' comes after a change, name every finding before the changes"),
            (("F-02", "status=wontfix", "reason=one", "reason=two"), "'reason=two' sets reason a second time, give each field once"),
            (("f-02", "status=wontfix", "reason=no"), "'f-02' names a finding in lower case, write it F-02"),
        ):
            with self.subTest(words=words):
                code, out, err = self.run_cli("ledger", "set", key, *words)

                self.assertEqual((code, out), (brain.REFUSED, ""))
                self.assertEqual(err.strip(), said)
                self.assertEqual(path.read_text(encoding="utf-8"), text)

        for words in (
            ("F-02", "status=wontfix", "reason=no"),
            ("f-02", "F-03", "status=wontfix", "reason=no"),
        ):
            with self.subTest(words=words):
                code, out, err = self.run_cli(
                    "ledger",
                    "set",
                    *words,
                )

                self.assertEqual((code, out), (brain.REFUSED, ""))
                self.assertEqual(err.strip(), f"'{words[0]}' names a finding, give the ledger's key before the findings")
                self.assertEqual(path.read_text(encoding="utf-8"), text)

    def test_standard_input_that_is_not_utf8_is_refused_not_a_traceback(self) -> None:
        _, out, _ = self.run_cli("ledger", "open", "fix/login", "--summary", "The login branch.")
        key = out.strip().split("/")[1]

        for argv in (("ledger", "add", key, "-", "--severity", "bug"), ("ledger", "open", "fix/other", "--summary", "-")):
            with self.subTest(argv=argv):
                code, out, err = self.run_cli(*argv, raw=b"caf\xe9 claim\n")

                self.assertEqual((code, out), (brain.REFUSED, ""))
                self.assertEqual(err.strip(), "standard input is not UTF-8 at byte 3")

    def test_an_argument_that_is_not_utf8_is_refused_not_a_traceback(self) -> None:
        code, out, err = self.run_cli("ledger", "open", "fix/login", "--summary", "caf\udce9 claim")

        self.assertEqual((code, out), (brain.REFUSED, ""))
        self.assertEqual(err.strip(), "an argument is not UTF-8, 'caf\\udce9 claim'")

    def test_ledger_open_takes_the_summary_from_standard_input_as_written(self) -> None:
        code, out, _ = self.run_cli("ledger", "open", "fix/login", "--summary", "-", stdin="adds `brain ledger report` and $HOME handling\n")
        key = out.strip().split("/")[1]

        self.assertEqual(code, 0)
        text = (self.root / "reviews" / key / "ledger.md").read_text(encoding="utf-8")
        self.assertIn("## Summary\n\nadds `brain ledger report` and $HOME handling\n", text)

        for argv, stdin, said in (
            (("--summary", "-"), "", "the summary is empty"),
            (("--summary", "   "), "", "the summary is empty"),
            (("--summary", "-"), "adds the ledger\n## Batches\n", "the summary is one line"),
            (("--summary", "-"), "## Batches\n", "the summary is prose, not a heading"),
        ):
            with self.subTest(argv=argv, stdin=stdin):
                code, out, err = self.run_cli("ledger", "open", "fix/other", *argv, stdin=stdin)

                self.assertEqual((code, out), (brain.REFUSED, ""))
                self.assertEqual(err.strip(), said)

    def test_ack_sets_the_standing_word_with_an_end_and_takes_it_back(self) -> None:
        code, out, _ = self.run_cli("ack")

        self.assertEqual(code, 0)
        self.assertRegex(out, r"^now: .+\nauto ack off\n$")

        code, out, err = self.run_cli("ack", "on")

        self.assertEqual((code, out), (brain.REFUSED, ""))
        self.assertEqual(err.strip(), "auto ack needs an end, a time with --for or --until, a task with --until-done, or both")

        for flag, value in (("--for", "2h"), ("--until", "22:00"), ("--until-done", "TODO-001")):
            with self.subTest(flag=flag):
                code, out, err = self.run_cli("ack", flag, value)

                self.assertEqual((code, out), (brain.REFUSED, ""))
                self.assertEqual(err.strip(), "give on to set the word, its end goes with on")

        code, out, _ = self.run_cli("ack", "on", "--for", "2h", "--until-done", "TODO-001")

        self.assertEqual(code, 0)
        self.assertRegex(out, r"\nauto ack on, (2 h|1 h 59 min) left, until \d{2}:\d{2} (today|tomorrow), or until TODO-001 is done\n$")

        code, out, _ = self.run_cli("ack", "on", "--until", "2099-01-01 09:30")

        self.assertEqual(code, 0)
        self.assertRegex(out, r"\nauto ack on, \d+ h(?: \d+ min)? left, until 2099-01-01 09:30\n$")

        for flag, value in (("--for", "2h"), ("--until", "22:00"), ("--until-done", "TODO-001"), ("--for", "")):
            with self.subTest(flag=flag, value=value):
                code, out, err = self.run_cli("ack", "off", flag, value)

                self.assertEqual((code, out), (brain.REFUSED, ""))
                self.assertEqual(err.strip(), "off takes no end, give it alone")
                _, out, _ = self.run_cli("ack")
                self.assertRegex(out, r"\nauto ack on, ")

        code, out, err = self.run_cli("ack", "on", "--for", "", "--until-done", "TODO-001")

        self.assertEqual((code, out), (brain.REFUSED, ""))
        self.assertEqual(err.strip(), "'' is not a duration like 8h, 90m or 2d")

        code, out, _ = self.run_cli("ack", "on", "--until-done", ".agents/reviews/2026-09-feat-billing/ledger.md")

        self.assertEqual(code, 0)
        self.assertRegex(out, r"\nauto ack on until 2026-09-feat-billing is done\n$")

        code, out, err = self.run_cli("ack", "on", "--until-done", f"{self.root.parent.name}/.agents/reviews/2026-09-feat-billing/ledger.md")

        self.assertEqual((code, err), (0, ""))
        self.assertRegex(out, r"\nauto ack on until 2026-09-feat-billing is done\n$")

        code, out, err = self.run_cli("ack", "on", "--until-done", "../other/.agents/reviews/2026-09-feat-billing/ledger.md")

        self.assertEqual((code, out), (brain.REFUSED, ""))
        self.assertEqual(err.strip(), "nothing in the brain answers to '../other/.agents/reviews/2026-09-feat-billing/ledger.md', so auto ack cannot end with it")

        code, out, _ = self.run_cli("ack", "off")

        self.assertEqual(code, 0)
        self.assertTrue(out.endswith("auto ack off\n"))

    def test_a_batch_starts_on_the_standing_word_and_records_it_as_auto(self) -> None:
        _, out, _ = self.run_cli("ledger", "open", "fix/login", "--summary", "The login branch.")
        key = out.strip().split("/")[1]
        self.run_cli("ledger", "batch", key, "--theme", "correctness")
        self.run_cli("ledger", "add", key, "one", "--severity", "bug", "--batch", "1")

        code, _, err = self.run_cli("ledger", "batch", key, "1", "--status", "fixing", "--ack", "auto")

        self.assertEqual(code, brain.REFUSED)
        self.assertEqual(err.strip(), "auto ack is off, the owner's word starts a batch")

        self.run_cli("ack", "on", "--for", "1h")
        code, _, _ = self.run_cli("ledger", "batch", key, "1", "--status", "fixing", "--ack", "auto")

        self.assertEqual(code, 0)
        text = (self.root / "reviews" / key / "ledger.md").read_text(encoding="utf-8")
        self.assertRegex(text, r"\| 1 \| correctness \| fixing \| auto \d{4}-\d{2}-\d{2} \|")

    def test_ledger_report_prints_the_state(self) -> None:
        _, out, _ = self.run_cli("ledger", "open", "fix/login", "--summary", "The login branch.", "--level", "low")
        key = out.strip().split("/")[1]

        code, out, _ = self.run_cli("ledger", "report", key)

        self.assertEqual(code, 0)
        self.assertEqual(out, f"{key}, open, 0 batches, 0 findings\nnext: close the ledger\n")
        text = (self.root / "reviews" / key / "ledger.md").read_text(encoding="utf-8")
        self.assertIn("\nlevel: low\n", text)

    def test_new_files_the_prose_from_standard_input(self) -> None:
        body = "## What\n\nExport it, with a $dollar and `code`.\n\n## Done when\n\nIt opens.\n"

        code, out, _ = self.run_cli(
            "new", "todo",
            "--title", "Export the ledger",
            "--description", "export the ledger to csv",
            "--area", "reports",
            "--by", "owner",
            stdin=body,
        )

        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "todos/TODO-004-export-the-ledger.md")
        written = (self.root / "todos" / "TODO-004-export-the-ledger.md").read_text(encoding="utf-8")
        self.assertIn("Export it, with a $dollar and `code`.", written)

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root ignores file modes")
    def test_a_write_still_lands_and_indexes_when_the_skills_folder_cannot_be_listed(self) -> None:
        skills = self.root / "skills"
        skills.chmod(0o111)

        try:
            code, out, err = self.run_cli("new", "todo", "--title", "Probe it", "--description", "a probe", "--area", "a", "--by", "owner")
        finally:
            skills.chmod(0o755)

        self.assertEqual((code, err), (0, ""))
        self.assertTrue((self.root / out.strip()).is_file())
        self.assertIn("TODO-004-probe-it.md", (self.root / "todos" / "INDEX.md").read_text(encoding="utf-8"))

    def test_standard_input_with_windows_line_breaks_reads_as_lines(self) -> None:
        body = b"## What\r\n\r\nExport it.\r\n\r\n## Done when\r\n\r\nIt opens.\r\n"

        code, out, err = self.run_cli("new", "todo", "--title", "Export it", "--description", "export", "--area", "a", "--by", "owner", raw=body)

        self.assertEqual((code, err), (0, ""))
        written = (self.root / out.strip()).read_text(encoding="utf-8")
        self.assertIn("## Done when\n\nIt opens.\n", written)
        self.assertNotIn("\r", written)

    def test_new_with_standard_input_closed_files_the_headings(self) -> None:
        out, err = io.StringIO(), io.StringIO()

        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with mock.patch("sys.stdin", None):
                code = brain.main(["new", "todo", "--title", "Export", "--description", "x", "--area", "a", "--by", "owner"])

        self.assertEqual(code, 0)
        self.assertIn("## Done when", (self.root / "todos" / "TODO-004-export.md").read_text(encoding="utf-8"))

    def test_new_rebuilds_the_indexes(self) -> None:
        self.run_cli("new", "todo", "--title", "Export", "--description", "x", "--area", "a", "--by", "owner")

        self.assertIn("TODO-004", (self.root / "todos" / "INDEX.md").read_text(encoding="utf-8"))

    def test_a_category_cannot_climb_out_of_the_brain(self) -> None:
        code, out, err = self.run_cli("new", "doc", "--title", "Escaped", "--description", "x", "--category", "../../escaped")

        self.assertEqual((code, out), (1, ""))
        self.assertIn("is not a category", err)
        self.assertFalse((self.root.parent / "escaped").exists())

    def test_new_gets_through_a_broken_index_and_rebuilds_it(self) -> None:
        target = self.root / "bugs" / "INDEX.md"
        target.write_text("no fence at all\n", encoding="utf-8")

        code, out, _ = self.run_cli(
            "new", "bug",
            "--title", "Search flashes",
            "--description", "results flash on first paint",
            "--severity", "nit",
            "--area", "search",
        )

        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "bugs/BUG-003-search-flashes.md")
        self.assertIn("BUG-003", target.read_text(encoding="utf-8"))
        self.assertEqual(self.run_cli("doctor")[0], 0)

    def test_set_refuses_a_bad_pair(self) -> None:
        code, _, err = self.run_cli("set", "TODO-001", "status")

        self.assertEqual(code, 1)
        self.assertIn("field=value", err)

    def test_refile_wants_exactly_one_of_title_or_kind(self) -> None:
        code, _, err = self.run_cli("refile", "TODO-001")

        self.assertEqual(code, 1)
        self.assertIn("one of the two", err)

    def test_a_refusal_from_the_library_comes_out_on_stderr(self) -> None:
        code, _, err = self.run_cli("set", "TODO-001", "status=done")

        self.assertEqual(code, 1)
        self.assertIn("done=YYYY-MM-DD", err)


if __name__ == "__main__":
    unittest.main()
