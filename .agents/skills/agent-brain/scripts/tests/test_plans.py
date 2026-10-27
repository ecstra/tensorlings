"""The steps table on a plan, a todo or a bug, never renumbered, marked from what each step points at."""

import os
import time
import unittest
from unittest import mock

from brainlib import doctor, index, ledger, plans, schema, sections, steps, task, writes
from tests import fixtures

PROSE = """## Why

The rest of phase 7, the four items still open.

## Steps

1. TODO-001, auto ack and the hook line.
2. TODO-003, compaction and the current task line.
3. An independent pass over main, the owner's run.

## Done when

Every step is done and the kit is stamped 1.4.0.
"""

TODO_PROSE = """## What

Export the table to csv.

## Steps

1. Read the table.
2. BUG-002, the login loop first.
3. Write the file.

## Done when

The file opens in a spreadsheet.
"""


class Filing(fixtures.BrainCase):
    def test_a_plan_is_filed_with_its_listed_steps_as_the_table(self) -> None:
        entry = plans.create(brain=self.brain, title="Kit phase 7", description="the rest of phase 7", body=PROSE, stamp="2026-09-24")

        self.assertEqual(entry.relative, "plans/kit-phase-7.md")
        self.assertEqual((entry.field("status"), entry.field("by"), entry.field("born")), ("open", "owner", "2026-09-24"))
        rows = steps.table(entry.document.body)
        self.assertEqual([(step.number, step.text, step.points_at, step.status) for step in rows], [
            (1, "auto ack and the hook line.", "TODO-001", "pending"),
            (2, "compaction and the current task line.", "TODO-003", "done"),
            (3, "An independent pass over main, the owner's run.", "", "pending"),
        ])
        self.assertIn("## Why\n\nThe rest of phase 7", entry.document.body)
        self.assertTrue(entry.document.body.rstrip().endswith("stamped 1.4.0."))

    def test_a_plan_with_no_steps_is_a_goal_to_find_out(self) -> None:
        body = "## Why\n\nA goal.\n\n## Done when\n\nThe goal is met.\n"

        entry = plans.create(brain=self.brain, title="A goal", description="a goal alone", body=body, values={"by": "agent"})

        self.assertEqual(steps.table(entry.document.body), [])
        self.assertEqual(entry.field("by"), "agent")
        self.assertIn("## Steps\n\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n\n## Done when", entry.document.body)

        closed = writes.update(brain=self.brain, key="a-goal", changes={"status": "done", "done": "2026-09-24"})
        self.assertEqual(closed.field("status"), "done")

    def test_a_step_that_points_at_nothing_the_brain_knows_is_refused(self) -> None:
        body = PROSE.replace("TODO-003, compaction", "TODO-099, compaction")

        with self.assertRaises(steps.StepsError) as caught:
            plans.create(brain=self.brain, title="Kit phase 7", description="the rest", body=body)

        self.assertEqual(str(caught.exception), "step 2 points at 'TODO-099', which is not a todo, a bug or a review the brain knows")

    def test_a_step_may_point_at_a_bug_or_a_review_and_a_word_of_prose_is_not_a_pointer(self) -> None:
        body = PROSE.replace(
            "1. TODO-001, auto ack and the hook line.\n2. TODO-003, compaction and the current task line.\n3. An independent pass over main, the owner's run.",
            "1. BUG-002, the login loop.\n2. 2026-09-feat-billing, the billing review.\n3. Later, the docs.\n4. TODO-003, compaction.\n5. An independent pass over main, the owner's run.",
        )

        entry = plans.create(brain=self.brain, title="Mixed", description="mixed pointers", body=body)

        pointers = [step.points_at for step in steps.table(entry.document.body)]
        self.assertEqual(pointers, ["BUG-002", "2026-09-feat-billing", "", "TODO-003", ""])

    def test_the_prose_without_a_done_when_or_with_no_prose_is_refused(self) -> None:
        with self.assertRaises(plans.PlanError):
            plans.create(brain=self.brain, title="Bare", description="bare", body="")
        with self.assertRaises(writes.WriteError) as caught:
            plans.create(brain=self.brain, title="Bare", description="bare", body="## Why\n\nNo close.\n")
        self.assertEqual(str(caught.exception), "the body is missing the Done when heading")

    def test_a_todo_filed_with_listed_steps_carries_the_same_table_set_by_whoever_filed_it(self) -> None:
        entry = writes.create(
            brain=self.brain,
            kind=schema.kind_named("todo"),
            title="Export to csv again",
            description="the export",
            values={"area": "reports", "by": "owner"},
            body=TODO_PROSE,
        )

        rows = steps.rows_of(brain=self.brain, entry=entry)
        self.assertEqual([(step.number, step.text, step.points_at, step.by, step.status) for step in rows], [
            (1, "Read the table.", "", "owner", "pending"),
            (2, "the login loop first.", "BUG-002", "owner", "pending"),
            (3, "Write the file.", "", "owner", "pending"),
        ])
        self.assertIn("## Steps\n\n| # | Step | Points at | By | Status |", entry.document.body)
        self.assertTrue(entry.document.body.rstrip().endswith("spreadsheet."))

    def test_steps_under_a_heading_the_readers_would_not_find_are_refused_rather_than_left_as_prose(self) -> None:
        for heading in ("### Steps", "## Steps:", "# STEPS"):
            with self.subTest(heading=heading):
                body = PROSE.replace("## Steps", heading)

                with self.assertRaises(steps.StepsError) as caught:
                    plans.create(brain=self.brain, title="Misplaced", description="misplaced", body=body)

                self.assertEqual(str(caught.exception), f"the steps go under a level two heading, ## Steps, and {heading!r} is not one")

        body = PROSE.replace("## Steps", "## Not the steps") + "\n```\n### Steps\n```\n"
        entry = plans.create(brain=self.brain, title="Fenced", description="fenced", body=body)
        self.assertEqual(steps.rows_of(brain=self.brain, entry=entry), [])

        body = "## Problem\n\nLoops.\n\n## Evidence\n\n### Steps\n\n1. open\n2. click\n"
        entry = writes.create(brain=self.brain, kind=schema.kind_named("bug"), title="Repro", description="repro", values={"severity": "bug", "area": "auth"}, body=body)
        self.assertEqual(steps.rows_of(brain=self.brain, entry=entry), [])
        self.assertIn("### Steps\n\n1. open\n2. click", entry.document.body)

    def test_a_wrapped_step_is_joined_and_any_other_line_under_steps_is_refused_rather_than_dropped(self) -> None:
        wrapped = TODO_PROSE.replace("1. Read the table.", "1. Read the table, every column of it,\n   and the header row.")
        entry = writes.create(brain=self.brain, kind=schema.kind_named("todo"), title="Wrapped", description="wrapped", values={"area": "reports", "by": "owner"}, body=wrapped)
        self.assertEqual(steps.rows_of(brain=self.brain, entry=entry)[0].text, "Read the table, every column of it, and the header row.")

        for text, line in (
            ("- Read the table.\n- Write the file.", "'- Read the table.'"),
            ("Read the table first, it drives the rest.\n\n1. Read the table.", "'Read the table first, it drives the rest.'"),
            ("1. Read the table.\n3. Write the file.", None),
        ):
            with self.subTest(text=text):
                body = TODO_PROSE.replace("1. Read the table.\n2. BUG-002, the login loop first.\n3. Write the file.", text)

                with self.assertRaises(steps.StepsError) as caught:
                    writes.create(brain=self.brain, kind=schema.kind_named("todo"), title="Loose", description="loose", values={"area": "reports", "by": "owner"}, body=body)

                if line is not None:
                    self.assertEqual(str(caught.exception), f"the Steps section holds a line that is not a numbered step, one per line with its wrapped lines indented under it: {line}")
                else:
                    self.assertEqual(str(caught.exception), "the steps are numbered 1 onward in order, and 3 sits where 2 should")

        self.assertIsNone(self.brain.find("loose"))

    def test_a_todo_or_a_bug_without_steps_carries_no_table_and_reads_as_none(self) -> None:
        self.assertEqual(steps.rows_of(brain=self.brain, entry=self.brain.find("TODO-001")), [])
        self.assertEqual(steps.rows_of(brain=self.brain, entry=self.brain.find("BUG-002")), [])
        self.assertNotIn("## Steps", self.brain.find("TODO-001").document.body)


class Growing(fixtures.BrainCase):
    def setUp(self) -> None:
        super().setUp()
        # filed on an earlier day, so a stamp a later write leaves is told from the one filing gave
        self.entry = plans.create(brain=self.brain, title="Kit phase 7", description="the rest of phase 7", body=PROSE, stamp="2026-09-01")

    def rows(self) -> list[steps.Step]:
        return steps.table(self.brain.find("kit-phase-7").document.body)

    def test_steps_are_appended_after_the_last_and_say_who_set_them(self) -> None:
        before = writes.today()
        numbers = steps.add(brain=self.brain, key="kit-phase-7", lines=["the six nits, one sweep branch", "BUG-002, the login loop"], by="agent")

        self.assertEqual(numbers, [4, 5])
        added = self.rows()[3:]
        self.assertEqual([(step.by, step.points_at, step.status) for step in added], [("agent", "", "pending"), ("agent", "BUG-002", "pending")])
        self.assertIn(self.brain.find("kit-phase-7").field("updated"), {before, writes.today()})

    def test_a_step_with_a_line_break_in_it_is_refused_before_it_splits_a_row(self) -> None:
        for text in ("Read the table.\nWrite the file.", "Read the table.\r\nWrite the file."):
            with self.subTest(text=text):
                with self.assertRaises(steps.StepsError) as caught:
                    steps.add(brain=self.brain, key="kit-phase-7", lines=[text])

                self.assertEqual(str(caught.exception), "step 4 is one line, and a line break would split its row")

        self.assertEqual(len(self.rows()), 3)

    def test_a_plain_step_is_marked_by_hand_and_a_pointed_one_is_not(self) -> None:
        steps.mark(brain=self.brain, key="kit-phase-7", number=3, status="done")
        self.assertEqual(self.rows()[2].status, "done")

        with self.assertRaises(steps.StepsError) as caught:
            steps.mark(brain=self.brain, key="kit-phase-7", number=1, status="done")
        self.assertEqual(str(caught.exception), "step 1 points at TODO-001, so its mark follows that entry, and goes done once TODO-001 is done")

        with self.assertRaises(steps.StepsError) as caught:
            steps.mark(brain=self.brain, key="kit-phase-7", number=9, status="done")
        self.assertEqual(str(caught.exception), "kit-phase-7 has 3 steps, and there is no step 9")

    def test_a_pointed_steps_refusal_says_what_moves_its_mark_for_the_mark_asked_and_the_entrys_state(self) -> None:
        review = ledger.open_ledger(brain=self.brain, branch="feat/x", summary="the x", stamp="2026-09-24")
        steps.add(brain=self.brain, key="kit-phase-7", lines=[f"{review.name}, the review.", "BUG-002, the loop."])
        fixtures.write(path=self.root / "todos" / "TODO-001-other-thing.md", text=(self.root / "todos" / "TODO-001-export-to-csv.md").read_text(encoding="utf-8").replace("TODO-001-export-to-csv", "TODO-001-other-thing"))
        path = self.entry.path
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace("| auto ack and the hook line. | TODO-001 |", "| auto ack and the hook line. | TODO-001-export-to-csv |"))
        self.brain.refresh()

        for number, status, refusal in (
            (1, "done", "step 1 points at TODO-001-export-to-csv, so its mark follows that entry, and goes done once TODO-001-export-to-csv is done"),
            (1, "skipped", "step 1 points at TODO-001-export-to-csv, so its mark follows that entry, and goes skipped once TODO-001-export-to-csv is dropped, or follows the entry filed in its place once TODO-001-export-to-csv is refiled"),
            (5, "skipped", "step 5 points at BUG-002, so its mark follows that entry, and follows the entry filed in its place once BUG-002 is refiled, a bug being never dropped"),
            (1, "pending", "step 1 points at TODO-001-export-to-csv, so its mark follows that entry, and is pending while TODO-001-export-to-csv is open"),
            (2, "done", "step 2 points at TODO-003, which is done, so its mark follows that entry and is done already"),
            (2, "skipped", "step 2 points at TODO-003, which is done, so its mark follows that entry and stands done, not skipped"),
            (2, "pending", "step 2 points at TODO-003, which is done, so its mark follows that entry, and brain set TODO-003 status=open reopens it, unless its steps and another open entry's would wait on each other, or its table or one they lead to does not read, which the set refuses"),
            (4, "skipped", f"step 4 points at {review.name}, so its mark follows that entry, and a step at a review is never skipped"),
            (5, "done", "step 5 points at BUG-002, so its mark follows that entry, and goes done once BUG-002 is fixed or wontfix"),
        ):
            with self.subTest(number=number, status=status):
                with self.assertRaises(steps.StepsError) as caught:
                    steps.mark(brain=self.brain, key="kit-phase-7", number=number, status=status)
                self.assertEqual(str(caught.exception), refusal)

        bug = self.root / "bugs" / "BUG-002-login-loop.md"
        fixtures.write(path=bug, text=bug.read_text(encoding="utf-8").replace("status: open\n", "status: paused\n"))
        self.brain.refresh()
        with self.assertRaises(steps.StepsError) as caught:
            steps.mark(brain=self.brain, key="kit-phase-7", number=5, status="done")
        self.assertEqual(str(caught.exception), "step 5 points at BUG-002, whose status is not one the tool knows, which brain doctor reports, and its mark follows that entry")

    def test_a_pointed_step_takes_its_mark_from_the_entry_when_that_entry_changes(self) -> None:
        self.assertEqual([step.status for step in self.rows()], ["pending", "done", "pending"])

        writes.update(brain=self.brain, key="TODO-001", changes={"status": "done", "done": "2026-09-24"}, stamp="2026-09-24")
        self.assertEqual([step.status for step in self.rows()], ["done", "done", "pending"])
        self.assertEqual(self.brain.find("kit-phase-7").field("updated"), "2026-09-24")

        writes.update(brain=self.brain, key="TODO-001", changes={"status": "dropped"}, stamp="2026-09-24")
        self.assertEqual([step.status for step in self.rows()], ["skipped", "done", "pending"])

        writes.update(brain=self.brain, key="TODO-001", changes={"area": "kit"}, stamp="2026-09-25")
        self.assertEqual(self.brain.find("kit-phase-7").field("updated"), "2026-09-24")

    def test_a_plan_closes_only_once_every_step_is_settled(self) -> None:
        with self.assertRaises(writes.WriteError) as caught:
            writes.update(brain=self.brain, key="kit-phase-7", changes={"status": "done", "done": "2026-09-24"})
        self.assertEqual(str(caught.exception), "a plan goes done once every step is done or skipped, and one of its steps still stands")

        writes.update(brain=self.brain, key="TODO-001", changes={"status": "done", "done": "2026-09-24"})
        steps.mark(brain=self.brain, key="kit-phase-7", number=3, status="skipped")

        closed = writes.update(brain=self.brain, key="kit-phase-7", changes={"status": "done", "done": "2026-09-24"})
        self.assertEqual(closed.field("status"), "done")

        with self.assertRaises(steps.StepsError) as caught:
            steps.add(brain=self.brain, key="kit-phase-7", lines=["one more"])
        self.assertEqual(str(caught.exception), "kit-phase-7 is done, a step is added while it is open")

    def test_the_view_reads_the_table_back_in_lines(self) -> None:
        steps.mark(brain=self.brain, key="kit-phase-7", number=3, status="done")

        self.assertEqual(
            steps.view(brain=self.brain, key="kit-phase-7"),
            "kit-phase-7, open, 2 of 3 steps settled\n"
            "1. auto ack and the hook line., TODO-001, pending\n"
            "2. compaction and the current task line., TODO-003, done\n"
            "3. An independent pass over main, the owner's run., done",
        )
        self.assertEqual(steps.current_step(self.rows()), 1)

    def test_a_key_that_carries_no_steps_table_is_refused(self) -> None:
        with self.assertRaises(steps.StepsError) as caught:
            steps.add(brain=self.brain, key="login", lines=["x"])

        self.assertEqual(str(caught.exception), "no plan, todo or bug answers to 'login'")


class Closing(fixtures.BrainCase):
    def setUp(self) -> None:
        super().setUp()
        self.plan = plans.create(brain=self.brain, title="Kit phase 7", description="the rest of phase 7", body=PROSE, stamp="2026-09-24")

    def test_a_closed_entry_is_marked_no_more_and_reads_as_its_file_holds_it(self) -> None:
        writes.update(brain=self.brain, key="TODO-001", changes={"status": "done", "done": "2026-09-24"})
        steps.mark(brain=self.brain, key="kit-phase-7", number=3, status="skipped")
        writes.update(brain=self.brain, key="kit-phase-7", changes={"status": "done", "done": "2026-09-24"})

        with self.assertRaises(steps.StepsError) as caught:
            steps.mark(brain=self.brain, key="kit-phase-7", number=3, status="pending")
        self.assertEqual(str(caught.exception), "kit-phase-7 is done, a step is marked while it is open")

        writes.update(brain=self.brain, key="TODO-001", changes={"status": "open", "done": ""})
        self.assertEqual([step.status for step in steps.rows_of(brain=self.brain, entry=self.brain.find("kit-phase-7"))], ["done", "done", "skipped"])
        self.assertIn("| TODO-001 | owner | done |", self.plan.path.read_text(encoding="utf-8"))

    def test_a_close_settles_a_stale_file_and_a_second_done_is_no_close(self) -> None:
        path = self.plan.path
        writes.update(brain=self.brain, key="TODO-001", changes={"status": "done", "done": "2026-09-24"})
        steps.mark(brain=self.brain, key="kit-phase-7", number=3, status="skipped")
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace("| TODO-001 | owner | done |", "| TODO-001 | owner | pending |"))
        self.brain.refresh()

        writes.update(brain=self.brain, key="kit-phase-7", changes={"status": "done", "done": "2026-09-24"})
        self.assertIn("| TODO-001 | owner | done |", path.read_text(encoding="utf-8"))

        writes.update(brain=self.brain, key="TODO-001", changes={"status": "open", "done": ""})
        again = writes.update(brain=self.brain, key="kit-phase-7", changes={"status": "done", "description": "the rest, again"})
        self.assertEqual(again.field("description"), "the rest, again")

    def test_a_move_from_dropped_to_done_settles_the_file_too(self) -> None:
        steps.add(brain=self.brain, key="TODO-001", lines=["BUG-002, the loop first."])
        writes.update(brain=self.brain, key="TODO-001", changes={"status": "dropped", "done": "2026-09-24"})
        writes.update(brain=self.brain, key="BUG-002", changes={"status": "wontfix"})
        path = self.root / "todos" / "TODO-001-export-to-csv.md"
        self.assertIn("| BUG-002 | agent | pending |", path.read_text(encoding="utf-8"))

        writes.update(brain=self.brain, key="TODO-001", changes={"status": "done"})

        self.assertIn("| BUG-002 | agent | done |", path.read_text(encoding="utf-8"))
        self.assertEqual([finding.message for finding in doctor.run(self.brain) if finding.check == "steps"], [])

    def test_the_list_repair_on_a_closed_entry_settles_its_pointed_rows_too(self) -> None:
        path = self.root / "todos" / "TODO-003-dark-mode.md"
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace("## Done when", "## Steps\n\n1. BUG-002, the login loop first.\n2. Paint it.\n\n## Done when"))
        self.brain.refresh()

        doctor.run(brain=self.brain, fix=True)

        self.assertEqual([(step.points_at, step.status) for step in steps.rows_of(brain=self.brain, entry=self.brain.find("TODO-003"))], [("BUG-002", "done"), ("", "done")])
        self.assertEqual([finding.message for finding in doctor.run(self.brain) if finding.check == "steps"], [])

    def test_a_done_entry_with_a_pending_step_is_told_to_reopen_and_close_again(self) -> None:
        path = self.plan.path
        writes.update(brain=self.brain, key="TODO-001", changes={"status": "done", "done": "2026-09-24"})
        steps.mark(brain=self.brain, key="kit-phase-7", number=3, status="done")
        writes.update(brain=self.brain, key="kit-phase-7", changes={"status": "done", "done": "2026-09-24"})
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace("| 3 | An independent pass over main, the owner's run. | | owner | done |", "| 3 | An independent pass over main, the owner's run. | | owner | pending |"))
        self.brain.refresh()

        self.assertEqual([finding.message for finding in doctor.run(self.brain) if finding.check == "steps"], ["the entry is done with step 3 still pending, so reopen it and close it again once the step is settled, and when the set to open is refused for steps that do not read, mend them first"])

    def test_a_retitle_rewrites_a_name_cell_in_a_closed_holder_too(self) -> None:
        path = self.plan.path
        writes.update(brain=self.brain, key="kit-phase-7", changes={"status": "dropped", "done": "2026-09-24"})
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace("| TODO-001 | owner | pending |", "| TODO-001-export-to-csv | owner | pending |"))
        self.brain.refresh()

        writes.retitle(brain=self.brain, key="TODO-001", title="Export to tsv")

        self.assertIn("| TODO-001 | owner | pending |", path.read_text(encoding="utf-8"))

    def test_dropped_refiled_and_wontfix_close_with_pending_steps_unflagged(self) -> None:
        steps.add(brain=self.brain, key="TODO-001", lines=["one", "two"])
        writes.update(brain=self.brain, key="TODO-001", changes={"status": "dropped", "done": "2026-09-24"})
        steps.add(brain=self.brain, key="BUG-002", lines=["find it"])
        writes.update(brain=self.brain, key="BUG-002", changes={"status": "wontfix"})

        self.assertEqual([finding.message for finding in doctor.run(self.brain) if finding.check == "steps"], [])

    def test_an_entry_filed_closed_with_a_step_pending_is_refused(self) -> None:
        with self.assertRaises(writes.WriteError) as caught:
            writes.create(
                brain=self.brain,
                kind=schema.kind_named("bug"),
                title="Filed fixed",
                description="filed fixed",
                values={"severity": "bug", "area": "auth", "status": "fixed"},
                body="## Problem\n\nBroken.\n\n## Evidence\n\nA log line.\n\n## Steps\n\n1. Find it.\n\n## Fix\n\nThe cause, in commit abc1234, on 2026-09-24.\n",
            )
        self.assertEqual(str(caught.exception), "a bug goes fixed once every step is done or skipped, and one of its steps still stands")

    def test_every_rule_a_close_falls_short_of_is_named_at_once(self) -> None:
        with self.assertRaises(writes.WriteError) as dropped:
            writes.update(brain=self.brain, key="kit-phase-7", changes={"status": "dropped"})
        self.assertEqual(str(dropped.exception), "a plan goes dropped with its closing date, done=YYYY-MM-DD")

        with self.assertRaises(writes.WriteError) as caught:
            writes.update(brain=self.brain, key="kit-phase-7", changes={"status": "done"})

        self.assertEqual(
            str(caught.exception),
            "a plan goes done with its closing date, done=YYYY-MM-DD, and a plan goes done once every step is done or skipped, and one of its steps still stands",
        )

    @unittest.skipUnless(os.name == "posix" and os.geteuid() != 0, "needs a folder the process cannot write")
    def test_a_write_that_lands_before_a_pointing_step_fails_to_follow_is_said_and_never_undone(self) -> None:
        folder = self.root / "plans"
        mode = folder.stat().st_mode
        os.chmod(folder, 0o555)
        try:
            with self.assertRaises(writes.Partial) as caught:
                writes.convert(
                    brain=self.brain,
                    key="TODO-001",
                    kind=schema.kind_named("bug"),
                    values={"severity": "bug", "area": "auth"},
                    description="the export, as a bug",
                    body="## Problem\n\nBroken.\n\n## Evidence\n\nA log line.\n",
                )
        finally:
            os.chmod(folder, mode)

        self.assertTrue(str(caught.exception).startswith("bugs/BUG-003-export-to-csv.md was filed and TODO-001-export-to-csv was written, and a step that points at it could not be refreshed"))
        self.brain.refresh()
        self.assertEqual(self.brain.find("TODO-001").field("status"), "refiled")
        self.assertIsNotNone(self.brain.find("BUG-003"))


class OnATodo(fixtures.BrainCase):
    def rows(self) -> list[steps.Step]:
        return steps.rows_of(brain=self.brain, entry=self.brain.find("TODO-001"))

    def test_steps_added_to_a_todo_without_any_open_its_table_before_done_when(self) -> None:
        numbers = steps.add(brain=self.brain, key="TODO-001", lines=["Read the table.", "BUG-002, the login loop first."], by="owner")

        self.assertEqual(numbers, [1, 2])
        self.assertEqual([(step.text, step.points_at, step.by) for step in self.rows()], [("Read the table.", "", "owner"), ("the login loop first.", "BUG-002", "owner")])
        body = self.brain.find("TODO-001").document.body
        self.assertLess(body.index("## Steps"), body.index("## Done when"))
        self.assertEqual(steps.view(brain=self.brain, key="TODO-001").split("\n")[0], "TODO-001-export-to-csv, open, 0 of 2 steps settled")

        steps.add(brain=self.brain, key="BUG-002", lines=["Find the loop."])
        self.assertTrue(self.brain.find("BUG-002").document.body.rstrip().endswith("| 1 | Find the loop. | | agent | pending |"))

    def test_add_turns_a_list_written_by_hand_into_the_table_before_it_appends(self) -> None:
        path = self.root / "todos" / "TODO-001-export-to-csv.md"
        text = path.read_text(encoding="utf-8").replace("## Done when", "## Steps\n\n1. one\n2. BUG-002, two\n\n## Done when")
        fixtures.write(path=path, text=text)
        self.brain.refresh()

        numbers = steps.add(brain=self.brain, key="TODO-001", lines=["three"])

        self.assertEqual(numbers, [3])
        self.assertEqual([(step.text, step.points_at, step.by) for step in self.rows()], [("one", "", "owner"), ("two", "BUG-002", "owner"), ("three", "", "agent")])

    def test_a_closed_entry_with_a_list_under_steps_is_still_edited_and_only_its_close_is_refused(self) -> None:
        path = self.root / "todos" / "TODO-003-dark-mode.md"
        text = path.read_text(encoding="utf-8").replace("## Done when", "## Steps\n\n- pick colours\n\n## Done when")
        fixtures.write(path=path, text=text)
        self.brain.refresh()

        changed = writes.update(brain=self.brain, key="TODO-003", changes={"description": "a dark theme, both modes"})
        self.assertEqual(changed.field("description"), "a dark theme, both modes")

        writes.update(brain=self.brain, key="TODO-003", changes={"status": "open", "done": ""})
        with self.assertRaises(writes.WriteError):
            writes.update(brain=self.brain, key="TODO-003", changes={"status": "done", "done": "2026-09-24"})

    def test_a_todo_goes_done_only_once_its_steps_are_settled_and_one_without_steps_goes_freely(self) -> None:
        steps.add(brain=self.brain, key="TODO-001", lines=["Read the table.", "Write the file."])

        with self.assertRaises(writes.WriteError) as caught:
            writes.update(brain=self.brain, key="TODO-001", changes={"status": "done", "done": "2026-09-24"})
        self.assertEqual(str(caught.exception), "a todo goes done once every step is done or skipped, and one of its steps still stands")

        steps.mark(brain=self.brain, key="TODO-001", number=1, status="done")
        steps.mark(brain=self.brain, key="TODO-001", number=2, status="skipped")
        writes.update(brain=self.brain, key="TODO-001", changes={"status": "done", "done": "2026-09-24"})
        self.assertEqual(self.brain.find("TODO-001").field("status"), "done")

        loose = writes.create(brain=self.brain, kind=schema.kind_named("todo"), title="Loose", description="loose", values={"area": "reports", "by": "owner"}, body="## What\n\nWhat.\n\n## Done when\n\nShipped, and it was.\n")
        closed = writes.update(brain=self.brain, key=loose.identifier, changes={"status": "done", "done": "2026-09-24"})
        self.assertEqual(closed.field("status"), "done")

    def test_a_step_at_an_entry_refiled_with_no_successor_reads_skipped_and_an_empty_step_is_refused(self) -> None:
        steps.add(brain=self.brain, key="TODO-001", lines=["BUG-002, the login loop first."])
        bug = self.root / "bugs" / "BUG-002-login-loop.md"
        fixtures.write(path=bug, text=bug.read_text(encoding="utf-8").replace("status: open\n", "status: refiled\n"))
        self.brain.refresh()

        self.assertEqual([(step.points_at, step.status) for step in self.rows()], [("BUG-002", "skipped")])

        with self.assertRaises(steps.StepsError) as caught:
            steps.add(brain=self.brain, key="TODO-001", lines=["Write it.", "   "])
        self.assertEqual(str(caught.exception), "step 3 has no text")

    def test_a_plan_is_set_by_the_owner_or_the_agent_through_its_fields(self) -> None:
        with self.assertRaises(plans.PlanError) as caught:
            plans.create(brain=self.brain, title="Robotic", description="r", body="## Why\n\nW.\n\n## Done when\n\nD.\n", values={"by": "robot"})
        self.assertEqual(str(caught.exception), "a plan is set by owner or agent, not 'robot'")

        agentic = plans.create(brain=self.brain, title="Agentic", description="a", body="## Why\n\nW.\n\n## Done when\n\nD.\n", values={"by": "agent"})
        self.assertEqual(agentic.field("by"), "agent")

    def test_the_plans_index_lists_the_open_ones_first_by_status(self) -> None:
        plans.create(brain=self.brain, title="Zed done", description="done", body="## Why\n\nW.\n\n## Done when\n\nD.\n", stamp="2026-09-24")
        plans.create(brain=self.brain, title="Able open", description="open", body="## Why\n\nW.\n\n## Done when\n\nD.\n", stamp="2026-09-24")
        writes.update(brain=self.brain, key="zed-done", changes={"status": "done", "done": "2026-09-24"})
        index.write(brain=self.brain, stamp="2026-09-24")

        text = (self.root / "plans" / "INDEX.md").read_text(encoding="utf-8")
        self.assertLess(text.index("## Open"), text.index("## Done"))
        self.assertLess(text.index("## Open"), text.index("[able-open]"))
        self.assertLess(text.index("## Done"), text.index("[zed-done]"))
        self.assertLess(text.index("[able-open]"), text.index("## Done"))

        plans.create(brain=self.brain, title="Gone dropped", description="dropped", body="## Why\n\nW.\n\n## Done when\n\nD.\n", stamp="2026-09-24")
        writes.update(brain=self.brain, key="gone-dropped", changes={"status": "dropped", "done": "2026-09-24"})
        parked = plans.create(brain=self.brain, title="Odd parked", description="parked", body="## Why\n\nW.\n\n## Done when\n\nD.\n", stamp="2026-09-24")
        fixtures.write(path=parked.path, text=parked.path.read_text(encoding="utf-8").replace("status: open\n", "status: parked\n"))
        self.brain.refresh()
        index.write(brain=self.brain, stamp="2026-09-24")

        text = (self.root / "plans" / "INDEX.md").read_text(encoding="utf-8")
        self.assertLess(text.index("## Done"), text.index("## Dropped"))
        self.assertLess(text.index("## Dropped"), text.index("[gone-dropped]"))
        self.assertLess(text.index("[gone-dropped]"), text.index("## Unknown"))
        self.assertLess(text.index("## Unknown"), text.index("[odd-parked]"))

    def test_a_pointed_step_of_a_todo_follows_its_entry_and_a_self_pointer_is_refused(self) -> None:
        steps.add(brain=self.brain, key="TODO-001", lines=["BUG-002, the login loop first."])
        self.assertEqual(self.rows()[0].status, "pending")

        writes.update(brain=self.brain, key="BUG-002", changes={"status": "wontfix"}, stamp="2026-09-24")
        self.assertEqual(self.rows()[0].status, "done")

        with self.assertRaises(steps.StepsError) as caught:
            steps.add(brain=self.brain, key="TODO-001", lines=["TODO-001, myself."])
        self.assertEqual(str(caught.exception), "step 2 points at TODO-001-export-to-csv itself, and a step points at other work")


class Pointers(fixtures.BrainCase):
    def setUp(self) -> None:
        super().setUp()
        self.plan = plans.create(brain=self.brain, title="Kit phase 7", description="the rest of phase 7", body=PROSE, stamp="2026-09-24")

    def stored(self) -> list[steps.Step]:
        return steps.table(self.brain.find("kit-phase-7").document.body)

    def test_a_cell_naming_a_review_by_its_folder_is_refreshed_when_the_review_closes(self) -> None:
        review = ledger.open_ledger(brain=self.brain, branch="feat/task", summary="the task chain", stamp="2026-09-24")
        steps.add(brain=self.brain, key="kit-phase-7", lines=[f"{review.name}, the review."])
        fixtures.write(path=review.path, text=review.path.read_text(encoding="utf-8").replace(f"name: {review.name}\n", "name: task-pass\n"))
        self.brain.refresh()

        ledger.close(brain=self.brain, key="task-pass", stamp="2026-09-24")

        self.assertEqual((self.stored()[3].points_at, self.stored()[3].status), ("task-pass", "done"))

    def test_closing_a_review_with_the_ledger_marks_the_step_that_points_at_it(self) -> None:
        review = ledger.open_ledger(brain=self.brain, branch="feat/task", summary="the task chain", stamp="2026-09-24")
        steps.add(brain=self.brain, key="kit-phase-7", lines=[f"{review.name}, the review."])

        ledger.close(brain=self.brain, key=review.name, stamp="2026-09-24")

        self.assertEqual(self.stored()[3].status, "done")

        with self.assertRaises(steps.StepsError) as caught:
            steps.mark(brain=self.brain, key="kit-phase-7", number=4, status="pending")
        self.assertEqual(str(caught.exception), f"step 4 points at {review.name}, which is closed, so its mark follows that entry, and brain set {review.name} status=open reopens it")

    def test_a_mark_stale_in_the_file_is_read_off_its_entry_by_every_reader(self) -> None:
        path = self.plan.path
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace("| TODO-003 | owner | done |", "| TODO-003 | owner | pending |"))
        self.brain.refresh()

        self.assertEqual([step.status for step in self.stored()], ["pending", "pending", "pending"])
        self.assertEqual([step.status for step in steps.rows_of(brain=self.brain, entry=self.brain.find("kit-phase-7"))], ["pending", "done", "pending"])
        self.assertIn("2. compaction and the current task line., TODO-003, done", steps.view(brain=self.brain, key="kit-phase-7"))

        writes.update(brain=self.brain, key="TODO-001", changes={"status": "done", "done": "2026-09-24"})
        steps.mark(brain=self.brain, key="kit-phase-7", number=3, status="skipped")
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace("| TODO-003 | owner | done |", "| TODO-003 | owner | pending |"))
        self.brain.refresh()

        closed = writes.update(brain=self.brain, key="kit-phase-7", changes={"status": "done", "done": "2026-09-24"})
        self.assertEqual(closed.field("status"), "done")

    def test_a_pointer_is_stored_as_the_id_so_a_retitle_never_breaks_it(self) -> None:
        numbers = steps.add(brain=self.brain, key="kit-phase-7", lines=["TODO-001-export-to-csv, by name.", "todo-001: in lower case."])
        self.assertEqual(numbers, [4, 5])
        self.assertEqual([step.points_at for step in self.stored()[3:]], ["TODO-001", "TODO-001"])

        writes.retitle(brain=self.brain, key="TODO-001", title="Export to xlsx")
        writes.update(brain=self.brain, key="TODO-001", changes={"status": "done", "done": "2026-09-24"})

        self.assertEqual([step.status for step in self.stored()], ["done", "done", "pending", "done", "done"])

    def test_a_review_is_pointed_at_by_its_name_and_never_by_the_stem_every_ledger_shares(self) -> None:
        numbers = steps.add(brain=self.brain, key="kit-phase-7", lines=["2026-09-feat-billing, the billing review.", "ledger: add the ack column."])

        self.assertEqual(numbers, [4, 5])
        self.assertEqual([(step.points_at, step.text) for step in self.stored()[3:]], [("2026-09-feat-billing", "the billing review."), ("", "ledger: add the ack column.")])

    def test_a_refile_from_before_the_became_field_is_followed_through_the_links_that_name_each_other(self) -> None:
        steps.add(brain=self.brain, key="kit-phase-7", lines=["TODO-001, the export."])
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"
        bug = self.root / "bugs" / "BUG-002-login-loop.md"
        fixtures.write(path=todo, text=todo.read_text(encoding="utf-8").replace("status: open\n", "status: refiled\n").replace("links: []", "links: [bugs/BUG-002-login-loop.md]"))
        self.brain.refresh()

        self.assertIsNone(steps.refiled_to(brain=self.brain, entry=self.brain.find("TODO-001")))
        rows = steps.rows_of(brain=self.brain, entry=self.brain.find("kit-phase-7"))
        self.assertEqual((rows[3].points_at, rows[3].status), ("TODO-001", "skipped"))

        with self.assertRaises(steps.StepsError) as caught:
            steps.mark(brain=self.brain, key="kit-phase-7", number=4, status="pending")
        self.assertEqual(str(caught.exception), "step 4 points at TODO-001, which was refiled as nothing the brain knows, so its mark stands skipped")

        fixtures.write(path=self.plan.path, text=self.plan.path.read_text(encoding="utf-8").replace("| 4 | the export. | TODO-001 | agent | pending |", "| 4 | the export. | todo-001 | agent | pending |"))
        self.brain.refresh()

        self.assertEqual([finding.message for finding in doctor.run(self.brain) if finding.path == "plans/kit-phase-7.md" and "step 4" in finding.message], ["step 4 points at todo-001, which was refiled as nothing the brain knows, so the tool keeps the key TODO-001 and the step stands skipped"])

        fixtures.write(path=bug, text=bug.read_text(encoding="utf-8").replace("links: []", "links: [todos/TODO-001-export-to-csv.md]"))
        self.brain.refresh()

        self.assertEqual(steps.refiled_to(brain=self.brain, entry=self.brain.find("TODO-001")).identifier, "BUG-002")
        rows = steps.rows_of(brain=self.brain, entry=self.brain.find("kit-phase-7"))
        self.assertEqual((rows[3].points_at, rows[3].status), ("BUG-002", "pending"))

    def test_a_step_follows_a_todo_refiled_into_a_bug(self) -> None:
        bug = writes.convert(brain=self.brain, key="TODO-001", kind=schema.kind_named("bug"), values={"severity": "bug", "area": "auth"}, description="the export, as a bug", body="## Problem\n\nThe export breaks, and is not worth fixing.\n\n## Evidence\n\nA log line.\n")

        rows = self.stored()
        self.assertEqual((rows[0].points_at, rows[0].status), (bug.identifier, "pending"))

        writes.update(brain=self.brain, key=bug.identifier, changes={"status": "wontfix"})
        self.assertEqual(self.stored()[0].status, "done")

    def test_a_step_whose_hand_off_is_held_is_told_it_follows_the_successor_once_it_closes_and_doctor_names_the_successor(self) -> None:
        other = self.root / "todos" / "TODO-003-dark-mode.md"
        fixtures.write(path=other, text=other.read_text(encoding="utf-8").replace("status: done", "status: open"))
        self.brain.refresh()
        bug = writes.convert(brain=self.brain, key="TODO-001", kind=schema.kind_named("bug"), values={"severity": "bug", "area": "auth"}, description="the export, as a bug", body="## Problem\n\nBroken.\n\n## Steps\n\n1. TODO-003, the theme first.\n\n## Evidence\n\nA log line.\n")
        steps.add(brain=self.brain, key="TODO-003", lines=["TODO-001, the export."])

        rows = steps.rows_of(brain=self.brain, entry=self.brain.find("TODO-003"))
        self.assertEqual([(step.points_at, step.status) for step in rows], [("TODO-001", "skipped")])

        with self.assertRaises(steps.StepsError) as caught:
            steps.mark(brain=self.brain, key="TODO-003", number=1, status="pending")
        self.assertEqual(str(caught.exception), f"step 1 points at TODO-001, which was refiled as {bug.identifier}, whose steps lead back here, so its mark stands skipped until {bug.identifier} closes or its steps no longer lead back, and follows {bug.identifier} then")

        fixtures.write(path=other, text=other.read_text(encoding="utf-8").replace("| TODO-001 |", "| todo-001 |"))
        self.brain.refresh()

        self.assertEqual([finding.message for finding in doctor.run(self.brain) if finding.path == "todos/TODO-003-dark-mode.md" and finding.check == "steps"], [f"step 1 points at todo-001, which was refiled as {bug.identifier}, whose steps lead back here, so the tool keeps the key TODO-001 and the step stands skipped"])

        fixtures.write(path=bug.path, text=bug.path.read_text(encoding="utf-8").replace("| Points at |", "| Points At |"))
        self.brain.refresh()

        self.assertEqual([finding.message for finding in doctor.run(self.brain) if finding.path == "todos/TODO-003-dark-mode.md" and finding.check == "steps"], [f"step 1 points at todo-001, which was refiled as {bug.identifier}, whose steps do not read, so the tool keeps the key TODO-001 and the step stands skipped until they are put right"])

        with self.assertRaises(steps.StepsError) as caught:
            steps.mark(brain=self.brain, key="TODO-003", number=1, status="pending")
        self.assertEqual(str(caught.exception), f"step 1 points at TODO-001, which was refiled as {bug.identifier}, whose steps do not read, so its mark stands skipped until they are put right, which brain doctor names, and follows {bug.identifier} then")

        fixtures.write(path=bug.path, text=bug.path.read_text(encoding="utf-8").replace("| Points At |", "| Points at |"))
        self.brain.refresh()
        writes.retitle(brain=self.brain, key=bug.identifier, title="Export it", stamp="2026-09-25")
        again = writes.convert(brain=self.brain, key=bug.identifier, kind=schema.kind_named("todo"), values={"area": "reports", "by": "owner"}, description="the export, once more", body="## What\n\nWhat.\n\n## Done when\n\nDone.\n")

        self.assertEqual([finding.message for finding in doctor.run(self.brain) if finding.path == "todos/TODO-003-dark-mode.md" and finding.check == "steps"], [f"step 1 points at todo-001, which was refiled as {again.identifier}"])

        fixtures.write(path=again.path, text=again.path.read_text(encoding="utf-8").replace("status: open\n", f"status: refiled\nbecame: {bug.identifier}\n"))
        self.brain.refresh()

        self.assertEqual([finding.message for finding in doctor.run(self.brain) if finding.path == "todos/TODO-003-dark-mode.md" and finding.check == "steps"], [f"step 1 points at todo-001, which sits in a loop of refiles through {again.identifier} and {bug.identifier}, so the tool keeps the key {again.identifier} and the step stands skipped until the loop is undone"])

        with self.assertRaises(steps.StepsError) as caught:
            steps.mark(brain=self.brain, key="TODO-003", number=1, status="pending")
        self.assertEqual(str(caught.exception), f"step 1 points at {again.identifier}, which sits in a loop of refiles through {again.identifier} and {bug.identifier}, so its mark stands skipped until the loop is undone")

        fixtures.write(path=again.path, text=again.path.read_text(encoding="utf-8").replace(f"status: refiled\nbecame: {bug.identifier}\n", "status: open\n"))
        fixtures.write(path=self.root / "todos" / f"{again.identifier}-other-thing.md", text=again.path.read_text(encoding="utf-8").replace(again.name, f"{again.identifier}-other-thing"))
        self.brain.refresh()

        self.assertEqual([finding.message for finding in doctor.run(self.brain) if finding.path == "todos/TODO-003-dark-mode.md" and finding.check == "steps"], [f"step 1 points at todo-001, which was refiled as {again.identifier}, which two files carry, so the tool keeps the key {bug.identifier} and the step stands skipped while the id is doubled"])

    @unittest.skipUnless(os.name == "posix" and os.geteuid() != 0, "needs a folder the process cannot write")
    def test_a_retitle_that_lands_before_a_became_field_fails_to_follow_is_said_and_never_undone(self) -> None:
        bug = self.root / "bugs" / "BUG-002-login-loop.md"
        fixtures.write(path=bug, text=bug.read_text(encoding="utf-8").replace("status: open\n", "status: refiled\nbecame: TODO-001-export-to-csv\n"))
        self.brain.refresh()
        folder = self.root / "bugs"
        mode = folder.stat().st_mode
        os.chmod(folder, 0o555)
        try:
            with self.assertRaises(writes.Partial) as caught:
                writes.retitle(brain=self.brain, key="TODO-001", title="Export it", stamp="2026-09-25")
        finally:
            os.chmod(folder, mode)

        self.assertTrue(str(caught.exception).startswith("TODO-001-export-it was written, and a became field that named it could not follow"), str(caught.exception))
        self.brain.refresh()
        self.assertTrue((self.root / "todos" / "TODO-001-export-it.md").exists())
        self.assertEqual(self.brain.find("BUG-002").field("became"), "TODO-001-export-to-csv")

    def test_a_retitle_moves_a_became_field_that_named_the_file_by_its_name(self) -> None:
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"
        fixtures.write(path=todo, text=todo.read_text(encoding="utf-8").replace("status: open\n", "status: refiled\nbecame: BUG-002-login-loop\n").replace("links: []", "links: [bugs/BUG-002-login-loop.md]"))
        self.brain.refresh()

        writes.retitle(brain=self.brain, key="BUG-002", title="Login redirect", stamp="2026-09-25")

        self.assertEqual(self.brain.find("TODO-001").field("became"), "BUG-002")
        self.assertEqual(self.brain.find("TODO-001").field("updated"), "2026-09-25")
        self.assertEqual([finding.message for finding in doctor.run(self.brain) if "became" in finding.message], [])

        fixtures.write(path=self.root / "bugs" / "BUG-002-other-thing.md", text=(self.root / "bugs" / "BUG-002-login-redirect.md").read_text(encoding="utf-8").replace("BUG-002-login-redirect", "BUG-002-other-thing"))
        fixtures.write(path=todo, text=todo.read_text(encoding="utf-8").replace("became: BUG-002\n", "became: BUG-002-login-redirect\n"))
        self.brain.refresh()

        writes.retitle(brain=self.brain, key="bugs/BUG-002-login-redirect.md", title="Login bounce", stamp="2026-09-25")

        self.assertEqual(self.brain.find("TODO-001").field("became"), "BUG-002-login-bounce")

    @unittest.skipUnless(os.name == "posix" and os.geteuid() != 0, "needs a folder the process cannot write")
    def test_a_retitle_whose_step_cells_cannot_follow_still_moves_the_chain_and_the_became_field(self) -> None:
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"
        fixtures.write(path=self.root / "todos" / "export-to-csv.md", text=todo.read_text(encoding="utf-8").replace("TODO-001-export-to-csv", "export-to-csv"))
        bug = self.root / "bugs" / "BUG-002-login-loop.md"
        fixtures.write(path=bug, text=bug.read_text(encoding="utf-8").replace("status: open\n", "status: refiled\nbecame: export-to-csv\n"))
        path = self.plan.path
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace("| An independent pass over main, the owner's run. | | owner | pending |", "| An independent pass over main, the owner's run. | export-to-csv | owner | pending |"))
        self.brain.refresh()
        task.turn_on(brain=self.brain, key="export-to-csv")
        folder = self.root / "plans"
        mode = folder.stat().st_mode
        os.chmod(folder, 0o555)
        try:
            with self.assertRaises(writes.Partial) as caught:
                writes.retitle(brain=self.brain, key="todos/export-to-csv.md", title="Export to JSON", stamp="2026-09-25")
        finally:
            os.chmod(folder, mode)

        self.assertTrue(str(caught.exception).startswith("export-to-json was written, and a step that named it could not follow"), str(caught.exception))
        self.assertNotIn("the chain could not follow", str(caught.exception))
        self.brain.refresh()
        self.assertEqual(task.describe(self.brain), "task export-to-json")
        self.assertEqual(self.brain.find("BUG-002").field("became"), "export-to-json")
        self.assertEqual(self.stored()[2].points_at, "export-to-csv")

    @unittest.skipUnless(os.name == "posix" and os.geteuid() != 0, "needs a folder the process cannot write")
    def test_a_repoint_that_fails_on_one_holder_still_moves_the_cells_of_the_holders_after_it(self) -> None:
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"
        fixtures.write(path=self.root / "todos" / "export-to-csv.md", text=todo.read_text(encoding="utf-8").replace("TODO-001-export-to-csv", "export-to-csv"))
        bug = self.root / "bugs" / "BUG-002-login-loop.md"
        fixtures.write(path=bug, text=bug.read_text(encoding="utf-8").replace("## Evidence", "## Steps\n\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n| 1 | the export. | export-to-csv | agent | pending |\n\n## Evidence"))
        path = self.plan.path
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace("| An independent pass over main, the owner's run. | | owner | pending |", "| An independent pass over main, the owner's run. | export-to-csv | owner | pending |"))
        self.brain.refresh()
        folder = self.root / "plans"
        mode = folder.stat().st_mode
        os.chmod(folder, 0o555)
        try:
            with self.assertRaises(writes.Partial):
                writes.retitle(brain=self.brain, key="todos/export-to-csv.md", title="Export to JSON", stamp="2026-09-25")
        finally:
            os.chmod(folder, mode)

        self.brain.refresh()
        self.assertEqual(steps.stored_rows(self.brain.find("BUG-002"))[0].points_at, "export-to-json")
        self.assertEqual(self.stored()[2].points_at, "export-to-csv")

    @unittest.skipUnless(os.name == "posix" and os.geteuid() != 0, "needs a folder the process cannot write")
    def test_a_retitle_says_every_follow_up_write_that_failed_and_still_moves_the_chain(self) -> None:
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"
        fixtures.write(path=self.root / "todos" / "export-to-csv.md", text=todo.read_text(encoding="utf-8").replace("TODO-001-export-to-csv", "export-to-csv"))
        bug = self.root / "bugs" / "BUG-002-login-loop.md"
        fixtures.write(path=bug, text=bug.read_text(encoding="utf-8").replace("status: open\n", "status: refiled\nbecame: export-to-csv\n"))
        path = self.plan.path
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace("| An independent pass over main, the owner's run. | | owner | pending |", "| An independent pass over main, the owner's run. | export-to-csv | owner | pending |"))
        self.brain.refresh()
        task.turn_on(brain=self.brain, key="export-to-csv")
        folders = [self.root / "plans", self.root / "bugs"]
        modes = [folder.stat().st_mode for folder in folders]
        for folder in folders:
            os.chmod(folder, 0o555)
        try:
            with self.assertRaises(writes.Partial) as caught:
                writes.retitle(brain=self.brain, key="todos/export-to-csv.md", title="Export to JSON", stamp="2026-09-25")
        finally:
            for folder, mode in zip(folders, modes):
                os.chmod(folder, mode)

        message = str(caught.exception)
        self.assertTrue(message.startswith("export-to-json was written, and a step that named it could not follow"), message)
        self.assertIn(", and a became field that named it could not follow", message)
        self.assertNotIn("the chain could not follow", message)
        self.assertIsInstance(caught.exception.__cause__, OSError)
        self.brain.refresh()
        self.assertEqual(task.describe(self.brain), "task export-to-json")

    @unittest.skipUnless(os.name == "posix" and os.geteuid() != 0, "needs a folder the process cannot write")
    def test_a_refresh_that_fails_on_one_holder_still_refreshes_the_holders_after_it(self) -> None:
        steps.add(brain=self.brain, key="kit-phase-7", lines=["BUG-002, the loop."])
        steps.add(brain=self.brain, key="TODO-001", lines=["BUG-002, the loop."])
        folder = self.root / "plans"
        mode = folder.stat().st_mode
        os.chmod(folder, 0o555)
        try:
            with self.assertRaises(writes.Partial):
                writes.update(brain=self.brain, key="BUG-002", changes={"status": "wontfix"})
        finally:
            os.chmod(folder, mode)

        self.brain.refresh()
        self.assertEqual(steps.stored_rows(self.brain.find("TODO-001"))[0].status, "done")
        self.assertEqual(self.stored()[3].status, "pending")

    def test_a_retitle_of_a_file_with_no_id_moves_the_chain_that_holds_it_by_its_stem(self) -> None:
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"
        fixtures.write(path=self.root / "todos" / "export-to-csv.md", text=todo.read_text(encoding="utf-8").replace("TODO-001-export-to-csv", "export-to-csv"))
        self.brain.refresh()
        task.turn_on(brain=self.brain, key="export-to-csv")

        writes.retitle(brain=self.brain, key="todos/export-to-csv.md", title="Export to JSON", stamp="2026-09-25")

        self.assertEqual(task.describe(self.brain), "task export-to-json")

    def test_two_entries_pointing_at_each_other_are_refused_before_they_wait_on_each_other_for_good(self) -> None:
        steps.add(brain=self.brain, key="TODO-001", lines=["BUG-002, fix first."])

        with self.assertRaises(steps.StepsError) as caught:
            steps.add(brain=self.brain, key="BUG-002", lines=["TODO-001, export first."])
        self.assertEqual(str(caught.exception), "step 1 points at TODO-001, whose steps point back here, and the two would wait on each other for good")

        path = self.root / "bugs" / "BUG-002-login-loop.md"
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").rstrip("\n") + "\n\n## Steps\n\n1. TODO-001, export first.\n")
        self.brain.refresh()
        with self.assertRaises(steps.StepsError):
            steps.add(brain=self.brain, key="BUG-002", lines=["x"])

    def test_a_refile_whose_successor_would_point_back_at_what_points_at_it_is_refused(self) -> None:
        steps.add(brain=self.brain, key="TODO-001", lines=["BUG-002, the loop first."])

        with self.assertRaises(steps.StepsError) as caught:
            writes.convert(
                brain=self.brain,
                key="BUG-002",
                kind=schema.kind_named("todo"),
                values={"area": "auth", "by": "owner"},
                description="the loop, as a todo",
                body="## What\n\nThe loop.\n\n## Steps\n\n1. TODO-001, the export first.\n\n## Done when\n\nDone.\n",
            )
        self.assertEqual(str(caught.exception), "step 1 points at TODO-001, whose steps point back here, and the two would wait on each other for good")
        self.assertEqual(self.brain.find("BUG-002").field("status"), "open")
        self.assertEqual(steps.rows_of(brain=self.brain, entry=self.brain.find("TODO-001"))[0].status, "pending")

    def test_a_pointer_at_a_closed_entry_whose_steps_point_back_is_no_cycle_until_it_reopens(self) -> None:
        steps.add(brain=self.brain, key="BUG-002", lines=["TODO-001, export first."])
        writes.update(brain=self.brain, key="BUG-002", changes={"status": "wontfix"})

        numbers = steps.add(brain=self.brain, key="TODO-001", lines=["BUG-002, confirm the wontfix."])

        self.assertEqual(numbers, [1])
        self.assertEqual(steps.rows_of(brain=self.brain, entry=self.brain.find("TODO-001"))[0].status, "done")

        with self.assertRaises(writes.WriteError) as caught:
            writes.update(brain=self.brain, key="BUG-002", changes={"status": "open"})
        self.assertEqual(str(caught.exception), "reopening BUG-002-login-loop would leave it and another entry waiting on each other, step 1 points at TODO-001, whose steps point back here, and the two would wait on each other for good")

        path = self.root / "bugs" / "BUG-002-login-loop.md"
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace("status: wontfix", "status: open"))
        self.brain.refresh()
        found = [finding.message for finding in doctor.run(self.brain) if finding.check == "steps"]
        self.assertIn("step 1 points at TODO-001, whose steps point back here, and the two wait on each other for good", found)
        self.assertIn("step 1 points at BUG-002, whose steps point back here, and the two wait on each other for good", found)

    def test_a_refile_whose_new_body_would_close_a_loop_is_refused(self) -> None:
        steps.add(brain=self.brain, key="BUG-002", lines=["TODO-001, export first."])

        with self.assertRaises(steps.StepsError) as caught:
            writes.convert(
                brain=self.brain,
                key="TODO-001",
                kind=schema.kind_named("bug"),
                values={"severity": "bug", "area": "auth"},
                description="the export, as a bug",
                body="## Problem\n\nBroken.\n\n## Evidence\n\nA log line.\n\n## Steps\n\n1. BUG-002, the login loop first.\n",
            )
        self.assertEqual(str(caught.exception), "step 1 points at BUG-002, whose steps point back here, and the two would wait on each other for good")
        self.assertEqual(self.brain.find("TODO-001").field("status"), "open")

    def test_a_refiled_entry_hands_its_steps_to_the_entry_of_the_other_kind_alone(self) -> None:
        bug = writes.convert(brain=self.brain, key="TODO-001", kind=schema.kind_named("bug"), values={"severity": "bug", "area": "auth"}, description="the export, as a bug")
        self.assertEqual(self.brain.find("TODO-001").field("became"), bug.identifier)
        related = writes.create(brain=self.brain, kind=schema.kind_named("bug"), title="Csv encoding off", description="related", values={"severity": "bug", "area": "auth"}, links=["TODO-001"])
        writes.update(brain=self.brain, key="TODO-001", changes={"links": f"[bugs/{bug.path.name}, todos/TODO-003-dark-mode.md, bugs/{related.path.name}]"})
        writes.retitle(brain=self.brain, key=bug.identifier, title="Csv export fails")

        numbers = steps.add(brain=self.brain, key="BUG-002", lines=["TODO-001, the export, now a bug."])

        self.assertEqual(numbers, [1])
        rows = steps.rows_of(brain=self.brain, entry=self.brain.find("BUG-002"))
        self.assertEqual((rows[0].points_at, rows[0].status), (bug.identifier, "pending"))

    def test_a_key_opening_a_step_without_a_comma_or_a_colon_is_refused_rather_than_kept_as_prose(self) -> None:
        opens = "step 4 opens with TODO-001 but not as 'TODO-001, what to do', so it would point at nothing"
        for line, reason in (
            ("TODO-001", opens),
            ("TODO-001 - export first", opens),
            ("TODO-001's export lands", opens),
            ("TODO-001,export first", opens),
            ("TODO-001/BUG-002 both", opens),
            ("TODO-001.", opens),
            ("TODO-001. export first", opens),
            ("TODO-001.export first", opens),
            ("TODO-001_export first", opens),
            ("TODO-001.: export first", opens),
            ("TODO-001_old: export first", opens),
            ("TODO-001.1: the first half", opens),
            ("TODO-099 is gone", "step 4 points at 'TODO-099', which is not a todo, a bug or a review the brain knows"),
            ("TODO-099. fix", "step 4 points at 'TODO-099', which is not a todo, a bug or a review the brain knows"),
            ("TODO-099's fix lands", "step 4 points at 'TODO-099', which is not a todo, a bug or a review the brain knows"),
            ("TODO-099_old: fix", "step 4 points at 'TODO-099', which is not a todo, a bug or a review the brain knows"),
        ):
            with self.subTest(line=line):
                with self.assertRaises(steps.StepsError) as caught:
                    steps.add(brain=self.brain, key="kit-phase-7", lines=[line])

                self.assertEqual(str(caught.exception), reason)

        self.assertEqual(steps.add(brain=self.brain, key="kit-phase-7", lines=["DEC-001 is the storage decision."]), [4])

        review = ledger.open_ledger(brain=self.brain, branch="123-fix-login", summary="a branch named by its issue", stamp="2026-09-24")
        with self.assertRaises(steps.StepsError) as caught:
            steps.add(brain=self.brain, key="kit-phase-7", lines=[f"{review.name} review it"])
        self.assertEqual(str(caught.exception), f"step 5 opens with {review.name} but not as '{review.name}, what to do', so it would point at nothing")
        with self.assertRaises(steps.StepsError):
            steps.add(brain=self.brain, key="kit-phase-7", lines=["2026-09-999-other: review it"])
        self.assertEqual(steps.add(brain=self.brain, key="kit-phase-7", lines=[f"{review.name}, review it"]), [5])

        other = ledger.open_ledger(brain=self.brain, branch="64bit-build", summary="a branch that opens with digits and a letter", stamp="2026-09-24")
        with self.assertRaises(steps.StepsError):
            steps.add(brain=self.brain, key="kit-phase-7", lines=[f"{other.name} review it"])
        with self.assertRaises(steps.StepsError):
            steps.add(brain=self.brain, key="kit-phase-7", lines=["2026-09-10x-other: review it"])

    def test_a_cell_in_another_case_is_refreshed_when_its_entry_changes(self) -> None:
        path = self.plan.path
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace("| TODO-001 | owner | pending |", "| todo-001 | owner | pending |"))
        self.brain.refresh()

        writes.update(brain=self.brain, key="TODO-001", changes={"status": "done", "done": "2026-09-24"})

        self.assertEqual((self.stored()[0].points_at, self.stored()[0].status), ("TODO-001", "done"))

    def test_a_refile_whose_successor_points_at_the_entry_it_replaces_is_refused_as_a_self_pointer(self) -> None:
        with self.assertRaises(steps.StepsError) as caught:
            writes.convert(
                brain=self.brain,
                key="BUG-002",
                kind=schema.kind_named("todo"),
                values={"area": "auth", "by": "owner"},
                description="the loop, as a todo",
                body="## What\n\nThe loop.\n\n## Steps\n\n1. BUG-002, reread the old report.\n\n## Done when\n\nDone.\n",
            )
        self.assertEqual(str(caught.exception), "step 1 points at BUG-002 itself, and a step points at other work")
        self.assertEqual(self.brain.find("BUG-002").field("status"), "open")

    def test_a_review_name_in_the_wrong_case_and_a_cell_holding_a_name_are_read_as_the_entry(self) -> None:
        numbers = steps.add(brain=self.brain, key="kit-phase-7", lines=["2026-09-Feat-Billing, the review."])
        self.assertEqual(numbers, [4])
        self.assertEqual(self.stored()[3].points_at, "2026-09-feat-billing")

        with self.assertRaises(steps.StepsError):
            steps.add(brain=self.brain, key="kit-phase-7", lines=["2026-09-Feat-Nowhere, nothing."])

        path = self.plan.path
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace("| TODO-001 | owner | pending |", "| TODO-001-export-to-csv | owner | pending |"))
        self.brain.refresh()

        self.assertEqual(steps.rows_of(brain=self.brain, entry=self.brain.find("kit-phase-7"))[0].points_at, "TODO-001")
        found = [finding for finding in doctor.run(self.brain) if finding.check == "steps"]
        self.assertEqual([finding.message for finding in found], ["step 1 points at TODO-001-export-to-csv, and the tool keeps the id TODO-001, which a retitle never moves"])

        writes.retitle(brain=self.brain, key="TODO-001", title="Export to xlsx")
        self.assertEqual(self.stored()[0].points_at, "TODO-001")
        self.assertEqual([finding.message for finding in doctor.run(self.brain) if finding.check == "steps"], [])

    def test_a_step_whose_id_a_merge_doubled_is_marked_by_hand_and_doctor_says_why(self) -> None:
        fixtures.write(path=self.root / "bugs" / "BUG-002-other-thing.md", text=(self.root / "bugs" / "BUG-002-login-loop.md").read_text(encoding="utf-8").replace("BUG-002-login-loop", "BUG-002-other-thing"))
        steps.add(brain=self.brain, key="kit-phase-7", lines=["the loop, by hand"])
        path = self.plan.path
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace("| the loop, by hand | | agent | pending |", "| the loop, by hand | BUG-002 | agent | pending |"))
        self.brain.refresh()

        found = [finding.message for finding in doctor.run(self.brain) if finding.check == "steps"]
        self.assertIn("step 4 points at 'BUG-002', which two files carry, so it reaches nothing while the id is doubled, and brain steps <key> step 4 done or skipped marks it by hand meanwhile", found)

        steps.mark(brain=self.brain, key="kit-phase-7", number=4, status="skipped")
        self.assertEqual((self.stored()[3].points_at, self.stored()[3].status), ("BUG-002", "skipped"))

    def test_a_cell_naming_a_file_whose_id_is_doubled_is_kept_as_written_when_the_table_is_rewritten(self) -> None:
        fixtures.write(path=self.root / "todos" / "TODO-001-other-thing.md", text=(self.root / "todos" / "TODO-001-export-to-csv.md").read_text(encoding="utf-8").replace("TODO-001-export-to-csv", "TODO-001-other-thing"))
        path = self.plan.path
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace("| An independent pass over main, the owner's run. | | owner | pending |", "| An independent pass over main, the owner's run. | TODO-001-export-to-csv | owner | pending |"))
        self.brain.refresh()
        self.assertEqual(self.stored()[2].points_at, "TODO-001-export-to-csv")

        steps.mark(brain=self.brain, key="kit-phase-7", number=1, status="done")

        self.assertEqual((self.stored()[2].points_at, self.stored()[2].status), ("TODO-001-export-to-csv", "pending"))
        self.assertEqual([finding.message for finding in doctor.run(self.brain) if finding.check == "steps" and "step 3" in finding.message], [])

        writes.retitle(brain=self.brain, key="todos/TODO-001-export-to-csv.md", title="Export it", stamp="2026-09-24")

        self.assertEqual(self.stored()[2].points_at, "TODO-001-export-it")
        self.assertEqual((self.stored()[0].points_at, self.stored()[0].status), ("TODO-001", "done"))
        self.assertEqual([finding.message for finding in doctor.run(self.brain) if finding.check == "steps" and "step 3" in finding.message], [])

    def test_a_retitle_that_fails_leaves_the_cells_that_named_the_file_as_they_were(self) -> None:
        fixtures.write(path=self.root / "todos" / "TODO-001-other-thing.md", text=(self.root / "todos" / "TODO-001-export-to-csv.md").read_text(encoding="utf-8").replace("TODO-001-export-to-csv", "TODO-001-other-thing"))
        path = self.plan.path
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace("| An independent pass over main, the owner's run. | | owner | pending |", "| An independent pass over main, the owner's run. | TODO-001-export-to-csv | owner | pending |"))
        self.brain.refresh()

        with mock.patch("pathlib.Path.rename", side_effect=OSError(30, "Read-only file system")):
            with self.assertRaises(OSError):
                writes.retitle(brain=self.brain, key="todos/TODO-001-export-to-csv.md", title="Export it", stamp="2026-09-24")

        self.brain.refresh()
        self.assertEqual(self.stored()[2].points_at, "TODO-001-export-to-csv")

    def test_a_listed_pointer_at_a_refiled_entry_whose_successor_points_back_stays_on_it_as_skipped(self) -> None:
        successor = writes.convert(
            brain=self.brain,
            key="TODO-003",
            kind=schema.kind_named("bug"),
            values={"severity": "bug", "area": "theme"},
            description="the theme, as a bug",
            body="## Problem\n\nBroken.\n\n## Evidence\n\nA log line.\n\n## Steps\n\n1. TODO-001, the export first.\n",
        )
        self.brain.refresh()

        self.assertEqual(steps.add(brain=self.brain, key="TODO-001", lines=["TODO-003, the old one."]), [1])
        self.assertEqual([(step.points_at, step.status) for step in steps.rows_of(brain=self.brain, entry=self.brain.find("TODO-001"))], [("TODO-003", "skipped")])
        self.assertEqual(successor.identifier, "BUG-003")

    def test_a_cell_holding_an_id_two_files_carry_points_back_at_neither(self) -> None:
        fixtures.write(path=self.root / "todos" / "TODO-001-other-thing.md", text=(self.root / "todos" / "TODO-001-export-to-csv.md").read_text(encoding="utf-8").replace("TODO-001-export-to-csv", "TODO-001-other-thing"))
        bug = self.root / "bugs" / "BUG-002-login-loop.md"
        fixtures.write(path=bug, text=bug.read_text(encoding="utf-8").replace("## Evidence", "## Steps\n\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n| 1 | the export. | TODO-001 | owner | pending |\n\n## Evidence"))
        self.brain.refresh()

        self.assertEqual(steps.add(brain=self.brain, key="todos/TODO-001-export-to-csv.md", lines=["BUG-002, the bug first."]), [1])
        self.assertEqual([finding.message for finding in doctor.run(self.brain) if "point back" in finding.message], [])

        fixtures.write(path=bug, text=bug.read_text(encoding="utf-8").replace("| TODO-001 |", "| TODO-001-other-thing |"))
        self.brain.refresh()

        self.assertEqual([finding.message for finding in doctor.run(self.brain) if "point back" in finding.message], [])

        with self.assertRaises(steps.StepsError) as caught:
            steps.add(brain=self.brain, key="todos/TODO-001-other-thing.md", lines=["BUG-002, the bug first."])
        self.assertEqual(str(caught.exception), "step 1 points at BUG-002, whose steps point back here, and the two would wait on each other for good")

    def test_a_retitle_of_a_file_with_no_id_moves_the_cells_that_named_it_to_the_new_name(self) -> None:
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"
        fixtures.write(path=self.root / "todos" / "export-to-csv.md", text=todo.read_text(encoding="utf-8").replace("TODO-001-export-to-csv", "export-to-csv"))
        path = self.plan.path
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace("| An independent pass over main, the owner's run. | | owner | pending |", "| An independent pass over main, the owner's run. | export-to-csv | owner | pending |"))
        self.brain.refresh()

        writes.retitle(brain=self.brain, key="todos/export-to-csv.md", title="Export to JSON", stamp="2026-09-25")

        self.assertEqual(self.stored()[2].points_at, "export-to-json")
        self.assertEqual([finding.message for finding in doctor.run(self.brain) if finding.check == "steps" and "step 3" in finding.message], [])

        bug = self.root / "bugs" / "BUG-002-login-loop.md"
        fixtures.write(path=self.root / "bugs" / "export-to-json.md", text=bug.read_text(encoding="utf-8").replace("BUG-002-login-loop", "export-to-json"))
        fixtures.write(path=bug, text=bug.read_text(encoding="utf-8").replace("status: open\n", "status: refiled\nbecame: export-to-json\n"))
        other_bug = self.root / "bugs" / "export-to-json.md"
        fixtures.write(path=other_bug, text=other_bug.read_text(encoding="utf-8").replace("status: open\n", "status: refiled\nbecame: todos/export-to-json.md\n"))
        self.brain.refresh()

        writes.retitle(brain=self.brain, key="todos/export-to-json.md", title="Export to YAML", stamp="2026-09-25")

        self.assertEqual(self.stored()[2].points_at, "export-to-json")
        self.assertEqual(self.brain.find("BUG-002").field("became"), "export-to-json")
        self.assertEqual(self.brain.find("bugs/export-to-json.md").field("became"), "export-to-yaml")

    def test_a_retitle_writes_the_list_lines_that_named_the_file_as_its_id(self) -> None:
        other = self.root / "todos" / "TODO-003-dark-mode.md"
        fixtures.write(path=other, text=other.read_text(encoding="utf-8").replace("status: done", "status: open").replace("## Done when", "## Steps\n\n1. TODO-001-export-to-csv, check the flow.\n2. todo-001-export-to-csv: and again.\n3. Ship it.\n\n## Done when"))
        self.brain.refresh()

        writes.retitle(brain=self.brain, key="TODO-001", title="Export it", stamp="2026-09-24")

        text = other.read_text(encoding="utf-8")
        self.assertIn("## Steps\n\n1. TODO-001, check the flow.\n2. TODO-001: and again.\n3. Ship it.\n\n## Done when", text)
        self.assertIn("updated: 2026-09-24", text)
        self.assertEqual(steps.add(brain=self.brain, key="TODO-003", lines=["Rest."]), [4])

    def test_a_closed_entry_whose_own_table_does_not_read_is_not_reopened_on_a_guess(self) -> None:
        other = self.root / "todos" / "TODO-003-dark-mode.md"
        fixtures.write(path=other, text=other.read_text(encoding="utf-8").replace("## Done when", "## Steps\n\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n| 1 | first | TODO-001 | agent | finished |\n\n## Done when"))
        self.brain.refresh()

        with self.assertRaises(writes.WriteError) as caught:
            writes.update(brain=self.brain, key="TODO-003", changes={"status": "open"})
        self.assertEqual(str(caught.exception), "reopening TODO-003-dark-mode waits on its steps table, which does not read, step 1 is pending or done or skipped, not 'finished', so the tool cannot tell whether it and another entry would wait on each other, and brain doctor names what is wrong with it")
        self.assertEqual(self.brain.find("TODO-003").field("status"), "done")

        fixtures.write(path=other, text=other.read_text(encoding="utf-8").replace("| Points at |", "| Points At |").replace("| finished |", "| done |"))
        self.brain.refresh()

        with self.assertRaises(writes.WriteError) as caught:
            writes.update(brain=self.brain, key="TODO-003", changes={"status": "open"})
        self.assertEqual(str(caught.exception), "reopening TODO-003-dark-mode waits on its steps table, which does not read, the Steps section holds table lines without the header the tool reads, | # | Step | Points at | By | Status |, so the tool cannot tell whether it and another entry would wait on each other, and brain doctor names what is wrong with it")

    def test_a_closed_entry_whose_list_points_at_an_entry_whose_steps_do_not_read_is_not_reopened_on_a_guess(self) -> None:
        other = self.root / "todos" / "TODO-003-dark-mode.md"
        fixtures.write(path=other, text=other.read_text(encoding="utf-8").replace("## Done when", "## Steps\n\n1. BUG-002, first.\n\n## Done when"))
        bug = self.root / "bugs" / "BUG-002-login-loop.md"
        fixtures.write(path=bug, text=bug.read_text(encoding="utf-8").replace("## Evidence", "## Steps\n\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n| 1 | the theme. | TODO-003 | agent | finished |\n\n## Evidence"))
        self.brain.refresh()

        with self.assertRaises(writes.WriteError) as caught:
            writes.update(brain=self.brain, key="TODO-003", changes={"status": "open"})
        self.assertEqual(str(caught.exception), "reopening TODO-003-dark-mode waits on the steps of BUG-002, which do not read, so the tool cannot tell whether it and TODO-003-dark-mode would wait on each other, and brain doctor names what is wrong with them")

        fixtures.write(path=bug, text=bug.read_text(encoding="utf-8").replace("| finished |", "| pending |"))
        self.brain.refresh()

        with self.assertRaises(writes.WriteError) as caught:
            writes.update(brain=self.brain, key="TODO-003", changes={"status": "open"})
        self.assertEqual(str(caught.exception), "reopening TODO-003-dark-mode would leave it and another entry waiting on each other, step 1 points at BUG-002, whose steps point back here, and the two would wait on each other for good")

        fixtures.write(path=other, text=other.read_text(encoding="utf-8").replace("1. BUG-002, first.\n", "- a note that is not a step\n2. BUG-002, first.\n"))
        self.brain.refresh()

        with self.assertRaises(writes.WriteError) as caught:
            writes.update(brain=self.brain, key="TODO-003", changes={"status": "open"})
        self.assertEqual(str(caught.exception), "reopening TODO-003-dark-mode would leave it and another entry waiting on each other, step 2 points at BUG-002, whose steps point back here, and the two would wait on each other for good")

    def test_a_refile_hand_off_onto_a_successor_whose_steps_do_not_read_is_not_made_on_a_guess(self) -> None:
        steps.add(brain=self.brain, key="TODO-001", lines=["TODO-003, the theme."])
        bug = self.root / "bugs" / "BUG-002-login-loop.md"
        fixtures.write(path=bug, text=bug.read_text(encoding="utf-8").replace("## Evidence", "## Steps\n\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n| 1 | the export. | TODO-001 | agent | pending |\n| 2 | broken | | agent | doing |\n\n## Evidence"))
        other = self.root / "todos" / "TODO-003-dark-mode.md"
        fixtures.write(path=other, text=other.read_text(encoding="utf-8").replace("status: done\n", "status: refiled\nbecame: BUG-002\n"))
        self.brain.refresh()

        rows = steps.rows_of(brain=self.brain, entry=self.brain.find("TODO-001"))
        self.assertEqual([(step.points_at, step.status) for step in rows], [("TODO-003", "skipped")])
        self.assertEqual([finding.message for finding in doctor.run(self.brain) if finding.path == "todos/TODO-001-export-to-csv.md" and "BUG-002" in finding.message], [])

    def test_a_table_that_does_not_read_one_step_past_the_pointed_entry_is_named_on_the_way(self) -> None:
        other = self.root / "todos" / "TODO-003-dark-mode.md"
        fixtures.write(path=other, text=other.read_text(encoding="utf-8").replace("status: done", "status: open").replace("## Done when", "## Steps\n\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n| 1 | the loop. | BUG-002 | agent | pending |\n\n## Done when"))
        bug = self.root / "bugs" / "BUG-002-login-loop.md"
        fixtures.write(path=bug, text=bug.read_text(encoding="utf-8").replace("## Evidence", "## Steps\n\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n| 1 | broken | | agent | finished |\n\n## Evidence"))
        self.brain.refresh()

        with self.assertRaises(steps.StepsError) as caught:
            steps.add(brain=self.brain, key="TODO-001", lines=["TODO-003, the theme."])
        self.assertEqual(str(caught.exception), "step 1 points at TODO-003, whose steps lead to BUG-002, whose steps do not read, so the tool cannot tell whether they point back here, and brain doctor names what is wrong with them")

    def test_a_successor_two_files_carry_holds_the_hand_off_and_doctor_says_so(self) -> None:
        steps.add(brain=self.brain, key="kit-phase-7", lines=["TODO-001, the export."])
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"
        fixtures.write(path=todo, text=todo.read_text(encoding="utf-8").replace("status: open\n", "status: refiled\nbecame: bugs/BUG-002-login-loop.md\n").replace("links: []", "links: [bugs/BUG-002-login-loop.md]"))
        fixtures.write(path=self.root / "bugs" / "BUG-002-other-thing.md", text=(self.root / "bugs" / "BUG-002-login-loop.md").read_text(encoding="utf-8").replace("BUG-002-login-loop", "BUG-002-other-thing"))
        self.brain.refresh()

        rows = steps.rows_of(brain=self.brain, entry=self.brain.find("kit-phase-7"))
        self.assertEqual((rows[3].points_at, rows[3].status), ("TODO-001", "skipped"))

        fixtures.write(path=self.plan.path, text=self.plan.path.read_text(encoding="utf-8").replace("| 4 | the export. | TODO-001 | agent | pending |", "| 4 | the export. | todo-001 | agent | pending |"))
        self.brain.refresh()

        self.assertEqual([finding.message for finding in doctor.run(self.brain) if finding.path == "plans/kit-phase-7.md" and "step 4" in finding.message], ["step 4 points at todo-001, which was refiled as BUG-002, which two files carry, so the tool keeps the key TODO-001 and the step stands skipped while the id is doubled"])

    def test_a_pointed_entry_whose_steps_hold_table_lines_without_the_header_is_unreadable_too(self) -> None:
        other = self.root / "todos" / "TODO-003-dark-mode.md"
        fixtures.write(path=other, text=other.read_text(encoding="utf-8").replace("status: done", "status: open").replace("## Done when", "## Steps\n\n| # | Step | Points At | By | Status |\n|---|---|---|---|---|\n| 1 | first | TODO-001 | agent | pending |\n\n## Done when"))
        self.brain.refresh()

        with self.assertRaises(steps.StepsError) as caught:
            steps.add(brain=self.brain, key="TODO-001", lines=["TODO-003, the other."])
        self.assertEqual(str(caught.exception), "step 1 points at TODO-003, whose steps do not read, so the tool cannot tell whether they point back here, and brain doctor names what is wrong with them")

    def test_a_step_at_an_entry_whose_list_leads_back_is_refused_before_the_list_is_turned(self) -> None:
        other = self.root / "todos" / "TODO-003-dark-mode.md"
        fixtures.write(path=other, text=other.read_text(encoding="utf-8").replace("status: done", "status: open").replace("## Done when", "## Steps\n\n1. TODO-001, the export first.\n\n## Done when"))
        self.brain.refresh()

        with self.assertRaises(steps.StepsError) as caught:
            steps.add(brain=self.brain, key="TODO-001", lines=["TODO-003, the theme first."])
        self.assertEqual(str(caught.exception), "step 1 points at TODO-003, whose steps point back here, and the two would wait on each other for good")

        self.assertEqual(steps.pointers_in(other.read_text(encoding="utf-8")), ["TODO-001"])
        self.assertEqual(steps.pointers_in("## Steps\n\n1. Ship it, then rest.\n2. BUG-002: and the loop.\n- not a step\n"), ["BUG-002"])
        self.assertEqual(steps.pointers_in("## Steps\n\n1. TODO-001,\n   the export first.\n2. Check the log\n   | grep the loop.\n"), ["TODO-001"])

        with self.assertRaises(steps.StepsError) as caught:
            steps.pointers_in("## Steps\n\n  | # | Step | Points at | By | Status |\n  |---|---|---|---|---|\n  | 1 | first | BUG-002 | agent | done |\n")
        self.assertEqual(str(caught.exception), "the Steps section holds table lines without the header the tool reads, | # | Step | Points at | By | Status |")

    def test_a_step_at_an_entry_whose_steps_do_not_read_is_refused_rather_than_guessed_at(self) -> None:
        other = self.root / "todos" / "TODO-003-dark-mode.md"
        fixtures.write(path=other, text=other.read_text(encoding="utf-8").replace("status: done", "status: open").replace("## Done when", "## Steps\n\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n| 1 | first | TODO-001 | agent | finished |\n\n## Done when"))
        self.brain.refresh()

        with self.assertRaises(steps.StepsError) as caught:
            steps.add(brain=self.brain, key="TODO-001", lines=["TODO-003, the other."])
        self.assertEqual(str(caught.exception), "step 1 points at TODO-003, whose steps do not read, so the tool cannot tell whether they point back here, and brain doctor names what is wrong with them")

    def test_a_step_at_a_doubled_key_is_refused_as_doubled_whether_typed_as_the_id_or_the_name(self) -> None:
        fixtures.write(path=self.root / "bugs" / "BUG-002-other-thing.md", text=(self.root / "bugs" / "BUG-002-login-loop.md").read_text(encoding="utf-8").replace("BUG-002-login-loop", "BUG-002-other-thing"))
        review = ledger.open_ledger(brain=self.brain, branch="feat/x", summary="the x", stamp="2026-09-24")
        fixtures.write(path=self.root / "reviews" / review.name.replace("feat-x", "feat-y") / "ledger.md", text=review.path.read_text(encoding="utf-8"))
        self.brain.refresh()

        for line, refusal in (
            ("BUG-002, the loop again.", "step 4 points at 'BUG-002', which two files carry, so it reaches nothing while the id is doubled"),
            ("BUG-002-login-loop, the loop again.", "step 4 points at 'BUG-002-login-loop', and its id BUG-002 two files carry, so it would reach nothing"),
            (f"{review.name}, the review.", f"step 4 points at '{review.name}', which two files carry, so it reaches nothing while the name is doubled"),
            ("BUG-002 the loop again.", "step 4 points at 'BUG-002', which two files carry, so it reaches nothing while the id is doubled"),
        ):
            with self.subTest(line=line):
                with self.assertRaises(steps.StepsError) as caught:
                    steps.add(brain=self.brain, key="kit-phase-7", lines=[line])
                self.assertEqual(str(caught.exception), refusal)

    def test_a_step_whose_review_name_two_files_answer_to_is_said_as_a_name(self) -> None:
        review = ledger.open_ledger(brain=self.brain, branch="feat/x", summary="the x", stamp="2026-09-24")
        steps.add(brain=self.brain, key="kit-phase-7", lines=[f"{review.name}, the review."])
        copied = self.root / "reviews" / review.name.replace("feat-x", "feat-y") / "ledger.md"
        fixtures.write(path=copied, text=review.path.read_text(encoding="utf-8"))
        self.brain.refresh()

        found = [finding.message for finding in doctor.run(self.brain) if finding.check == "steps"]
        self.assertIn(f"step 4 points at '{review.name}', which two files carry, so it reaches nothing while the name is doubled, and brain steps <key> step 4 done or skipped marks it by hand meanwhile", found)

        typed = review.name.replace("feat", "Feat")
        path = self.plan.path
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace(f"| {review.name} |", f"| {typed} |"))
        self.brain.refresh()

        found = [finding.message for finding in doctor.run(self.brain) if finding.check == "steps"]
        self.assertIn(f"step 4 points at '{typed}', which two files carry, so it reaches nothing while the name is doubled, and brain steps <key> step 4 done or skipped marks it by hand meanwhile", found)

    def test_a_closed_entry_with_a_list_is_repaired_though_an_open_entry_points_back(self) -> None:
        path = self.root / "todos" / "TODO-003-dark-mode.md"
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace("## Done when", "## Steps\n\n1. TODO-001, the export first.\n\n## Done when"))
        self.brain.refresh()
        steps.add(brain=self.brain, key="TODO-001", lines=["TODO-003, after dark mode."])

        found = [finding for finding in doctor.run(self.brain) if finding.check == "steps" and finding.path.startswith("todos/TODO-003")]
        self.assertEqual([finding.fixable for finding in found], [True])

        doctor.run(brain=self.brain, fix=True)
        self.assertEqual([(step.points_at, step.status) for step in steps.rows_of(brain=self.brain, entry=self.brain.find("TODO-003"))], [("TODO-001", "done")])

    def test_a_typed_row_whose_step_cell_opens_with_a_key_is_refused_like_a_listed_step(self) -> None:
        for cell, reason in (
            ("TODO-001, the export first.", "step 1 opens with TODO-001 in its Step cell, and a row points through its Points at cell"),
            ("BUG-099 from the old tracker", "step 1 points at 'BUG-099', which is not a todo, a bug or a review the brain knows"),
        ):
            with self.subTest(cell=cell):
                body = "## Why\n\nWhy.\n\n## Steps\n\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n| 1 | " + cell + " | | owner | pending |\n\n## Done when\n\nDone.\n"

                with self.assertRaises(steps.StepsError) as caught:
                    plans.create(brain=self.brain, title="Typed", description="typed", body=body)

                self.assertEqual(str(caught.exception), reason)

    def test_every_became_slip_is_named_by_doctor_and_a_doubled_became_is_refused_on_set(self) -> None:
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"
        other = self.root / "todos" / "TODO-003-dark-mode.md"
        fixtures.write(path=other, text=other.read_text(encoding="utf-8").replace("status: done\n", "status: open\nbecame: BUG-002\n"))
        fixtures.write(path=todo, text=todo.read_text(encoding="utf-8").replace("status: open\n", "status: refiled\nbecame: TODO-003\n").replace("links: []", "links: [todos/TODO-003-dark-mode.md]"))
        self.brain.refresh()

        found = [finding.message for finding in doctor.run(self.brain) if "became" in finding.message]
        self.assertIn("became names BUG-002 while the entry is not refiled", found)
        self.assertIn("became names TODO-003, which is a todo where a refiled todo becomes a bug", found)

        fixtures.write(path=self.root / "bugs" / "BUG-002-other-thing.md", text=(self.root / "bugs" / "BUG-002-login-loop.md").read_text(encoding="utf-8").replace("BUG-002-login-loop", "BUG-002-other-thing"))
        fixtures.write(path=todo, text=todo.read_text(encoding="utf-8").replace("became: TODO-003\n", "became: BUG-002\n"))
        self.brain.refresh()

        found = [finding.message for finding in doctor.run(self.brain) if "became" in finding.message]
        self.assertIn("became names BUG-002, which two files carry, so the steps that point here follow nothing until one is refiled", found)

        with self.assertRaises(writes.WriteError) as caught:
            writes.update(brain=self.brain, key="TODO-003", changes={"status": "refiled", "became": "BUG-002", "done": "2026-09-25", "links": "[bugs/BUG-002-login-loop.md]"})
        self.assertEqual(str(caught.exception), "became names BUG-002, which two files carry")

    def test_became_is_checked_on_set_and_by_doctor(self) -> None:
        bug = writes.convert(brain=self.brain, key="TODO-001", kind=schema.kind_named("bug"), values={"severity": "bug", "area": "auth"}, description="the export, as a bug")

        for value, reason in (
            ("BUG-05", "became names BUG-05, which is not a bug the brain knows"),
            ("TODO-003", "became names TODO-003, which is not a bug the brain knows"),
        ):
            with self.subTest(value=value):
                with self.assertRaises(writes.WriteError) as caught:
                    writes.update(brain=self.brain, key="TODO-001", changes={"became": value})

                self.assertEqual(str(caught.exception), reason)

        with self.assertRaises(writes.WriteError) as caught:
            writes.update(brain=self.brain, key="TODO-003", changes={"became": bug.identifier})
        self.assertEqual(str(caught.exception), "became goes with status refiled, since it names what the entry was refiled as")

        path = self.root / "todos" / "TODO-001-export-to-csv.md"
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace(f"became: {bug.identifier}", "became: BUG-05"))
        self.brain.refresh()
        found = [finding.message for finding in doctor.run(self.brain) if finding.check == "steps"]
        self.assertIn("became names BUG-05, which reaches nothing, so the steps that point here follow nothing", found)

        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace("became: BUG-05", f"became: {bug.identifier}"))
        self.brain.refresh()
        reopened = writes.update(brain=self.brain, key="TODO-001", changes={"status": "open", "done": ""})
        self.assertIsNone(reopened.field("became"))
        self.assertEqual([finding.message for finding in doctor.run(self.brain) if finding.check == "steps" and "became" in finding.message], [])

    def test_a_date_a_decision_and_a_word_of_prose_open_a_plain_step(self) -> None:
        numbers = steps.add(brain=self.brain, key="kit-phase-7", lines=["2026-10-01, freeze the release branch.", "DEC-001, apply the storage decision.", "Later: the docs.", "2026-10-01T09:00 cut the release branch."])

        self.assertEqual(numbers, [4, 5, 6, 7])
        self.assertEqual([(step.points_at, step.text) for step in self.stored()[3:]], [
            ("", "2026-10-01, freeze the release branch."),
            ("", "DEC-001, apply the storage decision."),
            ("", "Later: the docs."),
            ("", "2026-10-01T09:00 cut the release branch."),
        ])

        with self.assertRaises(steps.StepsError) as caught:
            steps.add(brain=self.brain, key="kit-phase-7", lines=["TODO-099, nothing."])
        self.assertEqual(str(caught.exception), "step 8 points at 'TODO-099', which is not a todo, a bug or a review the brain knows")

    def test_doctor_repairs_a_pointer_left_behind_by_a_refile_and_a_stale_mark(self) -> None:
        bug = writes.convert(brain=self.brain, key="TODO-001", kind=schema.kind_named("bug"), values={"severity": "bug", "area": "auth"}, description="the export, as a bug")
        path = self.plan.path
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace(f"| {bug.identifier} | owner | pending |", "| TODO-001 | owner | pending |"))
        self.brain.refresh()

        found = [finding for finding in doctor.run(self.brain) if finding.check == "steps"]
        self.assertEqual([finding.message for finding in found], [f"step 1 points at TODO-001, which was refiled as {bug.identifier}"])
        self.assertTrue(found[0].fixable)

        doctor.run(brain=self.brain, fix=True)
        self.assertEqual(self.stored()[0].points_at, bug.identifier)


class TheTable(fixtures.BrainCase):
    HEAD = "## Why\n\nWhy.\n\n## Steps\n\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n| 1 | one | | agent | pending |\n"
    TAIL = "\n## Done when\n\nDone.\n"

    def test_a_write_keeps_the_prose_and_the_code_after_the_table(self) -> None:
        body = self.HEAD + "\nA row looks like this:\n\n```\n| 1 | example | | agent | pending |\n```\n" + self.TAIL

        rows = steps.table(body)
        written = steps.rewrite(body=body, rows=rows + [steps.Step(number=2, text="two", points_at="", by="agent", status="pending")])

        self.assertIn("| 1 | one | | agent | pending |\n| 2 | two | | agent | pending |\n\nA row looks like this:\n\n```\n| 1 | example | | agent | pending |\n```\n", written)
        self.assertEqual([step.text for step in steps.table(written)], ["one", "two"])

    def test_a_write_reads_past_a_fenced_header_above_the_table(self) -> None:
        body = "## Why\n\nWhy.\n\n## Steps\n\n```\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n```\n\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n| 1 | one | | agent | pending |\n" + self.TAIL

        written = steps.rewrite(body=body, rows=[steps.Step(number=1, text="one", points_at="", by="agent", status="done")])

        self.assertIn("```\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n```\n\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n| 1 | one | | agent | done |\n", written)

    def test_the_table_goes_before_the_next_heading_whatever_space_follows_its_hashes(self) -> None:
        kind = schema.kind_named("plan")

        for body, ending in (
            ("##\tWhy\n\nbecause\n\n##\tDone when\n\nshipped\n", "Done when\n\nshipped\n"),
            ("## Why\n\nbecause\n\n##\tDone when\n\nshipped\n", "Done when\n\nshipped\n"),
            ("## Why\n\nbecause\n\n## Done when\n\nshipped", "Done when\n\nshipped"),
        ):
            with self.subTest(body=body):
                written = steps.with_table(body=body, rows=[], kind=kind)

                self.assertEqual([title.lower() for title in sections.headings(written)], ["why", "steps", "done when"])
                self.assertTrue(written.endswith(ending), written[-30:])

    def test_a_row_that_does_not_match_fails_at_once_whatever_its_padding(self) -> None:
        padding = " " * 60
        short = f"|{padding}1{padding}|{padding}one{padding}|{padding}|{padding}agent{padding}|"
        started = time.perf_counter()

        self.assertIsNone(steps.ROW.match(short))

        self.assertLess(time.perf_counter() - started, 1.0)
        whole = f"|{padding}1{padding}|{padding}one \\| two{padding}|{padding}|{padding}agent{padding}|{padding}pending{padding}|"
        self.assertEqual(steps.table(f"## Steps\n\n{steps.TABLE_HEADER}{whole}\n")[0].text, "one | two")

        self.assertIsNone(steps.ROW.match("| 1 | see \\| TODO-002 | agent | pending |"))
        started = time.perf_counter()

        self.assertIsNone(steps.ROW.match("| 1 | " + "a \\| " * 120 + "x"))

        self.assertLess(time.perf_counter() - started, 1.0)
        started = time.perf_counter()

        self.assertIsNone(steps.ROW.match("| 1 |" + (" " * 120 + "|") * 4 + " extra |"))
        self.assertIsNone(steps.ROW.match("| 1 |" + (" " * 120 + "|") * 3 + " " * 60 + "x" + " " * 60 + "| extra |"))

        self.assertLess(time.perf_counter() - started, 1.0)
        self.assertEqual(steps.table(f"## Steps\n\n{steps.TABLE_HEADER}| 1 |\t|  | agent |pending|\n")[0], steps.Step(number=1, text="", points_at="", by="agent", status="pending"))
        self.assertEqual(steps.table(f"## Steps\n\n{steps.TABLE_HEADER}| 1 | a \\ b | | agent | pending |\n")[0].text, "a \\ b")

    def test_a_table_typed_by_hand_is_checked_and_its_marks_derived_on_filing(self) -> None:
        kind = schema.kind_named("todo")
        values = {"area": "reports", "by": "owner"}

        for row, reason in (
            ("| 1 | x | TODO-099 | agent | pending |", "step 1 points at 'TODO-099', which is not a todo, a bug or a review the brain knows"),
            ("| 1 | x | | robot | finished |", "step 1 is pending or done or skipped, not 'finished'"),
            ("| 1 | x | | robot | pending |", "step 1 is set by owner or agent, not 'robot'"),
            ("| 1 |  | | agent | pending |", "step 1 has no text"),
            ("| 1 | x | | agent | pending |\n| 2 | y | agent |", "a line of the steps table is not a whole row, a row is | number | step | points at | by | status |, a pipe inside a cell is written \\|, and a backslash never ends a cell before its pipe"),
            ("| 1 | C:\\temp\\| | agent | pending |", "a line of the steps table is not a whole row, a row is | number | step | points at | by | status |, a pipe inside a cell is written \\|, and a backslash never ends a cell before its pipe"),
            ("| 2 | x | | agent | pending |", "the steps are numbered 1 onward in order, and row 2 sits where 1 should"),
            ("| 1 | x | | agent | pending |\n\n1. do y", "step lines go in the table, and '1. do y' sits outside it"),
            ("above", "step lines go in the table, and '2. b' sits outside it"),
        ):
            with self.subTest(row=row):
                body = "## What\n\nWhat.\n\n## Steps\n\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n" + row + "\n\n## Done when\n\nDone.\n"
                if row == "above":
                    body = "## What\n\nWhat.\n\n## Steps\n\n2. b\n\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n| 1 | x | | agent | pending |\n\n## Done when\n\nDone.\n"

                with self.assertRaises(steps.StepsError) as caught:
                    writes.create(brain=self.brain, kind=kind, title="Typed", description="typed", values=values, body=body)

                self.assertEqual(str(caught.exception), reason)

        body = "## What\n\nWhat.\n\n## Steps\n\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n| 1 | x | TODO-003 | agent | pending |\n\n## Done when\n\nDone.\n"
        entry = writes.create(brain=self.brain, kind=kind, title="Typed", description="typed", values=values, body=body)
        self.assertEqual([(step.points_at, step.status) for step in steps.rows_of(brain=self.brain, entry=entry)], [("TODO-003", "done")])

    def test_a_pipe_in_a_step_is_written_escaped_and_read_back_as_a_pipe(self) -> None:
        body = "## Problem\n\nLoops.\n\n## Evidence\n\nA log line.\n\n## Steps\n\n1. Run `tail log | grep 500` and read it.\n"
        entry = writes.create(brain=self.brain, kind=schema.kind_named("bug"), title="Piped", description="piped", values={"severity": "bug", "area": "auth"}, body=body)

        self.assertIn("| 1 | Run `tail log \\| grep 500` and read it. | | agent | pending |", entry.document.body)
        self.assertEqual(steps.rows_of(brain=self.brain, entry=entry)[0].text, "Run `tail log | grep 500` and read it.")

        numbers = steps.add(brain=self.brain, key=entry.name, lines=["a | b", "Copy it to C:\\", "grep 'a\\|b'"])
        self.assertEqual(numbers, [2, 3, 4])
        self.assertEqual([step.text for step in steps.rows_of(brain=self.brain, entry=self.brain.find(entry.name))], ["Run `tail log | grep 500` and read it.", "a | b", "Copy it to C:\\", "grep 'a\\|b'"])

        typed = "## Problem\n\nLoops.\n\n## Evidence\n\nA log line.\n\n## Steps\n\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n| 1 | match `\\d+` in C:\\temp | | agent | pending |\n"
        entry = writes.create(brain=self.brain, kind=schema.kind_named("bug"), title="Typed", description="typed", values={"severity": "bug", "area": "auth"}, body=typed)
        self.assertEqual(steps.rows_of(brain=self.brain, entry=entry)[0].text, "match `\\d+` in C:\\temp")

        steps.add(brain=self.brain, key=entry.name, lines=["two"])
        self.assertEqual(steps.rows_of(brain=self.brain, entry=self.brain.find(entry.name))[0].text, "match `\\d+` in C:\\temp")
        self.assertIn("| 1 | match `\\d+` in C:\\temp | | agent | pending |", self.brain.find(entry.name).document.body)

    def test_a_subheading_inside_steps_belongs_to_the_section_and_is_refused_as_a_line_that_is_not_a_step(self) -> None:
        body = "## Why\n\nWhy.\n\n## Steps\n\n### Phase 1\n\n1. a\n2. b\n\n### Phase 2\n\n1. c\n\n## Done when\n\nDone.\n"

        with self.assertRaises(steps.StepsError) as caught:
            plans.create(brain=self.brain, title="Phased", description="phased", body=body)
        self.assertEqual(str(caught.exception), "the Steps section holds a line that is not a numbered step, one per line with its wrapped lines indented under it: '### Phase 1'")

        with self.assertRaises(steps.StepsError):
            writes.create(brain=self.brain, kind=schema.kind_named("todo"), title="Phased", description="phased", values={"area": "reports", "by": "owner"}, body=body.replace("## Why\n\nWhy.", "## What\n\nWhat."))
        self.assertIsNone(self.brain.find("phased"))

    def test_a_step_line_typed_beside_an_existing_table_is_refused_by_every_reader(self) -> None:
        body = self.HEAD + "\n2. a step typed by hand\n" + self.TAIL

        with self.assertRaises(steps.StepsError) as caught:
            steps.table(body)
        self.assertEqual(str(caught.exception), "step lines go in the table, and '2. a step typed by hand' sits outside it")

        entry = plans.create(brain=self.brain, title="Typed beside", description="typed beside", body="## Why\n\nWhy.\n\n## Done when\n\nDone.\n")
        fixtures.write(path=entry.path, text=entry.path.read_text(encoding="utf-8").replace("|---|---|---|---|---|\n", "|---|---|---|---|---|\n| 1 | one | | agent | pending |\n\n2. a step typed by hand\n"))
        self.brain.refresh()
        self.assertEqual(steps.gaps(brain=self.brain, entry=self.brain.find("typed-beside"), fields={"status": "done"}), [str(caught.exception)])

    def test_a_list_pointing_at_the_entry_itself_is_refused_when_it_becomes_the_table(self) -> None:
        path = self.root / "todos" / "TODO-001-export-to-csv.md"
        text = path.read_text(encoding="utf-8").replace("## Done when", "## Steps\n\n1. TODO-001, wire it\n\n## Done when")
        fixtures.write(path=path, text=text)
        self.brain.refresh()

        with self.assertRaises(steps.StepsError) as caught:
            steps.add(brain=self.brain, key="TODO-001", lines=["x"])
        self.assertEqual(str(caught.exception), "step 1 points at TODO-001 itself, and a step points at other work")

    def test_a_plan_left_with_a_lookalike_heading_is_refused_on_add_as_on_filing(self) -> None:
        path = self.root / "plans" / "phase-7.md"
        fixtures.write(path=path, text="---\nname: phase-7\nkind: plan\ndescription: the rest\nupdated: 2026-09-24\nlinks: []\nstatus: open\nby: owner\nborn: 2026-09-24\n---\n\n## Why\n\nWhy.\n\n### Steps\n\n1. a\n2. b\n\n## Done when\n\nDone.\n")
        self.brain.refresh()

        with self.assertRaises(steps.StepsError) as caught:
            steps.add(brain=self.brain, key="phase-7", lines=["c"])
        self.assertEqual(str(caught.exception), "the steps go under a level two heading, ## Steps, and '### Steps' is not one")

    def test_a_row_apart_from_the_table_is_refused_whatever_its_number(self) -> None:
        body = self.HEAD + "\nA note kept by hand.\n\n| 1 | one | | agent | pending |\n" + self.TAIL

        with self.assertRaises(steps.StepsError) as caught:
            steps.table(body)
        self.assertEqual(str(caught.exception), "a step row sits apart from the table, put it directly under the header")

        with self.assertRaises(steps.StepsError):
            steps.rewrite(body=body, rows=[])


class TheKind(unittest.TestCase):
    def test_a_plan_is_indexed_and_carries_the_three_sections(self) -> None:
        kind = schema.kind_named("plan")

        self.assertTrue(kind.indexed)
        self.assertEqual([section.title for section in kind.sections], ["Why", "Steps", "Done when"])
        self.assertFalse(kind.has_ids)
