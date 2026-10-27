"""One chain of work at a time, a plan down to its step, kept in a local file and read back from the marks."""

import json
import os
import re
import unittest
from pathlib import Path
from unittest import mock

from brainlib import ledger, plans, schema, steps, task, writes
from tests import fixtures

# the close hint stamps the day it is read, so a test matches its shape and never the day
A_DATE = r"\d{4}-\d{2}-\d{2}"

PLAN = """## Why

The rest of phase 7.

## Steps

1. TODO-001, the export.
2. BUG-002, the login loop.
3. The docs.

## Done when

Every step is done.
"""


class TaskCase(fixtures.BrainCase):
    def setUp(self) -> None:
        super().setUp()
        self.file = self.root / task.FILE

    def chain(self) -> list[tuple[str, str]]:
        return [(item.kind, item.key) for item in task.current(self.brain)]

    def file_chain(self) -> list[dict]:
        return json.loads(self.file.read_text(encoding="utf-8"))["chain"]

    def with_steps(self) -> None:
        steps.add(brain=self.brain, key="TODO-001", lines=["Read the table.", "Write the file."], by="owner")

    def with_plan(self) -> None:
        plans.create(brain=self.brain, title="Kit phase 7", description="the rest of phase 7", body=PLAN, stamp="2026-09-24")

    def with_review(self) -> str:
        entry = ledger.open_ledger(brain=self.brain, branch="feat/task", summary="the task chain", stamp="2026-09-24")
        ledger.add_batch(brain=self.brain, key=entry.name, theme="correctness")
        return entry.name


class Starting(TaskCase):
    def test_a_todo_becomes_the_work_and_the_file_holds_the_chain(self) -> None:
        self.assertEqual(task.describe(self.brain), "task none")
        self.assertFalse(self.file.exists())

        chain = task.turn_on(brain=self.brain, key="TODO-001")

        self.assertEqual(chain, [task.Item(kind="todo", key="TODO-001")])
        self.assertEqual(self.file_chain(), [{"kind": "todo", "key": "TODO-001"}])
        self.assertIn("since", json.loads(self.file.read_text(encoding="utf-8")))
        self.assertEqual(task.describe(self.brain), "task TODO-001")

    def test_a_todo_with_steps_stands_on_the_first_pending_one_and_the_line_names_it(self) -> None:
        self.with_steps()

        task.turn_on(brain=self.brain, key="TODO-001")
        self.assertEqual(task.describe(self.brain), "task TODO-001, step 1 of 2, Read the table.")

        steps.mark(brain=self.brain, key="TODO-001", number=1, status="done")
        self.assertEqual(task.describe(self.brain), "task TODO-001, step 2 of 2, Write the file.")

    def test_a_closed_entry_a_decision_and_an_unknown_key_are_refused(self) -> None:
        for key, reason in (
            ("TODO-003", "TODO-003-dark-mode is done, and the work is on something open"),
            ("DEC-001", "the work is a plan, a todo, a bug or a review, and DEC-001-postgres-over-sqlite is a decision"),
            ("TODO-099", "nothing in the brain answers to 'TODO-099'"),
        ):
            with self.subTest(key=key):
                with self.assertRaises(task.TaskError) as caught:
                    task.turn_on(brain=self.brain, key=key)

                self.assertEqual(str(caught.exception), reason)

        self.assertFalse(self.file.exists())

    def test_a_second_piece_of_work_is_refused_while_one_stands_unless_it_replaces_it(self) -> None:
        task.turn_on(brain=self.brain, key="TODO-001")

        with self.assertRaises(task.TaskError) as caught:
            task.turn_on(brain=self.brain, key="BUG-002")
        self.assertEqual(str(caught.exception), "the work is on TODO-001, finish it or drop it with brain task off first")
        self.assertEqual(self.chain(), [("todo", "TODO-001")])

        task.turn_on(brain=self.brain, key="BUG-002", replace=True)
        self.assertEqual(self.chain(), [("bug", "BUG-002")])
        self.assertEqual(task.describe(self.brain), "bug BUG-002")


class Marking(TaskCase):
    def test_done_marks_the_step_the_work_stands_on_and_moves_to_the_next(self) -> None:
        self.with_steps()
        task.turn_on(brain=self.brain, key="TODO-001")

        task.mark(brain=self.brain, status="done")
        self.assertEqual([step.status for step in steps.rows_of(brain=self.brain, entry=self.brain.find("TODO-001"))], ["done", "pending"])
        self.assertEqual(task.describe(self.brain), "task TODO-001, step 2 of 2, Write the file.")

        task.mark(brain=self.brain, status="skipped")
        self.assertEqual([step.status for step in steps.rows_of(brain=self.brain, entry=self.brain.find("TODO-001"))], ["done", "skipped"])
        self.assertRegex(task.describe(self.brain), rf"^task TODO-001, every step settled, close it with brain set TODO-001 status=done done={A_DATE} once Done when names what closed it$")

        with self.assertRaises(task.TaskError) as caught:
            task.mark(brain=self.brain, status="done")
        self.assertEqual(str(caught.exception), "TODO-001-export-to-csv has no step pending")

    def test_the_close_hint_is_a_command_that_goes_through_for_each_kind(self) -> None:
        steps.add(brain=self.brain, key="BUG-002", lines=["Find it."], by="owner")
        task.turn_on(brain=self.brain, key="BUG-002")
        task.mark(brain=self.brain, status="done")

        self.assertEqual(task.describe(self.brain), "bug BUG-002, every step settled, close it with brain set BUG-002 status=fixed once Fix names the cause and the commit")

        goal = plans.create(brain=self.brain, title="A goal", description="a goal alone", body="## Why\n\nA goal.\n\n## Steps\n\n1. Find out.\n\n## Done when\n\nMet.\n")
        task.turn_on(brain=self.brain, key=goal.name, replace=True)
        task.mark(brain=self.brain, status="done")

        self.assertRegex(task.describe(self.brain), rf"^plan a-goal, every step settled, close it with brain set a-goal status=done done={A_DATE}$")

        fixtures.write(path=self.root / "docs" / "a-goal.md", text="---\nname: a-goal\nkind: doc\ndescription: a doc of the same name\nupdated: 2026-09-16\nlinks: []\n---\n\n# A goal\n")
        self.brain.refresh()

        self.assertRegex(task.describe(self.brain), rf"^plan a-goal, every step settled, close it with brain set plans/a-goal.md status=done done={A_DATE}$")

    def test_a_step_reopened_is_where_the_work_stands_again(self) -> None:
        self.with_steps()
        task.turn_on(brain=self.brain, key="TODO-001")
        task.mark(brain=self.brain, status="done")
        task.mark(brain=self.brain, status="done")

        task.reopen(brain=self.brain, number=1)
        self.assertEqual(task.describe(self.brain), "task TODO-001, step 1 of 2, Read the table.")

        task.mark(brain=self.brain, status="done")
        self.assertRegex(task.describe(self.brain), rf"^task TODO-001, every step settled, close it with brain set TODO-001 status=done done={A_DATE} once Done when names what closed it$")

        with self.assertRaises(steps.StepsError) as caught:
            task.reopen(brain=self.brain, number=3)
        self.assertEqual(str(caught.exception), "TODO-001-export-to-csv has 2 steps, and there is no step 3")

    def test_step_refuses_a_pending_step_and_an_entry_with_none_in_words(self) -> None:
        task.turn_on(brain=self.brain, key="TODO-001")
        with self.assertRaises(task.TaskError) as caught:
            task.reopen(brain=self.brain, number=1)
        self.assertEqual(str(caught.exception), "TODO-001-export-to-csv has no steps, add some with brain steps TODO-001 add")

        self.with_steps()
        with self.assertRaises(task.TaskError) as caught:
            task.reopen(brain=self.brain, number=1)
        self.assertEqual(str(caught.exception), "step 1 is pending already, and the work stands on it")

        task.mark(brain=self.brain, status="done")
        with self.assertRaises(task.TaskError) as caught:
            task.reopen(brain=self.brain, number=2)
        self.assertEqual(str(caught.exception), "step 2 is pending already, and the work stands on it")

        task.reopen(brain=self.brain, number=1)
        with self.assertRaises(task.TaskError) as caught:
            task.reopen(brain=self.brain, number=2)
        self.assertEqual(str(caught.exception), "step 2 is pending already, and the work stands on step 1")

    def test_a_pointed_step_a_review_and_work_without_steps_take_no_mark(self) -> None:
        self.with_plan()
        task.turn_on(brain=self.brain, key="kit-phase-7")

        with self.assertRaises(steps.StepsError) as caught:
            task.mark(brain=self.brain, status="done")
        self.assertEqual(str(caught.exception), "step 1 points at TODO-001, so its mark follows that entry, and goes done once TODO-001 is done")

        task.turn_on(brain=self.brain, key="TODO-001")
        with self.assertRaises(task.TaskError) as caught:
            task.mark(brain=self.brain, status="done")
        self.assertEqual(str(caught.exception), "TODO-001-export-to-csv has no steps, add some with brain steps TODO-001 add")

        review = self.with_review()
        task.turn_on(brain=self.brain, key=review)
        with self.assertRaises(task.TaskError) as caught:
            task.mark(brain=self.brain, status="done")
        self.assertEqual(str(caught.exception), "a review's step is its batch, read off the ledger, so it takes no mark here")

        task.turn_off(brain=self.brain, everything=True)
        with self.assertRaises(task.TaskError) as caught:
            task.mark(brain=self.brain, status="done")
        self.assertEqual(str(caught.exception), "no work is on, start it with brain task on")


class Chaining(TaskCase):
    def test_a_review_goes_under_the_todo_and_the_todo_under_the_plan_step_that_points_at_it(self) -> None:
        self.with_plan()
        review = self.with_review()

        task.turn_on(brain=self.brain, key="kit-phase-7")
        task.turn_on(brain=self.brain, key="TODO-001")
        task.turn_on(brain=self.brain, key=review)

        self.assertEqual(self.chain(), [("plan", "kit-phase-7"), ("todo", "TODO-001"), ("review", review)])
        self.assertEqual(task.describe(self.brain), "\n".join((
            "plan kit-phase-7, step 1 of 3, the export.",
            "task TODO-001",
            f"review {review}, batch 1 of 1, pending",
        )))

    def test_what_the_plan_step_does_not_point_at_stays_off_and_so_does_a_todo_under_a_todo(self) -> None:
        self.with_plan()
        task.turn_on(brain=self.brain, key="kit-phase-7")

        with self.assertRaises(task.TaskError) as caught:
            task.turn_on(brain=self.brain, key="BUG-002")
        self.assertEqual(str(caught.exception), "the work is on kit-phase-7, step 1 of 3, the export., finish it or drop it with brain task off first")

        writes.update(brain=self.brain, key="TODO-001", changes={"status": "done", "done": "2026-09-24"})
        self.assertEqual(task.describe(self.brain), "plan kit-phase-7, step 2 of 3, the login loop.")
        task.turn_on(brain=self.brain, key="BUG-002")
        self.assertEqual(self.chain(), [("plan", "kit-phase-7"), ("bug", "BUG-002")])

        with self.assertRaises(task.TaskError) as caught:
            task.turn_on(brain=self.brain, key="kit-phase-7")
        self.assertEqual(str(caught.exception), "kit-phase-7 is on the chain already, above BUG-002, and brain task off drops back to it")

        other = writes.create(brain=self.brain, kind=schema.kind_named("todo"), title="Another todo", description="another", values={"area": "reports", "by": "owner"}, body="## What\n\nWhat.\n\n## Done when\n\nDone.\n")
        third = writes.create(brain=self.brain, kind=schema.kind_named("todo"), title="A third todo", description="a third", values={"area": "reports", "by": "owner"}, body="## What\n\nWhat.\n\n## Done when\n\nDone.\n")
        steps.add(brain=self.brain, key=other.identifier, lines=["BUG-002, the loop."], by="owner")
        task.turn_on(brain=self.brain, key=other.identifier, replace=True)
        with self.assertRaises(task.TaskError) as caught:
            task.turn_on(brain=self.brain, key=third.identifier)
        self.assertEqual(str(caught.exception), f"the work is on {other.identifier}, step 1 of 1, the loop., finish it or drop it with brain task off first")

        task.turn_on(brain=self.brain, key="BUG-002")
        self.assertEqual(self.chain(), [("todo", other.identifier), ("bug", "BUG-002")])

    def test_the_entry_a_todos_current_step_points_at_goes_under_the_todo(self) -> None:
        steps.add(brain=self.brain, key="TODO-001", lines=["BUG-002, the login loop.", "Write it up."], by="owner")
        task.turn_on(brain=self.brain, key="TODO-001")

        task.turn_on(brain=self.brain, key="BUG-002")

        self.assertEqual(self.chain(), [("todo", "TODO-001"), ("bug", "BUG-002")])
        self.assertEqual(task.describe(self.brain), "task TODO-001, step 1 of 2, the login loop.\nbug BUG-002")

        task.turn_off(self.brain)
        self.with_plan()
        with self.assertRaises(task.TaskError) as caught:
            task.turn_on(brain=self.brain, key="kit-phase-7")
        self.assertEqual(str(caught.exception), "the work is on TODO-001, step 1 of 2, the login loop., finish it or drop it with brain task off first")

    def test_on_under_a_bug_whose_steps_do_not_read_is_refused_with_the_link_that_stands(self) -> None:
        path = self.root / "bugs" / "BUG-002-login-loop.md"
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace("## Evidence", "## Steps\n\n1. Open the app.\n2. Log in twice.\n\n## Evidence"))
        self.brain.refresh()
        task.turn_on(brain=self.brain, key="BUG-002")

        with self.assertRaises(task.TaskError) as caught:
            task.turn_on(brain=self.brain, key="TODO-001")
        self.assertEqual(str(caught.exception), "the work is on BUG-002, the steps table cannot be read, finish it or drop it with brain task off first")

    def test_on_for_the_innermost_link_leaves_the_chain_as_it_is(self) -> None:
        task.turn_on(brain=self.brain, key="TODO-001")
        before = self.file.read_text(encoding="utf-8")

        self.assertEqual(task.turn_on(brain=self.brain, key="TODO-001"), [task.Item(kind="todo", key="TODO-001")])
        self.assertEqual(self.file.read_text(encoding="utf-8"), before)

    def test_on_for_an_outer_link_says_it_is_on_the_chain_and_how_to_get_back_to_it(self) -> None:
        self.with_plan()
        task.turn_on(brain=self.brain, key="kit-phase-7")
        task.turn_on(brain=self.brain, key="TODO-001")

        with self.assertRaises(task.TaskError) as caught:
            task.turn_on(brain=self.brain, key="kit-phase-7")
        self.assertEqual(str(caught.exception), "kit-phase-7 is on the chain already, above TODO-001, and brain task off drops back to it")

    def test_step_on_a_step_settled_by_its_entry_says_reopening_the_entry_reopens_it(self) -> None:
        self.with_plan()
        task.turn_on(brain=self.brain, key="kit-phase-7")
        writes.update(brain=self.brain, key="TODO-001", changes={"status": "done", "done": "2026-09-24"})

        with self.assertRaises(steps.StepsError) as caught:
            task.reopen(brain=self.brain, number=1)
        self.assertEqual(str(caught.exception), "step 1 points at TODO-001, which is done, so its mark follows that entry, and brain set TODO-001 status=open reopens it, unless its steps and another open entry's would wait on each other, or its table or one they lead to does not read, which the set refuses")

    def test_a_link_whose_entry_closes_drops_off_with_everything_under_it(self) -> None:
        self.with_plan()
        review = self.with_review()
        task.turn_on(brain=self.brain, key="kit-phase-7")
        task.turn_on(brain=self.brain, key="TODO-001")
        task.turn_on(brain=self.brain, key=review)

        writes.update(brain=self.brain, key=review, changes={"status": "closed"}, stamp="2026-09-24")
        self.assertEqual(self.chain(), [("plan", "kit-phase-7"), ("todo", "TODO-001")])
        self.assertEqual(len(self.file_chain()), 3)

        writes.update(brain=self.brain, key="TODO-001", changes={"status": "done", "done": "2026-09-24"}, stamp="2026-09-24")
        self.assertEqual(self.chain(), [("plan", "kit-phase-7")])
        self.assertEqual(task.describe(self.brain), "plan kit-phase-7, step 2 of 3, the login loop.")

    def test_a_plan_stands_on_its_first_pending_step_and_with_no_steps_says_so(self) -> None:
        goal = plans.create(brain=self.brain, title="A goal", description="a goal alone", body="## Why\n\nA goal.\n\n## Done when\n\nMet.\n")

        task.turn_on(brain=self.brain, key=goal.name)
        self.assertEqual(task.describe(self.brain), "plan a-goal, no steps yet")

        with self.assertRaises(task.TaskError) as caught:
            task.turn_on(brain=self.brain, key="BUG-002")
        self.assertEqual(str(caught.exception), "the work is on a-goal, no steps yet, finish it or drop it with brain task off first")

        steps.add(brain=self.brain, key="a-goal", lines=["Find out.", "BUG-002, the login loop."])
        self.assertEqual(task.describe(self.brain), "plan a-goal, step 1 of 2, Find out.")

        task.mark(brain=self.brain, status="done")
        self.assertEqual(task.describe(self.brain), "plan a-goal, step 2 of 2, the login loop.")
        task.turn_on(brain=self.brain, key="BUG-002")
        self.assertEqual(self.chain(), [("plan", "a-goal"), ("bug", "BUG-002")])

        writes.update(brain=self.brain, key="BUG-002", changes={"status": "wontfix"})
        self.assertRegex(task.describe(self.brain), rf"^plan a-goal, every step settled, close it with brain set a-goal status=done done={A_DATE}$")


class Staying(TaskCase):
    def test_a_retitled_plan_keeps_its_place_on_the_chain(self) -> None:
        self.with_plan()
        task.turn_on(brain=self.brain, key="kit-phase-7")
        task.turn_on(brain=self.brain, key="TODO-001")

        writes.retitle(brain=self.brain, key="kit-phase-7", title="Kit phase seven")

        self.assertEqual(self.chain(), [("plan", "kit-phase-seven"), ("todo", "TODO-001")])
        self.assertEqual(task.describe(self.brain).split("\n")[0], "plan kit-phase-seven, step 1 of 3, the export.")

    @unittest.skipUnless(os.name == "posix" and os.geteuid() != 0, "needs a file the process cannot write")
    def test_a_retitle_whose_chain_cannot_follow_is_said_as_partial_and_the_file_stays_retitled(self) -> None:
        self.with_plan()
        task.turn_on(brain=self.brain, key="kit-phase-7")
        self.file.chmod(0o444)

        try:
            with self.assertRaises(writes.Partial) as caught:
                writes.retitle(brain=self.brain, key="kit-phase-7", title="Kit phase seven")
        finally:
            self.file.chmod(0o644)

        self.assertTrue(str(caught.exception).startswith("kit-phase-seven was written, and the chain could not follow the name, "), str(caught.exception))
        self.assertTrue(str(caught.exception).endswith(", so it says plan kit-phase-7, not in this checkout, until .task can be written again and brain task off drops it, link by link, for brain task on kit-phase-seven to start it"))
        self.assertTrue((self.root / "plans" / "kit-phase-seven.md").exists())
        self.brain.refresh()
        self.assertEqual(task.describe(self.brain), "plan kit-phase-7, not in this checkout")

    def test_a_link_whose_id_two_files_carry_stays_and_says_so(self) -> None:
        task.turn_on(brain=self.brain, key="TODO-001")
        fixtures.write(path=self.root / "todos" / "TODO-001-other-thing.md", text=(self.root / "todos" / "TODO-001-export-to-csv.md").read_text(encoding="utf-8").replace("TODO-001-export-to-csv", "TODO-001-other-thing"))
        self.brain.refresh()

        self.assertEqual(task.describe(self.brain), "task TODO-001, two files carry the id")
        self.assertEqual(self.file_chain(), [{"kind": "todo", "key": "TODO-001"}])
        with self.assertRaises(task.TaskError) as caught:
            task.mark(brain=self.brain, status="done")
        self.assertEqual(str(caught.exception), "TODO-001, two files carry the id")

    def test_a_plan_goes_on_the_chain_by_its_file_so_a_copy_that_kept_the_name_does_not_double_it(self) -> None:
        plan = plans.create(brain=self.brain, title="Kit", description="kit", body="## Why\n\nWhy.\n\n## Done when\n\nDone.\n")
        fixtures.write(path=self.root / "plans" / "kit-v2.md", text=plan.path.read_text(encoding="utf-8"))
        self.brain.refresh()

        task.turn_on(brain=self.brain, key="kit-v2")

        self.assertEqual(self.file_chain(), [{"kind": "plan", "key": "kit-v2"}])
        self.assertEqual(task.describe(self.brain), "plan kit-v2, no steps yet")

        task.turn_on(brain=self.brain, key="plans/kit.md", replace=True)

        self.assertEqual(self.file_chain(), [{"kind": "plan", "key": "kit"}])
        self.assertEqual(task.describe(self.brain), "plan kit, no steps yet")

    def test_a_plan_whose_name_another_file_answers_to_says_so_in_its_own_words(self) -> None:
        plan = plans.create(brain=self.brain, title="Kit", description="kit", body="## Why\n\nWhy.\n\n## Done when\n\nDone.\n")
        task.turn_on(brain=self.brain, key="kit")
        fixtures.write(path=self.root / "docs" / "sub" / "kit.md", text="---\nname: sub-kit\nkind: doc\ndescription: a doc\nupdated: 2026-09-16\nlinks: []\n---\n\n# Kit\n")
        self.brain.refresh()

        self.assertEqual(task.describe(self.brain), "plan kit, no steps yet")

        fixtures.write(path=self.root / "plans" / "sub" / "kit.md", text=plan.path.read_text(encoding="utf-8"))
        self.brain.refresh()

        self.assertEqual(task.describe(self.brain), "plan kit, two files answer to the name")

    def test_a_link_whose_status_the_tool_does_not_know_stays_and_says_so(self) -> None:
        review = self.with_review()
        task.turn_on(brain=self.brain, key="TODO-001")
        task.turn_on(brain=self.brain, key=review)
        path = self.root / "todos" / "TODO-001-export-to-csv.md"

        for status_line in ("", "status: paused\n"):
            with self.subTest(status_line=status_line):
                fixtures.write(path=path, text=re.sub(r"^status: open\n", status_line, path.read_text(encoding="utf-8"), count=1, flags=re.MULTILINE))
                self.brain.refresh()

                self.assertEqual(task.describe(self.brain).split("\n")[0], "task TODO-001, the status is not one the tool knows, which brain doctor reports")
                self.assertEqual(len(self.file_chain()), 2)

                task.turn_off(self.brain)
                with self.assertRaises(task.TaskError) as caught:
                    task.mark(brain=self.brain, status="done")
                self.assertEqual(str(caught.exception), "TODO-001, the status is not one the tool knows, which brain doctor reports")
                task.turn_on(brain=self.brain, key=review)

    def test_an_unreadable_todo_under_a_category_is_said_to_be_unreadable(self) -> None:
        created = writes.create(
            brain=self.brain,
            kind=schema.kind_named("todo"),
            title="Cat thing",
            description="a thing in a category",
            values={"by": "owner", "area": "reports"},
            category="area",
            stamp="2026-09-24",
        )
        task.turn_on(brain=self.brain, key=created.identifier)
        fixtures.write(path=created.path, text="# no header\n")
        self.brain.refresh()

        self.assertEqual(task.describe(self.brain), f"task {created.identifier}, cannot be read right now")

    def test_a_link_whose_entry_is_in_another_checkout_stays_and_says_so(self) -> None:
        self.with_plan()
        task.turn_on(brain=self.brain, key="kit-phase-7")
        task.turn_on(brain=self.brain, key="TODO-001")
        path = self.root / "todos" / "TODO-001-export-to-csv.md"
        away = self.root.parent / "TODO-001-export-to-csv.md"
        path.rename(away)
        self.brain.refresh()

        self.assertEqual(task.describe(self.brain), "plan kit-phase-7, step 1 of 3, the export.\ntask TODO-001, not in this checkout")
        self.assertEqual(len(self.file_chain()), 2)
        with self.assertRaises(task.TaskError) as caught:
            task.mark(brain=self.brain, status="done")
        self.assertEqual(str(caught.exception), "TODO-001, not in this checkout")

        away.rename(path)
        self.brain.refresh()
        self.assertEqual(task.describe(self.brain), "plan kit-phase-7, step 1 of 3, the export.\ntask TODO-001")

    def test_off_drops_the_link_as_written_so_a_close_just_before_it_takes_nothing_more(self) -> None:
        self.with_plan()
        task.turn_on(brain=self.brain, key="kit-phase-7")
        task.turn_on(brain=self.brain, key="TODO-001")
        writes.update(brain=self.brain, key="TODO-001", changes={"status": "done", "done": "2026-09-24"}, stamp="2026-09-24")

        remaining = task.turn_off(self.brain)

        self.assertEqual(remaining, [task.Item(kind="plan", key="kit-phase-7")])

    def test_a_mark_after_the_innermost_link_closed_is_refused_until_off_drops_it(self) -> None:
        plans.create(brain=self.brain, title="Kit", description="kit", body="## Why\n\nWhy.\n\n## Steps\n\n1. TODO-001, the export.\n2. Check the export by hand.\n\n## Done when\n\nDone.\n", stamp="2026-09-24")
        task.turn_on(brain=self.brain, key="kit")
        task.turn_on(brain=self.brain, key="TODO-001")
        writes.update(brain=self.brain, key="TODO-001", changes={"status": "done", "done": "2026-09-24"}, stamp="2026-09-24")

        self.assertEqual(task.describe(self.brain), "plan kit, step 2 of 2, Check the export by hand.")
        for move in (lambda: task.mark(brain=self.brain, status="done"), lambda: task.reopen(brain=self.brain, number=1)):
            with self.assertRaises(task.TaskError) as caught:
                move()
            self.assertEqual(str(caught.exception), "TODO-001 has closed and holds the place until brain task off drops it")
        self.assertEqual([step.status for step in steps.rows_of(brain=self.brain, entry=self.brain.find("kit"))], ["done", "pending"])

        task.turn_off(self.brain)
        task.mark(brain=self.brain, status="done")
        self.assertRegex(task.describe(self.brain), rf"^plan kit, every step settled, close it with brain set kit status=done done={A_DATE}$")

    def test_a_reading_between_the_close_and_off_writes_nothing_so_off_still_takes_only_the_closed_link(self) -> None:
        self.with_plan()
        task.turn_on(brain=self.brain, key="kit-phase-7")
        task.turn_on(brain=self.brain, key="TODO-001")
        writes.update(brain=self.brain, key="TODO-001", changes={"status": "done", "done": "2026-09-24"}, stamp="2026-09-24")

        self.assertEqual(task.describe(self.brain), "plan kit-phase-7, step 2 of 3, the login loop.")
        self.assertEqual(len(self.file_chain()), 2)
        self.assertEqual(task.turn_off(self.brain), [task.Item(kind="plan", key="kit-phase-7")])

    def test_a_chain_set_meanwhile_is_never_written_over_by_an_older_reading(self) -> None:
        self.with_plan()
        task.turn_on(brain=self.brain, key="kit-phase-7")
        task.turn_on(brain=self.brain, key="TODO-001")
        older = task.read(self.brain)
        writes.update(brain=self.brain, key="TODO-001", changes={"status": "done", "done": "2026-09-24"}, stamp="2026-09-24")
        task.turn_on(brain=self.brain, key="BUG-002", replace=True)

        task.current(brain=self.brain, chain=older)

        self.assertEqual(self.file_chain(), [{"kind": "bug", "key": "BUG-002"}])

    def test_a_broken_file_elsewhere_with_a_matching_name_does_not_hold_a_gone_link_as_unreadable(self) -> None:
        plan = plans.create(brain=self.brain, title="Kit", description="kit", body="## Why\n\nWhy.\n\n## Done when\n\nDone.\n")
        task.turn_on(brain=self.brain, key="kit")
        plan.path.unlink()

        for elsewhere in (self.root / "docs" / "kit-colors.md", self.root / "plans" / "kit-colors.md"):
            with self.subTest(elsewhere=elsewhere.relative_to(self.root)):
                fixtures.write(path=elsewhere, text="# no header\n")
                self.brain.refresh()

                self.assertEqual(task.describe(self.brain), "plan kit, not in this checkout")

                elsewhere.unlink()

        fixtures.write(path=self.root / "plans" / "kit.md", text="# no header\n")
        self.brain.refresh()
        self.assertEqual(task.describe(self.brain), "plan kit, cannot be read right now")

    def test_a_folder_in_the_files_place_is_named_and_never_written_through(self) -> None:
        self.file.mkdir()
        folder = ".task is a folder the tool never made, move it away by hand"

        self.assertEqual(task.describe(self.brain), f"task none, {folder}")
        for move in (
            lambda: task.turn_off(brain=self.brain, everything=True),
            lambda: task.turn_off(self.brain),
            lambda: task.turn_on(brain=self.brain, key="TODO-001", replace=True),
            lambda: task.turn_on(brain=self.brain, key="TODO-001"),
            lambda: task.mark(brain=self.brain, status="done"),
            lambda: task.reopen(brain=self.brain, number=1),
        ):
            with self.assertRaises(task.TaskError) as caught:
                move()
            self.assertEqual(str(caught.exception), folder)
        self.assertTrue(self.file.is_dir())

        self.file.rmdir()
        task.write(brain=self.brain, chain=[])
        task.write(brain=self.brain, chain=[])
        self.assertFalse(self.file.exists())


class Clearing(TaskCase):
    def test_off_drops_the_innermost_link_and_off_all_takes_the_file_away(self) -> None:
        self.with_plan()
        task.turn_on(brain=self.brain, key="kit-phase-7")
        task.turn_on(brain=self.brain, key="TODO-001")

        self.assertEqual(task.turn_off(self.brain), [task.Item(kind="plan", key="kit-phase-7")])
        self.assertEqual(self.chain(), [("plan", "kit-phase-7")])

        self.assertEqual(task.turn_off(self.brain), [])
        self.assertFalse(self.file.exists())
        self.assertEqual(task.turn_off(self.brain), [])

        task.turn_on(brain=self.brain, key="kit-phase-7")
        task.turn_on(brain=self.brain, key="TODO-001")
        self.assertEqual(task.turn_off(brain=self.brain, everything=True), [])
        self.assertFalse(self.file.exists())

    def test_a_file_that_does_not_read_as_a_chain_is_said_and_only_off_all_clears_it(self) -> None:
        broken = "task none, .task does not read as a chain and brain task off --all clears it"

        for text in ("not json", '{"chain": [{"kind": "doc", "key": "login"}]}', '{"chain": [{"kind": "todo", "key": ""}]}', '{"chain": "TODO-001"}'):
            with self.subTest(text=text):
                self.file.write_text(text, encoding="utf-8")

                self.assertEqual(task.describe(self.brain), broken)
                with self.assertRaises(task.TaskError):
                    task.turn_on(brain=self.brain, key="TODO-001")
                with self.assertRaises(task.TaskError):
                    task.mark(brain=self.brain, status="done")
                with self.assertRaises(task.TaskError):
                    task.turn_off(self.brain)
                self.assertEqual(self.file.read_text(encoding="utf-8"), text)

        task.turn_on(brain=self.brain, key="TODO-001", replace=True)
        self.assertEqual(self.chain(), [("todo", "TODO-001")])

        self.file.write_text("not json", encoding="utf-8")
        task.turn_off(brain=self.brain, everything=True)
        self.assertFalse(self.file.exists())

    def test_off_all_over_a_broken_file_cleared_meanwhile_is_no_error(self) -> None:
        self.file.write_text("not json", encoding="utf-8")

        with mock.patch.object(Path, "unlink", side_effect=FileNotFoundError("gone already")):
            self.assertEqual(task.turn_off(brain=self.brain, everything=True), [])

    @unittest.skipUnless(os.name == "posix" and os.geteuid() != 0, "needs a folder the process cannot list")
    def test_a_link_the_brain_cannot_read_this_time_stays_on_the_chain(self) -> None:
        self.with_plan()
        task.turn_on(brain=self.brain, key="kit-phase-7")
        task.turn_on(brain=self.brain, key="TODO-001")
        folder = self.root / "todos"
        mode = folder.stat().st_mode

        os.chmod(folder, 0)
        try:
            self.brain.refresh()
            self.assertEqual(task.describe(self.brain), "plan kit-phase-7, step 1 of 3, the export.\ntask TODO-001, cannot be read right now")
        finally:
            os.chmod(folder, mode)

        self.brain.refresh()
        self.assertEqual(self.chain(), [("plan", "kit-phase-7"), ("todo", "TODO-001")])


class TheReview(TaskCase):
    def test_a_review_that_opens_goes_on_the_chain_and_says_why_when_it_cannot(self) -> None:
        task.turn_on(brain=self.brain, key="TODO-001")
        review = self.with_review()

        self.assertIsNone(task.push_review(brain=self.brain, name=review))
        self.assertEqual(self.chain(), [("todo", "TODO-001"), ("review", review)])

        self.assertIsNone(task.push_review(brain=self.brain, name=review))
        self.assertEqual(self.chain(), [("todo", "TODO-001"), ("review", review)])

        self.with_plan()
        task.turn_on(brain=self.brain, key="kit-phase-7", replace=True)
        self.assertEqual(
            task.push_review(brain=self.brain, name=review),
            "the work is on kit-phase-7, step 1 of 3, the export., finish it or drop it with brain task off first",
        )

        task.turn_off(brain=self.brain, everything=True)
        self.assertIsNone(task.push_review(brain=self.brain, name=review))
        self.assertEqual(self.chain(), [("review", review)])

    def test_a_review_goes_under_the_plan_whose_current_step_points_at_it_by_name(self) -> None:
        review = self.with_review()
        plans.create(brain=self.brain, title="Review plan", description="the review", body=f"## Why\n\nWhy.\n\n## Steps\n\n1. {review}, the review.\n\n## Done when\n\nDone.\n", stamp="2026-09-25")
        task.turn_on(brain=self.brain, key="review-plan")

        task.turn_on(brain=self.brain, key=review)

        self.assertEqual(self.chain(), [("plan", "review-plan"), ("review", review)])

        task.turn_off(brain=self.brain, everything=True)
        task.turn_on(brain=self.brain, key="review-plan")
        self.assertIsNone(task.push_review(brain=self.brain, name=review))
        self.assertEqual(self.chain(), [("plan", "review-plan"), ("review", review)])

    @unittest.skipUnless(os.name == "posix" and os.geteuid() != 0, "needs a folder the process cannot write")
    def test_a_review_the_chain_cannot_take_says_why_without_the_machines_path(self) -> None:
        task.turn_on(brain=self.brain, key="TODO-001")
        review = self.with_review()
        mode = self.root.stat().st_mode
        os.chmod(self.root, 0o555)
        try:
            reason = task.push_review(brain=self.brain, name=review)
        finally:
            os.chmod(self.root, mode)

        self.assertIsNotNone(reason)
        self.assertIn("Permission denied", reason)
        self.assertNotIn(str(self.root.parent), reason)

    def test_the_review_line_follows_the_batches(self) -> None:
        review = self.with_review()
        task.turn_on(brain=self.brain, key=review)

        ledger.set_batch(brain=self.brain, key=review, number=1, status="done")
        self.assertEqual(task.describe(self.brain), f"review {review}, every batch done, 1 in all")

        ledger.add_batch(brain=self.brain, key=review, theme="closing pass")
        ledger.set_batch(brain=self.brain, key=review, number=2, status="fixing", ack="2026-09-24")
        self.assertEqual(task.describe(self.brain), f"review {review}, batch 2 of 2, fixing")

        task.turn_on(brain=self.brain, key="2026-09-feat-billing", replace=True)
        self.assertEqual(task.describe(self.brain), "review 2026-09-feat-billing, the batch table cannot be read")
