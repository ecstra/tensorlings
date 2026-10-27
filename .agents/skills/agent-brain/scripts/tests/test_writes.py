"""Filing something new, and changing something already filed."""

import unittest
from unittest import mock

from brainlib import frontmatter, schema, sections, store, writes
from tests import fixtures

TODO_BODY = """## What

Export the ledger to CSV, the way the owner wrote it:
  keep   this   spacing.

## Done when

The button is there and the file opens in Numbers.
"""


class Filing(fixtures.BrainCase):
    def file_todo(
        self,
        body: str = TODO_BODY,
        **values: str,
    ) -> store.Entry:
        return writes.create(
            brain=self.brain,
            kind=schema.kind_named("todo"),
            title="Export the ledger",
            description="export the ledger to csv",
            values=values or {"by": "owner", "area": "reports"},
            body=body,
            stamp="2026-09-16",
        )

    def test_it_lands_at_the_next_id(self) -> None:
        entry = self.file_todo()

        self.assertEqual(entry.relative, "todos/TODO-004-export-the-ledger.md")

    def test_the_prose_arrives_exactly_as_it_was_written(self) -> None:
        entry = self.file_todo()
        written = entry.path.read_text(encoding="utf-8")

        self.assertIn("  keep   this   spacing.", written)
        self.assertTrue(written.endswith("Numbers.\n"))
        self.assertIn("---\n\n## What\n", written)

    def test_the_fields_are_written_in_spec_order(self) -> None:
        entry = self.file_todo()

        self.assertEqual(
            [line.split(":")[0] for line in entry.document.lines],
            ["name", "kind", "description", "updated", "links", "status", "by", "area", "born"],
        )

    def test_what_a_new_todo_can_fill_in_it_fills_in(self) -> None:
        entry = self.file_todo()

        self.assertEqual(entry.field("status"), "open")
        self.assertEqual(entry.field("born"), "2026-09-16")

    def test_an_empty_body_becomes_the_required_headings(self) -> None:
        entry = self.file_todo(body="")

        self.assertEqual(
            sections.headings(entry.document.body),
            ["What", "Done when"],
        )

    def test_a_body_of_nothing_but_whitespace_gets_the_headings_too(self) -> None:
        entry = self.file_todo(body="  \n\n\t\n")

        self.assertEqual(sections.headings(entry.document.body), ["What", "Done when"])

    def test_a_field_with_a_line_break_is_refused_and_nothing_is_written(self) -> None:
        for description in ("first\nsecond: no", "x\n---\nboom"):
            with self.subTest(description=description):
                with self.assertRaises(writes.WriteError) as caught:
                    writes.create(
                        brain=self.brain,
                        kind=schema.kind_named("todo"),
                        title="Export the ledger",
                        description=description,
                        values={"by": "owner", "area": "reports"},
                        stamp="2026-09-16",
                    )

                self.assertIn("description is one line", str(caught.exception))
                self.assertEqual(list((self.root / "todos").glob("TODO-004-*")), [])

        with self.assertRaises(writes.WriteError):
            writes.update(brain=self.brain, key="BUG-002", changes={"area": "a\nb"})

        self.assertEqual(self.brain.find("BUG-002").field("area"), "auth")

    def test_a_padded_value_lands_trimmed(self) -> None:
        entry = self.file_todo(body="", by="owner", area="  reports  ")

        self.assertIn("\narea: reports\n", entry.path.read_text(encoding="utf-8"))

    def test_a_description_of_nothing_but_whitespace_is_refused(self) -> None:
        with self.assertRaises(writes.WriteError) as caught:
            writes.create(
                brain=self.brain,
                kind=schema.kind_named("todo"),
                title="Export the ledger",
                description="   ",
                values={"by": "owner", "area": "reports"},
                stamp="2026-09-16",
            )

        self.assertIn("needs description filled in", str(caught.exception))

    def test_a_new_file_holds_to_its_rules_from_the_start(self) -> None:
        with self.assertRaises(writes.WriteError) as caught:
            writes.create(
                brain=self.brain,
                kind=schema.kind_named("bug"),
                title="Ghost crash",
                description="crashes on nothing",
                values={"severity": "bug", "area": "auth", "status": "fixed"},
                stamp="2026-09-16",
            )

        self.assertEqual(str(caught.exception), "a bug goes fixed once its Fix section names the cause and the commit")
        self.assertEqual(list((self.root / "bugs").glob("BUG-003-*")), [])

    def test_every_rule_a_change_falls_short_of_is_named_at_once(self) -> None:
        with self.assertRaises(writes.WriteError) as caught:
            writes.update(brain=self.brain, key="TODO-001", changes={"status": "refiled"})

        self.assertEqual(
            str(caught.exception),
            "a todo goes refiled with its closing date, done=YYYY-MM-DD, and a refiled todo links to the bug it became",
        )

    def test_a_body_missing_a_required_heading_is_refused(self) -> None:
        with self.assertRaises(writes.WriteError) as caught:
            self.file_todo(body="## What\n\nJust this.\n")

        self.assertIn("Done when", str(caught.exception))

    def test_a_missing_field_is_refused(self) -> None:
        with self.assertRaises(writes.WriteError) as caught:
            self.file_todo(by="owner")

        self.assertIn("area", str(caught.exception))

    def test_a_value_outside_the_spec_is_refused(self) -> None:
        with self.assertRaises(writes.WriteError) as caught:
            self.file_todo(by="nobody", area="reports")

        self.assertIn("owner", str(caught.exception))

    def test_a_field_the_kind_does_not_carry_is_refused(self) -> None:
        with self.assertRaises(writes.WriteError) as caught:
            self.file_todo(by="owner", area="reports", severity="bug")

        self.assertIn("severity", str(caught.exception))

    def test_a_date_that_is_not_a_date_is_refused(self) -> None:
        with self.assertRaises(writes.WriteError) as caught:
            self.file_todo(by="owner", area="reports", born="last tuesday")

        self.assertIn("YYYY-MM-DD", str(caught.exception))

    def test_filing_the_same_title_twice_is_refused(self) -> None:
        self.file_todo()

        with self.assertRaises(writes.WriteError):
            self.file_todo()

    def test_a_name_one_of_the_kits_own_files_carries_is_refused_too(self) -> None:
        fixtures.write(path=self.root / "spec" / "conventions.md", text="---\nname: conventions\nkind: doc\ndescription: the law\nupdated: 2026-09-16\nlinks: []\n---\n\n# Conventions\n")

        with self.assertRaises(writes.WriteError) as caught:
            writes.create(
                brain=self.brain,
                kind=schema.kind_named("doc"),
                title="Conventions",
                description="a doc about them",
                stamp="2026-09-16",
            )

        self.assertEqual(str(caught.exception), "conventions is already the name of spec/conventions.md, and a name is unique across the brain")

    def test_a_name_another_kind_already_carries_is_refused(self) -> None:
        with self.assertRaises(writes.WriteError) as caught:
            writes.create(
                brain=self.brain,
                kind=schema.kind_named("memory"),
                title="Login",
                description="how login goes",
                values={"type": "project"},
                stamp="2026-09-16",
            )

        self.assertEqual(str(caught.exception), "login is already the name of docs/systems/login.md, and a name is unique across the brain")
        self.assertFalse((self.root / "memory" / "login.md").exists())

    def test_a_header_value_the_house_style_bans_is_refused(self) -> None:
        with self.assertRaises(writes.WriteError) as caught:
            writes.create(
                brain=self.brain,
                kind=schema.kind_named("doc"),
                title="Auth flow",
                description="the token exchange — end to end",
                values={},
                stamp="2026-09-16",
            )
        self.assertEqual(str(caught.exception), "description carries a dash the house style bans")

        with self.assertRaises(writes.WriteError) as caught:
            writes.update(brain=self.brain, key="TODO-001", changes={"area": "reports; exports"})
        self.assertEqual(str(caught.exception), "area carries a semicolon")

    def test_a_link_is_filed_as_a_path_whatever_form_it_came_in(self) -> None:
        entry = writes.create(
            brain=self.brain,
            kind=schema.kind_named("todo"),
            title="Export the ledger",
            description="export the ledger to csv",
            values={"by": "owner", "area": "reports"},
            links=["BUG-002", str(self.root / "decisions" / "DEC-001-postgres-over-sqlite.md")],
            stamp="2026-09-16",
        )

        self.assertEqual(
            entry.document.get_list("links"),
            ["bugs/BUG-002-login-loop.md", "decisions/DEC-001-postgres-over-sqlite.md"],
        )

    def test_one_file_linked_twice_over_is_written_once(self) -> None:
        entry = writes.create(
            brain=self.brain,
            kind=schema.kind_named("todo"),
            title="Export the ledger",
            description="export the ledger to csv",
            values={"by": "owner", "area": "reports"},
            links=["login", "docs/systems/login.md", "BUG-002"],
            stamp="2026-09-16",
        )

        self.assertEqual(entry.document.get_list("links"), ["docs/systems/login.md", "bugs/BUG-002-login-loop.md"])

        again = writes.update(brain=self.brain, key=entry.identifier, changes={"links": "[docs/systems/login.md, login, BUG-002, BUG-002]"})

        self.assertEqual(again.document.get_list("links"), ["docs/systems/login.md", "bugs/BUG-002-login-loop.md"])

    def test_a_kept_name_form_link_and_its_path_form_are_one_link(self) -> None:
        path = self.root / "todos" / "TODO-001-export-to-csv.md"
        document = frontmatter.parse(frontmatter.load(path))
        document.set_list(key="links", values=["login"])
        frontmatter.write(path=path, document=document)
        self.brain.refresh()

        again = writes.update(brain=self.brain, key="TODO-001", changes={"links": "[login, docs/systems/login.md]"})

        self.assertEqual(again.document.get_list("links"), ["login"])

    def test_a_link_that_reaches_nothing_in_the_brain_is_refused(self) -> None:
        outside = self.root.parent / ".claude" / "output-styles" / "plain-and-short.md"

        for reference in (str(outside), "../.claude/output-styles/plain-and-short.md", "BUG-999"):
            with self.subTest(reference=reference):
                with self.assertRaises(writes.WriteError) as caught:
                    writes.create(
                        brain=self.brain,
                        kind=schema.kind_named("todo"),
                        title="Export the ledger",
                        description="export the ledger to csv",
                        values={"by": "owner", "area": "reports"},
                        links=[reference],
                        stamp="2026-09-16",
                    )

                self.assertIn("reaches nothing inside .agents/", str(caught.exception))
                self.assertFalse((self.root / "todos" / "TODO-004-export-the-ledger.md").exists())

    def test_a_doc_lands_in_its_category_with_a_title(self) -> None:
        entry = writes.create(
            brain=self.brain,
            kind=schema.kind_named("doc"),
            title="Billing",
            description="how billing works",
            category="systems",
            stamp="2026-09-16",
        )

        self.assertEqual(entry.relative, "docs/systems/billing.md")
        self.assertIn("# Billing", entry.document.body)

    def test_a_review_not_titled_month_then_branch_is_refused(self) -> None:
        with self.assertRaises(writes.WriteError) as caught:
            writes.create(
                brain=self.brain,
                kind=schema.kind_named("review"),
                title="fix login",
                description="the login review",
                values={"branch": "fix/login"},
            )

        self.assertIn("<yyyy-mm>-<branch-slug>", str(caught.exception))

    def test_a_review_is_named_for_its_folder(self) -> None:
        entry = writes.create(
            brain=self.brain,
            kind=schema.kind_named("review"),
            title="2026-10 fix login",
            description="the login review",
            values={"branch": "fix/login"},
            body="## Summary\n\nOne branch.\n\n## Batches\n\nOne.\n\n## Findings\n\nNone yet.\n",
            stamp="2026-09-16",
        )

        self.assertEqual(entry.relative, "reviews/2026-10-fix-login/ledger.md")
        self.assertEqual(entry.name, "2026-10-fix-login")


class Changing(fixtures.BrainCase):
    def test_a_change_bumps_the_updated_date(self) -> None:
        entry = writes.update(
            brain=self.brain,
            key="TODO-001",
            changes={"area": "exports"},
            stamp="2026-10-01",
        )

        self.assertEqual(entry.field("area"), "exports")
        self.assertEqual(entry.field("updated"), "2026-10-01")

    def test_a_change_leaves_the_body_alone(self) -> None:
        before = (self.root / "todos" / "TODO-001-export-to-csv.md").read_text(
            encoding="utf-8"
        )

        entry = writes.update(
            brain=self.brain,
            key="TODO-001",
            changes={"area": "exports"},
            stamp="2026-10-01",
        )

        after = entry.path.read_text(encoding="utf-8")

        self.assertEqual(after.split("---\n")[2], before.split("---\n")[2])

    def test_links_set_by_name_are_written_as_paths(self) -> None:
        entry = writes.update(
            brain=self.brain,
            key="TODO-001",
            changes={"links": "[BUG-002]"},
            stamp="2026-10-01",
        )

        self.assertEqual(entry.document.get_list("links"), ["bugs/BUG-002-login-loop.md"])

    def test_a_link_the_file_already_carried_stays_while_a_new_one_must_reach_a_file(self) -> None:
        source = self.root / "todos" / "TODO-001-export-to-csv.md"
        document = frontmatter.read(str(source))
        document.set_list(key="links", values=["docs/gone.md"])
        frontmatter.write(path=str(source), document=document)
        self.brain.refresh()

        kept = writes.update(brain=self.brain, key="TODO-001", changes={"links": "[docs/gone.md, BUG-002]"})

        self.assertEqual(kept.document.get_list("links"), ["docs/gone.md", "bugs/BUG-002-login-loop.md"])

        with self.assertRaises(writes.WriteError):
            writes.update(brain=self.brain, key="TODO-001", changes={"links": "[docs/gone.md, docs/also-gone.md]"})

    def test_a_link_set_to_something_outside_the_brain_is_refused(self) -> None:
        with self.assertRaises(writes.WriteError) as caught:
            writes.update(
                brain=self.brain,
                key="TODO-001",
                changes={"links": f"[{self.root.parent / 'README.md'}]"},
            )

        self.assertIn("reaches nothing inside .agents/", str(caught.exception))

    def test_a_change_to_a_symlinked_entry_lands_in_the_brain_not_through_the_link(self) -> None:
        elsewhere = self.root.parent / "shared" / "TODO-001-export-to-csv.md"
        target = self.root / "todos" / "TODO-001-export-to-csv.md"
        elsewhere.parent.mkdir()
        target.rename(elsewhere)
        target.symlink_to(elsewhere)
        theirs = elsewhere.read_text(encoding="utf-8")

        writes.update(
            brain=self.brain,
            key="TODO-001",
            changes={"area": "exports"},
            stamp="2026-10-01",
        )

        self.assertFalse(target.is_symlink())
        self.assertIn("area: exports", target.read_text(encoding="utf-8"))
        self.assertEqual(elsewhere.read_text(encoding="utf-8"), theirs)

    def test_a_refiled_file_must_link_to_an_entry_of_the_other_kind_not_just_its_folder(self) -> None:
        for links in ("[todos/INDEX.md]", "[docs/systems/login.md]"):
            with self.subTest(links=links):
                with self.assertRaises(writes.WriteError) as caught:
                    writes.update(brain=self.brain, key="BUG-002", changes={"status": "refiled", "links": links})

                self.assertIn("links to the todo it became", str(caught.exception))

        todo = schema.kind_named("todo")
        self.assertFalse(sections.links_into(fields={"links": "[todos/BUG-002-login-loop.md]"}, kind=todo))
        self.assertTrue(sections.links_into(fields={"links": "[todos/TODO-001-export-to-csv.md]"}, kind=todo))

        entry = writes.update(brain=self.brain, key="BUG-002", changes={"status": "refiled", "links": "[todos/TODO-001-export-to-csv.md]"})

        self.assertEqual(entry.field("status"), "refiled")

    def test_a_change_that_leaves_a_file_breaking_its_rule_is_refused_whatever_field_moved(self) -> None:
        with self.assertRaises(writes.WriteError) as caught:
            writes.update(brain=self.brain, key="TODO-003", changes={"done": ""})

        self.assertEqual(str(caught.exception), "a todo goes done with its closing date, done=YYYY-MM-DD")
        self.assertEqual(self.brain.find("TODO-003").field("done"), "2026-09-10")

    def test_a_bug_cannot_go_fixed_with_an_empty_fix(self) -> None:
        with self.assertRaises(writes.WriteError) as caught:
            writes.update(
                brain=self.brain,
                key="BUG-002",
                changes={"status": "fixed"},
            )

        self.assertIn("Fix", str(caught.exception))

    def test_a_bug_goes_fixed_once_the_fix_is_written(self) -> None:
        path = self.root / "bugs" / "BUG-002-login-loop.md"
        path.write_text(
            path.read_text(encoding="utf-8") + "\n## Fix\n\nThe guard was inverted, abc1234.\n",
            encoding="utf-8",
        )

        entry = writes.update(
            brain=self.brain,
            key="BUG-002",
            changes={"status": "fixed"},
        )

        self.assertEqual(entry.field("status"), "fixed")

    def test_a_todo_cannot_go_done_without_the_date(self) -> None:
        with self.assertRaises(writes.WriteError) as caught:
            writes.update(
                brain=self.brain,
                key="TODO-001",
                changes={"status": "done"},
            )

        self.assertIn("done", str(caught.exception))

    def test_a_decision_cannot_be_superseded_by_nothing(self) -> None:
        with self.assertRaises(writes.WriteError) as caught:
            writes.update(
                brain=self.brain,
                key="DEC-001",
                changes={"status": "superseded"},
            )

        self.assertIn("replaced", str(caught.exception))

    def test_changing_something_two_files_answer_to_is_refused_in_the_writers_words(self) -> None:
        fixtures.write(
            path=self.root / "reviews" / "2026-10-fix-login" / "ledger.md",
            text=(self.root / "reviews" / "2026-09-feat-billing" / "ledger.md").read_text(encoding="utf-8").replace("2026-09-feat-billing", "2026-10-fix-login"),
        )

        with self.assertRaises(writes.WriteError) as caught:
            writes.update(brain=self.brain, key="ledger", changes={"status": "closed"})

        self.assertIn("'ledger' names 2 files, use the path", str(caught.exception))

    def test_changing_something_that_is_not_filed_is_refused(self) -> None:
        with self.assertRaises(writes.WriteError) as caught:
            writes.update(
                brain=self.brain,
                key="BUG-404",
                changes={"status": "fixed"},
            )

        self.assertIn("BUG-404", str(caught.exception))


class Retitling(fixtures.BrainCase):
    def setUp(self) -> None:
        super().setUp()
        path = self.root / "todos" / "TODO-001-export-to-csv.md"
        document = frontmatter.read(str(path))
        document.set_list(key="links", values=["bugs/BUG-002-login-loop.md"])
        frontmatter.write(path=str(path), document=document)

    def test_the_id_stays_and_the_slug_changes(self) -> None:
        entry = writes.retitle(
            brain=self.brain,
            key="BUG-002",
            title="Login loops on expiry",
            stamp="2026-10-01",
        )

        self.assertEqual(entry.relative, "bugs/BUG-002-login-loops-on-expiry.md")
        self.assertEqual(entry.name, "BUG-002-login-loops-on-expiry")
        self.assertEqual(entry.field("updated"), "2026-10-01")
        self.assertFalse((self.root / "bugs" / "BUG-002-login-loop.md").exists())

    def test_every_link_to_it_follows(self) -> None:
        writes.retitle(
            brain=self.brain,
            key="BUG-002",
            title="Login loops on expiry",
        )

        self.assertEqual(
            self.brain.find("TODO-001").document.get_list("links"),
            ["bugs/BUG-002-login-loops-on-expiry.md"],
        )

    def test_a_file_without_an_id_is_just_its_slug(self) -> None:
        entry = writes.retitle(
            brain=self.brain,
            key="stripe-retries-twice",
            title="Stripe retries a webhook twice",
        )

        self.assertEqual(entry.relative, "memory/stripe-retries-a-webhook-twice.md")

    def test_a_link_in_prose_follows_a_retitle(self) -> None:
        memory = self.root / "memory" / "stripe-retries-twice.md"
        memory.write_text(memory.read_text(encoding="utf-8") + "\nSee [the bug](../bugs/BUG-002-login-loop.md).\n", encoding="utf-8")

        writes.retitle(
            brain=self.brain,
            key="BUG-002",
            title="Login loops on expiry",
        )

        self.assertIn("[the bug](../bugs/BUG-002-login-loops-on-expiry.md)", memory.read_text(encoding="utf-8"))

    def test_every_link_that_follows_gets_the_date_too(self) -> None:
        writes.retitle(
            brain=self.brain,
            key="BUG-002",
            title="Login loops on expiry",
            stamp="2026-10-01",
        )

        self.assertEqual(self.brain.find("TODO-001").field("updated"), "2026-10-01")
        self.assertEqual(self.brain.find("TODO-003").field("updated"), "2026-09-16")

    def test_a_retitle_that_cannot_write_its_own_file_moves_no_link(self) -> None:
        bug = self.root / "bugs" / "BUG-002-login-loop.md"
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"
        before = todo.read_text(encoding="utf-8")
        bug.chmod(0o444)

        try:
            with self.assertRaises(PermissionError):
                writes.retitle(
                    brain=self.brain,
                    key="BUG-002",
                    title="Sign in loop",
                )
        finally:
            bug.chmod(0o644)

        self.assertEqual(todo.read_text(encoding="utf-8"), before)
        self.assertTrue(bug.is_file())
        self.assertEqual(self.brain.find("BUG-002").field("updated"), "2026-09-16")

    def test_a_retitle_whose_rename_fails_puts_everything_back(self) -> None:
        bug = self.root / "bugs" / "BUG-002-login-loop.md"
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"
        before = (bug.read_text(encoding="utf-8"), todo.read_text(encoding="utf-8"))

        with mock.patch("pathlib.Path.rename", side_effect=OSError(30, "Read-only file system")):
            with self.assertRaises(OSError):
                writes.retitle(
                    brain=self.brain,
                    key="BUG-002",
                    title="Sign in loop",
                )

        self.assertEqual((bug.read_text(encoding="utf-8"), todo.read_text(encoding="utf-8")), before)

    def test_a_retitle_that_cannot_put_a_link_back_names_both_failures(self) -> None:
        stuck = self.root / "todos" / "TODO-001-export-to-csv.md"

        with mock.patch("pathlib.Path.rename", side_effect=OSError(30, "Read-only file system")):
            with mock.patch("brainlib.links.put_back_all", return_value=[stuck]):
                with self.assertRaises(OSError) as caught:
                    writes.retitle(
                        brain=self.brain,
                        key="BUG-002",
                        title="Sign in loop",
                    )

        self.assertEqual(
            str(caught.exception),
            "[Errno 30] Read-only file system, and .agents/todos/TODO-001-export-to-csv.md could not be put back",
        )

    def test_a_retitle_to_its_own_title_is_a_restamp_not_a_refusal(self) -> None:
        entry = writes.retitle(
            brain=self.brain,
            key="BUG-002",
            title="Login loop",
            stamp="2026-10-01",
        )

        self.assertEqual(entry.relative, "bugs/BUG-002-login-loop.md")
        self.assertEqual(entry.field("updated"), "2026-10-01")

    def test_a_retitle_that_only_changes_case_goes_through(self) -> None:
        path = self.root / "bugs" / "BUG-002-login-loop.md"
        path.rename(path.with_name("BUG-002-Login-Loop.md"))
        self.brain.refresh()

        entry = writes.retitle(
            brain=self.brain,
            key="BUG-002",
            title="Login loop",
        )

        self.assertEqual(entry.relative, "bugs/BUG-002-login-loop.md")
        self.assertEqual(sorted(p.name for p in (self.root / "bugs").glob("BUG-002-*")), ["BUG-002-login-loop.md"])

    def test_a_file_already_on_disk_is_never_written_over(self) -> None:
        loose = self.root / "memory" / "loose.md"
        loose.write_text("# Just prose, no frontmatter\n", encoding="utf-8")

        with self.assertRaises(writes.WriteError) as caught:
            writes.retitle(
                brain=self.brain,
                key="stripe-retries-twice",
                title="Loose",
            )

        self.assertIn("memory/loose.md is already on disk", str(caught.exception))
        self.assertEqual(loose.read_text(encoding="utf-8"), "# Just prose, no frontmatter\n")
        self.assertTrue((self.root / "memory" / "stripe-retries-twice.md").is_file())

    def test_a_title_another_file_carries_is_refused(self) -> None:
        with self.assertRaises(writes.WriteError):
            writes.retitle(
                brain=self.brain,
                key="TODO-003",
                title="Export to CSV",
            )

    def test_a_retitle_may_not_take_a_name_another_kind_carries(self) -> None:
        with self.assertRaises(writes.WriteError) as caught:
            writes.retitle(
                brain=self.brain,
                key="stripe-retries-twice",
                title="Login",
            )

        self.assertEqual(str(caught.exception), "login is already the name of docs/systems/login.md, and a name is unique across the brain")
        self.assertTrue((self.root / "memory" / "stripe-retries-twice.md").is_file())

    def test_a_review_keeps_its_branch_name(self) -> None:
        with self.assertRaises(writes.WriteError):
            writes.retitle(
                brain=self.brain,
                key="2026-09-feat-billing",
                title="something else",
            )


class Converting(fixtures.BrainCase):
    def test_a_todo_becomes_a_bug_and_both_point_at_each_other(self) -> None:
        bug = writes.convert(
            brain=self.brain,
            key="TODO-001",
            kind=schema.kind_named("bug"),
            description="the export silently drops rows",
            values={"severity": "bug", "area": "reports"},
            body="## Problem\n\nRows go missing.\n\n## Evidence\n\nA diff.\n",
            stamp="2026-10-01",
        )
        todo = self.brain.find("TODO-001")

        self.assertEqual(bug.relative, "bugs/BUG-003-export-to-csv.md")
        self.assertEqual(bug.document.get_list("links"), ["todos/TODO-001-export-to-csv.md"])
        self.assertEqual(todo.field("status"), "refiled")
        self.assertEqual(todo.field("done"), "2026-10-01")
        self.assertEqual(todo.document.get_list("links"), ["bugs/BUG-003-export-to-csv.md"])
        self.assertEqual(sections.lifecycle_gaps(kind=todo.kind, fields=todo.document.fields(), body=todo.document.body), [])

    def test_a_bug_becomes_a_todo_the_same_way(self) -> None:
        todo = writes.convert(
            brain=self.brain,
            key="BUG-002",
            kind=schema.kind_named("todo"),
            description="make the login screen remember the session",
            values={"by": "agent", "area": "auth"},
            body="## What\n\nRemember the session.\n\n## Done when\n\nNo loop.\n",
        )

        self.assertEqual(todo.relative, "todos/TODO-004-login-loop.md")
        bug = self.brain.find("BUG-002")
        self.assertEqual(bug.field("status"), "refiled")
        self.assertEqual(sections.lifecycle_gaps(kind=bug.kind, fields=bug.document.fields(), body=bug.document.body), [])

    def test_a_convert_that_cannot_close_the_source_files_nothing_and_can_be_run_again(self) -> None:
        source = self.root / "todos" / "TODO-001-export-to-csv.md"
        before = source.read_text(encoding="utf-8")
        source.chmod(0o444)

        try:
            with self.assertRaises(PermissionError):
                writes.convert(
                    brain=self.brain,
                    key="TODO-001",
                    kind=schema.kind_named("bug"),
                    description="the export silently drops rows",
                    values={"severity": "bug", "area": "reports"},
                    body="## Problem\n\nRows go missing.\n\n## Evidence\n\nA diff.\n",
                )
        finally:
            source.chmod(0o644)

        self.assertEqual(source.read_text(encoding="utf-8"), before)
        self.assertEqual(list((self.root / "bugs").glob("BUG-003-*")), [])

        again = writes.convert(
            brain=self.brain,
            key="TODO-001",
            kind=schema.kind_named("bug"),
            description="the export silently drops rows",
            values={"severity": "bug", "area": "reports"},
            body="## Problem\n\nRows go missing.\n\n## Evidence\n\nA diff.\n",
        )

        self.assertEqual(again.relative, "bugs/BUG-003-export-to-csv.md")
        self.assertEqual(self.brain.find("TODO-001").field("status"), "refiled")

    def test_a_stale_link_the_source_already_carried_does_not_block_a_refile(self) -> None:
        source = self.root / "todos" / "TODO-001-export-to-csv.md"
        document = frontmatter.read(str(source))
        document.set_list(key="links", values=["docs/gone.md"])
        frontmatter.write(path=str(source), document=document)
        self.brain.refresh()

        bug = writes.convert(
            brain=self.brain,
            key="TODO-001",
            kind=schema.kind_named("bug"),
            description="the export silently drops rows",
            values={"severity": "bug", "area": "reports"},
            body="## Problem\n\nRows go missing.\n\n## Evidence\n\nA diff.\n",
        )

        self.assertEqual(self.brain.find("TODO-001").document.get_list("links"), ["docs/gone.md", bug.relative])

    def test_converting_to_its_own_kind_is_refused(self) -> None:
        with self.assertRaises(writes.WriteError):
            writes.convert(
                brain=self.brain,
                key="TODO-001",
                kind=schema.kind_named("todo"),
                description="x",
            )

    def test_only_todos_and_bugs_convert(self) -> None:
        with self.assertRaises(writes.WriteError):
            writes.convert(
                brain=self.brain,
                key="DEC-001",
                kind=schema.kind_named("bug"),
                description="x",
            )


if __name__ == "__main__":
    unittest.main()
