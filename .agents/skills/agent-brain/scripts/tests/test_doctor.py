"""Every doctor check catches the one defect it exists for, and fixes what it can."""

import errno
import os
import shutil
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

from brainlib import doctor, frontmatter, index, ledger, plans, schema, steps, store, task, writes
from tests import fixtures


class DoctorCase(fixtures.BrainCase):
    def setUp(self) -> None:
        super().setUp()
        index.write(brain=self.brain, stamp="2026-09-16")
        doctor.run(brain=self.brain, fix=True)
        self.brain.refresh()

    def messages(
        self,
        check: str,
    ) -> list[str]:
        return [finding.message for finding in doctor.run(self.brain) if finding.check == check]

    def rewrite(
        self,
        relative: str,
        **fields: str,
    ) -> None:
        path = self.root / relative
        document = frontmatter.read(str(path))
        for key, value in fields.items():
            document.set(key=key.replace("_", "-"), value=value)
        frontmatter.write(path=str(path), document=document)
        self.brain.refresh()

    def strip(
        self,
        relative: str,
        key: str,
    ) -> None:
        path = self.root / relative
        document = frontmatter.read(str(path))
        document.lines = [line for line in document.lines if not line.startswith(f"{key}:")]
        frontmatter.write(path=str(path), document=document)
        self.brain.refresh()

    def append(
        self,
        relative: str,
        text: str,
    ) -> None:
        path = self.root / relative
        path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")
        self.brain.refresh()


class Repairing(DoctorCase):
    def test_one_fix_settles_a_wrong_name_and_the_index_that_lists_it(self) -> None:
        self.rewrite(relative="memory/stripe-retries-twice.md", name="wrong-name")

        left = doctor.run(brain=self.brain, fix=True)

        self.assertEqual(left, [])
        self.assertIn("[stripe-retries-twice]", (self.root / "memory" / "INDEX.md").read_text(encoding="utf-8"))

    def test_a_repair_that_fails_becomes_a_flag_that_says_why(self) -> None:
        self.strip(relative="memory/stripe-retries-twice.md", key="updated")
        target = self.root / "memory" / "stripe-retries-twice.md"
        target.chmod(0o444)

        try:
            left = doctor.run(brain=self.brain, fix=True)
        finally:
            target.chmod(0o644)

        self.assertEqual(
            [finding.line() for finding in left if finding.check == "frontmatter"],
            [
                "memory/stripe-retries-twice.md: frontmatter: no updated date, and the repair failed: "
                "Permission denied: .agents/memory/stripe-retries-twice.md [flag]"
            ],
        )

    def test_a_failed_repair_marks_its_own_finding_and_no_other_on_the_same_file(self) -> None:
        self.rewrite(relative="memory/stripe-retries-twice.md", links="[login, nowhere-at-all]")
        target = self.root / "memory" / "stripe-retries-twice.md"
        target.chmod(0o444)

        try:
            left = doctor.run(brain=self.brain, fix=True)
        finally:
            target.chmod(0o644)

        self.assertEqual(
            [finding.line() for finding in left if finding.check == "link"],
            [
                "memory/stripe-retries-twice.md: link: 'nowhere-at-all' reaches nothing [flag]",
                "memory/stripe-retries-twice.md: link: links not written as paths relative to .agents/, each once, and the repair failed: "
                "Permission denied: .agents/memory/stripe-retries-twice.md [flag]",
            ],
        )

    def test_a_repair_that_runs_and_clears_nothing_comes_back_as_a_flag_that_says_so(self) -> None:
        self.strip(relative="memory/stripe-retries-twice.md", key="updated")

        with mock.patch("brainlib.doctor.stamp_today", return_value=lambda: None):
            left = doctor.run(brain=self.brain, fix=True)

        self.assertEqual(
            [finding.line() for finding in left if finding.check == "frontmatter"],
            ["memory/stripe-retries-twice.md: frontmatter: no updated date, and the repair ran without clearing it [flag]"],
        )

    def test_a_missing_skill_is_for_a_person_not_the_link_repair(self) -> None:
        fixtures.ship_review(self.root)
        shutil.rmtree(self.root / "skills")

        found = [finding for finding in doctor.run(brain=self.brain, fix=True) if finding.check == "layout"]

        self.assertEqual(
            [finding.line() for finding in found],
            [
                ".agents/skills/agent-brain: layout: the skill itself is missing from .agents/skills/agent-brain, reinstall the kit [flag]",
                ".agents/skills/branch-review: layout: the skill itself is missing from .agents/skills/branch-review, reinstall the kit [flag]",
            ],
        )

    def test_a_skills_folder_the_process_cannot_write_names_itself_in_the_failed_repair(self) -> None:
        skills = self.root.parent / ".claude" / "skills"
        (skills / "agent-brain").unlink()
        skills.chmod(0o555)

        try:
            left = [finding.line() for finding in doctor.run(brain=self.brain, fix=True) if finding.check == "layout"]
        finally:
            skills.chmod(0o755)

        self.assertEqual(
            left,
            [".claude/skills/agent-brain: layout: the skill link is missing, so Claude Code cannot see the skill, and the repair failed: Permission denied: .claude/skills [flag]"],
        )

    def test_a_real_copy_of_the_skill_passes_even_with_nothing_left_to_link_to(self) -> None:
        fixtures.ship_review(self.root)
        doctor.run(brain=self.brain, fix=True)
        for name in schema.KIT_SKILL_NAMES:
            link = self.root.parent / ".claude" / "skills" / name
            link.unlink()
            shutil.copytree(self.root / "skills" / name, link)
        shutil.rmtree(self.root / "skills")

        self.assertEqual(self.messages("layout"), [])

    def test_a_fix_stops_when_a_pass_changes_nothing(self) -> None:
        self.strip(relative="memory/stripe-retries-twice.md", key="updated")
        target = self.root / "memory" / "stripe-retries-twice.md"
        target.chmod(0o444)

        try:
            with mock.patch("brainlib.doctor.check_all", wraps=doctor.check_all) as checking:
                doctor.run(brain=self.brain, fix=True)
        finally:
            target.chmod(0o644)

        self.assertLessEqual(checking.call_count, 3)


class ACleanBrain(DoctorCase):
    def test_reports_nothing(self) -> None:
        self.assertEqual([finding.line() for finding in doctor.run(self.brain)], [])

    def test_a_finding_reads_as_one_line(self) -> None:
        finding = doctor.Finding(path="bugs/x.md", check="fields", message="a bug needs a severity")

        self.assertEqual(finding.line(), "bugs/x.md: fields: a bug needs a severity [flag]")


class Leftovers(DoctorCase):
    def test_what_an_interrupted_write_left_is_reported_and_removed(self) -> None:
        fixtures.write(path=self.root / "todos" / ".TODO-001-export-to-csv.md.ab12cd34.tmp", text="half\n")
        fixtures.write(path=self.root / "..ack.k3j2_x9a.tmp", text='{"since": "2026-09-23T12:00+05:30"\n')
        fixtures.write(path=self.root / "..task.k3j2_x9a.tmp", text='{"chain": [\n')
        fixtures.write(path=self.root / ".VERSION.k3j2_x9a.tmp", text="1.3.\n")
        fixtures.write(path=self.root / "skills" / "agent-brain" / ".SKILL.md.k3j2_x9a.tmp", text="half\n")
        fixtures.write(path=self.root / "skills" / "agent-brain" / "scripts" / "hooks.py", text="# the tool\n")
        fixtures.write(path=self.root / "skills" / "agent-brain" / "scripts" / ".hooks.py.k3j2_x9a.tmp", text="# half\n")
        fixtures.write(path=self.root / ".cache.tmp", text="a person's own\n")
        fixtures.write(path=self.root / "docs" / ".notes.md.tmp", text="a person's own\n")
        fixtures.write(path=self.root / "docs" / ".notes.markdown.tmp", text="a person's own\n")
        (self.root / "memory" / ".putback.xyz").mkdir()
        (self.root / "memory" / ".putback.xyz" / "stripe-retries-twice.md").symlink_to(self.root / "memory" / "stripe-retries-twice.md")

        found = doctor.run(self.brain)

        self.assertEqual(
            [finding.line() for finding in found],
            [
                "..ack.k3j2_x9a.tmp: leftover: left by an interrupted write, nothing in it is the brain's [fixable]",
                "..task.k3j2_x9a.tmp: leftover: left by an interrupted write, nothing in it is the brain's [fixable]",
                ".VERSION.k3j2_x9a.tmp: leftover: left by an interrupted write, nothing in it is the brain's [fixable]",
                "memory/.putback.xyz: leftover: left by an interrupted write, nothing in it is the brain's [fixable]",
                "skills/agent-brain/.SKILL.md.k3j2_x9a.tmp: leftover: left by an interrupted write, nothing in it is the brain's [fixable]",
                "skills/agent-brain/scripts/.hooks.py.k3j2_x9a.tmp: leftover: left by an interrupted write, nothing in it is the brain's [fixable]",
                "todos/.TODO-001-export-to-csv.md.ab12cd34.tmp: leftover: left by an interrupted write, nothing in it is the brain's [fixable]",
            ],
        )
        self.assertEqual(len(self.brain.entries()), 7)

        doctor.run(brain=self.brain, fix=True)

        self.assertEqual(self.messages("leftover"), [])
        self.assertFalse((self.root / "memory" / ".putback.xyz").exists())
        self.assertFalse((self.root / "todos" / ".TODO-001-export-to-csv.md.ab12cd34.tmp").exists())
        self.assertFalse((self.root / "..ack.k3j2_x9a.tmp").exists())
        self.assertFalse((self.root / ".VERSION.k3j2_x9a.tmp").exists())
        self.assertFalse((self.root / "skills" / "agent-brain" / ".SKILL.md.k3j2_x9a.tmp").exists())
        self.assertFalse((self.root / "skills" / "agent-brain" / "scripts" / ".hooks.py.k3j2_x9a.tmp").exists())
        self.assertTrue((self.root / ".cache.tmp").exists())
        self.assertTrue((self.root / "docs" / ".notes.md.tmp").exists())
        self.assertTrue((self.root / "docs" / ".notes.markdown.tmp").exists())

    def test_a_leftover_outside_the_brain_is_found_wherever_the_tool_writes(self) -> None:
        fixtures.write(path=self.root.parent / ".codex" / ".plain-and-short.md.q1w2e3r4.tmp", text="half\n")
        fixtures.write(path=self.root.parent / ".codex" / ".hooks.json.q1w2e3r4.tmp", text="{\n")
        fixtures.write(path=self.root.parent / ".claude" / ".settings.json.q1w2e3r4.tmp", text="{\n")
        fixtures.write(path=self.root.parent / ".claude" / "agents" / ".review-hunter.md.q1w2e3r4.tmp", text="half\n")
        fixtures.write(path=self.root.parent / ".CONTRIBUTING.md.q1w2e3r4.tmp", text="half\n")
        fixtures.write(path=self.root.parent / "..gitignore.q1w2e3r4.tmp", text="half\n")
        fixtures.write(path=self.root.parent / ".claude" / ".settings.json.tmp", text="another tool's\n")
        fixtures.write(path=self.root.parent / ".claude" / "commands" / ".deploy.markdown.tmp", text="a person's own\n")
        fixtures.write(path=self.root.parent / ".env.20260923.tmp", text="a person's own\n")
        fixtures.write(path=self.root.parent / ".notes.q1w2e3r4.tmp", text="a person's own\n")

        found = [finding.line() for finding in doctor.run(self.brain) if finding.check == "leftover"]

        self.assertEqual(
            found,
            [
                "..gitignore.q1w2e3r4.tmp: leftover: left by an interrupted write, nothing in it is the brain's [fixable]",
                ".CONTRIBUTING.md.q1w2e3r4.tmp: leftover: left by an interrupted write, nothing in it is the brain's [fixable]",
                ".claude/.settings.json.q1w2e3r4.tmp: leftover: left by an interrupted write, nothing in it is the brain's [fixable]",
                ".claude/agents/.review-hunter.md.q1w2e3r4.tmp: leftover: left by an interrupted write, nothing in it is the brain's [fixable]",
                ".codex/.hooks.json.q1w2e3r4.tmp: leftover: left by an interrupted write, nothing in it is the brain's [fixable]",
                ".codex/.plain-and-short.md.q1w2e3r4.tmp: leftover: left by an interrupted write, nothing in it is the brain's [fixable]",
            ],
        )

        doctor.run(brain=self.brain, fix=True)

        for relative in (
            ".codex/.plain-and-short.md.q1w2e3r4.tmp",
            ".codex/.hooks.json.q1w2e3r4.tmp",
            ".claude/.settings.json.q1w2e3r4.tmp",
            ".claude/agents/.review-hunter.md.q1w2e3r4.tmp",
            ".CONTRIBUTING.md.q1w2e3r4.tmp",
            "..gitignore.q1w2e3r4.tmp",
        ):
            self.assertFalse((self.root.parent / relative).exists(), relative)
        for relative in (".claude/.settings.json.tmp", ".claude/commands/.deploy.markdown.tmp", ".env.20260923.tmp", ".notes.q1w2e3r4.tmp"):
            self.assertTrue((self.root.parent / relative).exists(), relative)

    def test_a_scratch_shaped_folder_outside_the_brain_is_not_the_tools(self) -> None:
        theirs = self.root.parent / ".codex" / ".putback.abc"
        fixtures.write(path=theirs / "keep.txt", text="theirs\n")

        doctor.run(brain=self.brain, fix=True)

        self.assertEqual(self.messages("leftover"), [])
        self.assertTrue((theirs / "keep.txt").is_file())

    def test_a_persons_own_dotfile_is_nobodys_to_delete(self) -> None:
        fixtures.write(path=self.root / "docs" / ".mine.tmp", text="mine\n")
        fixtures.write(path=self.root / "docs" / ".gitkeep", text="")
        fixtures.write(path=self.root / "docs" / ".login.md.swp", text="swap\n")
        (self.root / "docs" / ".putback.link").symlink_to(self.root / "memory")

        doctor.run(brain=self.brain, fix=True)

        self.assertEqual(self.messages("leftover"), [])
        for name in (".mine.tmp", ".gitkeep", ".login.md.swp", ".putback.link"):
            with self.subTest(name=name):
                self.assertTrue((self.root / "docs" / name).is_symlink() or (self.root / "docs" / name).exists())


class Strays(DoctorCase):
    def test_a_note_beside_a_ledger_is_a_stray_and_nothing_else(self) -> None:
        fixtures.write(path=self.root / "reviews" / "2026-09-feat-billing" / "notes.md", text="# Working notes\n")
        self.brain.refresh()

        found = [finding.line() for finding in doctor.run(self.brain)]

        self.assertEqual(found, ["reviews/2026-09-feat-billing/notes.md: stray: a review folder holds ledger.md and nothing else [flag]"])

    def test_an_index_where_none_belongs_is_a_stray(self) -> None:
        fixtures.write(path=self.root / "handover" / "INDEX.md", text="---\nname: handover-index\n---\n\n# Handover\n")
        fixtures.write(path=self.root / "spec" / "INDEX.md", text="---\nname: spec-index\n---\n\n# Spec\n")
        self.brain.refresh()

        self.assertEqual(self.messages("stray"), ["no index belongs here, the tool builds one per tracker, docs/ and memory/"] * 2)

    def test_a_ledger_any_deeper_than_its_folder_is_a_stray_with_the_reason(self) -> None:
        fixtures.write(
            path=self.root / "reviews" / "a" / "b" / "ledger.md",
            text=(self.root / "reviews" / "2026-09-feat-billing" / "ledger.md").read_text(encoding="utf-8"),
        )
        self.brain.refresh()

        self.assertEqual(self.messages("stray"), ["a ledger sits at reviews/<folder>/ledger.md, one folder down, no more and no less"])

    def test_a_review_folder_not_named_for_its_month_and_branch_is_reported(self) -> None:
        folder = self.root / "reviews" / "2026-09-feat-billing"

        for name in ("Feat Billing!", "billing"):
            with self.subTest(name=name):
                folder.rename(folder.with_name(name))
                folder = folder.with_name(name)
                self.brain.refresh()

                self.assertIn(f"a review folder is named <yyyy-mm>-<branch-slug>, not '{name}'", self.messages("filename"))

    def test_a_nested_spec_file_is_a_stray_with_the_reason(self) -> None:
        fixtures.write(path=self.root / "spec" / "reviews" / "batches.md", text="# Batches\n")
        self.brain.refresh()

        self.assertEqual(self.messages("stray"), ["spec/ holds its files directly, nothing nested"])

    def test_a_file_at_the_root_the_brain_does_not_list_is_a_stray(self) -> None:
        fixtures.write(path=self.root / "notes.md", text="# Loose\n")
        self.brain.refresh()

        self.assertEqual(self.messages("stray"), ["not a file the brain lists, file it with brain new or move it out"])
        self.assertEqual(self.messages("frontmatter"), [])

    def test_a_stray_is_still_scanned_for_secrets_and_machine_paths(self) -> None:
        fixtures.write(path=self.root / "scratch.md", text="token: AKIAIOSFODNN7EXAMPLE, from /Users/someone/keys.txt\n")
        self.brain.refresh()

        self.assertEqual(
            sorted(self.messages("hygiene")),
            ["carries something shaped like a secret", "names a machine's local path"],
        )

    def test_a_readme_inside_docs_is_judged_as_a_doc(self) -> None:
        fixtures.write(
            path=self.root / "docs" / "README.md",
            text="---\nname: README\nkind: doc\ndescription: a map\nupdated: 2026-09-16\nlinks: []\n---\n\n# Map\n",
        )
        self.brain.refresh()

        self.assertEqual(self.messages("stray"), [])
        self.assertIn("'README' is not a slug", self.messages("filename"))


class Frontmatter(DoctorCase):
    def test_a_file_without_a_fence_is_reported(self) -> None:
        fixtures.write(path=self.root / "memory" / "loose.md", text="# Just prose\n")
        self.brain.refresh()

        self.assertIn("the file does not open with a --- fence", self.messages("frontmatter"))

    def test_a_file_that_is_not_utf8_is_reported_and_every_check_still_runs(self) -> None:
        (self.root / "memory" / "latin.md").write_bytes(b"---\nname: latin\n---\n\ncaf\xe9\n")
        self.brain.refresh()

        found = doctor.run(self.brain)

        self.assertEqual(
            [finding.line() for finding in found],
            ["memory/latin.md: frontmatter: the file is not UTF-8 text [flag]"],
        )

    def test_a_dangling_symlink_and_a_folder_named_like_a_file_are_reported_not_fatal(self) -> None:
        (self.root / "memory" / "gone.md").symlink_to("nowhere.md")
        (self.root / "docs" / "notes.md").mkdir()
        self.brain.refresh()

        found = [finding.line() for finding in doctor.run(self.brain)]

        self.assertEqual(len(found), 2)
        self.assertTrue(all("frontmatter: the file cannot be read: " in line for line in found))

    def test_a_blank_common_field_is_reported_like_a_missing_one(self) -> None:
        self.rewrite(relative="memory/stripe-retries-twice.md", description="", updated="", links="")

        self.assertEqual(
            sorted(self.messages("frontmatter")),
            ["description is blank", "links is blank, none is written as []", "no updated date"],
        )

        doctor.run(brain=self.brain, fix=True)

        self.assertEqual(self.messages("frontmatter"), ["description is blank"])
        self.assertEqual(self.brain.find("stripe-retries-twice").document.get_list("links"), [])

    def test_a_missing_updated_date_is_fixed(self) -> None:
        self.strip(relative="memory/stripe-retries-twice.md", key="updated")

        self.assertIn("no updated date", self.messages("frontmatter"))

        doctor.run(brain=self.brain, fix=True)

        self.assertEqual(self.messages("frontmatter"), [])


class Names(DoctorCase):
    def test_a_name_that_disagrees_with_the_filename_is_fixed(self) -> None:
        self.rewrite(relative="bugs/BUG-002-login-loop.md", name="something-else")

        self.assertIn("name is 'something-else', the file says 'BUG-002-login-loop'", self.messages("name"))

        doctor.run(brain=self.brain, fix=True)

        self.assertEqual(self.messages("name"), [])

    def test_a_plans_name_repair_moves_its_link_on_the_chain_too(self) -> None:
        plan = plans.create(brain=self.brain, title="Kit", description="kit", body="## Why\n\nWhy.\n\n## Done when\n\nDone.\n", stamp="2026-09-16")
        task.turn_on(brain=self.brain, key="kit")
        plan.path.rename(self.root / "plans" / "kit-v2.md")
        self.brain.refresh()

        self.assertIn("name is 'kit', the file says 'kit-v2'", self.messages("name"))

        doctor.run(brain=self.brain, fix=True)
        self.brain.refresh()

        self.assertEqual(self.messages("name"), [])
        self.assertEqual(task.describe(self.brain), "plan kit-v2, no steps yet")

    def test_a_copied_plan_that_kept_the_name_does_not_take_the_link_with_it(self) -> None:
        plan = plans.create(brain=self.brain, title="Kit", description="kit", body="## Why\n\nWhy.\n\n## Done when\n\nDone.\n", stamp="2026-09-16")
        task.turn_on(brain=self.brain, key="kit")
        fixtures.write(path=self.root / "plans" / "kit-copy.md", text=plan.path.read_text(encoding="utf-8"))
        self.brain.refresh()

        self.assertIn("name is 'kit', the file says 'kit-copy'", self.messages("name"))

        doctor.run(brain=self.brain, fix=True)
        self.brain.refresh()

        self.assertEqual(self.messages("name"), [])
        self.assertEqual(task.describe(self.brain), "plan kit, no steps yet")

    def test_a_cell_naming_a_review_in_another_case_is_reported_as_a_name_the_review_keeps(self) -> None:
        review = ledger.open_ledger(brain=self.brain, branch="feat/x", summary="the x", stamp="2026-09-16")
        plan = plans.create(brain=self.brain, title="Kit", description="kit", body=f"## Why\n\nWhy.\n\n## Steps\n\n1. {review.name}, the review.\n\n## Done when\n\nDone.\n", stamp="2026-09-16")
        fixtures.write(path=plan.path, text=plan.path.read_text(encoding="utf-8").replace(f"| {review.name} |", f"| {review.name.upper()} |"))
        self.brain.refresh()

        self.assertEqual(self.messages("steps"), [f"step 1 points at {review.name.upper()}, and the tool keeps the name {review.name}, which a review keeps as its branch's"])

        doctor.run(brain=self.brain, fix=True)
        self.brain.refresh()

        self.assertEqual(self.messages("steps"), [])
        self.assertEqual(steps.stored_rows(self.brain.find("kit"))[0].points_at, review.name)

    def test_a_cell_naming_a_file_with_no_id_in_another_case_is_reported_as_a_name_not_an_id(self) -> None:
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"
        fixtures.write(path=self.root / "todos" / "export-to-csv.md", text=todo.read_text(encoding="utf-8").replace("TODO-001-export-to-csv", "export-to-csv"))
        plan = plans.create(brain=self.brain, title="Kit", description="kit", body="## Why\n\nWhy.\n\n## Steps\n\n1. export-to-csv, the export.\n\n## Done when\n\nDone.\n", stamp="2026-09-16")
        fixtures.write(path=plan.path, text=plan.path.read_text(encoding="utf-8").replace("| export-to-csv |", "| EXPORT-TO-CSV |"))
        self.brain.refresh()

        self.assertEqual(self.messages("steps"), ["step 1 points at EXPORT-TO-CSV, and the tool keeps the name export-to-csv, the file carrying no id, which doctor reports"])

    def test_a_name_repair_on_a_file_with_no_id_moves_the_cells_that_named_it(self) -> None:
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"
        fixtures.write(path=self.root / "todos" / "export-as-csv.md", text=todo.read_text(encoding="utf-8").replace("TODO-001-export-to-csv", "export-to-csv"))
        plans.create(brain=self.brain, title="Kit", description="kit", body="## Why\n\nWhy.\n\n## Steps\n\n1. export-to-csv, the export.\n\n## Done when\n\nDone.\n", stamp="2026-09-16")
        self.brain.refresh()

        self.assertIn("name is 'export-to-csv', the file says 'export-as-csv'", self.messages("name"))

        doctor.run(brain=self.brain, fix=True)
        self.brain.refresh()

        self.assertEqual(self.messages("name"), [])
        self.assertEqual(steps.stored_rows(self.brain.find("kit"))[0].points_at, "export-as-csv")
        self.assertEqual([message for message in self.messages("steps") if "step 1" in message], [])

    def test_a_reviews_name_repair_moves_its_link_on_the_chain_and_the_steps_that_point_at_it(self) -> None:
        review = ledger.open_ledger(brain=self.brain, branch="feat/x", summary="the x", stamp="2026-09-16")
        plans.create(brain=self.brain, title="Kit", description="kit", body=f"## Why\n\nWhy.\n\n## Steps\n\n1. {review.name}, the review.\n\n## Done when\n\nDone.\n", stamp="2026-09-16")
        task.turn_on(brain=self.brain, key="TODO-001")
        task.turn_on(brain=self.brain, key=review.name)
        moved = review.name.replace("feat-x", "feat-y")
        review.path.parent.rename(self.root / "reviews" / moved)
        self.brain.refresh()

        self.assertIn(f"name is '{review.name}', the file says '{moved}'", self.messages("name"))

        doctor.run(brain=self.brain, fix=True)
        self.brain.refresh()

        self.assertEqual(self.messages("name"), [])
        self.assertEqual(self.messages("steps"), [])
        self.assertEqual(task.describe(self.brain), f"task TODO-001\nreview {moved}, no batches yet")
        self.assertEqual(steps.stored_rows(self.brain.find("kit"))[0].points_at, moved)

    def test_a_reviews_name_repair_leaves_a_link_that_another_reviews_folder_answers_to(self) -> None:
        first = ledger.open_ledger(brain=self.brain, branch="feat/a", summary="the a", stamp="2026-09-16")
        second = ledger.open_ledger(brain=self.brain, branch="feat/b", summary="the b", stamp="2026-09-16")
        self.rewrite(relative=first.relative, name=second.name)
        self.rewrite(relative=second.relative, name=second.name.replace("feat-b", "feat-c"))
        task.turn_on(brain=self.brain, key=second.relative)

        self.assertEqual(task.describe(self.brain), f"review {second.name}, no batches yet")

        doctor.run(brain=self.brain, fix=True)
        self.brain.refresh()

        self.assertEqual(self.messages("name"), [])
        self.assertEqual(task.read(self.brain), [task.Item(kind="review", key=second.name)])
        self.assertEqual(task.describe(self.brain), f"review {second.name}, no batches yet")

    @unittest.skipUnless(os.name == "posix" and os.geteuid() != 0, "needs a file the process cannot write")
    def test_a_reviews_name_repair_whose_field_cannot_be_written_leaves_its_links_resolving(self) -> None:
        review = ledger.open_ledger(brain=self.brain, branch="feat/x", summary="the x", stamp="2026-09-16")
        plans.create(brain=self.brain, title="Kit", description="kit", body=f"## Why\n\nWhy.\n\n## Steps\n\n1. {review.name}, the review.\n\n## Done when\n\nDone.\n", stamp="2026-09-16")
        task.turn_on(brain=self.brain, key="TODO-001")
        task.turn_on(brain=self.brain, key=review.name)
        moved = review.name.replace("feat-x", "feat-y")
        review.path.parent.rename(self.root / "reviews" / moved)
        ledger_file = self.root / "reviews" / moved / "ledger.md"
        ledger_file.chmod(0o444)
        self.brain.refresh()

        try:
            found = [finding.message for finding in doctor.run(brain=self.brain, fix=True) if finding.check in ("name", "steps")]
        finally:
            ledger_file.chmod(0o644)

        self.assertEqual(found, [f"name is '{review.name}', the file says '{moved}', and the repair failed: Permission denied: .agents/reviews/{moved}/ledger.md"])
        self.brain.refresh()
        self.assertEqual(task.describe(self.brain), f"task TODO-001\nreview {moved}, no batches yet")
        self.assertEqual(self.messages("steps"), [])

    @unittest.skipUnless(os.name == "posix" and os.geteuid() != 0, "needs a file the process cannot write")
    def test_a_name_repair_whose_chain_cannot_follow_leaves_the_finding_and_says_why(self) -> None:
        plan = plans.create(brain=self.brain, title="Kit", description="kit", body="## Why\n\nWhy.\n\n## Done when\n\nDone.\n", stamp="2026-09-16")
        task.turn_on(brain=self.brain, key="kit")
        plan.path.rename(self.root / "plans" / "kit-v2.md")
        chain = self.root / task.FILE
        chain.chmod(0o444)
        self.brain.refresh()

        try:
            found = [finding.message for finding in doctor.run(brain=self.brain, fix=True) if finding.check == "name"]
        finally:
            chain.chmod(0o644)

        self.assertEqual(found, ["name is 'kit', the file says 'kit-v2', and the repair failed: Permission denied: .agents/.task"])
        self.brain.refresh()
        self.assertEqual(task.describe(self.brain), "plan kit, no steps yet")

    def test_a_missing_name_or_kind_is_the_frontmatter_checks_alone(self) -> None:
        self.strip(relative="bugs/BUG-002-login-loop.md", key="name")
        self.strip(relative="bugs/BUG-002-login-loop.md", key="kind")
        self.strip(relative="memory/stripe-retries-twice.md", key="name")

        self.assertEqual(self.messages("name"), [])
        self.assertEqual(self.messages("kind"), [])
        self.assertEqual(self.messages("frontmatter"), ["name field is missing", "kind field is missing", "name field is missing"])

    def test_an_entry_named_like_one_of_the_kits_files_is_reported(self) -> None:
        fixtures.write(path=self.root / "spec" / "conventions.md", text="---\nname: conventions\nkind: doc\ndescription: the law\nupdated: 2026-09-16\nlinks: []\n---\n\n# Conventions\n")
        fixtures.write(path=self.root / "docs" / "conventions.md", text="---\nname: conventions\nkind: doc\ndescription: a doc\nupdated: 2026-09-16\nlinks: []\n---\n\n# Conventions\n")
        self.brain.refresh()

        self.assertIn("name 'conventions' is also carried by spec/conventions.md", self.messages("name"))

    def test_two_files_with_one_name_are_reported(self) -> None:
        self.rewrite(relative="todos/TODO-003-dark-mode.md", name="TODO-001-export-to-csv")

        self.assertTrue(any("also carried by" in message for message in self.messages("name")))


class Kinds(DoctorCase):
    def test_a_todo_filed_among_the_bugs_is_reported(self) -> None:
        self.rewrite(relative="bugs/BUG-002-login-loop.md", kind="todo")

        self.assertIn("kind is 'todo' but the file sits in bugs/", self.messages("kind"))


class Fields(DoctorCase):
    def test_a_value_outside_the_spec_is_reported(self) -> None:
        self.rewrite(relative="bugs/BUG-002-login-loop.md", severity="huge")

        self.assertIn("severity is breaks or bug or debt or nit, not 'huge'", self.messages("fields"))

    def test_a_missing_kind_field_is_reported(self) -> None:
        self.strip(relative="bugs/BUG-002-login-loop.md", key="area")

        self.assertIn("a bug needs area filled in", self.messages("fields"))

    def test_a_missing_common_field_is_reported_once_by_the_frontmatter_check(self) -> None:
        self.strip(relative="memory/stripe-retries-twice.md", key="updated")

        self.assertEqual(self.messages("fields"), [])
        self.assertEqual(self.messages("frontmatter"), ["no updated date"])


class Steps(DoctorCase):
    def test_a_conversion_refused_for_a_loop_an_earlier_repair_made_is_a_flag_and_the_run_goes_on(self) -> None:
        def refused() -> None:
            raise steps.StepsError("step 1 points at TODO-001, whose steps point back here, and the two would wait on each other for good")

        finding = doctor.Finding(path="todos/TODO-003-dark-mode.md", check="steps", message="the Steps section has no table", repair=refused)
        with mock.patch.object(doctor, "check_all", return_value=[finding]):
            found = doctor.run(brain=self.brain, fix=True)

        self.assertEqual([(one.path, one.message, one.fixable) for one in found], [("todos/TODO-003-dark-mode.md", "the Steps section has no table, and the repair failed: step 1 points at TODO-001, whose steps point back here, and the two would wait on each other for good", False)])

    def test_a_bugs_steps_of_prose_or_bullets_are_named_line_and_all_though_never_turned(self) -> None:
        bug = self.root / "bugs" / "BUG-002-login-loop.md"
        fixtures.write(path=bug, text=bug.read_text(encoding="utf-8").replace("## Evidence", "## Steps\n\n- pick colours\n\n## Evidence"))
        self.brain.refresh()

        found = [(finding.message, finding.fixable) for finding in doctor.run(brain=self.brain, fix=True) if finding.path == "bugs/BUG-002-login-loop.md" and finding.check == "steps"]

        self.assertEqual(
            found,
            [("the Steps section has no table, its header is | # | Step | Points at | By | Status |, and a numbered list there becomes one through brain steps <key> add with a step to add, or through brain doctor --fix on a plan or a todo, and doctor says what to do on a bug, a bug's Steps being the fix's steps and a reproduction going under Evidence, so move it there by hand, or make it a numbered list and on an open bug run brain steps <key> add to keep it as the fix's, and the list is refused as it stands, the Steps section holds a line that is not a numbered step, one per line with its wrapped lines indented under it: '- pick colours'", False)],
        )
        self.assertIn("- pick colours", bug.read_text(encoding="utf-8"))

    def test_a_row_with_no_text_is_reported(self) -> None:
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"
        fixtures.write(path=todo, text=todo.read_text(encoding="utf-8").replace("## Done when", "## Steps\n\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n| 1 | first | | agent | pending |\n| 2 |  | | agent | pending |\n\n## Done when"))
        self.brain.refresh()

        self.assertEqual(
            [(finding.message, finding.fixable) for finding in doctor.run(self.brain) if finding.path == "todos/TODO-001-export-to-csv.md" and finding.check == "steps"],
            [("step 2 has no text, which no write should have let stand, so fill it in by hand", False)],
        )

    def test_two_lists_that_point_at_each_other_are_neither_turned_and_the_finding_says_why(self) -> None:
        todo = self.root / "todos" / "TODO-001-export-to-csv.md"
        fixtures.write(path=todo, text=todo.read_text(encoding="utf-8").replace("## Done when", "## Steps\n\n1. TODO-003, do y.\n\n## Done when"))
        other = self.root / "todos" / "TODO-003-dark-mode.md"
        fixtures.write(path=other, text=other.read_text(encoding="utf-8").replace("status: done", "status: open").replace("## Done when", "## Steps\n\n1. TODO-001, do x.\n\n## Done when"))
        self.brain.refresh()

        found = [(finding.path, finding.message, finding.fixable) for finding in doctor.run(brain=self.brain, fix=True) if finding.check == "steps"]
        self.brain.refresh()

        self.assertEqual(
            found,
            [
                ("todos/TODO-001-export-to-csv.md", "the Steps section has no table, its header is | # | Step | Points at | By | Status |, and a numbered list there becomes one through brain steps <key> add with a step to add, or through brain doctor --fix on a plan or a todo, and doctor says what to do on a bug, and the list is refused as it stands, step 1 points at TODO-003, whose steps point back here, and the two would wait on each other for good", False),
                ("todos/TODO-003-dark-mode.md", "the Steps section has no table, its header is | # | Step | Points at | By | Status |, and a numbered list there becomes one through brain steps <key> add with a step to add, or through brain doctor --fix on a plan or a todo, and doctor says what to do on a bug, and the list is refused as it stands, step 1 points at TODO-001, whose steps point back here, and the two would wait on each other for good", False),
            ],
        )
        self.assertIn("1. TODO-003, do y.", todo.read_text(encoding="utf-8"))
        self.assertIn("1. TODO-001, do x.", other.read_text(encoding="utf-8"))

    def test_a_plan_with_no_steps_heading_is_reported_once_with_the_repair_that_adds_the_table(self) -> None:
        path = self.root / "plans" / "bare.md"
        fixtures.write(path=path, text="---\nname: bare\nkind: plan\ndescription: a bare plan\nupdated: 2026-09-16\nlinks: []\nstatus: open\nby: owner\nborn: 2026-09-16\n---\n\n## Why\n\nWhy.\n\n## Done when\n\nDone.\n")
        self.brain.refresh()

        found = [finding for finding in doctor.run(self.brain) if finding.path == "plans/bare.md"]
        self.assertEqual([(finding.check, finding.message, finding.fixable) for finding in found], [("steps", "there is no Steps section", True)])

        doctor.run(brain=self.brain, fix=True)
        self.brain.refresh()

        self.assertIn("\n## Steps\n\n| # | Step | Points at | By | Status |\n", path.read_text(encoding="utf-8"))
        self.assertEqual([finding.message for finding in doctor.run(self.brain) if finding.path == "plans/bare.md"], [])

    def setUp(self) -> None:
        super().setUp()
        self.plan = plans.create(
            brain=self.brain,
            title="Kit phase 7",
            description="the rest of phase 7",
            body="## Why\n\nThe rest.\n\n## Steps\n\n1. TODO-001, auto ack.\n2. The owner's own run.\n\n## Done when\n\nAll of it.\n",
            stamp="2026-09-24",
        )
        index.write(brain=self.brain, stamp="2026-09-24")
        self.brain.refresh()

    def test_a_plan_that_reads_right_is_clean_and_so_is_a_goal_with_no_steps_yet(self) -> None:
        self.assertEqual(self.messages("steps"), [])

        plans.create(brain=self.brain, title="A goal", description="a goal alone", body="## Why\n\nA goal.\n\n## Done when\n\nMet.\n")

        self.assertEqual(self.messages("steps"), [])

    def test_a_pointed_mark_that_drifted_is_reported_and_repaired(self) -> None:
        text = self.plan.path.read_text(encoding="utf-8")
        fixtures.write(path=self.plan.path, text=text.replace("| TODO-001 | owner | pending |", "| TODO-001 | owner | done |"))
        self.brain.refresh()

        self.assertEqual(self.messages("steps"), ["step 1 is marked done while TODO-001 says pending"])

        doctor.run(brain=self.brain, fix=True)

        self.assertEqual(self.messages("steps"), [])
        self.assertIn("| TODO-001 | owner | pending |", self.plan.path.read_text(encoding="utf-8"))

    def test_a_step_pointing_at_nothing_and_a_broken_table_are_reported(self) -> None:
        text = self.plan.path.read_text(encoding="utf-8")
        fixtures.write(path=self.plan.path, text=text.replace("| TODO-001 |", "| TODO-099 |"))
        self.brain.refresh()
        self.assertEqual(self.messages("steps"), ["step 1 points at 'TODO-099', which reaches no todo, bug or review, and brain steps <key> step 1 done or skipped marks it by hand meanwhile"])

        fixtures.write(path=self.plan.path, text=text.replace("| 2 |", "| 3 |"))
        self.brain.refresh()
        self.assertEqual(self.messages("steps"), ["the steps are numbered 1 onward in order, and row 3 sits where 2 should"])

    def test_an_entry_left_open_with_every_step_settled_waits_on_its_done_when_unflagged(self) -> None:
        writes.update(brain=self.brain, key="TODO-001", changes={"status": "done", "done": "2026-09-24"})
        steps.mark(brain=self.brain, key="kit-phase-7", number=2, status="skipped")
        steps.add(brain=self.brain, key="BUG-002", lines=["Find the loop."])
        steps.mark(brain=self.brain, key="BUG-002", number=1, status="done")
        writes.update(brain=self.brain, key="BUG-002", changes={"status": "branched"})

        self.assertEqual(self.messages("steps"), [])

    def test_a_closed_entry_with_a_step_still_pending_is_flagged_and_its_table_is_left_as_it_is(self) -> None:
        path = self.plan.path
        writes.update(brain=self.brain, key="TODO-001", changes={"status": "done", "done": "2026-09-24"})
        steps.mark(brain=self.brain, key="kit-phase-7", number=2, status="done")
        writes.update(brain=self.brain, key="kit-phase-7", changes={"status": "done", "done": "2026-09-24"})

        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace("| 2 | The owner's own run. | | owner | done |", "| 2 | The owner's own run. | | owner | pending |"))
        self.brain.refresh()

        self.assertEqual(self.messages("steps"), ["the entry is done with step 2 still pending, so reopen it and close it again once the step is settled, and when the set to open is refused for steps that do not read, mend them first"])

        todo = self.root / "todos" / "TODO-001-export-to-csv.md"
        fixtures.write(path=todo, text=todo.read_text(encoding="utf-8").replace("## Done when", "## Steps\n\n| # | Step | Points at | By | Status |\n|---|---|---|---|---|\n| 1 | Read the table. | | owner | pending |\n\n## Done when"))
        self.brain.refresh()

        self.assertEqual([message for message in self.messages("steps") if "TODO-001" not in message and "step 1" in message], ["the entry is done with step 1 still pending, so reopen it and close it again once the step is settled, and when the set to open is refused, mend the steps it says do not read, or close the entry whose steps point back here first"])

        writes.update(brain=self.brain, key="TODO-001", changes={"status": "open", "done": ""})
        self.assertIn("| 1 | auto ack. | TODO-001 | owner | done |", path.read_text(encoding="utf-8"))

    def test_a_todos_steps_written_as_a_list_by_hand_are_reported_and_a_todo_without_any_is_clean(self) -> None:
        self.assertEqual([finding.path for finding in doctor.run(self.brain) if finding.check == "steps"], [])

        path = self.root / "todos" / "TODO-001-export-to-csv.md"
        document = frontmatter.read(str(path))
        document.body = "\n## What\n\nExport.\n\n## Steps\n\n1. one\n2. two\n\n## Done when\n\nDone.\n"
        frontmatter.write(path=str(path), document=document)
        self.brain.refresh()

        found = [finding for finding in doctor.run(self.brain) if finding.check == "steps"]
        self.assertEqual(
            [finding.line() for finding in found],
            ["todos/TODO-001-export-to-csv.md: steps: the Steps section has no table, its header is | # | Step | Points at | By | Status |, and a numbered list there becomes one through brain steps <key> add with a step to add, or through brain doctor --fix on a plan or a todo, and doctor says what to do on a bug [fixable]"],
        )

        doctor.run(brain=self.brain, fix=True)

        self.assertEqual(self.messages("steps"), [])
        self.assertEqual([(step.text, step.by, step.status) for step in steps.rows_of(brain=self.brain, entry=self.brain.find("TODO-001"))], [("one", "owner", "pending"), ("two", "owner", "pending")])

    def test_a_list_on_a_closed_entry_is_repaired_with_its_rows_settled_and_a_loose_list_is_only_flagged(self) -> None:
        path = self.root / "todos" / "TODO-003-dark-mode.md"
        document = frontmatter.read(str(path))
        document.body = "\n## What\n\nA dark theme.\n\n## Steps\n\n1. pick colours\n2. wire toggle\n\n## Done when\n\nDone.\n"
        frontmatter.write(path=str(path), document=document)
        self.brain.refresh()

        doctor.run(brain=self.brain, fix=True)

        self.assertEqual(self.messages("steps"), [])
        self.assertEqual([step.status for step in steps.rows_of(brain=self.brain, entry=self.brain.find("TODO-003"))], ["done", "done"])

        document = frontmatter.read(str(self.root / "bugs" / "BUG-002-login-loop.md"))
        document.body = "\n## Problem\n\nLoops.\n\n## Evidence\n\nA log line.\n\n## Steps\n\n- find the loop\n"
        frontmatter.write(path=str(self.root / "bugs" / "BUG-002-login-loop.md"), document=document)
        self.brain.refresh()

        found = [finding for finding in doctor.run(self.brain) if finding.check == "steps"]
        self.assertEqual(len(found), 1)
        self.assertFalse(found[0].fixable)

    def test_a_list_under_a_bugs_steps_is_flagged_and_never_repaired_since_it_may_be_the_reproduction(self) -> None:
        path = self.root / "bugs" / "BUG-002-login-loop.md"
        document = frontmatter.read(str(path))
        document.body = "\n## Problem\n\nLoops.\n\n## Evidence\n\nA log line.\n\n## Steps\n\n1. open /login\n2. submit the form\n"
        frontmatter.write(path=str(path), document=document)
        self.brain.refresh()

        found = [finding for finding in doctor.run(self.brain) if finding.check == "steps"]
        self.assertEqual(len(found), 1)
        self.assertFalse(found[0].fixable)
        self.assertIn("a bug's Steps being the fix's steps and a reproduction going under Evidence", found[0].message)

        doctor.run(brain=self.brain, fix=True)
        self.assertIn("1. open /login\n2. submit the form", self.brain.find("BUG-002").document.body)

        document.body = "\n## Problem\n\nLoops.\n\n## Evidence\n\nA log line.\n\n1. open /login\n2. submit the form\n\n## Steps\n"
        frontmatter.write(path=str(path), document=document)
        self.brain.refresh()

        self.assertEqual(self.messages("steps"), [])
        self.assertEqual(steps.rows_of(brain=self.brain, entry=self.brain.find("BUG-002")), [])
        self.assertEqual(writes.update(brain=self.brain, key="BUG-002", changes={"status": "wontfix"}).field("status"), "wontfix")

    def test_a_plan_with_a_lookalike_heading_and_no_steps_section_gets_no_repair(self) -> None:
        path = self.root / "plans" / "phase-8.md"
        fixtures.write(path=path, text="---\nname: phase-8\nkind: plan\ndescription: the rest\nupdated: 2026-09-24\nlinks: []\nstatus: open\nby: owner\nborn: 2026-09-24\n---\n\n## Why\n\nWhy.\n\n### Steps\n\n1. a\n\n## Done when\n\nDone.\n")
        self.brain.refresh()

        found = [finding for finding in doctor.run(self.brain) if finding.check == "steps"]
        self.assertEqual([finding.fixable for finding in found], [False])

        doctor.run(brain=self.brain, fix=True)
        self.assertNotIn("\n## Steps\n", path.read_text(encoding="utf-8"))


class Sections(DoctorCase):
    def test_a_missing_required_heading_is_reported(self) -> None:
        path = self.root / "bugs" / "BUG-002-login-loop.md"
        document = frontmatter.read(str(path))
        document.body = "\n## Problem\n\nIt loops.\n"
        frontmatter.write(path=str(path), document=document)
        self.brain.refresh()

        self.assertIn("no Evidence heading", self.messages("sections"))

    def test_a_fixed_bug_with_an_empty_fix_is_reported(self) -> None:
        self.rewrite(relative="bugs/BUG-002-login-loop.md", status="fixed")

        self.assertIn("a bug goes fixed once its Fix section names the cause and the commit", self.messages("lifecycle"))

    def test_a_done_todo_without_a_date_is_reported(self) -> None:
        self.strip(relative="todos/TODO-003-dark-mode.md", key="done")

        self.assertIn("a todo goes done with its closing date, done=YYYY-MM-DD", self.messages("lifecycle"))

    def test_a_dropped_todo_without_a_date_is_reported_and_refused_alike(self) -> None:
        self.rewrite(relative="todos/TODO-001-export-to-csv.md", status="dropped")

        self.assertIn("a todo goes dropped with its closing date, done=YYYY-MM-DD", self.messages("lifecycle"))

        with self.assertRaises(writes.WriteError) as caught:
            writes.update(brain=self.brain, key="TODO-003", changes={"status": "dropped", "done": ""})

        self.assertEqual(str(caught.exception), "a todo goes dropped with its closing date, done=YYYY-MM-DD")

    def test_a_rule_that_names_two_things_is_said_once(self) -> None:
        kind = doctor.schema.Kind(
            name="x",
            folder="x",
            rules=(doctor.schema.Rule(status="done", says="the one message", field="done", section="Done when"),),
        )

        gaps = doctor.sections.lifecycle_gaps(kind=kind, fields={"status": "done"}, body="")

        self.assertEqual(gaps, ["the one message"])

    def test_a_refiled_file_links_to_what_it_became(self) -> None:
        self.rewrite(relative="bugs/BUG-002-login-loop.md", status="refiled")
        self.rewrite(relative="todos/TODO-003-dark-mode.md", status="refiled", links="[docs/systems/login.md]")

        self.assertEqual(
            sorted(self.messages("lifecycle")),
            ["a refiled bug links to the todo it became", "a refiled todo links to the bug it became"],
        )

    def test_a_refile_through_the_tool_leaves_the_brain_clean(self) -> None:
        writes.convert(
            brain=self.brain,
            key="TODO-001",
            kind=self.brain.find("BUG-002").kind,
            description="the export drops rows",
            values={"severity": "bug", "area": "reports"},
            body="## Problem\n\nRows go missing.\n\n## Evidence\n\nA diff.\n",
        )
        doctor.run(brain=self.brain, fix=True)

        self.assertEqual([finding.line() for finding in doctor.run(self.brain)], [])
        self.assertIn("- reports: 0 done, 0 dropped, 1 refiled", (self.root / "todos" / "INDEX.md").read_text(encoding="utf-8"))

    def test_a_superseded_decision_names_its_successor(self) -> None:
        self.rewrite(relative="decisions/DEC-001-postgres-over-sqlite.md", status="superseded")

        self.assertIn("a superseded decision names the one that replaced it", self.messages("lifecycle"))


class Filenames(DoctorCase):
    def test_a_bug_without_an_id_is_reported(self) -> None:
        fixtures.write(
            path=self.root / "bugs" / "login-loop.md",
            text=(self.root / "bugs" / "BUG-002-login-loop.md").read_text(encoding="utf-8"),
        )
        self.brain.refresh()

        self.assertIn("a bug is named BUG-###-<slug>.md", self.messages("filename"))

    def test_one_id_on_two_files_is_reported(self) -> None:
        fixtures.write(
            path=self.root / "bugs" / "BUG-002-second.md",
            text=(self.root / "bugs" / "BUG-002-login-loop.md").read_text(encoding="utf-8"),
        )
        self.brain.refresh()

        self.assertTrue(any("BUG-002 is also" in message for message in self.messages("filename")))

    def test_a_slug_with_capitals_is_reported(self) -> None:
        fixtures.write(
            path=self.root / "memory" / "Stripe-Retries.md",
            text=(self.root / "memory" / "stripe-retries-twice.md").read_text(encoding="utf-8"),
        )
        self.brain.refresh()

        self.assertIn("'Stripe-Retries' is not a slug", self.messages("filename"))


class Links(DoctorCase):
    def test_a_link_to_nothing_is_reported(self) -> None:
        self.rewrite(relative="todos/TODO-001-export-to-csv.md", links="[BUG-999]")

        self.assertIn("'BUG-999' reaches nothing", self.messages("link"))

    def test_a_banned_character_in_a_header_value_is_a_field_finding_on_that_file(self) -> None:
        self.rewrite(relative="docs/systems/login.md", description="the sign in flow — end to end")

        lines = [finding.line() for finding in doctor.run(brain=self.brain, fix=True)]

        self.assertEqual(lines, ["docs/systems/login.md: fields: description carries a dash the house style bans [flag]"])

    def test_a_folder_that_is_not_a_slug_is_named_on_the_file_it_holds(self) -> None:
        doc = (self.root / "docs" / "systems" / "login.md").read_text(encoding="utf-8")
        fixtures.write(path=self.root / "docs" / "sub—system" / "signup.md", text=doc.replace("name: login", "name: signup"))
        self.brain.refresh()

        lines = [finding.line() for finding in doctor.run(brain=self.brain, fix=True)]

        self.assertEqual(lines, ["docs/sub—system/signup.md: filename: sits in a folder that is not a slug, 'sub—system' [flag]"])

        bug = (self.root / "bugs" / "BUG-002-login-loop.md").read_text(encoding="utf-8")
        fixtures.write(path=self.root / "bugs" / "sub—area" / "slow.md", text=bug.replace("name: BUG-002-login-loop", "name: slow"))
        self.brain.refresh()

        lines = [finding.line() for finding in doctor.run(brain=self.brain, fix=True)]

        self.assertIn("bugs/sub—area/slow.md: filename: sits in a folder that is not a slug, 'sub—area' [flag]", lines)
        self.assertIn("bugs/sub—area/slow.md: filename: a bug is named BUG-###-<slug>.md [flag]", lines)

    def test_a_kit_file_with_a_banned_character_in_its_description_is_named_itself(self) -> None:
        fixtures.write(
            path=self.root / "README.md",
            text="---\nname: agent-readme\nkind: doc\ndescription: the map; read it first\nupdated: 2026-09-16\nlinks: []\n---\n\n# Map\n",
        )
        self.brain.refresh()

        lines = [finding.line() for finding in doctor.run(brain=self.brain, fix=True)]

        self.assertEqual(lines, ["README.md: fields: description carries a semicolon [flag]"])

    def test_the_skills_docs_are_held_to_the_prose_and_boundary_rules_only(self) -> None:
        skill = self.root / "skills" / "agent-brain" / "SKILL.md"
        skill.write_text(skill.read_text(encoding="utf-8") + "\nA line; with a semicolon and /Users/someone/code.\n", encoding="utf-8")
        self.brain.refresh()

        self.assertEqual(
            [finding.line() for finding in doctor.run(self.brain)],
            [
                "skills/agent-brain/SKILL.md: style: carries a semicolon [flag]",
                "skills/agent-brain/SKILL.md: hygiene: names a machine's local path [flag]",
            ],
        )

    def test_a_link_written_as_a_name_is_turned_into_a_path(self) -> None:
        self.rewrite(relative="todos/TODO-001-export-to-csv.md", links="[BUG-002]")

        self.assertIn("links not written as paths relative to .agents/, each once", self.messages("link"))

        doctor.run(brain=self.brain, fix=True)

        repaired = self.brain.find("TODO-001")
        self.assertEqual(repaired.document.get_list("links"), ["bugs/BUG-002-login-loop.md"])
        self.assertEqual(repaired.field("updated"), date.today().isoformat())
        self.assertEqual(self.messages("link"), [])

    def test_one_file_linked_by_name_and_by_path_is_repaired_to_one_link(self) -> None:
        self.rewrite(relative="todos/TODO-001-export-to-csv.md", links="[BUG-002, bugs/BUG-002-login-loop.md, login]")

        doctor.run(brain=self.brain, fix=True)

        repaired = self.brain.find("TODO-001")
        self.assertEqual(repaired.document.get_list("links"), ["bugs/BUG-002-login-loop.md", "docs/systems/login.md"])
        self.assertEqual(self.messages("link"), [])

    def test_a_link_that_reaches_nothing_twice_is_reported_once_and_kept_once(self) -> None:
        self.rewrite(relative="todos/TODO-001-export-to-csv.md", links="[nowhere-at-all, bugs/BUG-002-login-loop.md, nowhere-at-all]")

        self.assertEqual(
            self.messages("link"),
            ["'nowhere-at-all' reaches nothing", "links not written as paths relative to .agents/, each once"],
        )

        doctor.run(brain=self.brain, fix=True)

        self.assertEqual(self.brain.find("TODO-001").document.get_list("links"), ["nowhere-at-all", "bugs/BUG-002-login-loop.md"])

    def test_a_kit_file_with_name_links_is_turned_to_paths_too(self) -> None:
        fixtures.write(
            path=self.root / "README.md",
            text="---\nname: agent-readme\nkind: doc\ndescription: the map\nupdated: 2026-09-16\nlinks: [bugs-index]\n---\n\n# Map\n",
        )
        self.brain.refresh()

        doctor.run(brain=self.brain, fix=True)

        document = frontmatter.read(str(self.root / "README.md"))
        self.assertEqual(document.get_list("links"), ["bugs/INDEX.md"])
        self.assertEqual(self.messages("link"), [])

    def test_a_body_link_to_nothing_is_reported(self) -> None:
        self.append(relative="todos/TODO-001-export-to-csv.md", text="\nSee [gone](../nowhere.md).\n")

        self.assertIn("body link '../nowhere.md' reaches nothing", self.messages("link"))

    def test_a_dangling_body_link_in_one_of_the_kits_own_files_is_reported(self) -> None:
        fixtures.write(
            path=self.root / "README.md",
            text="---\nname: agent-readme\nkind: doc\ndescription: the map\nupdated: 2026-09-16\nlinks: []\n---\n\nSee [gone](bugs/BUG-404-gone.md).\n",
        )
        self.brain.refresh()

        self.assertIn("body link 'bugs/BUG-404-gone.md' reaches nothing", self.messages("link"))

    def test_a_link_that_leaves_the_brain_is_reported_and_nothing_crashes(self) -> None:
        outside = self.root.parent / ".claude" / "output-styles" / "plain-and-short.md"
        self.rewrite(relative="todos/TODO-001-export-to-csv.md", links=f"[{outside}, ../README.md]")

        found = self.messages("link")

        self.assertIn(f"'{outside}' reaches nothing", found)
        self.assertIn("'../README.md' reaches nothing", found)


class Indexes(DoctorCase):
    def test_a_stale_index_is_reported_and_rebuilt(self) -> None:
        self.rewrite(relative="todos/TODO-001-export-to-csv.md", description="a new description")

        self.assertIn("differs from what the tool builds for its folder", self.messages("index"))

        doctor.run(brain=self.brain, fix=True)

        self.assertEqual(self.messages("index"), [])

    def test_one_fix_builds_the_brain_once_per_pass_however_many_indexes_are_stale(self) -> None:
        for path in index.build(self.brain):
            path.unlink()
        self.brain.refresh()

        with mock.patch("brainlib.index.build", wraps=index.build) as building:
            left = doctor.run(brain=self.brain, fix=True)

        self.assertEqual(left, [])
        self.assertEqual(building.call_count, 2)

    def test_an_index_with_a_hand_edited_header_is_reported_and_rebuilt(self) -> None:
        self.rewrite(relative="todos/INDEX.md", name="my-todos", links="[bugs/INDEX.md]")

        self.assertIn("differs from what the tool builds for its folder", self.messages("index"))

        doctor.run(brain=self.brain, fix=True)

        document = frontmatter.read(str(self.root / "todos" / "INDEX.md"))
        self.assertEqual(document.get("name"), "todos-index")
        self.assertEqual(document.get_list("links"), ["INDEX.md"])
        self.assertEqual(self.messages("index"), [])


class Style(DoctorCase):
    def test_a_dash_in_prose_is_reported(self) -> None:
        self.append(relative="docs/systems/login.md", text="\nThe flow — such as it is.\n")

        self.assertIn("carries a dash the house style bans", self.messages("style"))

    def test_a_semicolon_in_code_is_not(self) -> None:
        self.append(relative="docs/systems/login.md", text="\n```dart\nfinal x = 1;\n```\n")

        self.assertEqual(self.messages("style"), [])


class Hygiene(DoctorCase):
    def test_a_machine_path_is_reported(self) -> None:
        self.append(relative="docs/systems/login.md", text="\nThe file sits at /Users/someone/src/login.dart.\n")

        self.assertIn("names a machine's local path", self.messages("hygiene"))

    def test_something_shaped_like_a_key_is_reported(self) -> None:
        self.append(relative="docs/systems/login.md", text="\nUse AKIAIOSFODNN7EXAMPLE for the bucket.\n")

        self.assertIn("carries something shaped like a secret", self.messages("hygiene"))


class Layout(DoctorCase):
    def test_a_missing_version_stamp_is_reported(self) -> None:
        (self.root / "VERSION").unlink()

        self.assertIn("no kit version stamp, run the upgrade", self.messages("layout"))

    def test_the_skill_links_are_created_for_claude_code(self) -> None:
        self.assertTrue((self.root.parent / ".claude" / "skills" / "agent-brain").is_symlink())
        self.assertFalse((self.root.parent / ".claude" / "skills" / "branch-review").exists())
        self.assertEqual(self.messages("layout"), [])

        fixtures.ship_review(self.root)

        self.assertIn("the skill link is missing, so Claude Code cannot see the skill", self.messages("layout"))

        doctor.run(brain=self.brain, fix=True)

        self.assertTrue((self.root.parent / ".claude" / "skills" / "branch-review").is_symlink())
        self.assertEqual(self.messages("layout"), [])

    def test_a_folder_under_skills_without_a_skill_file_is_nobodys_skill(self) -> None:
        fixtures.write(path=self.root / "skills" / "scratch" / "notes.md", text="notes\n")

        self.assertEqual(self.messages("layout"), [])

        doctor.run(brain=self.brain, fix=True)

        self.assertFalse((self.root.parent / ".claude" / "skills" / "scratch").exists())

    def test_a_dead_link_of_a_kit_skill_is_named_by_its_own_path(self) -> None:
        fixtures.ship_review(self.root)
        doctor.run(brain=self.brain, fix=True)
        link = self.root.parent / ".claude" / "skills" / "branch-review"
        link.unlink()
        link.symlink_to("nowhere")

        found = [finding for finding in doctor.run(self.brain) if finding.check == "layout"]

        self.assertEqual([finding.path for finding in found], [".claude/skills/branch-review"])
        self.assertTrue(found[0].fixable)

    def test_a_link_to_a_project_skill_that_is_gone_is_removed(self) -> None:
        fixtures.write(path=self.root / "skills" / "deploy" / "SKILL.md", text="---\nname: deploy\ndescription: ships it\n---\n\nOurs.\n")
        doctor.run(brain=self.brain, fix=True)
        shutil.rmtree(self.root / "skills" / "deploy")
        link = self.root.parent / ".claude" / "skills" / "deploy"
        self.assertTrue(link.is_symlink())

        self.assertEqual(self.messages("layout"), [doctor.SKILL_GONE])

        doctor.run(brain=self.brain, fix=True)

        self.assertFalse(link.is_symlink())
        self.assertEqual(self.messages("layout"), [])

    def test_a_link_whose_target_runs_through_a_file_is_gone_too(self) -> None:
        fixtures.write(path=self.root / "skills" / "deploy", text="a file where the folder was\n")
        link = self.root.parent / ".claude" / "skills" / "deploy"
        link.symlink_to("../../.agents/skills/deploy/SKILL.md")

        self.assertEqual(self.messages("layout"), [doctor.SKILL_GONE])

        doctor.run(brain=self.brain, fix=True)

        self.assertFalse(link.is_symlink())

    def test_the_stray_pass_waits_while_the_skills_folder_cannot_be_listed(self) -> None:
        link = self.root.parent / ".claude" / "skills" / "old"
        link.symlink_to("../../.agents/skills/old")
        failing = OSError(errno.EIO, "Input/output error")

        with mock.patch.object(store, "skill_folders", side_effect=failing):
            found = [finding for finding in doctor.run(brain=self.brain, fix=True) if finding.check == "layout"]

        self.assertEqual([finding.path for finding in found], ["skills"])
        self.assertTrue(link.is_symlink())
        self.assertEqual(self.messages("layout"), [doctor.SKILL_GONE])

    def test_a_link_the_project_made_elsewhere_is_left_alone(self) -> None:
        elsewhere = self.root.parent / "elsewhere"
        elsewhere.mkdir()
        (self.root.parent / ".claude" / "skills" / "theirs").symlink_to(elsewhere / "gone")

        self.assertEqual(self.messages("layout"), [])

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root ignores file modes")
    def test_a_skills_folder_that_cannot_be_listed_is_reported_and_the_kit_links_still_made(self) -> None:
        skills = self.root / "skills"
        link = self.root.parent / ".claude" / "skills" / "agent-brain"
        fixtures.write(path=skills / "deploy" / "SKILL.md", text="---\nname: deploy\ndescription: ships it\n---\n\nOurs.\n")
        doctor.run(brain=self.brain, fix=True)
        link.unlink()
        skills.chmod(0o111)

        try:
            found = [finding for finding in doctor.run(brain=self.brain, fix=True) if finding.check in ("layout", "index")]
        finally:
            skills.chmod(0o755)

        self.assertEqual([finding.path for finding in found], ["INDEX.md", "skills"])
        self.assertEqual(found[0].message, "cannot be rebuilt while the skills folder cannot be listed")
        self.assertTrue(found[1].message.startswith("the skills folder cannot be listed, so only the kit's skill links were checked: "))
        self.assertFalse(any(finding.fixable for finding in found))
        self.assertTrue(link.is_symlink())
        self.assertIn("[skills/deploy/SKILL.md]", (self.root / "INDEX.md").read_text(encoding="utf-8"))

    def test_a_skills_folder_that_loops_does_not_crash_doctor(self) -> None:
        skills = self.root / "skills"
        skills.rename(self.root / "skills-real")
        skills.symlink_to("skills")

        self.assertIn("the skill itself is missing from .agents/skills/agent-brain, reinstall the kit", self.messages("layout"))

    def test_a_dangling_link_inside_the_brain_is_never_removed_through_a_linked_skills_folder(self) -> None:
        claude_skills = self.root.parent / ".claude" / "skills"
        (claude_skills / "agent-brain").unlink()
        claude_skills.rmdir()
        claude_skills.symlink_to("../.agents/skills")
        stray = self.root / "skills" / "notes"
        stray.symlink_to("old-notes")

        doctor.run(brain=self.brain, fix=True)

        self.assertTrue(stray.is_symlink())
        self.assertNotIn(doctor.SKILL_GONE, self.messages("layout"))

    def test_a_folder_under_skills_named_outside_the_slug_is_nobodys_skill(self) -> None:
        for name in (".review-old", "c#-style", "Release Notes"):
            fixtures.write(path=self.root / "skills" / name / "SKILL.md", text="---\nname: old\n---\n\nA draft.\n")

        self.assertEqual(self.messages("layout"), [])

        doctor.run(brain=self.brain, fix=True)

        self.assertEqual(sorted(path.name for path in (self.root.parent / ".claude" / "skills").iterdir()), ["agent-brain"])
        self.assertNotIn("c#-style", (self.root / "INDEX.md").read_text(encoding="utf-8"))

    def test_a_brain_stamped_before_a_kit_skill_is_not_asked_for_it(self) -> None:
        fixtures.write(path=self.root / "VERSION", text="1.1.0\n")

        self.assertEqual(self.messages("layout"), [])

        fixtures.write(path=self.root / "VERSION", text="1.2.0\n")

        self.assertEqual(self.messages("layout"), ["the skill itself is missing from .agents/skills/branch-review, reinstall the kit"])

    def test_a_brain_with_no_stamp_to_read_is_on_the_oldest_release(self) -> None:
        (self.root / "VERSION").unlink()

        self.assertEqual(self.messages("layout"), ["no kit version stamp, run the upgrade"])

        fixtures.write(path=self.root / "VERSION", text="v1.2.0\n")

        self.assertEqual(self.messages("layout"), ["the kit version stamp is not a version like 1.0.0, write the kit's release in it"])

    def test_a_skill_the_project_keeps_gets_a_link_too(self) -> None:
        fixtures.write(
            path=self.root / "skills" / "deploy" / "SKILL.md",
            text="---\nname: deploy\ndescription: ships it\n---\n\nA skill of the project's own.\n",
        )

        self.assertIn("the skill link is missing, so Claude Code cannot see the skill", self.messages("layout"))

        doctor.run(brain=self.brain, fix=True)

        self.assertEqual(self.messages("layout"), [])
        self.assertEqual(os.readlink(self.root.parent / ".claude" / "skills" / "deploy"), "../../.agents/skills/deploy")

    def test_a_link_that_points_at_nothing_is_reported_and_replaced(self) -> None:
        link = self.root.parent / ".claude" / "skills" / "agent-brain"
        link.unlink()
        link.symlink_to("../../.agent/skills/agent-brain")

        self.assertIn("the skill link does not reach the skill, so Claude Code cannot see the skill", self.messages("layout"))

        doctor.run(brain=self.brain, fix=True)

        self.assertEqual(self.messages("layout"), [])
        self.assertEqual(os.readlink(link), "../../.agents/skills/agent-brain")

    def test_a_link_to_the_wrong_place_is_replaced_too(self) -> None:
        link = self.root.parent / ".claude" / "skills" / "agent-brain"
        link.unlink()
        link.symlink_to("../../.agents/SYSTEM.md")

        doctor.run(brain=self.brain, fix=True)

        self.assertEqual(self.messages("layout"), [])
        self.assertEqual(os.readlink(link), "../../.agents/skills/agent-brain")

    def test_a_plain_file_in_the_links_place_is_reported_and_replaced(self) -> None:
        link = self.root.parent / ".claude" / "skills" / "agent-brain"
        link.unlink()
        link.write_text("../../.agents/skills/agent-brain", encoding="utf-8")

        self.assertIn("something that is not a link sits where the skill link belongs, so Claude Code cannot see the skill", self.messages("layout"))

        doctor.run(brain=self.brain, fix=True)

        self.assertEqual(self.messages("layout"), [])
        self.assertTrue(link.is_symlink())

    def test_a_symlinked_skills_folder_gets_a_link_that_resolves_from_where_it_lands(self) -> None:
        skills = self.root.parent / ".claude" / "skills"
        shared = self.root.parent / "shared-skills"
        shared.mkdir()
        fixtures.ship_review(self.root)
        doctor.run(brain=self.brain, fix=True)
        for name in schema.KIT_SKILL_NAMES:
            (skills / name).unlink()
        skills.rmdir()
        skills.symlink_to(shared)

        doctor.run(brain=self.brain, fix=True)

        self.assertEqual(self.messages("layout"), [])
        for name in schema.KIT_SKILL_NAMES:
            self.assertTrue((shared / name / "SKILL.md").is_file())

    def test_a_link_that_loops_is_reported_and_remade_on_every_floor(self) -> None:
        link = self.root.parent / ".claude" / "skills" / "agent-brain"
        link.unlink()
        link.symlink_to("agent-brain")

        self.assertIn("the skill link does not reach the skill, so Claude Code cannot see the skill", self.messages("layout"))

        doctor.run(brain=self.brain, fix=True)

        self.assertEqual(self.messages("layout"), [])
        self.assertEqual(os.readlink(link), "../../.agents/skills/agent-brain")

    def test_a_skills_folder_that_loops_counts_as_inside_the_project(self) -> None:
        skills = self.root.parent / ".claude" / "skills"
        (skills / "agent-brain").unlink()
        skills.rmdir()
        skills.symlink_to("skills")

        self.assertFalse(doctor.skills_folder_outside(self.root.parent))

    def test_a_skills_folder_outside_the_project_gets_no_link_written_into_it(self) -> None:
        skills = self.root.parent / ".claude" / "skills"
        outside = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, outside)
        (skills / "agent-brain").unlink()
        skills.rmdir()
        skills.symlink_to(outside)
        (outside / "deploy").symlink_to(self.root / "skills" / "deploy")

        found = [finding for finding in doctor.run(self.brain) if finding.check == "layout"]

        self.assertEqual(
            [finding.message for finding in found],
            [
                "the skill link is missing, so Claude Code cannot see the skill" + doctor.SHARED_FOLDER,
                doctor.SKILL_GONE + doctor.SHARED_REMOVAL,
            ],
        )
        self.assertFalse(any(finding.fixable for finding in found))

        doctor.run(brain=self.brain, fix=True)

        self.assertEqual([path.name for path in outside.iterdir()], ["deploy"])

    def test_a_link_to_something_still_there_is_never_removed(self) -> None:
        fixtures.write(path=self.root / "skills" / "deploy" / "SKILL.md", text="---\nname: deploy\ndescription: ships it\n---\n\nOurs.\n")
        doctor.run(brain=self.brain, fix=True)
        (self.root / "skills" / "deploy" / "SKILL.md").unlink()
        fixtures.write(path=self.root / "skills" / "deploy" / "draft.md", text="not yet a skill\n")
        links = self.root.parent / ".claude" / "skills"
        (links / "all").symlink_to("../../.agents/skills")
        (links / "readme").symlink_to("../../.agents/skills/agent-brain/SKILL.md")

        self.assertEqual(self.messages("layout"), [])

        doctor.run(brain=self.brain, fix=True)

        self.assertEqual(sorted(path.name for path in links.iterdir()), ["agent-brain", "all", "deploy", "readme"])

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root ignores file modes")
    def test_a_link_to_a_project_skill_the_process_cannot_read_is_kept(self) -> None:
        home = self.root / "skills" / "deploy"
        fixtures.write(path=home / "SKILL.md", text="---\nname: deploy\ndescription: ships it\n---\n\nOurs.\n")
        doctor.run(brain=self.brain, fix=True)
        home.chmod(0)

        try:
            doctor.run(brain=self.brain, fix=True)
        finally:
            home.chmod(0o755)

        self.assertTrue((self.root.parent / ".claude" / "skills" / "deploy").is_symlink())

    def test_a_claude_folder_the_process_cannot_search_reads_as_a_missing_link_on_every_floor(self) -> None:
        claude = self.root.parent / ".claude"
        claude.chmod(0)

        try:
            self.assertIn("the skill link is missing, so Claude Code cannot see the skill", self.messages("layout"))
        finally:
            claude.chmod(0o700)

    def test_a_real_folder_carrying_the_skill_passes(self) -> None:
        link = self.root.parent / ".claude" / "skills" / "agent-brain"
        link.unlink()
        shutil.copytree(self.root / "skills" / "agent-brain", link)

        self.assertEqual(self.messages("layout"), [])

    def test_a_skill_of_its_own_in_the_links_place_is_left_for_a_person(self) -> None:
        link = self.root.parent / ".claude" / "skills" / "agent-brain"
        link.unlink()
        fixtures.write(path=link / "SKILL.md", text="---\nname: deploy\n---\n\nThe project's own, not a copy.\n")

        found = [finding for finding in doctor.run(self.brain) if finding.check == "layout"]

        self.assertEqual([finding.message for finding in found], [doctor.ANOTHER_SKILL])
        self.assertFalse(found[0].fixable)

        doctor.run(brain=self.brain, fix=True)

        self.assertIn("not a copy", (link / "SKILL.md").read_text(encoding="utf-8"))

    def test_a_copy_of_the_skill_that_fell_behind_is_refreshed(self) -> None:
        link = self.root.parent / ".claude" / "skills" / "agent-brain"
        home = self.root / "skills" / "agent-brain"
        link.unlink()
        shutil.copytree(home, link)
        fixtures.write(path=home / "SKILL.md", text=(home / "SKILL.md").read_text(encoding="utf-8") + "\nA line the upgrade added.\n")
        fixtures.write(path=home / "references" / "new.md", text="new\n")

        found = [finding for finding in doctor.run(self.brain) if finding.check == "layout"]

        self.assertEqual([(finding.path, finding.message) for finding in found], [(".claude/skills/agent-brain", doctor.STALE_COPY)])
        self.assertTrue(found[0].fixable)

        doctor.run(brain=self.brain, fix=True)

        self.assertEqual(self.messages("layout"), [])
        self.assertTrue(link.is_dir() and not link.is_symlink())
        self.assertEqual((link / "SKILL.md").read_bytes(), (home / "SKILL.md").read_bytes())
        self.assertEqual((link / "references" / "new.md").read_text(encoding="utf-8"), "new\n")

    def test_a_shared_skills_folder_linking_this_skill_in_another_checkout_passes(self) -> None:
        skills = self.root.parent / ".claude" / "skills"
        outside = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, outside)
        other = outside / "other-checkout" / ".agents" / "skills" / "agent-brain"
        shutil.copytree(self.root / "skills" / "agent-brain", other)
        (skills / "agent-brain").unlink()
        skills.rmdir()
        skills.symlink_to(outside / "shared")
        (outside / "shared").mkdir()
        (outside / "shared" / "agent-brain").symlink_to(other)

        self.assertEqual(self.messages("layout"), [])

        fixtures.write(path=other / "SKILL.md", text="---\nname: deploy\n---\n\nAnother skill under the name.\n")

        self.assertEqual(
            self.messages("layout"),
            ["the skill link does not reach the skill, so Claude Code cannot see the skill" + doctor.SHARED_FOLDER],
        )

        (outside / "shared" / "agent-brain").unlink()
        shutil.copytree(self.root / "skills" / "agent-brain", outside / "shared" / "agent-brain")
        copy = outside / "shared" / "agent-brain" / "SKILL.md"
        fixtures.write(path=copy, text=copy.read_text(encoding="utf-8") + "\nA line the other checkout has.\n")

        self.assertEqual(self.messages("layout"), [])

    def test_a_project_skills_own_copy_is_left_for_a_person_not_refreshed(self) -> None:
        for name in ("deploy", "branch-review"):
            with self.subTest(name=name):
                home = self.root / "skills" / name
                fixtures.write(path=home / "SKILL.md", text=f"---\nname: {name}\ndescription: ships it\n---\n\nThe brain's text.\n")
                copy = self.root.parent / ".claude" / "skills" / name
                fixtures.write(path=copy / "SKILL.md", text=f"---\nname: {name}\ndescription: ships it\n---\n\nThe text a person edited here.\n")

                found = [finding for finding in doctor.run(self.brain) if finding.check == "layout"]

                self.assertEqual([finding.message for finding in found], [doctor.ANOTHER_SKILL])
                self.assertFalse(found[0].fixable)

                doctor.run(brain=self.brain, fix=True)

                self.assertIn("a person edited here", (copy / "SKILL.md").read_text(encoding="utf-8"))
                shutil.rmtree(home)
                shutil.rmtree(copy)

    def test_a_link_into_another_skills_folder_is_replaced(self) -> None:
        fixtures.ship_review(self.root)
        doctor.run(brain=self.brain, fix=True)
        link = self.root.parent / ".claude" / "skills" / "agent-brain"
        link.unlink()
        link.symlink_to("../../.agents/skills/branch-review")

        self.assertIn("the skill link does not reach the skill, so Claude Code cannot see the skill", self.messages("layout"))

        doctor.run(brain=self.brain, fix=True)

        self.assertEqual(self.messages("layout"), [])
        self.assertEqual(os.readlink(link), "../../.agents/skills/agent-brain")

    def test_a_real_folder_without_the_skill_is_left_for_a_person(self) -> None:
        link = self.root.parent / ".claude" / "skills" / "agent-brain"
        link.unlink()
        link.mkdir()

        found = [finding for finding in doctor.run(self.brain) if finding.check == "layout"]

        self.assertEqual([finding.message for finding in found], [doctor.REAL_FOLDER_WITHOUT_SKILL])
        self.assertFalse(found[0].fixable)

        doctor.run(brain=self.brain, fix=True)

        self.assertTrue(link.is_dir() and not link.is_symlink())


class FrontDoors(DoctorCase):
    """The imports in the root files Claude Code and Gemini CLI read, which a slip silently unloads."""

    def imports(self) -> list[str]:
        return [finding.line() for finding in doctor.run(self.brain) if finding.check == "import"]

    def doors(
        self,
        texts: dict[str, str],
    ) -> None:
        for name, text in texts.items():
            fixtures.write(
                path=self.root.parent / name,
                text=text,
            )

    def test_the_kits_own_front_doors_pass(self) -> None:
        kit = Path(__file__).resolve().parents[5]
        for name in ("AGENTS.md", "CLAUDE.md", "GEMINI.md"):
            shutil.copy(kit / name, self.root.parent / name)
        fixtures.write(path=self.root.parent / ".gemini" / "plain-and-short.md", text="# Plain\n")

        self.assertEqual(self.imports(), [])

        gemini = self.root.parent / "GEMINI.md"
        gemini.write_text(
            gemini.read_text(encoding="utf-8") + "@docs/missing.md\n",
            encoding="utf-8",
        )

        self.assertEqual(
            self.imports(),
            ["GEMINI.md: import: imports @docs/missing.md, which is not there [flag]"],
        )

    def test_code_in_tilde_and_long_fences_indented_blocks_and_double_backtick_spans_is_skipped(self) -> None:
        self.doors(
            {
                "AGENTS.md": "# AGENTS.md\n\nOurs.\n",
                "CLAUDE.md": (
                    "@AGENTS.md\n\n"
                    "~~~python\n@functools.cache\n~~~\n\n"
                    "````markdown\n```\n@fenced.md.\n```\n````\n\n"
                    "    @indented.md.\n\n"
                    "And `` @spanned.md `code` @spanned.md. `` then @outside.md too.\n"
                ),
            }
        )

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @outside.md, which is not there [flag]"],
        )

    def test_a_fence_never_closed_runs_to_the_end_and_hides_the_rules_import(self) -> None:
        self.doors(
            {
                "AGENTS.md": "# AGENTS.md\n\nOurs.\n",
                "CLAUDE.md": "# CLAUDE.md\n\n```bash\nmake test\n\n@AGENTS.md\n",
            }
        )

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: does not import AGENTS.md, so the rules never load [flag]"],
        )

    def test_a_list_paragraph_indented_four_columns_is_read_and_not_taken_for_code(self) -> None:
        self.doors({"CLAUDE.md": "# CLAUDE.md\n\n- Rules\n\n    @AGENTS.md\n\n- Style\n\n    Follow @docs/gone.md\n"})

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @docs/gone.md, which is not there [flag]"],
        )

    def test_a_fence_inside_an_html_comment_opens_nothing(self) -> None:
        self.doors({"CLAUDE.md": "<!--\n```\n-->\n@AGENTS.md\n@docs/gone.md\n"})

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @docs/gone.md, which is not there [flag]"],
        )

    def test_an_indented_block_right_after_a_heading_is_code(self) -> None:
        self.doors({"CLAUDE.md": "@AGENTS.md\n\n# T\n    @x.md.\n\nThen @docs/gone.md\n"})

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @docs/gone.md, which is not there [flag]"],
        )

    def test_a_backtick_with_no_closer_in_its_list_item_is_plain_text(self) -> None:
        self.doors({"CLAUDE.md": "- Quote with a stray ` here\n- Rules: @AGENTS.md\n- Run `make test`\n- See @docs/gone.md\n"})

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @docs/gone.md, which is not there [flag]"],
        )

    def test_a_backtick_with_no_closer_in_its_heading_is_plain_text(self) -> None:
        self.doors({"CLAUDE.md": "# The ` key\n@AGENTS.md and `x`, see @docs/gone.md\n"})

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @docs/gone.md, which is not there [flag]"],
        )

    def test_a_fence_opened_on_a_list_items_own_line_closes_at_its_closing_line(self) -> None:
        self.doors({"CLAUDE.md": "- ```bash\n  pnpm add @docs/x.md\n  ```\n\n@AGENTS.md\n@docs/gone.md\n"})

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @docs/gone.md, which is not there [flag]"],
        )

    def test_an_indented_block_after_a_setext_heading_is_code(self) -> None:
        self.doors({"CLAUDE.md": "@AGENTS.md\n\nSetup\n=====\n    npm i @docs/gone.md.\n\nSee @docs/gone.md\n"})

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @docs/gone.md, which is not there [flag]"],
        )

    def test_a_heading_at_the_margin_ends_the_list_before_an_indented_block(self) -> None:
        self.doors({"CLAUDE.md": "@AGENTS.md\n\n- item\n## Next\n\n    npm i @docs/gone.md.\n\nSee @docs/gone.md\n"})

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @docs/gone.md, which is not there [flag]"],
        )

    def test_a_fence_at_the_margin_ends_the_list_before_an_indented_block(self) -> None:
        self.doors({"CLAUDE.md": "@AGENTS.md\n\n- item\n```\nx\n```\n    npm i @docs/gone.md.\n\nSee @docs/gone.md\n"})

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @docs/gone.md, which is not there [flag]"],
        )

    def test_a_comment_that_never_closes_runs_to_the_end_and_hides_the_rules_import(self) -> None:
        self.doors({"CLAUDE.md": "<!-- a note that lost its end\n\n@AGENTS.md\n"})

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: does not import AGENTS.md, so the rules never load [flag]"],
        )

    def test_a_backtick_after_a_backslash_opens_no_span(self) -> None:
        self.doors({"CLAUDE.md": "@AGENTS.md\n\nUse \\` to quote, then @docs/gone.md. and `x`\n"})

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @docs/gone.md. with the punctuation glued on, so the parser looks for that name and the file never loads, end the path with a space or a line break [flag]"],
        )

    def test_a_token_that_leads_nowhere_and_names_no_markdown_file_is_a_mention(self) -> None:
        self.doors({"CLAUDE.md": "@AGENTS.md\n\nWe pin @tanstack/query@5.59.0 and use @pytest.fixture for setup. Ask @jane.doe, then see @docs/SETUP.\n"})

        self.assertEqual(self.imports(), [])

        fixtures.write(
            path=self.root.parent / "docs" / "SETUP",
            text="Setup.\n",
        )

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @docs/SETUP. with the punctuation glued on, so the parser looks for that name and the file never loads, end the path with a space or a line break [flag]"],
        )

    def test_an_import_found_only_under_another_case_is_named_the_same_on_every_machine(self) -> None:
        fixtures.write(
            path=self.root.parent / "docs" / "a.md",
            text="# A\n",
        )
        self.doors({"AGENTS.md": "# AGENTS.md\n\n@Docs/a.md\n"})

        self.assertEqual(
            self.imports(),
            ["AGENTS.md: import: imports @Docs/a.md, which is spelled docs/a.md, so it loads only where the file system ignores case [flag]"],
        )

    def test_the_rules_imported_under_another_case_give_one_finding(self) -> None:
        self.doors({"CLAUDE.md": "@Agents.md\n"})

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @Agents.md, which is spelled AGENTS.md, so it loads only where the file system ignores case [flag]"],
        )

    def test_gemini_cli_reads_an_html_comment_raw(self) -> None:
        (self.root.parent / ".gemini").mkdir()
        self.doors({"GEMINI.md": "<!-- once @old.md -->\n@AGENTS.md\n"})

        self.assertEqual(
            self.imports(),
            ["GEMINI.md: import: imports @old.md, which is not there [flag]"],
        )

    def test_the_rules_imported_in_a_sentence_through_a_dot_slash_or_by_a_symlink_load(self) -> None:
        (self.root.parent / ".gemini").mkdir()
        self.doors(
            {
                "AGENTS.md": "# AGENTS.md\n\nOurs.\n",
                "CLAUDE.md": "# CLAUDE.md\n\nSee @AGENTS.md for the rules.\n",
                "GEMINI.md": "# GEMINI.md\n\n@./AGENTS.md\n",
            }
        )

        self.assertEqual(self.imports(), [])

        claude = self.root.parent / "CLAUDE.md"
        claude.unlink()
        claude.symlink_to("AGENTS.md")

        self.assertEqual(self.imports(), [])

        claude.unlink()
        self.doors({"CLAUDE.md": "# CLAUDE.md\n\nSee AGENTS.md for the rules.\n"})

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: does not import AGENTS.md, so the rules never load [flag]"],
        )

    def test_claude_code_reads_an_import_without_its_fragment(self) -> None:
        fixtures.write(
            path=self.root.parent / "docs" / "guide.md",
            text="# Guide\n",
        )
        self.doors({"CLAUDE.md": "@AGENTS.md\n\nSee @docs/guide.md#setup first, and @docs/guide.md#usage. Then @docs/gone.md#setup\n"})

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @docs/gone.md, which is not there [flag]"],
        )

    def test_gemini_cli_keeps_a_fragment_in_the_name_it_looks_for(self) -> None:
        (self.root.parent / ".gemini").mkdir()
        fixtures.write(
            path=self.root.parent / "docs" / "style.md",
            text="# Style\n",
        )
        self.doors(
            {
                "AGENTS.md": "# AGENTS.md\n\nSee @docs/style.md#setup first.\n",
                "CLAUDE.md": "@AGENTS.md @docs/style.md#setup\n",
                "GEMINI.md": "@AGENTS.md @docs/style.md#setup\n",
            }
        )

        self.assertEqual(
            self.imports(),
            [
                "AGENTS.md: import: imports @docs/style.md#setup, which Gemini CLI looks for with the fragment in its name, so it is not there [flag]",
                "GEMINI.md: import: imports @docs/style.md#setup, which Gemini CLI looks for with the fragment in its name, so it is not there [flag]",
            ],
        )

        shutil.rmtree(self.root.parent / ".gemini")

        self.assertEqual(self.imports(), [])

    def test_a_lone_emphasis_marker_opens_no_import_in_claude_code(self) -> None:
        self.doors({"CLAUDE.md": "@AGENTS.md\n\nDrafts are named *@notes.md or snake_@todo.md here.\n\n**@docs/gone.md**\n"})

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @docs/gone.md, which is not there [flag]"],
        )

    def test_a_trailing_marker_with_no_opener_stays_in_the_path_claude_code_reads(self) -> None:
        fixtures.write(
            path=self.root.parent / "docs" / "setup.md",
            text="# Setup\n",
        )
        self.doors({"CLAUDE.md": "@AGENTS.md\n\nRead @docs/setup.md* first.\n"})

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @docs/setup.md*, which is not there [flag]"],
        )

    def test_emphasis_pairs_across_the_lines_of_a_paragraph_and_never_across_a_blank_line(self) -> None:
        fixtures.write(
            path=self.root.parent / "docs" / "setup.md",
            text="# Setup\n",
        )
        self.doors({"CLAUDE.md": "@AGENTS.md\n\n**Before any change, read\n@docs/setup.md**\n\n**Never across a blank line\n\n@docs/gone.md**\n"})

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @docs/gone.md**, which is not there [flag]"],
        )

    def test_two_openers_with_no_closer_open_no_import_in_claude_code(self) -> None:
        self.doors({"CLAUDE.md": "@AGENTS.md\n\nDrafts are named *@notes.md and *@todo.md.\n\n**@docs/gone.md**\n"})

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @docs/gone.md, which is not there [flag]"],
        )

    def test_a_list_bullet_is_no_opener_for_a_trailing_marker(self) -> None:
        fixtures.write(
            path=self.root.parent / "docs" / "setup.md",
            text="# Setup\n",
        )
        self.doors({"CLAUDE.md": "@AGENTS.md\n\n* Read @docs/setup.md* first.\n"})

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @docs/setup.md*, which is not there [flag]"],
        )

    def test_a_pair_already_closed_lends_no_opener_to_a_trailing_marker(self) -> None:
        fixtures.write(
            path=self.root.parent / "docs" / "setup.md",
            text="# Setup\n",
        )
        self.doors({"CLAUDE.md": "@AGENTS.md\n\nRead *this* and @docs/setup.md* first.\n"})

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @docs/setup.md*, which is not there [flag]"],
        )

    def test_an_import_right_after_an_emphasis_marker_is_named_for_gemini_cli_in_a_file_both_agents_read(self) -> None:
        (self.root.parent / ".gemini").mkdir()
        fixtures.write(
            path=self.root.parent / "docs" / "style.md",
            text="# Style\n",
        )
        self.doors(
            {
                "AGENTS.md": "# AGENTS.md\n\n**@docs/style.md**\n",
                "GEMINI.md": "@AGENTS.md\n",
            }
        )

        self.assertEqual(
            self.imports(),
            ["AGENTS.md: import: imports @docs/style.md right after an emphasis marker, where Gemini CLI opens no import [flag]"],
        )

        shutil.rmtree(self.root.parent / ".gemini")

        self.assertEqual(self.imports(), [])

    def test_a_backslash_in_an_import_path_is_named(self) -> None:
        fixtures.write(
            path=self.root.parent / "docs" / "setup.md",
            text="# Setup\n",
        )
        self.doors({"CLAUDE.md": "@AGENTS.md\n@docs\\setup.md\n"})

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @docs\\setup.md with a backslash in its path, which ends the path there, so write it with slashes [flag]"],
        )

    def test_a_backslash_import_of_a_file_that_is_not_markdown_is_named(self) -> None:
        fixtures.write(
            path=self.root.parent / "config" / "settings.json",
            text="{}\n",
        )
        self.doors({"CLAUDE.md": "@AGENTS.md\n@config\\settings.json\n"})

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @config\\settings.json with a backslash in its path, which ends the path there, so write it with slashes [flag]"],
        )

    def test_a_backslash_in_an_import_is_named_for_gemini_cli_which_reads_it_as_part_of_the_name(self) -> None:
        shutil.rmtree(self.root.parent / ".claude")
        (self.root.parent / ".gemini").mkdir()
        fixtures.write(
            path=self.root.parent / "config" / "settings.json",
            text="{}\n",
        )
        self.doors(
            {
                "AGENTS.md": "# AGENTS.md\n\n@config\\settings.json\n",
                "GEMINI.md": "@AGENTS.md\n@config\\settings.json\n",
            }
        )
        gemini = "imports @config\\settings.json with a backslash in its path, which Gemini CLI reads as part of the name, so write it with slashes [flag]"

        self.assertEqual(
            self.imports(),
            [
                f"AGENTS.md: import: {gemini}",
                f"GEMINI.md: import: {gemini}",
            ],
        )

        (self.root.parent / ".claude").mkdir()

        self.assertEqual(
            self.imports(),
            [
                "AGENTS.md: import: imports @config\\settings.json with a backslash in its path, which ends the path there, so write it with slashes [flag]",
                f"AGENTS.md: import: {gemini}",
                f"GEMINI.md: import: {gemini}",
            ],
        )

    def test_an_import_that_climbs_out_and_back_in_by_the_folders_name_is_named_and_never_counts_as_the_rules(self) -> None:
        folder = self.root.parent.name
        self.doors({"CLAUDE.md": f"@../{folder}/AGENTS.md\n@../{folder}-shared/rules.md\n"})

        self.assertEqual(
            self.imports(),
            [
                f"CLAUDE.md: import: imports @../{folder}/AGENTS.md out of the project's folder and back in by its name, which another checkout may not share, so write it from the project's folder [flag]",
                f"CLAUDE.md: import: imports @../{folder}-shared/rules.md, which is not there [flag]",
                "CLAUDE.md: import: does not import AGENTS.md, so the rules never load [flag]",
            ],
        )

    def test_an_import_comes_back_in_only_by_retracing_the_folders_it_climbed_out_of(self) -> None:
        folder = self.root.parent.name
        parent = self.root.parent.parent.name
        self.doors({"CLAUDE.md": f"@AGENTS.md\n@../../{parent}/{folder}/AGENTS.md\n@../../{parent}-other/{folder}/README.md\n"})

        self.assertEqual(
            self.imports(),
            [
                f"CLAUDE.md: import: imports @../../{parent}/{folder}/AGENTS.md out of the project's folder and back in by its name, which another checkout may not share, so write it from the project's folder [flag]",
                f"CLAUDE.md: import: imports @../../{parent}-other/{folder}/README.md, which is not there [flag]",
            ],
        )

    def test_an_import_from_the_root_of_the_disk_is_named_and_never_counts_as_the_rules(self) -> None:
        absolute = self.root.parent / "AGENTS.md"
        self.doors({"CLAUDE.md": f"@{absolute}\n"})

        self.assertEqual(
            self.imports(),
            [
                f"CLAUDE.md: import: imports @{absolute} from the root of the disk, which is there on one machine and missing on the next, so write it from the project's folder [flag]",
                "CLAUDE.md: import: does not import AGENTS.md, so the rules never load [flag]",
            ],
        )

    def test_the_same_slip_written_twice_is_named_twice_however_many_agents_read_it(self) -> None:
        self.doors({"AGENTS.md": "# AGENTS.md\n\nSee @docs/gone.md. Then see @docs/gone.md. again.\n"})
        glued = "AGENTS.md: import: imports @docs/gone.md. with the punctuation glued on, so the parser looks for that name and the file never loads, end the path with a space or a line break [flag]"

        self.assertEqual(
            self.imports(),
            [glued, glued],
        )

        (self.root.parent / ".gemini").mkdir()
        self.doors({"GEMINI.md": "@AGENTS.md\n"})

        self.assertEqual(
            self.imports(),
            [glued, glued],
        )

    def test_a_mention_of_a_person_or_a_package_is_passed_over(self) -> None:
        self.doors({"AGENTS.md": "# AGENTS.md\n\nAsk @alice before touching billing, we use @tanstack/query.\nKeep @./scratch and @notes.md apart.\n"})

        self.assertEqual(
            self.imports(),
            [
                "AGENTS.md: import: imports @./scratch, which is not there [flag]",
                "AGENTS.md: import: imports @notes.md, which is not there [flag]",
            ],
        )

    def test_an_import_from_the_home_folder_is_checked_for_glue_and_never_for_being_there(self) -> None:
        self.doors({"AGENTS.md": "# AGENTS.md\n\n@~/.claude/my-project-instructions.md\nThen @~/notes.md, and work.\n"})
        home = {"HOME": str(self.root.parent / "home")}

        with mock.patch.dict(
            os.environ,
            home,
        ):
            found = self.imports()

        self.assertEqual(
            found,
            ["AGENTS.md: import: imports @~/notes.md, with the punctuation glued on, so the parser looks for that name and the file never loads, end the path with a space or a line break [flag]"],
        )

    def test_claude_code_leaves_emphasis_around_an_import_out_of_its_path(self) -> None:
        fixtures.write(
            path=self.root.parent / "docs" / "style.md",
            text="# Style\n",
        )
        self.doors({"CLAUDE.md": "@AGENTS.md\n\n**Always follow @docs/style.md**\n\n**@docs/style.md**.\n\n_@docs/gone.md_\n"})

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @docs/gone.md, which is not there [flag]"],
        )

    def test_gemini_cli_opens_no_import_after_an_emphasis_marker_and_reads_markers_after_one_as_its_name(self) -> None:
        (self.root.parent / ".gemini").mkdir()
        fixtures.write(
            path=self.root.parent / "docs" / "style.md",
            text="# Style\n",
        )
        self.doors(
            {
                "AGENTS.md": "# AGENTS.md\n\n**Always follow @docs/style.md**\n",
                "CLAUDE.md": "**@AGENTS.md**\n\n**Always follow @docs/style.md**\n",
                "GEMINI.md": "**@AGENTS.md**\n",
            }
        )

        self.assertEqual(
            self.imports(),
            [
                "AGENTS.md: import: imports @docs/style.md** with emphasis markers glued on, which Gemini CLI reads as part of the name [flag]",
                "GEMINI.md: import: does not import AGENTS.md, so the rules never load [flag]",
            ],
        )

    def test_an_escaped_space_is_a_finding_where_gemini_reads_and_a_path_in_claude_md(self) -> None:
        (self.root.parent / ".gemini").mkdir()
        fixtures.write(
            path=self.root.parent / "docs" / "team notes.md",
            text="# Notes\n",
        )
        self.doors(
            {
                "AGENTS.md": "# AGENTS.md\n\n@docs/team\\ notes.md\n",
                "CLAUDE.md": "@AGENTS.md\n@docs/team\\ notes.md\n",
                "GEMINI.md": "@AGENTS.md\n@docs/team\\ notes.md\n",
            }
        )

        self.assertEqual(
            self.imports(),
            [
                "AGENTS.md: import: imports @docs/team\\ notes.md with an escaped space, which Gemini CLI does not read, so the path ends at the backslash there [flag]",
                "GEMINI.md: import: imports @docs/team\\ notes.md with an escaped space, which Gemini CLI does not read, so the path ends at the backslash there [flag]",
            ],
        )

    def test_an_escaped_space_in_agents_md_is_a_path_while_the_project_keeps_no_gemini_folder(self) -> None:
        fixtures.write(
            path=self.root.parent / "docs" / "team notes.md",
            text="# Notes\n",
        )
        self.doors({"AGENTS.md": "# AGENTS.md\n\n@docs/team\\ notes.md\n"})

        self.assertEqual(self.imports(), [])

        (self.root.parent / ".gemini").mkdir()
        self.doors({"GEMINI.md": "@AGENTS.md\n"})

        self.assertEqual(
            self.imports(),
            ["AGENTS.md: import: imports @docs/team\\ notes.md with an escaped space, which Gemini CLI does not read, so the path ends at the backslash there [flag]"],
        )

    def test_an_escaped_space_in_agents_md_read_by_both_agents_is_looked_for_by_claude_code_and_named_for_gemini_cli(self) -> None:
        (self.root.parent / ".gemini").mkdir()
        self.doors(
            {
                "AGENTS.md": "# AGENTS.md\n\n@docs/team\\ notes.md\n",
                "GEMINI.md": "@AGENTS.md\n",
            }
        )

        self.assertEqual(
            self.imports(),
            [
                "AGENTS.md: import: imports @docs/team\\ notes.md, which is not there [flag]",
                "AGENTS.md: import: imports @docs/team\\ notes.md with an escaped space, which Gemini CLI does not read, so the path ends at the backslash there [flag]",
            ],
        )

    def test_an_escaped_space_path_in_agents_md_that_is_not_there_is_reported_while_no_gemini_folder_is_kept(self) -> None:
        self.doors({"AGENTS.md": "# AGENTS.md\n\n@docs/team\\ notes.md\n"})

        self.assertEqual(
            self.imports(),
            ["AGENTS.md: import: imports @docs/team\\ notes.md, which is not there [flag]"],
        )

    def test_a_front_door_that_is_not_utf8_is_still_read(self) -> None:
        self.doors({"AGENTS.md": "# AGENTS.md\n\nOurs.\n"})
        (self.root.parent / "CLAUDE.md").write_bytes(b"# CLAUDE.md\n\nCaf\xe9 rules.\n\n@AGENTS.md\n@docs/gone.md.\n")

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @docs/gone.md. with the punctuation glued on, so the parser looks for that name and the file never loads, end the path with a space or a line break [flag]"],
        )

    def test_a_front_door_that_cannot_be_read_is_named(self) -> None:
        self.doors({"CLAUDE.md": "@AGENTS.md\n"})
        (self.root.parent / "AGENTS.md").unlink()
        (self.root.parent / "AGENTS.md").mkdir()

        self.assertEqual(
            self.imports(),
            [
                "AGENTS.md: import: cannot be read: Is a directory: AGENTS.md [flag]",
                "CLAUDE.md: import: imports @AGENTS.md, which is not there [flag]",
            ],
        )

    def test_claude_md_and_gemini_md_are_read_only_beside_their_agents_folder(self) -> None:
        shutil.rmtree(self.root.parent / ".claude")
        self.doors(
            {
                "AGENTS.md": "# AGENTS.md\n\nOurs.\n",
                "CLAUDE.md": "# CLAUDE.md\n\nOurs.\n",
                "GEMINI.md": "@AGENTS.md\n@.gemini/plain-and-short.md\n",
            }
        )

        self.assertEqual(self.imports(), [])

        (self.root.parent / ".claude").mkdir()
        (self.root.parent / ".gemini").mkdir()

        self.assertEqual(
            self.imports(),
            [
                "CLAUDE.md: import: does not import AGENTS.md, so the rules never load [flag]",
                "GEMINI.md: import: imports @.gemini/plain-and-short.md, which is not there [flag]",
            ],
        )

    def test_a_front_door_spelled_in_another_case_is_named(self) -> None:
        (self.root.parent / "CLAUDE.md").unlink()
        self.doors(
            {
                "AGENTS.md": "# AGENTS.md\n\nOurs.\n",
                "Claude.md": "@AGENTS.md\n",
            }
        )

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: is spelled Claude.md, and only CLAUDE.md is read, so rename it [flag]"],
        )

    def test_a_missing_claude_md_is_named_while_the_claude_folder_is_there(self) -> None:
        (self.root.parent / "CLAUDE.md").unlink()

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: is not there, so Claude Code loads no rules, and the next upgrade writes it back [flag]"],
        )

        shutil.rmtree(self.root.parent / ".claude")

        self.assertEqual(self.imports(), [])

    def test_a_missing_gemini_md_is_named_once_the_gemini_folder_is_there(self) -> None:
        self.assertEqual(self.imports(), [])

        (self.root.parent / ".gemini").mkdir()

        self.assertEqual(
            self.imports(),
            ["GEMINI.md: import: is not there, so Gemini CLI loads no rules, and the next upgrade writes it back [flag]"],
        )

    def test_a_missing_agents_md_is_named_with_no_agent_folder_at_all(self) -> None:
        shutil.rmtree(self.root.parent / ".claude")
        (self.root.parent / "AGENTS.md").unlink()

        self.assertEqual(
            self.imports(),
            ["AGENTS.md: import: is not there, so no agent finds the rules, and the next upgrade writes it back [flag]"],
        )

    def test_an_import_glued_to_punctuation_is_reported(self) -> None:
        fixtures.write(path=self.root.parent / "AGENTS.md", text="# AGENTS.md\n\nThe rules are in @.agents/SYSTEM.md. Read them, then @.agents/INDEX.md, then work.\n")

        self.assertEqual(
            self.imports(),
            [
                "AGENTS.md: import: imports @.agents/SYSTEM.md. with the punctuation glued on, so the parser looks for that name and the file never loads, end the path with a space or a line break [flag]",
                "AGENTS.md: import: imports @.agents/INDEX.md, with the punctuation glued on, so the parser looks for that name and the file never loads, end the path with a space or a line break [flag]",
            ],
        )

    def test_an_import_that_reaches_nothing_is_reported(self) -> None:
        fixtures.write(path=self.root.parent / "CLAUDE.md", text="# CLAUDE.md\n\n@AGENTS.md\n@docs/missing.md\n")
        fixtures.write(path=self.root.parent / "AGENTS.md", text="# AGENTS.md\n\nOurs.\n")

        self.assertEqual(self.imports(), ["CLAUDE.md: import: imports @docs/missing.md, which is not there [flag]"])

    def test_a_front_door_without_the_rules_import_is_reported(self) -> None:
        (self.root.parent / ".gemini").mkdir()
        fixtures.write(path=self.root.parent / "AGENTS.md", text="# AGENTS.md\n\nOurs.\n")
        fixtures.write(path=self.root.parent / "CLAUDE.md", text="# CLAUDE.md\n\nThe rules are in @AGENTS.md.\n")
        fixtures.write(path=self.root.parent / "GEMINI.md", text="# GEMINI.md\n\nOurs alone.\n")

        self.assertEqual(
            self.imports(),
            [
                "CLAUDE.md: import: imports @AGENTS.md. with the punctuation glued on, so the parser looks for that name and the file never loads, end the path with a space or a line break [flag]",
                "CLAUDE.md: import: does not import AGENTS.md, so the rules never load [flag]",
                "GEMINI.md: import: does not import AGENTS.md, so the rules never load [flag]",
            ],
        )

    def test_imports_inside_code_and_comments_are_skipped_and_a_mail_address_is_not_one(self) -> None:
        fixtures.write(path=self.root.parent / "AGENTS.md", text="# AGENTS.md\n\nWrite to owner@example.com.\n")
        claude = "# CLAUDE.md\n\n<!-- once @old.md. -->\n\n@AGENTS.md\n\n```text\n@fenced.md.\n```\n\nAnd `@inline.md.` too.\n"
        self.doors({"CLAUDE.md": claude})

        self.assertEqual(self.imports(), [])

        self.doors({"CLAUDE.md": claude + "\nAnd @outside.md too.\n"})

        self.assertEqual(
            self.imports(),
            ["CLAUDE.md: import: imports @outside.md, which is not there [flag]"],
        )


if __name__ == "__main__":
    unittest.main()
