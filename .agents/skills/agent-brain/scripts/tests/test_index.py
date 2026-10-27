"""Every INDEX.md comes from the folder it lists, and says so."""

import unittest
from pathlib import Path
from unittest import mock

from brainlib import frontmatter, index, schema, store, writes
from tests import fixtures


class Building(fixtures.BrainCase):
    def setUp(self) -> None:
        super().setUp()
        self.built = index.build(self.brain)

    def body(
        self,
        relative: str,
    ) -> str:
        return frontmatter.parse(self.built[self.root / relative]).body

    def test_there_is_one_index_per_indexed_folder_plus_the_spine(self) -> None:
        self.assertEqual(len(self.built), len(schema.INDEXED_FOLDERS) + 1)

    def test_every_indexed_kind_has_a_renderer_and_no_other_does(self) -> None:
        indexed = {kind.name for kind in schema.KINDS.values() if kind.indexed}

        self.assertEqual(set(index.RENDERERS), indexed)

    def test_the_spine_counts_what_each_tracker_holds(self) -> None:
        spine = self.body("INDEX.md")

        self.assertIn("the plans: empty", spine)
        self.assertIn("the work tracker: 1 open, 1 done", spine)
        self.assertIn("the defect tracker: 1 open (1 bug)", spine)
        self.assertIn("the decision record: 1 standing", spine)
        self.assertIn("code review ledgers: 1 open", spine)
        self.assertIn("the knowledge library: 1 doc", spine)
        self.assertIn("project facts: 1 fact", spine)
        self.assertIn("handover/ is born with its first handoff", spine)

    def test_the_spine_names_the_rulebook_first(self) -> None:
        spine = self.body("INDEX.md")

        self.assertIn("- [SYSTEM.md](SYSTEM.md), the rulebook", spine)

    def test_the_spine_names_the_tool_and_the_version_stamp(self) -> None:
        spine = self.body("INDEX.md")

        self.assertIn("- [skills/agent-brain/SKILL.md](skills/agent-brain/SKILL.md), the brain tool", spine)
        self.assertNotIn("skills/branch-review", spine)

        fixtures.ship_review(self.root)
        self.built = index.build(self.brain)
        spine = self.body("INDEX.md")

        self.assertIn("- [skills/branch-review/SKILL.md](skills/branch-review/SKILL.md), the code review", spine)

    def test_a_projects_own_folder_under_a_kit_skills_name_is_the_projects_until_the_stamp_says_otherwise(self) -> None:
        fixtures.write(path=self.root / "VERSION", text="1.1.0\n")
        fixtures.write(path=self.root / "skills" / "branch-review" / "SKILL.md", text="---\nname: branch-review\ndescription: our own release checklist\n---\n\nOurs.\n")
        self.built = index.build(self.brain)
        spine = self.body("INDEX.md")

        self.assertIn("- [skills/branch-review/SKILL.md](skills/branch-review/SKILL.md), our own release checklist", spine)
        self.assertNotIn("the code review, run on the owner's word", spine)

        (self.root / "VERSION").unlink()
        self.built = index.build(self.brain)
        spine = self.body("INDEX.md")

        self.assertIn("- [skills/branch-review/SKILL.md](skills/branch-review/SKILL.md), our own release checklist", spine)

    def test_the_spine_names_a_skill_the_project_keeps(self) -> None:
        fixtures.write(path=self.root / "skills" / "deploy" / "SKILL.md", text="---\nname: deploy\ndescription: ships the app\n---\n\nOurs.\n")
        fixtures.write(path=self.root / "skills" / "odd" / "SKILL.md", text="no header here\n")
        self.built = index.build(self.brain)
        spine = self.body("INDEX.md")

        self.assertIn("- [skills/deploy/SKILL.md](skills/deploy/SKILL.md), ships the app", spine)
        self.assertIn("- [skills/odd/SKILL.md](skills/odd/SKILL.md), the project's own odd skill", spine)
        self.assertLess(spine.index("[skills/agent-brain/SKILL.md]"), spine.index("[skills/deploy/SKILL.md]"))
        self.assertIn("- [VERSION](VERSION), the kit release this brain is on", spine)
        self.assertLess(spine.index("[VERSION]"), spine.index("## The knowledge"))

    def test_open_todos_are_grouped_by_area_and_closed_ones_counted(self) -> None:
        todos = self.body("todos/INDEX.md")

        self.assertIn("### reports\n\n- [TODO-001](TODO-001-export-to-csv.md), export the table to csv", todos)
        self.assertIn("- theme: 1 done, 0 dropped", todos)
        self.assertNotIn("TODO-003-dark-mode.md", todos)

    def test_open_bugs_are_grouped_by_severity(self) -> None:
        bugs = self.body("bugs/INDEX.md")

        self.assertIn("### bug\n\n- [BUG-002](BUG-002-login-loop.md), the sign in screen loops", bugs)

    def test_docs_are_grouped_by_category(self) -> None:
        docs = self.body("docs/INDEX.md")

        self.assertIn("## systems\n\n- [login](systems/login.md), the sign in flow", docs)

    def test_a_review_line_reaches_its_ledger(self) -> None:
        reviews = self.body("reviews/INDEX.md")

        self.assertIn("- [2026-09-feat-billing](2026-09-feat-billing/ledger.md), the billing review", reviews)

    def test_a_file_outside_the_spec_still_shows_in_its_index_under_unknown(self) -> None:
        bug = (self.root / "bugs" / "BUG-002-login-loop.md").read_text(encoding="utf-8")
        fixtures.write(path=self.root / "bugs" / "BUG-003-no-severity.md", text=bug.replace("name: BUG-002-login-loop", "name: BUG-003-no-severity").replace("severity: bug\n", ""))
        fixtures.write(path=self.root / "bugs" / "BUG-004-odd-status.md", text=bug.replace("name: BUG-002-login-loop", "name: BUG-004-odd-status").replace("status: open", "status: weird"))
        fixtures.write(path=self.root / "bugs" / "BUG-005-odd-severity.md", text=bug.replace("name: BUG-002-login-loop", "name: BUG-005-odd-severity").replace("severity: bug", "severity: critical"))
        todo = (self.root / "todos" / "TODO-001-export-to-csv.md").read_text(encoding="utf-8")
        fixtures.write(path=self.root / "todos" / "TODO-005-no-status.md", text=todo.replace("name: TODO-001-export-to-csv", "name: TODO-005-no-status").replace("status: open\n", ""))
        self.brain.refresh()
        built = index.build(self.brain)

        spine = frontmatter.parse(built[self.root / "INDEX.md"]).body
        bugs = frontmatter.parse(built[self.root / "bugs" / "INDEX.md"]).body
        todos = frontmatter.parse(built[self.root / "todos" / "INDEX.md"]).body
        self.assertIn("the defect tracker: 3 open (1 bug, 2 unknown)", spine)
        self.assertIn("### unknown\n\n- [BUG-003](BUG-003-no-severity.md)", bugs)
        self.assertIn("- [BUG-005](BUG-005-odd-severity.md)", bugs)
        self.assertIn("## Unknown\n\n- [BUG-004](BUG-004-odd-status.md)", bugs)
        self.assertIn("- reports: 0 done, 0 dropped, 1 unknown", todos)

    def test_a_ledger_with_a_status_outside_the_spec_shows_under_unknown(self) -> None:
        ledger = (self.root / "reviews" / "2026-09-feat-billing" / "ledger.md").read_text(encoding="utf-8")
        fixtures.write(
            path=self.root / "reviews" / "2026-08-feat-search" / "ledger.md",
            text=ledger.replace("2026-09-feat-billing", "2026-08-feat-search").replace("status: open", "status: archived"),
        )
        self.brain.refresh()
        built = index.build(self.brain)

        spine = frontmatter.parse(built[self.root / "INDEX.md"]).body
        reviews = frontmatter.parse(built[self.root / "reviews" / "INDEX.md"]).body
        self.assertIn("code review ledgers: 1 archived, 1 open", spine)
        self.assertIn("## Unknown\n\n- [2026-08-feat-search](2026-08-feat-search/ledger.md)", reviews)

    def test_an_empty_folder_says_so(self) -> None:
        for path in (self.root / "memory").glob("*.md"):
            if path.name != "INDEX.md":
                path.unlink()
        self.brain.refresh()

        memory = frontmatter.parse(index.build(self.brain)[self.root / "memory" / "INDEX.md"]).body

        self.assertIn("Nothing here yet.", memory)

    def test_every_index_says_it_is_generated(self) -> None:
        for path, text in self.built.items():
            with self.subTest(path=path.relative_to(self.root)):
                self.assertIn("Built by the brain tool", frontmatter.parse(text).body)


class Writing(fixtures.BrainCase):
    def test_the_fixture_stubs_start_out_stale(self) -> None:
        self.assertEqual(len(index.stale(self.brain)), len(schema.INDEXED_FOLDERS) + 1)

    def test_writing_makes_them_current_and_keeps_their_names(self) -> None:
        index.write(brain=self.brain, stamp="2026-10-01")

        self.assertEqual(index.stale(self.brain), [])
        written = frontmatter.read(str(self.root / "todos" / "INDEX.md"))
        self.assertEqual(written.get("name"), "todos-index")
        self.assertEqual(written.get("updated"), "2026-10-01")

    def test_writing_builds_the_brain_once_however_many_are_stale(self) -> None:
        with mock.patch("brainlib.index.build", wraps=index.build) as building:
            written = index.write(brain=self.brain, stamp="2026-10-01")

        self.assertEqual(len(written), len(schema.INDEXED_FOLDERS) + 1)
        self.assertEqual(building.call_count, 1)

    def test_writing_twice_changes_nothing_the_second_time(self) -> None:
        index.write(brain=self.brain, stamp="2026-10-01")

        self.assertEqual(index.write(brain=self.brain, stamp="2026-10-02"), [])

    def test_the_header_is_the_tools_and_links_only_what_exists(self) -> None:
        index.write(brain=self.brain, stamp="2026-10-01")

        todos = frontmatter.read(str(self.root / "todos" / "INDEX.md"))
        spine = frontmatter.read(str(self.root / "INDEX.md"))
        self.assertEqual(todos.get("description"), "the work tracker's index, open items first")
        self.assertEqual(todos.get_list("links"), ["INDEX.md"])
        self.assertEqual(spine.get_list("links"), [])

    def test_a_spec_that_exists_gets_its_link_in_the_header_and_the_body(self) -> None:
        fixtures.write(path=self.root / "spec" / "todos.md", text="---\nname: spec-todos\n---\n\n# Todos\n")
        index.write(brain=self.brain, stamp="2026-10-01")

        todos = frontmatter.read(str(self.root / "todos" / "INDEX.md"))
        self.assertEqual(todos.get_list("links"), ["INDEX.md", "spec/todos.md"])
        self.assertIn("Filed per [spec/todos.md](../spec/todos.md).", todos.body)
        self.assertNotIn("Filed per", frontmatter.read(str(self.root / "bugs" / "INDEX.md")).body)

    def test_a_changed_header_makes_the_index_stale_and_is_rebuilt(self) -> None:
        index.write(brain=self.brain, stamp="2026-10-01")
        target = self.root / "bugs" / "INDEX.md"
        document = frontmatter.read(str(target))
        document.set(key="description", value="my own words")
        frontmatter.write(path=str(target), document=document)

        self.assertEqual(index.stale(self.brain), [target])

        index.write(brain=self.brain, stamp="2026-10-02")

        rebuilt = frontmatter.read(str(target))
        self.assertEqual(rebuilt.get("description"), "the defect tracker's index, open bugs first")
        self.assertEqual(rebuilt.get("updated"), "2026-10-02")

    def test_a_symlinked_index_becomes_a_real_file_and_nothing_outside_is_written(self) -> None:
        elsewhere = self.root.parent / "elsewhere-index.md"
        elsewhere.write_text("theirs\n", encoding="utf-8")
        target = self.root / "bugs" / "INDEX.md"
        target.unlink()
        target.symlink_to(elsewhere)

        index.write(brain=self.brain, stamp="2026-10-01")

        self.assertFalse(target.is_symlink())
        self.assertIn("BUG-002", target.read_text(encoding="utf-8"))
        self.assertEqual(elsewhere.read_text(encoding="utf-8"), "theirs\n")

    def test_an_index_in_a_folder_the_process_cannot_search_is_stale_not_fatal(self) -> None:
        index.write(brain=self.brain, stamp="2026-10-01")
        folder = self.root / "docs"
        folder.chmod(0)

        try:
            self.assertIn(folder / "INDEX.md", index.stale(self.brain))
        finally:
            folder.chmod(0o700)

    def test_the_date_alone_never_makes_an_index_stale(self) -> None:
        index.write(brain=self.brain, stamp="2026-10-01")
        target = self.root / "bugs" / "INDEX.md"
        document = frontmatter.read(str(target))
        document.set(key="updated", value="2020-01-01")
        frontmatter.write(path=str(target), document=document)

        self.assertEqual(index.stale(self.brain), [])

    def test_an_index_that_lost_its_frontmatter_is_rebuilt_whole(self) -> None:
        index.write(brain=self.brain, stamp="2026-10-01")
        target = self.root / "bugs" / "INDEX.md"
        target.write_text("# Bugs\n\nSomeone pasted over it.\n", encoding="utf-8")

        self.assertIn(target, index.stale(self.brain))

        index.write(brain=self.brain, stamp="2026-10-02")

        written = frontmatter.read(str(target))
        self.assertEqual(written.get("name"), "bugs-index")
        self.assertIn("BUG-002", written.body)

    def test_an_index_that_is_not_utf8_is_stale_and_rewritten(self) -> None:
        index.write(brain=self.brain, stamp="2026-10-01")
        target = self.root / "memory" / "INDEX.md"
        target.write_bytes(b"---\nname: memory-index\n---\n\ncaf\xe9\n")

        self.assertIn(target, index.stale(self.brain))

        index.write(brain=self.brain, stamp="2026-10-02")

        self.assertIn("stripe-retries-twice", target.read_text(encoding="utf-8"))

    def test_filing_something_new_makes_its_index_and_the_spine_stale(self) -> None:
        index.write(brain=self.brain, stamp="2026-10-01")
        writes.create(
            brain=self.brain,
            kind=schema.kind_named("bug"),
            title="Search flashes",
            description="results flash on first paint",
            values={"severity": "nit", "area": "search"},
            stamp="2026-10-02",
        )

        self.assertEqual(
            sorted(path.relative_to(self.root) for path in index.stale(self.brain)),
            [Path("INDEX.md"), Path("bugs/INDEX.md")],
        )


class TheKitItself(unittest.TestCase):
    def test_the_shipped_indexes_are_what_the_tool_would_write(self) -> None:
        brain = store.Brain(root=Path(__file__).resolve().parents[4])

        self.assertEqual(index.stale(brain), [])


if __name__ == "__main__":
    unittest.main()
