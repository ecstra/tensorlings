"""Links resolve by path, by id, or by name, and the tool can follow them back."""

import os
import unittest
from pathlib import Path
from unittest import mock

from brainlib import frontmatter, links, store
from tests import fixtures


def broken(
    entry: store.Entry,
) -> list[str]:
    """The entry's dangling body links, the way a test asks for them."""
    return links.broken_in_body(path=entry.path, document=entry.document)


def point(
    root: Path,
    relative: str,
    targets: list[str],
) -> None:
    """Set one file's frontmatter links, the way a test needs them."""
    path = root / relative
    document = frontmatter.read(str(path))
    document.set_list(key="links", values=targets)
    frontmatter.write(path=str(path), document=document)


class Resolving(fixtures.BrainCase):
    def test_a_path_resolves_to_its_file(self) -> None:
        found = links.resolve(brain=self.brain, reference="bugs/BUG-002-login-loop.md")

        self.assertEqual(found, self.root / "bugs" / "BUG-002-login-loop.md")

    def test_an_id_resolves_to_its_file(self) -> None:
        found = links.resolve(brain=self.brain, reference="BUG-002")

        self.assertEqual(found, self.root / "bugs" / "BUG-002-login-loop.md")

    def test_a_kit_file_resolves_by_its_name(self) -> None:
        found = links.resolve(brain=self.brain, reference="todos-index")

        self.assertEqual(found, self.root / "todos" / "INDEX.md")

    def test_nothing_resolves_to_none(self) -> None:
        self.assertIsNone(links.resolve(brain=self.brain, reference="BUG-999"))

    def test_canonical_turns_an_id_into_a_path(self) -> None:
        self.assertEqual(
            links.canonical(brain=self.brain, reference="TODO-003"),
            "todos/TODO-003-dark-mode.md",
        )

    def test_an_absolute_path_outside_the_brain_resolves_to_nothing(self) -> None:
        outside = self.root.parent / ".claude" / "output-styles" / "plain-and-short.md"

        self.assertTrue(outside.is_file())
        self.assertIsNone(links.resolve(brain=self.brain, reference=str(outside)))
        self.assertIsNone(links.canonical(brain=self.brain, reference=str(outside)))

    def test_a_path_that_climbs_out_of_the_brain_resolves_to_nothing(self) -> None:
        self.assertIsNone(links.canonical(brain=self.brain, reference="../.claude/output-styles/plain-and-short.md"))

    def test_a_symlinked_file_inside_the_brain_is_inside_it(self) -> None:
        shared = self.root.parent / "shared" / "runbook.md"
        fixtures.write(path=shared, text="---\nname: runbook\n---\n\n# Runbook\n")
        (self.root / "docs" / "runbook.md").symlink_to(shared)

        self.assertEqual(links.canonical(brain=self.brain, reference="docs/runbook.md"), "docs/runbook.md")

    def test_a_symlink_loop_resolves_to_nothing_without_raising(self) -> None:
        (self.root / "docs" / "loop-a").symlink_to("loop-b")
        (self.root / "docs" / "loop-b").symlink_to("loop-a")

        self.assertIsNone(links.canonical(brain=self.brain, reference="docs/loop-a/note.md"))

    def test_a_link_into_a_dotted_path_reaches_nothing(self) -> None:
        real = self.root / "memory" / "stripe-retries-twice.md"
        (self.root / "memory" / ".putback.xyz").mkdir()
        (self.root / "memory" / ".putback.xyz" / "stripe-retries-twice.md").symlink_to(real)
        fixtures.write(path=self.root / "docs" / ".draft.md", text=real.read_text(encoding="utf-8"))

        for reference in ("memory/.putback.xyz/stripe-retries-twice.md", "docs/.draft.md"):
            with self.subTest(reference=reference):
                self.assertIsNone(links.canonical(brain=self.brain, reference=reference))

    def test_an_absolute_path_inside_the_brain_canonicalizes_to_a_relative_one(self) -> None:
        absolute = str(self.root / "bugs" / "BUG-002-login-loop.md")

        self.assertEqual(links.canonical(brain=self.brain, reference=absolute), "bugs/BUG-002-login-loop.md")


class FollowingBack(fixtures.BrainCase):
    def setUp(self) -> None:
        super().setUp()
        point(root=self.root, relative="todos/TODO-001-export-to-csv.md", targets=["BUG-002", "DEC-001"])
        point(root=self.root, relative="memory/stripe-retries-twice.md", targets=["bugs/BUG-002-login-loop.md"])

    def test_backlinks_find_everything_that_points_here(self) -> None:
        bug = self.brain.find("BUG-002")

        pointing = links.backlinks(brain=self.brain, target=bug)

        self.assertEqual(
            sorted(entry.name for entry in pointing),
            ["TODO-001-export-to-csv", "stripe-retries-twice"],
        )

    def test_a_link_to_nothing_resolves_to_nothing(self) -> None:
        self.assertIsNone(links.resolve(brain=self.brain, reference="BUG-999"))
        self.assertIsNone(links.resolve(brain=self.brain, reference="docs/nowhere/at-all.md"))
        self.assertIsNotNone(links.resolve(brain=self.brain, reference="BUG-002"))

    def test_a_body_link_to_nothing_is_reported_and_a_url_is_not(self) -> None:
        path = self.root / "todos" / "TODO-001-export-to-csv.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\nSee [gone](../nowhere.md) and [site](https://example.com).\n",
            encoding="utf-8",
        )

        entry = self.brain.find("TODO-001")

        self.assertEqual(broken(entry), ["../nowhere.md"])

    def test_a_body_link_into_a_folder_the_process_cannot_search_is_broken_not_fatal(self) -> None:
        outside = self.root.parent / "outside"
        fixtures.write(path=outside / "note.md", text="# Note\n")
        path = self.root / "todos" / "TODO-001-export-to-csv.md"
        path.write_text(
            path.read_text(encoding="utf-8") + "\nSee [note](../../outside/note.md).\n",
            encoding="utf-8",
        )
        outside.chmod(0)

        try:
            self.assertEqual(broken(self.brain.find("TODO-001")), ["../../outside/note.md"])
        finally:
            outside.chmod(0o700)

    def test_an_absolute_body_link_is_broken_even_when_the_file_exists(self) -> None:
        path = self.root / "todos" / "TODO-001-export-to-csv.md"
        path.write_text(
            path.read_text(encoding="utf-8") + f"\nSee [the bug]({self.root / 'bugs' / 'BUG-002-login-loop.md'}).\n",
            encoding="utf-8",
        )

        entry = self.brain.find("TODO-001")

        self.assertEqual(broken(entry), [str(self.root / "bugs" / "BUG-002-login-loop.md")])


class Rewriting(fixtures.BrainCase):
    def setUp(self) -> None:
        super().setUp()
        point(root=self.root, relative="todos/TODO-001-export-to-csv.md", targets=["BUG-002"])
        point(root=self.root, relative="todos/TODO-003-dark-mode.md", targets=["DEC-001"])

    def test_every_link_to_the_old_path_now_names_the_new(self) -> None:
        changed = links.rewrite_references(
            brain=self.brain,
            old="bugs/BUG-002-login-loop.md",
            new="bugs/BUG-002-login-loops-on-expiry.md",
        )

        self.assertEqual([path.name for path in changed], ["TODO-001-export-to-csv.md"])
        self.assertEqual(
            self.brain.find("TODO-001").document.get_list("links"),
            ["bugs/BUG-002-login-loops-on-expiry.md"],
        )

    def test_a_file_that_cannot_be_written_puts_the_others_back(self) -> None:
        point(root=self.root, relative="todos/TODO-003-dark-mode.md", targets=["BUG-002"])
        first = self.root / "todos" / "TODO-001-export-to-csv.md"
        second = self.root / "todos" / "TODO-003-dark-mode.md"
        before = first.read_text(encoding="utf-8")
        second.chmod(0o444)

        try:
            with self.assertRaises(OSError):
                links.rewrite_references(
                    brain=self.brain,
                    old="bugs/BUG-002-login-loop.md",
                    new="bugs/BUG-002-renamed.md",
                    stamp="2026-10-01",
                )
        finally:
            second.chmod(0o644)

        self.assertEqual(first.read_text(encoding="utf-8"), before)
        self.assertEqual(self.brain.find("TODO-001").document.get_list("links"), ["BUG-002"])

    def test_a_file_that_cannot_be_read_stops_the_move_before_any_write(self) -> None:
        point(root=self.root, relative="todos/TODO-003-dark-mode.md", targets=["BUG-002"])
        self.brain.entries()
        first = self.root / "todos" / "TODO-001-export-to-csv.md"
        second = self.root / "todos" / "TODO-003-dark-mode.md"
        before = first.read_text(encoding="utf-8")
        second.chmod(0)

        try:
            with self.assertRaises(frontmatter.FrontmatterError):
                links.rewrite_references(
                    brain=self.brain,
                    old="bugs/BUG-002-login-loop.md",
                    new="bugs/BUG-002-renamed.md",
                )
        finally:
            second.chmod(0o644)

        self.assertEqual(first.read_text(encoding="utf-8"), before)

    def test_a_symlinked_file_put_back_is_the_symlink_it_was(self) -> None:
        point(root=self.root, relative="todos/TODO-003-dark-mode.md", targets=["BUG-002"])
        first = self.root / "todos" / "TODO-001-export-to-csv.md"
        shared = self.root.parent / "shared" / "TODO-001-export-to-csv.md"
        shared.parent.mkdir()
        first.rename(shared)
        first.symlink_to(shared)
        theirs = shared.read_text(encoding="utf-8")
        second = self.root / "todos" / "TODO-003-dark-mode.md"
        second.chmod(0o444)

        try:
            with self.assertRaises(OSError):
                links.rewrite_references(
                    brain=self.brain,
                    old="bugs/BUG-002-login-loop.md",
                    new="bugs/BUG-002-renamed.md",
                )
        finally:
            second.chmod(0o644)

        self.assertTrue(first.is_symlink())
        self.assertEqual(Path(os.readlink(str(first))), shared)
        self.assertEqual(shared.read_text(encoding="utf-8"), theirs)

    def test_a_symlink_goes_back_twice_over_with_a_stale_scratch_beside_it(self) -> None:
        first = self.root / "todos" / "TODO-001-export-to-csv.md"
        shared = self.root.parent / "shared" / "TODO-001-export-to-csv.md"
        shared.parent.mkdir()
        first.rename(shared)
        first.symlink_to(shared)
        (self.root / "todos" / ".putback.stale").mkdir()
        entry = self.brain.find("TODO-001")
        item = links.Planned(path=entry.path, document=entry.document, links=[], body=None, text="", link=str(shared))

        for _ in range(2):
            links.put_back(item)

        self.assertTrue(first.is_symlink())
        self.assertEqual(sorted(p.name for p in (self.root / "todos").iterdir() if p.name.startswith(".")), [".putback.stale"])

    def test_a_file_that_will_not_go_back_is_named_and_the_others_still_go_back(self) -> None:
        point(root=self.root, relative="todos/TODO-003-dark-mode.md", targets=["BUG-002"])
        point(root=self.root, relative="memory/stripe-retries-twice.md", targets=["BUG-002"])
        first = self.root / "memory" / "stripe-retries-twice.md"
        before = first.read_text(encoding="utf-8")
        real_put_back = links.put_back

        def put_back_but_not_the_first_todo(item: links.Planned) -> None:
            if item.path.name == "TODO-001-export-to-csv.md":
                raise OSError(5, "Input/output error", str(item.path))
            real_put_back(item)

        (self.root / "todos" / "TODO-003-dark-mode.md").chmod(0o444)

        try:
            with mock.patch("brainlib.links.put_back", side_effect=put_back_but_not_the_first_todo):
                with self.assertRaises(OSError) as caught:
                    links.rewrite_references(
                        brain=self.brain,
                        old="bugs/BUG-002-login-loop.md",
                        new="bugs/BUG-002-renamed.md",
                    )
        finally:
            (self.root / "todos" / "TODO-003-dark-mode.md").chmod(0o644)

        self.assertIn("Permission denied: .agents/todos/TODO-003-dark-mode.md, and .agents/todos/TODO-001-export-to-csv.md could not be put back", str(caught.exception))
        self.assertEqual(first.read_text(encoding="utf-8"), before)

    def test_a_name_form_and_a_path_form_of_one_file_move_as_one_link(self) -> None:
        point(root=self.root, relative="todos/TODO-001-export-to-csv.md", targets=["BUG-002", "bugs/BUG-002-login-loop.md", "login"])
        self.brain.refresh()

        links.rewrite_references(
            brain=self.brain,
            old="bugs/BUG-002-login-loop.md",
            new="bugs/BUG-002-renamed.md",
        )
        self.brain.refresh()

        self.assertEqual(self.brain.find("TODO-001").document.get_list("links"), ["bugs/BUG-002-renamed.md", "login"])

    def test_a_file_whose_link_moved_is_stamped(self) -> None:
        links.rewrite_references(
            brain=self.brain,
            old="bugs/BUG-002-login-loop.md",
            new="bugs/BUG-002-renamed.md",
            stamp="2026-10-01",
        )

        self.assertEqual(self.brain.find("TODO-001").field("updated"), "2026-10-01")

    def test_a_markdown_link_in_the_prose_follows_too_and_keeps_its_anchor(self) -> None:
        memory = self.root / "memory" / "stripe-retries-twice.md"
        memory.write_text(
            memory.read_text(encoding="utf-8")
            + "\nSee [the bug](../bugs/BUG-002-login-loop.md#evidence) and [the doc](../docs/systems/login.md).\n",
            encoding="utf-8",
        )

        changed = links.rewrite_references(
            brain=self.brain,
            old="bugs/BUG-002-login-loop.md",
            new="bugs/BUG-002-login-loops-on-expiry.md",
            stamp="2026-10-01",
        )

        self.assertIn(memory, changed)
        after = memory.read_text(encoding="utf-8")
        self.assertIn("[the bug](../bugs/BUG-002-login-loops-on-expiry.md#evidence)", after)
        self.assertIn("[the doc](../docs/systems/login.md)", after)
        self.assertIn("updated: 2026-10-01", after)

    def test_a_reference_style_link_follows_too_and_a_dangling_one_is_reported(self) -> None:
        memory = self.root / "memory" / "stripe-retries-twice.md"
        memory.write_text(
            memory.read_text(encoding="utf-8") + "\nSee [the bug][bug] and [gone][g].\n\n[bug]: ../bugs/BUG-002-login-loop.md#evidence\n[g]: ../nowhere.md\n",
            encoding="utf-8",
        )

        links.rewrite_references(
            brain=self.brain,
            old="bugs/BUG-002-login-loop.md",
            new="bugs/BUG-002-renamed.md",
        )

        after = memory.read_text(encoding="utf-8")
        self.assertIn("[bug]: ../bugs/BUG-002-renamed.md#evidence", after)
        self.assertIn("../nowhere.md", broken(self.brain.find("stripe-retries-twice")))

    def test_a_body_link_in_one_of_the_kits_own_files_follows_too(self) -> None:
        readme = self.root / "README.md"
        fixtures.write(
            path=readme,
            text="---\nname: agent-readme\nkind: doc\ndescription: the map\nupdated: 2026-09-16\nlinks: []\n---\n\nStart with [the bug](bugs/BUG-002-login-loop.md).\n",
        )

        changed = links.rewrite_references(
            brain=self.brain,
            old="bugs/BUG-002-login-loop.md",
            new="bugs/BUG-002-renamed.md",
            stamp="2026-10-01",
        )

        self.assertIn(readme, changed)
        after = readme.read_text(encoding="utf-8")
        self.assertIn("[the bug](bugs/BUG-002-renamed.md)", after)
        self.assertIn("updated: 2026-10-01", after)

    def test_a_link_whose_text_holds_code_is_still_a_link_to_the_move_and_to_doctor(self) -> None:
        memory = self.root / "memory" / "stripe-retries-twice.md"
        memory.write_text(
            memory.read_text(encoding="utf-8") + "\nSee [the `login` bug](../bugs/BUG-002-login-loop.md) and `x`[a]: ../bugs/BUG-002-login-loop.md\n",
            encoding="utf-8",
        )

        links.rewrite_references(
            brain=self.brain,
            old="bugs/BUG-002-login-loop.md",
            new="bugs/BUG-002-renamed.md",
        )

        after = memory.read_text(encoding="utf-8")
        self.assertIn("[the `login` bug](../bugs/BUG-002-renamed.md)", after)
        self.assertIn("`x`[a]: ../bugs/BUG-002-login-loop.md", after)
        self.assertIn("../bugs/BUG-002-renamed.md", broken(self.brain.find("stripe-retries-twice")))

    def test_a_target_in_angle_brackets_resolves_and_moves_with_its_brackets(self) -> None:
        memory = self.root / "memory" / "stripe-retries-twice.md"
        fixtures.write(path=self.root / "memory" / "a note.md", text="---\nname: a-note\n---\n\n# A note\n")
        memory.write_text(
            memory.read_text(encoding="utf-8") + "\nSee [it](<../bugs/BUG-002-login-loop.md>), [spaced](<a note.md>), [empty](<>).\n",
            encoding="utf-8",
        )

        self.assertEqual(broken(self.brain.find("stripe-retries-twice")), [""])

        links.rewrite_references(
            brain=self.brain,
            old="bugs/BUG-002-login-loop.md",
            new="bugs/BUG-002-renamed.md",
        )

        self.assertIn("[it](<../bugs/BUG-002-renamed.md>)", memory.read_text(encoding="utf-8"))

    def test_a_link_quoted_in_code_is_neither_moved_nor_reported(self) -> None:
        memory = self.root / "memory" / "stripe-retries-twice.md"
        quoted = "\n```md\n[old](../bugs/BUG-002-login-loop.md)\n```\n\nand `[x](../bugs/BUG-002-login-loop.md)` inline.\n"
        memory.write_text(memory.read_text(encoding="utf-8") + quoted, encoding="utf-8")

        changed = links.rewrite_references(
            brain=self.brain,
            old="bugs/BUG-002-login-loop.md",
            new="bugs/BUG-002-renamed.md",
        )

        self.assertNotIn(memory, changed)
        self.assertIn(quoted, memory.read_text(encoding="utf-8"))
        self.assertEqual(broken(self.brain.find("stripe-retries-twice")), [])

    def test_a_link_elsewhere_is_left_alone(self) -> None:
        links.rewrite_references(
            brain=self.brain,
            old="bugs/BUG-002-login-loop.md",
            new="bugs/BUG-002-renamed.md",
        )

        self.assertEqual(self.brain.find("TODO-003").document.get_list("links"), ["DEC-001"])


if __name__ == "__main__":
    unittest.main()
