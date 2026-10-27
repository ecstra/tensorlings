"""Finding the brain, listing what is in it, and picking the next id."""

import os
import tempfile
import unittest
from pathlib import Path
from typing import Callable, Iterator
from unittest import mock

from brainlib import links, schema, store
from tests import fixtures


class BrainOnDisk(fixtures.BrainCase):
    def test_it_is_found_from_a_folder_inside_the_project(self) -> None:
        found = store.find_brain(self.root.parent)

        self.assertEqual(found.root, self.root)

    def test_it_is_found_from_inside_itself(self) -> None:
        found = store.find_brain(self.root / "todos")

        self.assertEqual(found.root, self.root)

    def test_a_project_without_one_says_so(self) -> None:
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(store.StoreError) as caught:
                store.find_brain(Path(empty))

        self.assertIn(".agents", str(caught.exception))


class ListingEntries(fixtures.BrainCase):
    def test_an_index_is_not_an_entry(self) -> None:
        names = [entry.path.name for entry in self.brain.entries()]

        self.assertNotIn("INDEX.md", names)

    def test_nothing_inside_a_dot_folder_or_named_with_a_dot_is_an_entry(self) -> None:
        real = self.root / "memory" / "stripe-retries-twice.md"
        (self.root / "memory" / ".putback.xyz").mkdir()
        (self.root / "memory" / ".putback.xyz" / "stripe-retries-twice.md").symlink_to(real)
        fixtures.write(path=self.root / "memory" / ".draft.md", text=real.read_text(encoding="utf-8"))

        memory = self.brain.entries_of(schema.kind_named("memory"))

        self.assertEqual([entry.relative for entry in memory], ["memory/stripe-retries-twice.md"])

    def test_a_readme_inside_docs_is_a_doc_like_any_other(self) -> None:
        fixtures.write(
            path=self.root / "docs" / "README.md",
            text="---\nname: README\nkind: doc\ndescription: a map\nupdated: 2026-09-16\nlinks: []\n---\n\n# Map\n",
        )

        docs = self.brain.entries_of(schema.kind_named("doc"))

        self.assertEqual([entry.relative for entry in docs], ["docs/README.md", "docs/systems/login.md"])

    def test_known_paths_are_the_entries_and_what_could_not_be_read(self) -> None:
        fixtures.write(path=self.root / "todos" / "loose.md", text="# No fence here\n")

        known = self.brain.known()

        self.assertIn(self.root / "todos" / "loose.md", known)
        self.assertIn(self.root / "bugs" / "BUG-002-login-loop.md", known)
        self.assertNotIn(self.root / "todos" / "INDEX.md", known)

    def test_the_kit_files_are_the_ones_that_exist(self) -> None:
        fixtures.write(path=self.root / "spec" / "todos.md", text="---\nname: spec-todos\n---\n\n# Todos\n")
        fixtures.write(path=self.root / "spec" / "INDEX.md", text="---\nname: spec-index\n---\n\n# Not a kit file\n")
        fixtures.write(path=self.root / "spec" / ".draft.md", text="---\nname: ghost-spec\n---\n\n# Not one either\n")

        listed = [str(path.relative_to(self.root)) for path in self.brain.kit_files()]

        self.assertEqual(listed, ["INDEX.md", "spec/todos.md", "docs/INDEX.md", "memory/INDEX.md", "bugs/INDEX.md", "todos/INDEX.md", "decisions/INDEX.md", "plans/INDEX.md", "reviews/INDEX.md"])

    def test_todos_come_back_in_id_order(self) -> None:
        todos = self.brain.entries_of(schema.kind_named("todo"))

        self.assertEqual(
            [entry.identifier for entry in todos],
            ["TODO-001", "TODO-003"],
        )

    def test_a_doc_in_a_category_folder_is_found(self) -> None:
        docs = self.brain.entries_of(schema.kind_named("doc"))

        self.assertEqual([entry.relative for entry in docs], ["docs/systems/login.md"])

    def test_a_review_is_its_ledger(self) -> None:
        reviews = self.brain.entries_of(schema.kind_named("review"))

        self.assertEqual(
            [entry.relative for entry in reviews],
            ["reviews/2026-09-feat-billing/ledger.md"],
        )

    def test_a_ledger_any_deeper_is_not_a_review(self) -> None:
        fixtures.write(
            path=self.root / "reviews" / "a" / "b" / "ledger.md",
            text=(self.root / "reviews" / "2026-09-feat-billing" / "ledger.md").read_text(encoding="utf-8"),
        )

        reviews = self.brain.entries_of(schema.kind_named("review"))

        self.assertEqual([entry.relative for entry in reviews], ["reviews/2026-09-feat-billing/ledger.md"])

    def test_every_kind_together_is_the_whole_brain(self) -> None:
        self.assertEqual(len(self.brain.entries()), 7)

    def test_one_unreadable_file_does_not_hide_the_rest(self) -> None:
        fixtures.write(path=self.root / "todos" / "loose.md", text="# No fence here\n")

        todos = self.brain.entries_of(schema.kind_named("todo"))

        self.assertEqual(len(todos), 2)
        self.assertEqual(
            [path.name for path in self.brain.unreadable],
            ["loose.md"],
        )

    def test_a_file_that_is_not_utf8_is_noted_the_same_way(self) -> None:
        (self.root / "todos" / "TODO-005-latin.md").write_bytes(b"---\nname: caf\xe9\n---\n")

        todos = self.brain.entries_of(schema.kind_named("todo"))

        self.assertEqual(len(todos), 2)
        self.assertEqual(
            list(self.brain.unreadable.values()),
            [".agents/todos/TODO-005-latin.md: the file is not UTF-8 text"],
        )

    def test_a_dangling_symlink_is_noted_too(self) -> None:
        (self.root / "todos" / "TODO-006-gone.md").symlink_to("nowhere.md")

        todos = self.brain.entries_of(schema.kind_named("todo"))

        self.assertEqual(len(todos), 2)
        self.assertIn("the file cannot be read: ", list(self.brain.unreadable.values())[0])

    @unittest.skipUnless(os.name == "posix" and os.geteuid() != 0, "needs a folder the process cannot list")
    def test_a_folder_that_cannot_be_listed_is_noted_beside_the_files_that_cannot_be_read(self) -> None:
        folder = self.root / "reviews" / "2026-09-feat-billing"
        mode = folder.stat().st_mode

        os.chmod(folder, 0)
        try:
            reviews = self.brain.entries_of(schema.kind_named("review"))
        finally:
            os.chmod(folder, mode)

        self.assertEqual(reviews, [])
        self.assertEqual(self.brain.unreadable, {folder: "cannot be listed"})
        self.assertIn(folder, self.brain.known())

    @unittest.skipUnless(os.name == "posix" and os.geteuid() != 0, "needs a folder the process cannot list")
    def test_a_dotted_folder_is_never_entered_so_one_that_cannot_be_listed_is_nothing_to_note(self) -> None:
        cache = self.root / "todos" / ".cache"
        cache.mkdir()
        fixtures.write(path=cache / "TODO-009-stale.md", text="# never listed\n")

        os.chmod(cache, 0)
        try:
            todos = self.brain.entries_of(schema.kind_named("todo"))
        finally:
            os.chmod(cache, 0o755)

        self.assertEqual(len(todos), 2)
        self.assertEqual(self.brain.unreadable, {})

    def test_a_folder_gone_between_the_listing_and_the_descent_is_nothing_to_note(self) -> None:
        real_walk = os.walk

        def racing_walk(top: str, onerror: "Callable[[OSError], None]") -> "Iterator[tuple[str, list[str], list[str]]]":
            onerror(FileNotFoundError(2, "No such file or directory", str(Path(top) / ".putback.k3j2_x")))
            return real_walk(top, onerror=onerror)

        with mock.patch.object(os, "walk", racing_walk):
            todos = self.brain.entries_of(schema.kind_named("todo"))

        self.assertEqual(len(todos), 2)
        self.assertEqual(self.brain.unreadable, {})

    def test_the_suffix_follows_the_platforms_case_rule_like_the_doctors_glob(self) -> None:
        source = self.root / "todos" / "TODO-001-export-to-csv.md"
        fixtures.write(path=self.root / "todos" / "TODO-004-loud.MD", text=source.read_text(encoding="utf-8").replace("TODO-001-export-to-csv", "TODO-004-loud"))

        with mock.patch.object(os.path, "normcase", str.lower):
            self.brain.refresh()
            loud = [entry.name for entry in self.brain.entries_of(schema.kind_named("todo"))]

        self.brain.refresh()
        quiet = [entry.name for entry in self.brain.entries_of(schema.kind_named("todo"))]

        self.assertIn("TODO-004-loud", loud)
        self.assertEqual("TODO-004-loud" in quiet, os.path.normcase("A") == "a")


class NextId(fixtures.BrainCase):
    def test_a_gap_in_the_middle_stays_a_gap(self) -> None:
        self.assertEqual(self.brain.next_id(schema.kind_named("todo")), "TODO-004")

    def test_an_empty_folder_starts_at_one(self) -> None:
        for path in (self.root / "bugs").glob("BUG-*.md"):
            path.unlink()

        self.assertEqual(self.brain.next_id(schema.kind_named("bug")), "BUG-001")

    def test_it_follows_the_highest_id_not_the_count(self) -> None:
        fixtures.write(
            path=self.root / "bugs" / "BUG-050-something.md",
            text=(self.root / "bugs" / "BUG-002-login-loop.md").read_text(
                encoding="utf-8"
            ),
        )

        self.assertEqual(self.brain.next_id(schema.kind_named("bug")), "BUG-051")

    def test_an_id_on_a_file_the_store_cannot_read_is_still_taken(self) -> None:
        (self.root / "bugs" / "BUG-007-half-written.md").write_bytes(b"no fence, caf\xe9\n")

        self.assertEqual(self.brain.next_id(schema.kind_named("bug")), "BUG-008")

    def test_a_kind_without_ids_refuses_to_invent_one(self) -> None:
        with self.assertRaises(store.StoreError):
            self.brain.next_id(schema.kind_named("doc"))


class FindingOne(fixtures.BrainCase):
    def test_a_name_that_is_also_the_stem_is_one_file_not_two(self) -> None:
        found = self.brain.find("login")

        self.assertEqual(found.relative, "docs/systems/login.md")

    def test_each_caller_gets_its_own_copy_of_the_kit_documents(self) -> None:
        path, document = self.brain.kit_documents()[0]
        document.set(key="description", value="changed in memory")
        document.body = "gone"

        again = dict(self.brain.kit_documents())[path]

        self.assertNotEqual(again.get("description"), "changed in memory")
        self.assertNotEqual(again.body, "gone")

    def test_the_lookup_is_rebuilt_after_a_refresh(self) -> None:
        self.brain.find("BUG-002")
        (self.root / "bugs" / "BUG-002-login-loop.md").unlink()

        self.assertIsNotNone(self.brain.find("BUG-002"))
        self.brain.refresh()
        self.assertIsNone(self.brain.find("BUG-002"))

    def test_an_id_finds_its_file(self) -> None:
        found = self.brain.find("BUG-002")

        self.assertIsNotNone(found)
        self.assertEqual(found.relative, "bugs/BUG-002-login-loop.md")

    def test_a_name_finds_its_file(self) -> None:
        found = self.brain.find("stripe-retries-twice")

        self.assertEqual(found.relative, "memory/stripe-retries-twice.md")

    def test_a_path_inside_the_brain_finds_its_file(self) -> None:
        found = self.brain.find("docs/systems/login.md")

        self.assertEqual(found.name, "login")

    def test_a_path_from_the_repo_root_finds_its_file(self) -> None:
        found = self.brain.find(".agents/reviews/2026-09-feat-billing/ledger.md")

        self.assertEqual(found.relative, "reviews/2026-09-feat-billing/ledger.md")

    def test_a_path_from_a_git_top_above_the_project_finds_its_file(self) -> None:
        project: Path = self.root.parent
        for key in (
            f"{project.name}/.agents/reviews/2026-09-feat-billing/ledger.md",
            f"{project.parent.name}/{project.name}/.agents/reviews/2026-09-feat-billing/ledger.md",
        ):
            with self.subTest(key=key):
                found = self.brain.find(key)

                self.assertIsNotNone(found)
                self.assertEqual(found.relative, "reviews/2026-09-feat-billing/ledger.md")

    def test_an_absolute_path_finds_its_file(self) -> None:
        found = self.brain.find(str(self.root / "reviews" / "2026-09-feat-billing" / "ledger.md"))

        self.assertIsNotNone(found)
        self.assertEqual(found.relative, "reviews/2026-09-feat-billing/ledger.md")

    def test_a_path_spelled_with_backslashes_finds_its_file(self) -> None:
        for key in (
            "reviews\\2026-09-feat-billing\\ledger.md",
            ".agents\\reviews\\2026-09-feat-billing\\ledger.md",
        ):
            with self.subTest(key=key):
                found = self.brain.find(key)

                self.assertIsNotNone(found)
                self.assertEqual(found.relative, "reviews/2026-09-feat-billing/ledger.md")

    def test_a_share_path_the_way_windows_writes_a_checkout_inside_wsl_finds_its_file_from_inside_that_distro(self) -> None:
        project_folders: str = "\\".join(self.root.parent.parts[1:])
        with mock.patch.dict(
            os.environ,
            {"WSL_DISTRO_NAME": "ubuntu"},
        ):
            for share in ("\\\\wsl.localhost\\Ubuntu", "\\\\WSL$\\UBUNTU"):
                with self.subTest(share=share):
                    found = self.brain.find(f"{share}\\{project_folders}\\.agents\\reviews\\2026-09-feat-billing\\ledger.md")

                    self.assertIsNotNone(found)
                    self.assertEqual(found.relative, "reviews/2026-09-feat-billing/ledger.md")

    def test_a_wsl_share_of_this_project_answers_nothing_outside_that_distro(self) -> None:
        key: str = "\\\\wsl.localhost\\Ubuntu\\" + "\\".join(self.root.parent.parts[1:]) + "\\.agents\\reviews\\2026-09-feat-billing\\ledger.md"

        with mock.patch.dict(os.environ):
            os.environ.pop(
                "WSL_DISTRO_NAME",
                None,
            )
            found_with_no_distro = self.brain.find(key)

            os.environ["WSL_DISTRO_NAME"] = "Debian"
            found_from_another_distro = self.brain.find(key)

        self.assertIsNone(found_with_no_distro)
        self.assertIsNone(found_from_another_distro)

    def test_a_share_path_into_another_distro_or_onto_another_host_answers_nothing(self) -> None:
        project: Path = self.root.parent
        with mock.patch.dict(
            os.environ,
            {"WSL_DISTRO_NAME": "Ubuntu"},
        ):
            for share in (
                "\\\\wsl.localhost\\Debian\\" + "\\".join(project.parts[1:]),
                f"\\\\fileserver\\projects\\{project.name}",
            ):
                with self.subTest(share=share):
                    self.assertIsNone(self.brain.find(f"{share}\\.agents\\reviews\\2026-09-feat-billing\\ledger.md"))

    def test_a_drive_letter_path_answers_nothing_outside_wsl(self) -> None:
        key: str = "C:\\" + "\\".join(self.root.parent.parts[1:]) + "\\.agents\\reviews\\2026-09-feat-billing\\ledger.md"

        with mock.patch.dict(os.environ):
            os.environ.pop(
                "WSL_DISTRO_NAME",
                None,
            )
            found = self.brain.find(key)

        self.assertIsNone(found)

    def test_a_drive_letter_path_inside_wsl_answers_nothing_unless_its_folder_under_mnt_is_the_project(self) -> None:
        project: Path = self.root.parent
        with mock.patch.dict(
            os.environ,
            {"WSL_DISTRO_NAME": "Ubuntu"},
        ):
            for drive_path in (
                "D:\\" + "\\".join(project.parts[2:]),
                "C:\\" + "\\".join(project.parts[1:]),
            ):
                with self.subTest(drive_path=drive_path):
                    self.assertIsNone(self.brain.find(f"{drive_path}\\.agents\\reviews\\2026-09-feat-billing\\ledger.md"))

    def test_a_drive_letter_path_inside_wsl_is_the_folder_under_mnt_named_by_its_letter(self) -> None:
        # the answer needs nothing at this root on disk, so the brain can sit where a drive's folder under /mnt would be
        mounted = store.Brain(root=Path("/mnt/c/work/app/.agents"))

        with mock.patch.dict(
            os.environ,
            {"WSL_DISTRO_NAME": "Ubuntu"},
        ):
            answers: dict[str, bool] = {folder: mounted.is_this_project(folder) for folder in ("C:/work/app/", "c:/work/app/", "D:/work/app/")}

        self.assertEqual(
            answers,
            {"C:/work/app/": True, "c:/work/app/": True, "D:/work/app/": False},
        )

    def test_a_drive_letter_path_inside_wsl_in_any_case_finds_its_file_where_the_mount_ignores_case(self) -> None:
        drives: Path = Path(self.tmp.name).resolve()
        mounted = store.Brain(root=fixtures.build(drives / "c" / "Users" / "u" / "work" / "app"))
        if not (drives / "c" / "users").exists():
            self.skipTest("needs a file system that ignores case, and the temporary folder's tells case apart")

        with mock.patch.object(
            store,
            "WSL_DRIVES",
            drives,
        ), mock.patch.dict(
            os.environ,
            {"WSL_DISTRO_NAME": "Ubuntu"},
        ):
            for key in (
                "C:\\Users\\u\\work\\app\\.agents\\reviews\\2026-09-feat-billing\\ledger.md",
                "c:\\users\\u\\work\\app\\.agents\\reviews\\2026-09-feat-billing\\ledger.md",
            ):
                with self.subTest(key=key):
                    found = mounted.find(key)

                    self.assertIsNotNone(found)
                    self.assertEqual(found.relative, "reviews/2026-09-feat-billing/ledger.md")

    def test_a_path_from_the_home_folder_finds_its_file(self) -> None:
        home: Path = Path(self.tmp.name)
        app = store.Brain(root=fixtures.build(home / "work" / "app"))

        with mock.patch.dict(
            os.environ,
            {
                "HOME": str(home),
                "USERPROFILE": str(home),
            },
        ):
            found = app.find("~/work/app/.agents/reviews/2026-09-feat-billing/ledger.md")

        self.assertIsNotNone(found)
        self.assertEqual(found.relative, "reviews/2026-09-feat-billing/ledger.md")

    def test_another_projects_brain_by_an_absolute_path_is_not_found(self) -> None:
        other: Path = Path(self.tmp.name).resolve() / "other"
        (other / ".agents" / "reviews" / "2026-09-feat-billing").mkdir(parents=True)
        for key in (
            str(other / ".agents" / "reviews" / "2026-09-feat-billing" / "ledger.md"),
            "/some/other/project/.agents/reviews/2026-09-feat-billing/ledger.md",
            "/.agents/reviews/2026-09-feat-billing/ledger.md",
            "C:\\some\\other\\project\\.agents\\reviews\\2026-09-feat-billing\\ledger.md",
            "\\\\wsl.localhost\\Ubuntu\\home\\u\\other\\.agents\\reviews\\2026-09-feat-billing\\ledger.md",
        ):
            with self.subTest(key=key):
                self.assertIsNone(self.brain.find(key))

    def test_another_projects_brain_by_a_relative_path_is_not_found(self) -> None:
        for key in (
            "other/project/.agents/reviews/2026-09-feat-billing/ledger.md",
            f"other/{self.root.parent.name}x/.agents/reviews/2026-09-feat-billing/ledger.md",
            "../other/.agents/reviews/2026-09-feat-billing/ledger.md",
        ):
            with self.subTest(key=key):
                self.assertIsNone(self.brain.find(key))

    def test_a_link_into_another_projects_brain_reaches_nothing_here(self) -> None:
        reached = links.resolve(
            brain=self.brain,
            reference="../../../other/.agents/reviews/2026-09-feat-billing/ledger.md",
        )

        self.assertIsNone(reached)

    def test_something_that_is_not_filed_comes_back_empty(self) -> None:
        self.assertIsNone(self.brain.find("BUG-999"))

    def test_a_key_that_fits_two_files_is_refused_with_both_paths(self) -> None:
        fixtures.write(
            path=self.root / "reviews" / "2026-10-fix-login" / "ledger.md",
            text=(self.root / "reviews" / "2026-09-feat-billing" / "ledger.md").read_text(encoding="utf-8").replace("2026-09-feat-billing", "2026-10-fix-login"),
        )

        with self.assertRaises(store.Ambiguous) as caught:
            self.brain.find("ledger")

        self.assertEqual(
            str(caught.exception),
            "'ledger' names 2 files, use the path: reviews/2026-09-feat-billing/ledger.md, reviews/2026-10-fix-login/ledger.md",
        )
        self.assertEqual(self.brain.find("2026-10-fix-login").relative, "reviews/2026-10-fix-login/ledger.md")


class NamingTheNextFile(fixtures.BrainCase):
    def test_a_todo_gets_its_id_and_a_slug(self) -> None:
        path = self.brain.path_for(
            kind=schema.kind_named("todo"),
            title="Export to CSV",
        )

        self.assertEqual(
            path.relative_to(self.root),
            Path("todos/TODO-004-export-to-csv.md"),
        )

    def test_a_doc_lands_in_its_category(self) -> None:
        path = self.brain.path_for(
            kind=schema.kind_named("doc"),
            title="Billing",
            category="systems",
        )

        self.assertEqual(
            path.relative_to(self.root),
            Path("docs/systems/billing.md"),
        )

    def test_a_review_is_a_ledger_in_its_own_folder(self) -> None:
        path = self.brain.path_for(
            kind=schema.kind_named("review"),
            title="2026-10 fix login",
        )

        self.assertEqual(
            path.relative_to(self.root),
            Path("reviews/2026-10-fix-login/ledger.md"),
        )

    def test_a_category_is_slugs_and_nothing_else(self) -> None:
        for category in ("../../escaped", "/tmp/escaped", "Systems", "systems/../x", "a b"):
            with self.subTest(category=category):
                with self.assertRaises(store.StoreError) as caught:
                    self.brain.path_for(
                        kind=schema.kind_named("doc"),
                        title="Billing",
                        category=category,
                    )

                self.assertIn("is not a category", str(caught.exception))

    def test_a_category_may_be_more_than_one_folder_deep(self) -> None:
        path = self.brain.path_for(
            kind=schema.kind_named("doc"),
            title="Billing",
            category="systems/payments",
        )

        self.assertEqual(path.relative_to(self.root), Path("docs/systems/payments/billing.md"))

    def test_a_title_with_nothing_to_slug_is_refused(self) -> None:
        with self.assertRaises(store.StoreError):
            self.brain.path_for(
                kind=schema.kind_named("todo"),
                title="!!!",
            )


class Explaining(unittest.TestCase):
    def test_an_os_error_names_its_reason_and_the_path_from_the_brain_down(self) -> None:
        error = PermissionError(13, "Permission denied", "/somewhere/deep/.agents/bugs/BUG-001-x.md")

        self.assertEqual(store.explain(error), "Permission denied: .agents/bugs/BUG-001-x.md")

    def test_a_path_in_an_agent_folder_is_named_from_that_folder_down(self) -> None:
        self.assertEqual(store.near("/somewhere/deep/.claude/skills"), ".claude/skills")
        self.assertEqual(store.near("/somewhere/deep/.codex/plain-and-short.md"), ".codex/plain-and-short.md")
        self.assertEqual(store.near("/somewhere/deep/lib/main.dart"), "main.dart")

    def test_a_checkout_living_under_a_claude_folder_is_named_from_its_own_brain_down(self) -> None:
        self.assertEqual(store.near("/Users/x/.claude/projects/foo/.agents/todos/TODO-001-x.md"), ".agents/todos/TODO-001-x.md")
        self.assertEqual(store.near("/Users/x/.claude/projects/foo/.codex/plain-and-short.md"), ".codex/plain-and-short.md")

    def test_a_missing_key_reads_as_missing_not_as_a_quoted_repr(self) -> None:
        self.assertEqual(store.explain(KeyError("status")), "status is missing")
        self.assertEqual(store.explain(KeyError()), "a key is missing")

    def test_any_other_error_is_its_own_message_untouched(self) -> None:
        self.assertEqual(store.explain(schema.UnknownKind("unknown kind 'x'")), "unknown kind 'x'")
        self.assertEqual(store.explain(ValueError("it's fine")), "it's fine")


if __name__ == "__main__":
    unittest.main()
