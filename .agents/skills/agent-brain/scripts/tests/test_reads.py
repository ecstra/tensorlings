"""Every read answer is counted from the files, not recalled."""

import unittest

from brainlib import doctor, frontmatter, index, plans, reads, schema, store, upgrade
from tests import fixtures


class ReadCase(fixtures.BrainCase):
    def setUp(self) -> None:
        super().setUp()
        index.write(brain=self.brain, stamp="2026-09-16")
        doctor.run(brain=self.brain, fix=True)
        self.brain.refresh()


class Status(ReadCase):
    def test_the_version_comes_from_the_stamp(self) -> None:
        self.assertEqual(reads.version(self.brain), "1.0.0")

    def test_no_stamp_reads_as_unversioned(self) -> None:
        (self.root / "VERSION").unlink()

        self.assertEqual(reads.version(self.brain), "unversioned")

    def test_a_stamp_the_process_cannot_open_reads_as_unreadable(self) -> None:
        (self.root / "VERSION").chmod(0)

        self.assertEqual(reads.version(self.brain), "unreadable")
        self.assertIn("kit unreadable", reads.render_status(reads.status(self.brain)))

    def test_every_reader_of_the_stamp_reads_it_the_same_way(self) -> None:
        stamp = self.root / "VERSION"
        stamp.write_bytes(b"\xef\xbb\xbf1.2.0\r\n")

        self.assertEqual(reads.version(self.brain), "1.2.0")
        self.assertEqual(store.stamp_of(self.root), (1, 2, 0))
        self.assertEqual(upgrade.read_version(stamp), "1.2.0")

        stamp.write_bytes(b"1.2.0\xff\n")

        self.assertEqual(reads.version(self.brain), "unreadable")
        self.assertIsNone(store.stamp_of(self.root))
        with self.assertRaises(upgrade.UpgradeError):
            upgrade.read_version(stamp)

    def test_it_counts_every_tracker_by_status(self) -> None:
        data = reads.status(self.brain)

        self.assertEqual(data["counts"]["todo"], {"open": 1, "done": 1})
        self.assertEqual(data["counts"]["bug"], {"open": 1})
        self.assertEqual(data["counts"]["doc"], {"filed": 1})
        self.assertEqual(data["counts"]["handover"], {})
        self.assertEqual(data["open_reviews"], ["2026-09-feat-billing"])
        self.assertEqual(data["findings"], 0)

    def test_a_file_without_a_status_is_counted_by_the_same_word_everywhere(self) -> None:
        path = self.root / "decisions" / "DEC-001-postgres-over-sqlite.md"
        document = frontmatter.read(str(path))
        document.lines = [line for line in document.lines if not line.startswith("status:")]
        frontmatter.write(path=str(path), document=document)
        self.brain.refresh()

        self.assertEqual(reads.status(self.brain)["counts"]["decision"], {"unknown": 1})
        self.assertIn("the decision record: 1 unknown", index.build(self.brain)[self.root / "INDEX.md"])

    def test_it_renders_as_a_few_aligned_lines(self) -> None:
        text = reads.render_status(reads.status(self.brain))

        self.assertIn("kit 1.0.0    doctor clean\nplans      none\ntodos      1 open, 1 done", text)
        self.assertIn("reviewing  2026-09-feat-billing", text)

        plans.create(brain=self.brain, title="Phase 7", description="the rest", body="## Why\n\nWhy.\n\n## Done when\n\nDone.\n")
        text = reads.render_status(reads.status(self.brain))

        self.assertIn("plans      1 open\ntodos", text)

    def test_a_broken_file_shows_up_rather_than_crashing(self) -> None:
        fixtures.write(path=self.root / "bugs" / "loose.md", text="# no fence\n")
        self.brain.refresh()

        data = reads.status(self.brain)

        self.assertEqual(data["unreadable"], ["bugs/loose.md"])
        self.assertGreater(data["findings"], 0)


class Listing(ReadCase):
    def test_rows_narrow_by_field(self) -> None:
        found = reads.rows(
            brain=self.brain,
            kind=schema.kind_named("todo"),
            filters={"status": "open"},
        )

        self.assertEqual([row["id"] for row in found], ["TODO-001"])

    def test_rows_carry_the_kinds_columns(self) -> None:
        found = reads.rows(brain=self.brain, kind=schema.kind_named("bug"))

        self.assertEqual(found[0]["severity"], "bug")
        self.assertEqual(found[0]["area"], "auth")
        self.assertEqual(found[0]["path"], "bugs/BUG-002-login-loop.md")

    def test_rendering_lines_the_columns_up(self) -> None:
        kind = schema.kind_named("bug")
        text = reads.render_rows(kind=kind, found=reads.rows(brain=self.brain, kind=kind))

        self.assertEqual(text, "BUG-002  open    bug       auth  the sign in screen loops")

    def test_nothing_matching_says_so(self) -> None:
        kind = schema.kind_named("bug")
        text = reads.render_rows(
            kind=kind,
            found=reads.rows(brain=self.brain, kind=kind, filters={"status": "fixed"}),
        )

        self.assertEqual(text, "no bugs match")


class Showing(ReadCase):
    def test_show_hands_back_the_whole_file(self) -> None:
        text = reads.show(brain=self.brain, key="DEC-001")

        self.assertTrue(text.startswith("---\nname: DEC-001-postgres-over-sqlite"))
        self.assertIn("## Decision", text)

    def test_meta_only_is_the_frontmatter_lines(self) -> None:
        text = reads.show(brain=self.brain, key="DEC-001", meta_only=True)

        self.assertIn("status: standing", text)
        self.assertNotIn("## Decision", text)

    def test_an_unknown_key_is_none(self) -> None:
        self.assertIsNone(reads.show(brain=self.brain, key="DEC-404"))

    def test_path_of_resolves_an_id(self) -> None:
        self.assertEqual(
            reads.path_of(brain=self.brain, key="TODO-003"),
            self.root / "todos" / "TODO-003-dark-mode.md",
        )


class LinksOf(ReadCase):
    def test_it_lists_both_directions(self) -> None:
        path = self.root / "todos" / "TODO-001-export-to-csv.md"
        document = frontmatter.read(str(path))
        document.set_list(key="links", values=["BUG-002"])
        frontmatter.write(path=str(path), document=document)
        self.brain.refresh()

        data = reads.links_of(brain=self.brain, key="BUG-002")

        self.assertEqual(data["links"], [])
        self.assertEqual(data["backlinks"], ["todos/TODO-001-export-to-csv.md"])
        self.assertIn("linked from\n  todos/TODO-001-export-to-csv.md", reads.render_links(data))


if __name__ == "__main__":
    unittest.main()
