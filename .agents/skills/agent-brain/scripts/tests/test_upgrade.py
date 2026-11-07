"""Port and upgrade bring a project to the kit and leave its own words alone."""

import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Optional
from unittest import mock

import hooks
from brainlib import autoack, doctor, schema, store, task, upgrade
from tests import fixtures

KIT = upgrade.Kit(root=Path(__file__).resolve().parents[5])
KIT_IGNORES = [line for line in KIT.text(Path(".gitignore")).splitlines() if line.strip() and not line.startswith("#")]


def commit_all(
    root: Path,
) -> None:
    """A git repository with everything committed, the state a real project is in before a port.

    The repository takes no hook template and signs nothing, so neither
    a hook nor a signing key of this machine's git setup reaches into a
    test.
    """
    subprocess.run(["git", "-C", str(root), "init", "-q", "--template="], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.email=tests@example.com",
            "-c",
            "user.name=tests",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "before",
        ],
        check=True,
    )


def git_clone(
    source: str,
    target: Path,
    bare: bool = False,
) -> None:
    """A git clone that takes no hook template, for the reason commit_all takes none."""
    subprocess.run(["git", "clone", "-q", "--template=", *(["--bare"] if bare else []), source, str(target)], check=True)


def git_init(
    target: Path,
) -> None:
    """An empty git repository that takes no hook template, for the reason commit_all takes none."""
    subprocess.run(["git", "init", "-q", "--template=", str(target)], check=True)


def status_of(
    root: Path,
) -> str:
    return subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout


class ProjectCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "shell"
        self.root.mkdir()
        self.project = upgrade.Project(root=self.root)


class TheKitItself(unittest.TestCase):
    def test_the_kit_this_tool_runs_from_is_a_kit(self) -> None:
        self.assertEqual(upgrade.default_kit().root, KIT.root)
        self.assertEqual(KIT.version(), (KIT.brain / "VERSION").read_text(encoding="utf-8").strip())

    def test_no_skill_or_definition_is_dated_after_the_release_the_kit_stamps(self) -> None:
        stamp = schema.as_tuple(version=KIT.version())

        for skill in schema.KIT_SKILLS:
            for since in (skill.since, *(definition.since for definition in skill.definitions)):
                with self.subTest(skill=skill.name, since=since):
                    self.assertTrue(
                        schema.carries(
                            since=since,
                            stamp=stamp,
                        ),
                    )

    def test_every_definition_the_kit_ships_is_one_the_schema_lists(self) -> None:
        listed_paths = {definition.path for skill in schema.KIT_SKILLS for definition in skill.definitions}
        definition_folders = {Path(path).parent for path in listed_paths}

        shipped_paths = {path.as_posix() for path in KIT.agent_files() if path.parent in definition_folders}

        self.assertEqual(shipped_paths, listed_paths)

    def test_the_kits_own_settings_carry_no_read_guard(self) -> None:
        self.assertNotIn("permissions", json.loads(KIT.text(upgrade.CLAUDE_SETTINGS)))

    def test_the_kits_agents_md_says_once_where_its_words_start(self) -> None:
        text = KIT.text(Path("AGENTS.md"))

        self.assertEqual(text.count(upgrade.KIT_MARK), 1)
        self.assertTrue(text.startswith(f"# AGENTS.md\n\n{upgrade.KIT_MARK}\n\n"))

    def test_the_kit_owns_its_rulebook_specs_skill_and_stamp_but_not_the_indexes(self) -> None:
        files = {str(path) for path in KIT.brain_files()}

        self.assertTrue({"SYSTEM.md", "README.md", "house-rules.md", "spec/conventions.md", "skills/agent-brain/SKILL.md", "VERSION"} <= files)
        self.assertIn("skills/agent-brain/scripts/brainlib/upgrade.py", files)
        self.assertNotIn("INDEX.md", files)
        self.assertFalse(any("__pycache__" in name for name in files))
        self.assertFalse(any("/tests/" in name for name in files))

    def test_the_agent_files_are_the_configs_and_styles_never_the_link(self) -> None:
        files = {str(path) for path in KIT.agent_files()}

        self.assertTrue({".claude/settings.json", ".codex/hooks.json", ".gemini/settings.json", ".claude/output-styles/plain-and-short.md"} <= files)
        self.assertFalse(any("skills" in name or "settings.local" in name for name in files))
        self.assertNotIn(".codex/plain-and-short.md", files)
        self.assertNotIn(".gemini/plain-and-short.md", files)


class Porting(ProjectCase):
    def test_a_bare_project_gets_the_kit_and_a_list_of_what_to_file(self) -> None:
        fixtures.write(path=self.root / "README.md", text="# App\n")
        fixtures.write(path=self.root / "todo.md", text="- ship it\n")
        fixtures.write(path=self.root / ".hidden.md", text="- never filed\n")
        fixtures.write(path=self.root / "docs" / "design.md", text="# Design\n")

        report = upgrade.port(kit=KIT, project=self.project)

        self.assertEqual([finding.line() for finding in report.findings], [])
        self.assertEqual(report.candidates, ["todo.md", "docs/"])
        self.assertEqual((self.root / ".agents" / "VERSION").read_text(encoding="utf-8").strip(), KIT.version())
        self.assertEqual((self.root / "AGENTS.md").read_text(encoding="utf-8"), KIT.text(Path("AGENTS.md")))
        self.assertTrue((self.root / ".agents" / "todos" / "INDEX.md").is_file())
        for name in schema.KIT_SKILL_NAMES:
            self.assertTrue((self.root / ".claude" / "skills" / name).is_symlink())
        self.assertIn("Doctor is clean.", report.render())
        self.assertIn("  todo.md", report.render())
        self.assertEqual(report.lines[0], f"porting {self.root.resolve()}")

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_what_is_left_to_file_comes_from_git_root_files_one_by_one_and_folders_counted(self) -> None:
        for name in ("README.md", "CHANGELOG.md", "todo.md", "Notes.md", ".hidden.md", "docs/a.md", "docs/b.md", "docs/.draft.md", "notes/x.md", "src/README.md", ".github/PR.md", "build/out.md"):
            fixtures.write(path=self.root / name, text=f"# {name}\n")
        fixtures.write(path=self.root / ".gitignore", text="build/\n")
        commit_all(self.root)
        fixtures.write(path=self.root / "wiki" / "y.md", text="# y\n")

        report = upgrade.port(kit=KIT, project=self.project)

        self.assertEqual(
            report.candidates,
            [
                "Notes.md",
                "todo.md",
                "docs/ (2 markdown files, may stay per the rules, linked into the brain)",
                "notes/ (1 markdown file)",
                "src/ (1 markdown file)",
                "wiki/ (1 markdown file)",
            ],
        )

    def test_a_front_door_with_words_of_its_own_and_no_section_zero_is_carried_whole(self) -> None:
        fixtures.write(path=self.root / "CLAUDE.md", text="# CLAUDE.md\n\nRun `make test` before a review.\n\n## Style\n\nTabs.\n")
        fixtures.write(path=self.root / "GEMINI.md", text=KIT.text(Path("GEMINI.md")))

        report = upgrade.port(kit=KIT, project=self.project)

        written = (self.root / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("carried CLAUDE.md whole into section 0, read it against the rulebook", report.render())
        self.assertIn("\n## 0. Project Rules\n\nRun `make test` before a review.\n\n## Style\n\nTabs.\n\n" + upgrade.KIT_MARK, written)
        self.assertEqual(upgrade.section_zero_in(written), "## 0. Project Rules\n\nRun `make test` before a review.\n\n## Style\n\nTabs.\n")

    def test_a_byte_order_mark_never_reaches_section_zero(self) -> None:
        (self.root / "CLAUDE.md").write_bytes("﻿# CLAUDE.md\n\nRun `make test`.\n".encode("utf-8"))

        upgrade.port(kit=KIT, project=self.project)

        self.assertEqual(
            upgrade.section_zero_in((self.root / "AGENTS.md").read_text(encoding="utf-8")),
            "## 0. Project Rules\n\nRun `make test`.\n",
        )

    def ignored(
        self,
        name: str,
    ) -> bool:
        return subprocess.run(["git", "-C", str(self.root), "check-ignore", "-q", "--no-index", name], check=False).returncode == 0

    @unittest.skipUnless(os.name == "posix" and shutil.which("sh"), "needs a shell to run the launcher")
    def test_the_launcher_finds_the_script_from_the_project_variable_or_by_walking_up_and_does_nothing_without_one(self) -> None:
        fixtures.write(path=self.root / "README.md", text="# App\n")
        (self.root / "src" / "deep").mkdir(parents=True)
        upgrade.port(kit=KIT, project=self.project)
        config: dict = json.loads((self.root / ".codex" / "hooks.json").read_text(encoding="utf-8"))
        command: str = config["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]

        escaped = self.root.parent / "p\\tq" / "shell"
        escaped.mkdir(parents=True)
        fixtures.write(path=escaped / "README.md", text="# App\n")
        upgrade.port(kit=KIT, project=upgrade.Project(root=escaped))
        elsewhere = self.root.parent / "elsewhere"
        elsewhere.mkdir()

        def run(
            folder: Path,
            event_folder: Optional[Path] = None,
            **variables: str,
        ) -> "subprocess.CompletedProcess[str]":
            return subprocess.run(
                ["sh", "-c", command],
                cwd=str(folder),
                env={**os.environ, **variables},
                input=json.dumps({"cwd": str(event_folder or folder)}),
                capture_output=True,
                text=True,
                check=False,
            )

        for name in ("random.py", "types.py", "json.py", "pathlib.py", "runpy.py"):
            fixtures.write(path=self.root / name, text="raise ImportError('the session folder shadowed a standard module')\n")
            fixtures.write(path=self.root / "src" / "deep" / name, text="raise ImportError('the session folder shadowed a standard module')\n")

        for folder in (self.root, self.root / "src" / "deep", escaped):
            with self.subTest(folder=str(folder.relative_to(self.root.parent))):
                ran = run(folder)

                self.assertEqual((ran.returncode, ran.stderr), (0, ""))
                self.assertTrue(ran.stdout.startswith("now: "), ran.stdout)

        for variable in ("CLAUDE_PROJECT_DIR", "GEMINI_PROJECT_DIR"):
            with self.subTest(variable=variable):
                ran = run(elsewhere, event_folder=self.root, **{variable: str(self.root)})

                self.assertEqual((ran.returncode, ran.stderr), (0, ""))
                self.assertTrue(ran.stdout.startswith("now: "), ran.stdout)

        ran = run(elsewhere)
        self.assertEqual((ran.returncode, ran.stdout, ran.stderr), (0, "", ""))

        ran = run(self.root / "src" / "deep", CLAUDE_PROJECT_DIR=str(elsewhere))
        self.assertEqual((ran.returncode, ran.stdout, ran.stderr), (0, "", ""))

        broken = subprocess.run(
            ["sh", "-c", command.replace(" prompt ", " compact ")],
            cwd=str(self.root),
            input=json.dumps({"cwd": str(self.root)}),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(broken.returncode, 1)
        self.assertTrue(broken.stderr.startswith("the brain hook could not run: "), broken.stderr)

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_the_words_file_and_its_write_in_flight_are_ignored_by_the_kit_and_by_every_ported_project(self) -> None:
        self.assertIn(f".agents/{autoack.FILE}", KIT_IGNORES)
        self.assertIn(f".agents/.{autoack.FILE}.*.tmp", KIT_IGNORES)
        self.assertIn(f".agents/{task.FILE}", KIT_IGNORES)
        self.assertIn(f".agents/.{task.FILE}.*.tmp", KIT_IGNORES)
        fixtures.write(path=self.root / "README.md", text="# App\n")
        commit_all(self.root)

        upgrade.port(kit=KIT, project=self.project)

        with mock.patch.object(tempfile, "mkstemp", wraps=tempfile.mkstemp) as made:
            autoack.turn_on(brain=store.Brain(root=self.root / ".agents"), hours=1)
        in_flight = Path(made.call_args.kwargs["dir"]) / f"{made.call_args.kwargs['prefix']}k3j2_x{made.call_args.kwargs['suffix']}"

        self.assertTrue(self.ignored(f".agents/{autoack.FILE}"))
        self.assertTrue(self.ignored(str(in_flight.relative_to(self.root))))
        self.assertTrue(self.ignored(f".agents/{task.FILE}"))
        self.assertTrue(self.ignored(f".agents/.{task.FILE}.k3j2_x9a.tmp"))
        self.assertFalse(self.ignored(".agents/todos/TODO-001-mine.md"))

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_the_kits_block_sits_at_the_top_so_every_line_the_project_wrote_wins(self) -> None:
        own = ".env*\n!.env.example\n!.env.sample\n.claude/*\n!.claude/*.json\n/.env.local\n"
        fixtures.write(path=self.root / "README.md", text="# App\n")
        fixtures.write(path=self.root / ".gitignore", text=own)
        commit_all(self.root)

        report = upgrade.port(kit=KIT, project=self.project)

        written = (self.root / ".gitignore").read_text(encoding="utf-8")
        self.assertEqual(written, "# The agent kit's own ignores.\n" + "\n".join(KIT_IGNORES) + "\n\n" + own)
        self.assertFalse(self.ignored(".env.example"))
        self.assertFalse(self.ignored(".env.sample"))
        self.assertTrue(self.ignored(".env.local"))
        self.assertTrue(self.ignored(".env.prod"))
        self.assertFalse(self.ignored(".claude/settings.local.json"))
        self.assertTrue(self.ignored("tools/__pycache__/x.pyc"))
        self.assertIn("set the kit's ignore block at the top of .gitignore, the project's own lines below it win", report.render())

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_the_kits_lines_reach_what_the_project_says_nothing_about(self) -> None:
        fixtures.write(path=self.root / "README.md", text="# App\n")
        fixtures.write(path=self.root / ".gitignore", text="node_modules/\n")
        commit_all(self.root)

        upgrade.port(kit=KIT, project=self.project)

        self.assertFalse(self.ignored(".env.example"))
        self.assertTrue(self.ignored(".env.local"))
        self.assertTrue(self.ignored(".claude/settings.local.json"))
        self.assertTrue(self.ignored("node_modules/x.js"))

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_a_file_the_project_ignores_by_name_or_by_an_anchored_spelling_stays_ignored(self) -> None:
        fixtures.write(path=self.root / "README.md", text="# App\n")
        fixtures.write(path=self.root / ".gitignore", text=".env\n.env.example\n/.env.sample\n")
        commit_all(self.root)

        upgrade.port(kit=KIT, project=self.project)

        self.assertTrue(self.ignored(".env.example"))
        self.assertTrue(self.ignored(".env.sample"))

    def test_the_block_is_rebuilt_at_the_top_and_every_other_line_keeps_its_place(self) -> None:
        stale = "build/\n\n# The agent kit's own ignores.\n!.env.example\nmine/\n# a note of ours\n.env.*\n\n!keep.log\n\n# The agent kit's own ignores.\n*.pyc\n\n"
        fixtures.write(path=self.root / ".gitignore", text=stale)

        upgrade.port(kit=KIT, project=self.project)
        first = (self.root / ".gitignore").read_text(encoding="utf-8")
        again = upgrade.upgrade(kit=KIT, project=self.project)

        self.assertEqual(first, "# The agent kit's own ignores.\n" + "\n".join(KIT_IGNORES) + "\n\nbuild/\n\nmine/\n# a note of ours\n\n!keep.log\n")
        self.assertEqual((self.root / ".gitignore").read_text(encoding="utf-8"), first)
        self.assertNotIn("ignore block", again.render())

    def test_the_ignore_file_is_split_the_way_git_splits_it_and_any_heading_spelling_is_folded_at_once(self) -> None:
        stale = "# The agent kit's own ignores. \n.env\n# The agent kit's own ignores.\n*.pyc\nmine/\n\nweird\x0cname/\n"
        fixtures.write(path=self.root / ".gitignore", text=stale)

        report = upgrade.port(kit=KIT, project=self.project)
        first = (self.root / ".gitignore").read_text(encoding="utf-8")
        again = upgrade.upgrade(kit=KIT, project=self.project)

        self.assertEqual(first, "# The agent kit's own ignores.\n" + "\n".join(KIT_IGNORES) + "\n\nmine/\n\nweird\x0cname/\n")
        self.assertIn("ignore block", report.render())
        self.assertNotIn("ignore block", again.render())

    def test_a_stray_carriage_return_at_a_lines_end_never_flips_the_file_to_crlf(self) -> None:
        (self.root / ".gitignore").write_bytes(b"mine/\nbuild/\r")

        upgrade.port(kit=KIT, project=self.project)
        first = (self.root / ".gitignore").read_bytes()
        again = upgrade.upgrade(kit=KIT, project=self.project)

        self.assertTrue(first.endswith(b"\n\nmine/\nbuild/\n"))
        self.assertNotIn(b"\r", first)
        self.assertEqual((self.root / ".gitignore").read_bytes(), first)
        self.assertNotIn("ignore block", again.render())

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_a_line_the_project_added_under_an_earlier_block_never_climbs_above_its_own_earlier_lines(self) -> None:
        fixtures.write(path=self.root / "README.md", text="# App\n")
        fixtures.write(
            path=self.root / ".gitignore",
            text="# Ours.\n!creds.json\n\n# The agent kit's own ignores.\n.env\n!.env.example\ncreds.json\n",
        )
        commit_all(self.root)

        upgrade.port(kit=KIT, project=self.project)

        self.assertTrue(self.ignored("creds.json"))

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_a_byte_order_mark_stays_at_the_start_of_the_ignore_file(self) -> None:
        fixtures.write(path=self.root / "README.md", text="# App\n")
        (self.root / ".gitignore").write_bytes("﻿build/\nnode_modules/\n".encode("utf-8"))
        commit_all(self.root)

        upgrade.port(kit=KIT, project=self.project)

        raw = (self.root / ".gitignore").read_bytes()
        self.assertTrue(raw.startswith("﻿# The agent kit's own ignores.\n".encode("utf-8")))
        self.assertEqual(raw.count("﻿".encode("utf-8")), 1)
        self.assertTrue(self.ignored("build/out"))

    def test_a_grown_kit_puts_its_new_line_inside_the_block_in_its_own_order(self) -> None:
        upgrade.port(kit=KIT, project=self.project)
        grown = Path(self.tmp.name) / "grown"
        shutil.copytree(KIT.root / ".agents", grown / ".agents", symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("AGENTS.md", "CLAUDE.md", "GEMINI.md", "CONTRIBUTING.md"):
            shutil.copy(KIT.root / name, grown / name)
        for name in (".claude", ".codex", ".gemini"):
            shutil.copytree(KIT.root / name, grown / name, symlinks=True)
        fixtures.write(path=grown / ".gitignore", text=KIT.text(Path(".gitignore")).replace(".env.*\n", ".env.*\n.env.*.local\n"))

        upgrade.upgrade(kit=upgrade.Kit(root=grown), project=self.project)

        written = (self.root / ".gitignore").read_text(encoding="utf-8")
        self.assertEqual(written.count("# The agent kit's own ignores."), 1)
        self.assertIn(".env.*\n.env.*.local\n!.env.example\n", written)

    def test_the_kits_own_stubs_are_never_carried(self) -> None:
        for name in ("CLAUDE.md", "GEMINI.md"):
            fixtures.write(path=self.root / name, text=KIT.text(Path(name)))

        report = upgrade.port(kit=KIT, project=self.project)

        self.assertNotIn("carried", report.render())
        self.assertEqual((self.root / "AGENTS.md").read_text(encoding="utf-8"), KIT.text(Path("AGENTS.md")))

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_replaced_root_files_are_named_with_where_git_keeps_them(self) -> None:
        fixtures.write(path=self.root / "CONTRIBUTING.md", text="# Contributing\n\nOurs.\n")
        fixtures.write(path=self.root / "CLAUDE.md", text=KIT.text(Path("CLAUDE.md")))
        commit_all(self.root)

        report = upgrade.port(kit=KIT, project=self.project)
        commit_all(self.root)
        again = upgrade.upgrade(kit=KIT, project=self.project)

        self.assertIn("replaced CONTRIBUTING.md", report.render())
        self.assertIn("the old ones are still in git: git show HEAD:CONTRIBUTING.md", report.render())
        self.assertNotIn("replaced CLAUDE.md", report.render())
        self.assertNotIn("replaced", again.render())
        self.assertNotIn("still in git", again.render())

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_the_git_line_names_files_as_git_holds_them(self) -> None:
        repo = Path(self.tmp.name) / "repo"
        project = upgrade.Project(root=repo / "shell")
        fixtures.write(path=repo / "README.md", text="# Mono\n")
        fixtures.write(path=project.root / "Agents.md", text="# Agents.md\n\n## 0. Project Rules\n\nMine.\n")
        fixtures.write(path=project.root / "CONTRIBUTING.md", text="# Contributing\n\nOurs.\n")
        commit_all(repo)
        fixtures.write(path=project.root / "GEMINI.md", text=KIT.text(Path("GEMINI.md")))

        report = upgrade.port(kit=KIT, project=project)

        self.assertIn("the old ones are still in git: git show HEAD:shell/Agents.md, git show HEAD:shell/CONTRIBUTING.md", report.render())

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_a_replaced_file_git_never_held_is_not_named(self) -> None:
        fixtures.write(path=self.root / "CLAUDE.md", text="# CLAUDE.md\n\n## 0. Project Rules\n\nMine.\n")
        commit_all(self.root)
        fixtures.write(path=self.root / "AGENTS.md", text=KIT.text(Path("AGENTS.md")))

        report = upgrade.port(kit=KIT, project=self.project)

        self.assertIn("replaced AGENTS.md", report.render())
        self.assertIn("the old ones are still in git: git show HEAD:CLAUDE.md\n", report.render())

    def test_outside_git_nothing_points_at_a_saved_copy(self) -> None:
        fixtures.write(path=self.root / "CONTRIBUTING.md", text="# Contributing\n\nOurs.\n")

        report = upgrade.port(kit=KIT, project=self.project)

        self.assertIn("replaced CONTRIBUTING.md", report.render())
        self.assertNotIn("still in git", report.render())

    def test_a_project_with_a_brain_is_sent_to_upgrade(self) -> None:
        fixtures.build_old_project(self.root)

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.port(kit=KIT, project=self.project)

        self.assertIn("run brain upgrade instead", str(caught.exception))


class Upgrading(ProjectCase):
    def setUp(self) -> None:
        super().setUp()
        fixtures.build_old_project(self.root)

    def test_the_old_layout_is_carried_to_the_kit(self) -> None:
        report = upgrade.upgrade(kit=KIT, project=self.project)

        self.assertFalse((self.root / ".agent").exists())
        self.assertTrue((self.root / ".agents" / "bugs" / "BUG-001-scrollbar.md").is_file())
        agents = (self.root / "AGENTS.md").read_text(encoding="utf-8")
        self.assertTrue(agents.startswith("# AGENTS.md\n\n## 0. Project Rules\n"))
        self.assertIn("### 0.1 Gates", agents)
        self.assertIn("(.agents/docs/foundation/theme.md)", agents)
        self.assertNotIn("## 1. Coding Guideline", agents)
        self.assertIn("@.agents/SYSTEM.md and", agents)
        self.assertEqual([path.name for path in self.root.iterdir() if path.name.lower() == "agents.md"], ["AGENTS.md"])
        self.assertEqual((self.root / "CLAUDE.md").read_text(encoding="utf-8"), KIT.text(Path("CLAUDE.md")))
        self.assertEqual((self.root / ".agents" / "SYSTEM.md").read_text(encoding="utf-8"), KIT.text(Path(".agents/SYSTEM.md")))
        self.assertIn("(.agents/docs/foundation/theme.md)", (self.root / "README.md").read_text(encoding="utf-8"))
        self.assertEqual((self.root / ".agents" / "VERSION").read_text(encoding="utf-8").strip(), KIT.version())
        self.assertEqual([finding.line() for finding in report.findings], [])
        self.assertIn("renamed .agent/ to .agents/", report.render())
        self.assertIn("kept section 0 from CLAUDE.md", report.render())

    def test_the_ignore_file_keeps_its_line_endings_and_gains_the_kits_lines_once(self) -> None:
        upgrade.upgrade(kit=KIT, project=self.project)
        first = (self.root / ".gitignore").read_bytes()

        upgrade.upgrade(kit=KIT, project=self.project)

        self.assertEqual((self.root / ".gitignore").read_bytes(), first)
        self.assertTrue(first.startswith(b"# The agent kit's own ignores.\r\n.agents/.worktrees/\r\n"))
        self.assertTrue(first.endswith(b"\r\n\r\nbuild/\r\n.agents/.worktrees/\r\n"))
        self.assertIn(b"__pycache__/\r\n", first)
        self.assertNotIn(b"\n", first.replace(b"\r\n", b""))

    def test_running_it_again_leaves_the_project_as_it_was(self) -> None:
        upgrade.upgrade(kit=KIT, project=self.project)
        agents = (self.root / "AGENTS.md").read_text(encoding="utf-8")

        report = upgrade.upgrade(kit=KIT, project=self.project)

        self.assertEqual((self.root / "AGENTS.md").read_text(encoding="utf-8"), agents)
        self.assertNotIn("renamed", report.render())
        self.assertNotIn("migrated", report.render())
        self.assertEqual([finding.line() for finding in report.findings], [])

    def test_a_project_without_a_brain_is_sent_to_port(self) -> None:
        shutil.rmtree(self.root / ".agent")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=KIT, project=self.project)

        self.assertIn("run brain port instead", str(caught.exception))

    def test_an_older_kit_never_downgrades_a_newer_project(self) -> None:
        upgrade.upgrade(kit=KIT, project=self.project)
        fixtures.write(path=self.root / ".agents" / "VERSION", text="9.9.9\n")
        system = (self.root / ".agents" / "SYSTEM.md")
        system.write_text("the newer rulebook\n", encoding="utf-8")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=KIT, project=self.project)

        self.assertEqual(str(caught.exception), f"the project is on 9.9.9 and this kit is {KIT.version()}, use a kit at least that new")
        self.assertEqual((self.root / ".agents" / "VERSION").read_text(encoding="utf-8"), "9.9.9\n")
        self.assertEqual(system.read_text(encoding="utf-8"), "the newer rulebook\n")

    def test_the_kit_cannot_upgrade_itself(self) -> None:
        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=KIT, project=upgrade.Project(root=KIT.root))

        self.assertEqual(str(caught.exception), "the kit is the project, there is nothing to install")

    def test_the_projects_own_hooks_and_settings_stay_beside_the_kits(self) -> None:
        kit_call = "python3 .agent/skills/agent-brain/scripts/hooks.py pre-write --agent claude"
        mine = {
            "permissions": {"allow": ["Bash(flutter test)"]},
            "env": {"MINE": "1", "GREETING": "héllo"},
            "hooks": {
                "PreToolUse": [
                    {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo mine"}]},
                    {"matcher": "Write", "hooks": [{"type": "command", "command": "python3 old/agent-brain/scripts/hooks.py pre-write"}]},
                    {"matcher": "Write", "hooks": [{"type": "command", "command": f"bash guard.sh && {kit_call}"}]},
                    {"matcher": "Write", "hooks": [{"type": "command", "command": kit_call}]},
                ],
                "Notification": [
                    {"hooks": [{"type": "command", "command": "python3 .agent/skills/agent-brain/scripts/hooks.py notify --agent claude"}]},
                    {"hooks": [{"type": "command", "command": "say done"}]},
                ],
                "Stop": [
                    {"hooks": [{"type": "command", "command": "python3 .agent/skills/agent-brain/scripts/hooks.py stop --agent claude"}]},
                ],
            },
        }
        fixtures.write(path=self.root / ".claude" / "settings.json", text=json.dumps(mine, indent=4))

        upgrade.upgrade(kit=KIT, project=self.project)

        text = (self.root / ".claude" / "settings.json").read_text(encoding="utf-8")
        merged = json.loads(text)
        commands = [hook["command"] for group in merged["hooks"]["PreToolUse"] for hook in group["hooks"]]
        self.assertEqual(merged["permissions"], {"allow": ["Bash(flutter test)"], "deny": list(upgrade.READ_GUARDS)})
        self.assertEqual(merged["env"]["MINE"], "1")
        self.assertEqual(merged["env"]["CLAUDE_CODE_SUBAGENT_MODEL"], "opus")
        self.assertIn("echo mine", commands)
        self.assertIn("python3 old/agent-brain/scripts/hooks.py pre-write", commands)
        self.assertIn(f"bash guard.sh && {kit_call}", commands)
        self.assertEqual(sum(command.endswith("pre-write --agent claude") for command in commands), 2)
        self.assertEqual([hook["command"] for group in merged["hooks"]["Notification"] for hook in group["hooks"]], ["say done"])
        self.assertEqual(merged["hooks"]["Stop"], [])
        for event, argument in (("UserPromptSubmit", "prompt"), ("PostToolBatch", "tick")):
            with self.subTest(event=event):
                added = [hook["command"] for group in merged["hooks"][event] for hook in group["hooks"]]
                self.assertEqual(len(added), 1)
                self.assertEqual(added[0], f'python3 -c "{upgrade.LAUNCHER}" {argument} --agent claude')
        self.assertEqual(merged["outputStyle"], "Plain & Short")
        self.assertIn('"GREETING": "héllo"', text)
        self.assertIn('\n    "permissions"', text)

    def test_the_kits_deny_rules_replace_their_earlier_spelling_and_leave_the_projects_own(self) -> None:
        theirs = ["Read(./.env)", "Read(./vendor/agent-brain/scripts/**)", "Read(//opt/tools/agent-brain/scripts/secret.py)"]
        mine = {"permissions": {"allow": ["Bash(flutter test)"], "deny": [*theirs, "Read(/.agents/skills/agent-brain/scripts/*.py)", "Read(/.agent/skills/agent-brain/scripts/**)"]}}
        fixtures.write(path=self.root / ".claude" / "settings.json", text=json.dumps(mine))

        upgrade.upgrade(kit=KIT, project=self.project)
        upgrade.upgrade(kit=KIT, project=self.project)

        merged = json.loads((self.root / ".claude" / "settings.json").read_text(encoding="utf-8"))
        self.assertEqual(merged["permissions"], {"allow": ["Bash(flutter test)"], "deny": [*theirs, *upgrade.READ_GUARDS]})

    def test_a_deny_list_or_a_permissions_value_of_the_wrong_shape_is_refused_in_words(self) -> None:
        shapes = (
            ('{"permissions": {"deny": "Read(./.env)"}}', "permissions.deny in .claude/settings.json is not a JSON list"),
            ('{"permissions": {"deny": null}}', "permissions.deny in .claude/settings.json is not a JSON list"),
            ('{"permissions": []}', "permissions in .claude/settings.json is not a JSON object"),
            ('{"env": null}', "env in .claude/settings.json is not a JSON object"),
            ('{"env": "x"}', "env in .claude/settings.json is not a JSON object"),
        )
        for text, said in shapes:
            with self.subTest(text=text):
                fixtures.write(path=self.root / ".claude" / "settings.json", text=text)

                with self.assertRaises(upgrade.UpgradeError) as caught:
                    upgrade.upgrade(kit=KIT, project=self.project)

                self.assertEqual(str(caught.exception), said)

    def test_a_malformed_group_or_event_the_kit_never_touches_is_left_alone(self) -> None:
        mine = {"hooks": {"Notification": [{"matcher": "*", "hooks": None}], "PreCompact": {"note": "not a list"}, "Stop": None, "SubagentStop": []}}
        fixtures.write(path=self.root / ".claude" / "settings.json", text=json.dumps(mine))

        upgrade.upgrade(kit=KIT, project=self.project)

        merged = json.loads((self.root / ".claude" / "settings.json").read_text(encoding="utf-8"))
        self.assertEqual(merged["hooks"]["Notification"], [{"matcher": "*", "hooks": None}])
        self.assertEqual(merged["hooks"]["PreCompact"], {"note": "not a list"})
        self.assertIsNone(merged["hooks"]["Stop"])
        self.assertEqual(merged["hooks"]["SubagentStop"], [])
        self.assertIn("PreToolUse", merged["hooks"])

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_an_edited_style_copy_is_guarded_since_the_doctor_rewrites_it(self) -> None:
        commit_all(self.root)
        upgrade.upgrade(kit=KIT, project=self.project)
        commit_all(self.root)
        with open(self.root / ".codex" / "plain-and-short.md", "a", encoding="utf-8") as handle:
            handle.write("\nmine\n")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=KIT, project=self.project)

        self.assertIn(".codex/plain-and-short.md", str(caught.exception))

    def test_the_shipped_configs_carry_the_prompt_and_the_tick_for_their_own_agent(self) -> None:
        shipped = {
            "claude": (".claude/settings.json", None),
            "codex": (".codex/hooks.json", "Bash"),
            "gemini": (".gemini/settings.json", "run_shell_command"),
        }
        for agent, (name, tick_matcher) in shipped.items():
            with self.subTest(agent=agent):
                config: dict = json.loads(KIT.text(Path(name)))
                commands: list[str] = [hook["command"] for groups in config["hooks"].values() for group in groups for hook in group["hooks"]]
                self.assertTrue(all(upgrade.KIT_HOOK.match(command) for command in commands), commands)
                self.assertTrue(all(command.startswith(f'python3 -c "{upgrade.LAUNCHER}" ') for command in commands), commands)
                self.assertTrue(all(f"--agent {agent}" in command for command in commands), commands)

                prompts: list[dict] = [group for group in config["hooks"][hooks.PROMPT[agent]] if any('" prompt ' in hook["command"] for hook in group["hooks"])]
                ticks: list[dict] = [group for group in config["hooks"][hooks.TICK[agent]] if any('" tick ' in hook["command"] for hook in group["hooks"])]
                self.assertEqual((len(prompts), len(ticks)), (1, 1))
                self.assertIsNone(prompts[0].get("matcher"))
                self.assertEqual(ticks[0].get("matcher"), tick_matcher)

    def test_every_codex_hook_carries_a_windows_command_for_the_same_event(self) -> None:
        config: dict = json.loads(KIT.text(Path(".codex/hooks.json")))
        hooks_shipped: list[dict] = [hook for groups in config["hooks"].values() for group in groups for hook in group["hooks"]]

        self.assertEqual(len(hooks_shipped), 5)
        for hook in hooks_shipped:
            with self.subTest(command=hook["command"]):
                tail = hook["command"].split('" ', 1)[1]
                self.assertEqual(hook["commandWindows"], f'py -3 -c "{upgrade.LAUNCHER}" {tail}')

    def test_a_kit_group_is_known_by_the_whole_shape_of_its_command(self) -> None:
        kit_calls = [
            'python3 "$CLAUDE_PROJECT_DIR"/.agents/skills/agent-brain/scripts/hooks.py session-start --agent claude',
            "python3 .agents/skills/agent-brain/scripts/hooks.py session-start --agent codex --with-rules",
            f'python3 -c "{upgrade.LAUNCHER}" session-start --agent claude --with-rules',
            *[f'python3 -c "{text}" tick --agent codex' for text in upgrade.LAUNCHERS],
            "python3 -c \"import os,pathlib,runpy,sys;h='.agents/skills/agent-brain/scripts/hooks.py';c=pathlib.Path(os.environ.get('CLAUDE_PROJECT_DIR') or os.environ.get('GEMINI_PROJECT_DIR') or os.getcwd());p=next((d/h for d in [c,*c.parents] if (d/h).is_file()),None);p or sys.exit(0);sys.path.insert(0,str(p.parent));sys.argv[0]=str(p);runpy.run_path(str(p),run_name='__main__')\" prompt --agent claude",
            "python3 .agent/skills/agent-brain/scripts/hooks.py pre-write --agent gemini",
            'python3 "$GEMINI_PROJECT_DIR"/.agents/skills/agent-brain/scripts/hooks.py prompt --agent gemini',
            "python3 $CLAUDE_PROJECT_DIR/.agents/skills/agent-brain/scripts/hooks.py tick --agent claude",
            "python3 $GEMINI_PROJECT_DIR/.agent/skills/agent-brain/scripts/hooks.py post-write --agent gemini",
            "python3 .agents/skills/agent-brain/scripts/guard.py pre-write",
        ]
        own_calls = [
            "bash run.sh && python3 .agents/skills/agent-brain/scripts/hooks.py pre-write --agent claude",
            "python3 vendor/agent-brain/scripts/hooks.py pre-write",
            "python3 .agents/skills/agent-brain/scripts/hooks.py pre-write | tee log",
            "python3 .agents/skills/my-skill/scripts/hooks.py pre-write",
            'python3 -c "import runpy" tick --agent codex',
            f'python3 -c "{upgrade.LAUNCHER}" tick --agent codex | tee log',
            f'python3 -c "{upgrade.LAUNCHER.replace(".agents/", "vendor/")}" tick --agent codex',
            f'python3 -c "{upgrade.LAUNCHER.replace(".agents/", "vendor/copy/.agents/")}" tick --agent codex',
            f'python3 -c "{upgrade.LAUNCHER};open(\'mine.log\',\'a\').write(\'x\')" tick --agent codex',
            f'python3 -c "{upgrade.LAUNCHER.replace("import sys;", "import sys,os;")}" tick --agent codex',
            'python3 "$(git rev-parse --show-toplevel)"/.agents/skills/agent-brain/scripts/hooks.py tick --agent codex',
            'python3 "$HOME"/.agents/skills/agent-brain/scripts/hooks.py tick --agent codex',
            "python3 $CLAUDE_PROJECT_DIR_OLD/.agents/skills/agent-brain/scripts/hooks.py tick --agent claude",
            "python3 $GEMINI_PROJECT_DIR2/.agents/skills/agent-brain/scripts/hooks.py tick --agent gemini",
            "python3 $GEMINI_PROJECT_DIR/sub/.agents/skills/agent-brain/scripts/hooks.py tick --agent gemini",
            "python3 $CLAUDE_PROJECT_DIR/sub/.agents/skills/agent-brain/scripts/hooks.py tick --agent claude",
            'python3 "$CLAUDE_PROJECT_DIR"/vendor/.agents/skills/agent-brain/scripts/hooks.py tick --agent claude',
            'python3 "$GEMINI_PROJECT_DIR"/.agents/skills/agent-brain/scripts/hooks.py tick --agent gemini && say done',
            "echo mine",
        ]

        for command in kit_calls:
            self.assertTrue(upgrade.is_kit_group({"hooks": [{"type": "command", "command": command}]}), command)
        for command in own_calls:
            self.assertFalse(upgrade.is_kit_group({"hooks": [{"type": "command", "command": command}]}), command)

    def test_a_config_that_is_not_json_is_refused_by_name_before_anything_is_written(self) -> None:
        fixtures.write(path=self.root / ".gemini" / "settings.json", text="{not json")
        before = (self.root / "CLAUDE.md").read_text(encoding="utf-8")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=KIT, project=self.project)

        self.assertTrue(str(caught.exception).startswith(".gemini/settings.json is not valid JSON"))
        self.assertTrue((self.root / ".agent").is_dir())
        self.assertFalse((self.root / ".agents").exists())
        self.assertEqual((self.root / "CLAUDE.md").read_text(encoding="utf-8"), before)

    def test_hooks_that_are_not_an_object_are_refused_in_words(self) -> None:
        for text, said in (
            ('{"hooks": []}', "hooks in .claude/settings.json is not a JSON object"),
            ('{"hooks": null}', "hooks in .claude/settings.json is not a JSON object"),
            ('{"hooks": {"PreToolUse": {"matcher": "Bash"}}}', "hooks.PreToolUse in .claude/settings.json is not a JSON list"),
        ):
            with self.subTest(text=text):
                fixtures.write(path=self.root / ".claude" / "settings.json", text=text)

                with self.assertRaises(upgrade.UpgradeError) as caught:
                    upgrade.upgrade(kit=KIT, project=self.project)

                self.assertEqual(str(caught.exception), said)

    def test_a_kit_without_a_stamp_is_refused_before_a_port_writes(self) -> None:
        kit_root = Path(self.tmp.name) / "kit"
        shutil.copytree(KIT.root / ".agents", kit_root / ".agents", ignore=shutil.ignore_patterns("__pycache__"))
        (kit_root / ".agents" / "VERSION").unlink()
        bare = Path(self.tmp.name) / "bare"
        bare.mkdir()

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.port(kit=upgrade.Kit(root=kit_root), project=upgrade.Project(root=bare))

        self.assertIn("is not a kit checkout", str(caught.exception))
        self.assertEqual(list(bare.iterdir()), [])

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root ignores file modes")
    def test_the_report_streams_so_a_failure_leaves_the_record(self) -> None:
        out = io.StringIO()
        (self.root / ".gemini").mkdir()
        (self.root / ".gemini").chmod(0o555)

        try:
            with self.assertRaises(OSError):
                upgrade.upgrade(kit=KIT, project=self.project, out=out)
        finally:
            (self.root / ".gemini").chmod(0o755)

        self.assertIn("renamed .agent/ to .agents/", out.getvalue())
        self.assertIn("replaced CLAUDE.md", out.getvalue())

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_git_records_the_folder_move_and_the_case_change_as_renames(self) -> None:
        commit_all(self.root)

        upgrade.upgrade(kit=KIT, project=self.project)

        status = status_of(self.root)
        self.assertIn(".agent/bugs/BUG-001-scrollbar.md -> .agents/bugs/BUG-001-scrollbar.md", status)
        self.assertIn("Agents.md -> AGENTS.md", status)
        self.assertTrue(all(line.startswith("R") for line in status.splitlines() if "Agents.md -> AGENTS.md" in line))

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_uncommitted_work_in_a_file_it_would_replace_is_refused_first(self) -> None:
        commit_all(self.root)
        (self.root / "CLAUDE.md").write_text("edited since\n", encoding="utf-8")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=KIT, project=self.project)

        self.assertIn("CLAUDE.md", str(caught.exception))
        self.assertTrue((self.root / ".agent").is_dir())

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_the_guard_sees_a_front_door_in_another_case_and_the_old_folder(self) -> None:
        commit_all(self.root)
        (self.root / "Agents.md").write_text("edited since\n", encoding="utf-8")
        (self.root / ".agent" / "docs" / "foundation" / "theme.md").write_text("edited since, see .agent/docs\n", encoding="utf-8")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=KIT, project=self.project)

        self.assertEqual(
            str(caught.exception),
            "commit, stash or move these first, they would be written over or removed: .agent/docs/foundation/theme.md, Agents.md",
        )

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_the_guard_covers_the_kits_files_under_the_brain_and_the_skill_folder(self) -> None:
        commit_all(self.root)
        upgrade.upgrade(kit=KIT, project=self.project)
        commit_all(self.root)
        (self.root / ".agents" / "spec" / "conventions.md").write_text("a local edit\n", encoding="utf-8")
        fixtures.write(path=self.root / ".agents" / "skills" / "agent-brain" / "MY-NOTES.md", text="mine\n")
        fixtures.write(path=self.root / ".agents" / "skills" / "branch-review" / "MY-ANGLES.md", text="mine too\n")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=KIT, project=self.project)

        self.assertIn(".agents/skills/agent-brain/MY-NOTES.md", str(caught.exception))
        self.assertIn(".agents/skills/branch-review/MY-ANGLES.md", str(caught.exception))
        self.assertIn(".agents/spec/conventions.md", str(caught.exception))
        self.assertEqual((self.root / ".agents" / "spec" / "conventions.md").read_text(encoding="utf-8"), "a local edit\n")

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_files_the_run_keeps_are_no_reason_to_refuse(self) -> None:
        commit_all(self.root)
        (self.root / ".gitignore").write_bytes(b"build/\r\nlocal/\r\n")
        fixtures.write(
            path=self.root / ".claude" / "settings.json",
            text=json.dumps({"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "echo mine"}]}]}}),
        )

        report = upgrade.upgrade(kit=KIT, project=self.project)

        self.assertIn(b"local/\r\n", (self.root / ".gitignore").read_bytes())
        merged = json.loads((self.root / ".claude" / "settings.json").read_text(encoding="utf-8"))
        self.assertIn("echo mine", json.dumps(merged))
        self.assertNotIn("not a git repository", report.render())

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_the_guard_holds_for_a_project_below_the_repository_root(self) -> None:
        repo = Path(self.tmp.name) / "monorepo"
        below = repo / "apps" / "shell"
        shutil.move(str(self.root), str(below))
        fixtures.write(path=repo / "AGENTS.md", text="the repo's own front door\n")
        commit_all(repo)
        project = upgrade.Project(root=below)
        (below / "CLAUDE.md").write_text("edited below the root\n", encoding="utf-8")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=KIT, project=project)

        self.assertEqual(str(caught.exception), "commit, stash or move these first, they would be written over or removed: CLAUDE.md")

        commit_all(repo)
        (repo / "AGENTS.md").write_text("edited at the root, not the project's\n", encoding="utf-8")

        report = upgrade.upgrade(kit=KIT, project=project)

        self.assertTrue((below / ".agents" / "VERSION").is_file())
        self.assertNotIn("not a git repository", report.render())
        self.root = below

    def test_a_projects_own_skill_under_a_name_the_kit_now_ships_is_refused_not_replaced(self) -> None:
        own = self.root / ".agent" / "skills" / "branch-review"
        fixtures.write(path=own / "SKILL.md", text="---\nname: branch-review\ndescription: our own review\n---\n\nOurs.\n")
        fixtures.write(path=own / "checklist.md", text="ours too\n")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=KIT, project=self.project)

        self.assertEqual(
            str(caught.exception),
            "the kit now ships a skill named branch-review, and .agent/skills/branch-review is the project's own, "
            "move it aside first, or delete it if an interrupted upgrade left it",
        )
        self.assertEqual((own / "checklist.md").read_text(encoding="utf-8"), "ours too\n")
        self.assertIn("our own review", (own / "SKILL.md").read_text(encoding="utf-8"))
        self.assertFalse((self.root / ".agents").exists())

    def test_the_folder_migration_leaves_every_kit_skill_folder_alone(self) -> None:
        for name in schema.KIT_SKILL_NAMES:
            fixtures.write(path=self.root / ".agent" / "skills" / name / "notes.md", text="names .agent/docs\n")
        fixtures.write(path=self.root / ".agent" / "docs" / "note.md", text="names .agent/docs\n")

        listed = [path.relative_to(self.root) for path in upgrade.brain_markdown(self.root / ".agent")]

        self.assertIn(Path(".agent/docs/note.md"), listed)
        for name in schema.KIT_SKILL_NAMES:
            with self.subTest(name=name):
                notes = self.root / ".agent" / "skills" / name / "notes.md"
                self.assertNotIn(notes.relative_to(self.root), listed)
                self.assertTrue(upgrade.in_skill_folder(project=self.project, path=notes))

    def test_the_kits_own_review_folder_under_an_old_stamp_is_the_kits_still(self) -> None:
        shutil.copytree(KIT.brain / "skills" / "branch-review", self.root / ".agent" / "skills" / "branch-review")

        report = upgrade.upgrade(kit=KIT, project=self.project)

        self.assertEqual(report.findings, [])
        self.assertEqual((self.root / ".agents" / "VERSION").read_text(encoding="utf-8").strip(), KIT.version())

    def test_a_symlinked_skill_folder_on_the_old_layout_is_refused_too(self) -> None:
        shared = Path(self.tmp.name) / "shared"
        fixtures.write(path=shared / "SKILL.md", text="shared\n")
        (self.root / ".agent" / "skills").mkdir()
        (self.root / ".agent" / "skills" / "agent-brain").symlink_to(shared)

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=KIT, project=self.project)

        self.assertEqual(str(caught.exception), ".agent/skills/agent-brain is a symlink, replace it with a real folder first")
        self.assertTrue((self.root / ".agent").is_dir())
        self.assertEqual((shared / "SKILL.md").read_text(encoding="utf-8"), "shared\n")

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_metadata_under_the_old_folder_is_not_guarded_since_the_move_keeps_it(self) -> None:
        (self.root / ".agent" / "docs" / ".DS_Store").write_bytes(b"\x00\x01")
        fixtures.write(path=self.root / ".agent" / "docs" / "__pycache__" / "x.cpython-39.pyc", text="bytecode\n")
        (self.root / ".gitignore").write_bytes(b"build/\r\n.agent/.worktrees/\r\n.DS_Store\r\n__pycache__/\r\n")
        commit_all(self.root)

        report = upgrade.upgrade(kit=KIT, project=self.project)

        self.assertTrue((self.root / ".agents" / "docs" / ".DS_Store").is_file())
        self.assertIn("renamed .agent/ to .agents/", report.render())

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_a_symlink_under_the_skill_folder_is_guarded_since_retire_takes_it(self) -> None:
        commit_all(self.root)
        upgrade.upgrade(kit=KIT, project=self.project)
        commit_all(self.root)
        elsewhere = Path(self.tmp.name) / "elsewhere.py"
        elsewhere.write_text("print(1)\n", encoding="utf-8")
        (self.root / ".agents" / "skills" / "agent-brain" / "scripts" / "local.py").symlink_to(elsewhere)

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=KIT, project=self.project)

        self.assertIn(".agents/skills/agent-brain/scripts/local.py", str(caught.exception))

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_a_dotfile_under_the_old_folder_is_guarded(self) -> None:
        fixtures.write(path=self.root / ".agent" / ".notes.md", text="see .agent/docs/foundation/theme.md\n")
        commit_all(self.root)
        (self.root / ".agent" / ".notes.md").write_text("edited, see .agent/docs/foundation/theme.md\n", encoding="utf-8")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=KIT, project=self.project)

        self.assertIn(".agent/.notes.md", str(caught.exception))

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_the_old_skill_folder_is_guarded_by_its_old_name(self) -> None:
        skill = self.root / ".agent" / "skills" / "agent-brain"
        fixtures.write(path=skill / "SKILL.md", text="an older skill\n")
        fixtures.write(path=skill / "scripts" / "brainlib" / "local_helper.py", text="mine\n")
        commit_all(self.root)
        (skill / "SKILL.md").write_text("edited since\n", encoding="utf-8")
        (skill / "scripts" / "brainlib" / "local_helper.py").write_text("edited too\n", encoding="utf-8")
        (skill / "scripts" / "local.py").symlink_to(self.root / "README.md")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=KIT, project=self.project)

        self.assertEqual(
            str(caught.exception),
            "commit, stash or move these first, they would be written over or removed: "
            ".agent/skills/agent-brain/SKILL.md, .agent/skills/agent-brain/scripts/brainlib/local_helper.py, .agent/skills/agent-brain/scripts/local.py",
        )
        self.assertTrue((self.root / ".agent").is_dir())

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_a_projects_own_skill_is_pointed_at_the_new_name_and_guarded_like_any_file(self) -> None:
        commit_all(self.root)
        fixtures.write(path=self.root / ".agent" / "skills" / "my-skill" / "NOTES.md", text="see .agent/docs\n")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=KIT, project=self.project)

        self.assertIn(".agent/skills/my-skill/NOTES.md", str(caught.exception))

        commit_all(self.root)
        report = upgrade.upgrade(kit=KIT, project=self.project)

        self.assertIn("renamed .agent/ to .agents/", report.render())
        self.assertEqual((self.root / ".agents" / "skills" / "my-skill" / "NOTES.md").read_text(encoding="utf-8"), "see .agents/docs\n")

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_every_file_git_sees_that_names_the_old_folder_is_pointed_at_the_new_one(self) -> None:
        fixtures.write(path=self.root / "Makefile", text="docs:\n\tls .agent/docs\n")
        fixtures.write(path=self.root / ".github" / "workflows" / "ci.yml", text="run: cat .agent/INDEX.md\n")
        fixtures.write(path=self.root / "docs" / "guide.md", text="See .agent/docs/foundation/theme.md.\n")
        fixtures.write(
            path=self.root / ".claude" / "settings.json",
            text='{"hooks": {"PreToolUse": [{"hooks": [{"type": "command", "command": "python3 .agent/tools/check.py"}]}]}}\n',
        )
        (self.root / "logo.png").write_bytes(b"\x89PNG .agent/ \x00\xff")
        (self.root / "theme-link.md").symlink_to(".agent/docs/foundation/theme.md")
        commit_all(self.root)

        report = upgrade.upgrade(kit=KIT, project=self.project)

        self.assertIn(".agents/docs", (self.root / "Makefile").read_text(encoding="utf-8"))
        self.assertIn(".agents/INDEX.md", (self.root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
        self.assertIn(".agents/docs/foundation/theme.md", (self.root / "docs" / "guide.md").read_text(encoding="utf-8"))
        settings = (self.root / ".claude" / "settings.json").read_text(encoding="utf-8")
        self.assertIn("python3 .agents/tools/check.py", settings)
        self.assertIn('" pre-write --agent claude', settings)
        self.assertEqual((self.root / "logo.png").read_bytes(), b"\x89PNG .agent/ \x00\xff")
        self.assertEqual(os.readlink(self.root / "theme-link.md"), ".agent/docs/foundation/theme.md")
        self.assertIn("pointed", report.render())

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_an_uncommitted_file_naming_the_old_folder_is_refused_since_the_migration_rewrites_it(self) -> None:
        commit_all(self.root)
        fixtures.write(path=self.root / "Makefile", text="docs:\n\tls .agent/docs\n")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=KIT, project=self.project)

        self.assertEqual(str(caught.exception), "commit, stash or move these first, they would be written over or removed: Makefile")
        self.assertTrue((self.root / ".agent").is_dir())

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_the_file_listing_holds_whatever_grep_full_name_says(self) -> None:
        repo = Path(self.tmp.name) / "mono"
        package = repo / "packages" / "app"
        fixtures.build_old_project(package)
        fixtures.write(path=repo / "README.md", text="# Mono\n")
        commit_all(repo)
        config = Path(self.tmp.name) / "gitconfig"
        fixtures.write(path=config, text="[grep]\n\tfullName = true\n")

        with mock.patch.dict(os.environ, {"GIT_CONFIG_GLOBAL": str(config)}):
            report = upgrade.upgrade(kit=KIT, project=upgrade.Project(root=package))

        self.assertIn("pointed", report.render())
        self.assertIn(".agents/docs/foundation/theme.md", (package / "README.md").read_text(encoding="utf-8"))

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_a_file_git_ignores_that_names_the_old_folder_is_refused_rather_than_left_behind(self) -> None:
        (self.root / ".gitignore").write_bytes(b"build/\r\n.agent/.worktrees/\r\n.claude/settings.local.json\r\n")
        commit_all(self.root)
        fixtures.write(path=self.root / ".claude" / "settings.local.json", text='{"hooks": {"PreToolUse": [{"hooks": [{"type": "command", "command": "python3 .agent/tools/mine.py"}]}]}}\n')

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=KIT, project=self.project)

        self.assertEqual(
            str(caught.exception),
            "git ignores these and they would be written over or removed, move them out of the way first: .claude/settings.local.json",
        )
        self.assertTrue((self.root / ".agent").is_dir())

        (self.root / ".claude" / "settings.local.json").unlink()
        report = upgrade.upgrade(kit=KIT, project=self.project)

        self.assertIn("renamed .agent/ to .agents/", report.render())

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_a_project_folder_an_outer_repository_ignores_is_refused_rather_than_half_migrated(self) -> None:
        outer = Path(self.tmp.name) / "outer"
        fixtures.write(path=outer / "README.md", text="# Outer\n")
        fixtures.write(path=outer / ".gitignore", text="pkg/\n")
        fixtures.write(path=outer / "pkg" / "README.md", text="See .agent/docs/foundation/theme.md.\n")
        fixtures.write(path=outer / "pkg" / ".agent" / "INDEX.md", text="# Index\n\n- [theme](.agent/docs/foundation/theme.md)\n")
        fixtures.write(path=outer / "pkg" / ".agent" / "docs" / "foundation" / "theme.md", text=fixtures.OLD_DOC)
        commit_all(outer)

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=KIT, project=upgrade.Project(root=outer / "pkg"))

        self.assertEqual(
            str(caught.exception),
            "git ignores these and they would be written over or removed, move them out of the way first: "
            ".agent/INDEX.md, README.md",
        )
        self.assertTrue((outer / "pkg" / ".agent").is_dir())

    def test_outside_git_only_the_root_markdown_the_ignore_file_and_the_brain_are_pointed(self) -> None:
        fixtures.write(path=self.root / "Makefile", text="docs:\n\tls .agent/docs\n")

        upgrade.upgrade(kit=KIT, project=self.project)

        self.assertEqual((self.root / "Makefile").read_text(encoding="utf-8"), "docs:\n\tls .agent/docs\n")
        self.assertIn(".agents/docs/foundation/theme.md", (self.root / "README.md").read_text(encoding="utf-8"))
        self.assertIn(".agents/.worktrees/", (self.root / ".gitignore").read_text(encoding="utf-8"))

    def test_a_doc_folder_named_skills_is_still_pointed_at_the_new_name(self) -> None:
        fixtures.write(path=self.root / ".agent" / "docs" / "skills" / "notes.md", text="see .agent/docs/foundation/theme.md\n")
        commit_all(self.root)

        upgrade.upgrade(kit=KIT, project=self.project)

        self.assertEqual(
            (self.root / ".agents" / "docs" / "skills" / "notes.md").read_text(encoding="utf-8"),
            "see .agents/docs/foundation/theme.md\n",
        )

    def test_the_old_layout_takes_section_zero_from_claude_md_over_a_stale_copy(self) -> None:
        stale = fixtures.OLD_RULES.replace("# CLAUDE.md", "# Agents.md", 1).replace("Run `flutter test` before a review.", "Stale.")
        fixtures.write(path=self.root / "Agents.md", text=stale)

        report = upgrade.upgrade(kit=KIT, project=self.project)

        written = (self.root / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("kept section 0 from CLAUDE.md", report.render())
        self.assertIn("Run `flutter test` before a review.", written)
        self.assertNotIn("Stale.", written)

    def test_an_empty_stub_never_wins_over_a_full_section_zero(self) -> None:
        fixtures.write(path=self.root / "Agents.md", text="# Agents.md\n\n## 0. Project Rules\n\n## 1. Coding Guideline\n\nOld.\n")
        (self.root / "CLAUDE.md").write_text(fixtures.OLD_RULES.replace("## 0. Project Rules", "## 0. Rules"), encoding="utf-8")

        report = upgrade.upgrade(kit=KIT, project=self.project)

        self.assertIn("kept section 0 from CLAUDE.md", report.render())
        self.assertIn("### 0.1 Gates", (self.root / "AGENTS.md").read_text(encoding="utf-8"))

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_a_project_on_the_first_kit_keeps_its_section_zero_and_never_the_kits_old_words(self) -> None:
        first_kit = (
            f"{upgrade.FIRST_KIT_WORDS}\nRules that belong to this project alone go in a `## 0. Project Rules`\n"
            "section directly under the title above. Everything else in this file is\nthe kit's.\n"
        )
        bare = Path(self.tmp.name) / "bare"
        fixtures.write(path=bare / "README.md", text="# App\n")
        project = upgrade.Project(root=bare)
        upgrade.port(kit=KIT, project=project)

        (bare / "AGENTS.md").write_text(f"# AGENTS.md\n\n{first_kit}", encoding="utf-8")
        commit_all(bare)
        without = upgrade.upgrade(kit=KIT, project=project)
        plain = (bare / "AGENTS.md").read_text(encoding="utf-8")

        (bare / "AGENTS.md").write_text(f"# AGENTS.md\n\n## 0. Project Rules\n\nMine.\n\n{first_kit}", encoding="utf-8")
        commit_all(bare)
        upgrade.upgrade(kit=KIT, project=project)
        kept = (bare / "AGENTS.md").read_text(encoding="utf-8")

        self.assertNotIn("carried", without.render())
        self.assertEqual(plain, KIT.text(Path("AGENTS.md")))
        self.assertEqual(kept, upgrade.with_section_zero(text=KIT.text(Path("AGENTS.md")), zero="## 0. Project Rules\n\nMine.\n"))

    def test_the_old_layout_carries_nothing_whole_since_the_rest_was_the_old_kits(self) -> None:
        for name in ("CLAUDE.md", "Agents.md"):
            (self.root / name).write_text("# Rules\n\n## 1. Coding Guideline\n\nThe old kit's words.\n", encoding="utf-8")

        report = upgrade.upgrade(kit=KIT, project=self.project)

        self.assertNotIn("carried", report.render())
        self.assertIsNone(upgrade.section_zero_in((self.root / "AGENTS.md").read_text(encoding="utf-8")))

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_a_name_in_another_unicode_form_is_still_guarded(self) -> None:
        commit_all(self.root)
        upgrade.upgrade(kit=KIT, project=self.project)
        name = "références.md"
        fixtures.write(path=self.root / ".agents" / "skills" / "agent-brain" / name, text="mine\n")
        commit_all(self.root)
        (self.root / ".agents" / "skills" / "agent-brain" / name).write_text("edited\n", encoding="utf-8")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=KIT, project=self.project)

        self.assertIn("rences.md", str(caught.exception))

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_a_file_inside_an_ignored_folder_the_run_would_remove_is_refused(self) -> None:
        commit_all(self.root)
        upgrade.upgrade(kit=KIT, project=self.project)
        (self.root / ".gitignore").write_bytes((self.root / ".gitignore").read_bytes() + b"scratch/\r\n")
        commit_all(self.root)
        fixtures.write(path=self.root / ".agents" / "skills" / "agent-brain" / "scratch" / "notes.md", text="mine\n")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=KIT, project=self.project)

        self.assertIn(".agents/skills/agent-brain/scratch/notes.md", str(caught.exception))

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_a_brain_without_the_old_folder_is_not_migrated_whatever_its_stamp_says(self) -> None:
        (self.root / ".agent").rename(self.root / ".agents")
        fixtures.write(path=self.root / ".agents" / "VERSION", text="0.9.0\n")
        fixtures.write(path=self.root / ".agents" / "docs" / "note.md", text="The folder was .agent/docs once, a quote.\n")
        commit_all(self.root)
        (self.root / "README.md").write_text("edited, names .agent/docs still\n", encoding="utf-8")

        report = upgrade.upgrade(kit=KIT, project=self.project)

        self.assertNotIn("renamed .agent/", report.render())
        self.assertNotIn("pointed", report.render())
        self.assertEqual((self.root / "README.md").read_text(encoding="utf-8"), "edited, names .agent/docs still\n")
        self.assertIn(".agent/docs once", (self.root / ".agents" / "docs" / "note.md").read_text(encoding="utf-8"))

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_an_upgrade_straight_after_a_port_is_not_refused_over_identical_files(self) -> None:
        bare = Path(self.tmp.name) / "bare"
        fixtures.write(path=bare / "README.md", text="# App\n")
        commit_all(bare)
        project = upgrade.Project(root=bare)
        upgrade.port(kit=KIT, project=project)

        report = upgrade.upgrade(kit=KIT, project=project)

        self.assertIn("0 written, 0 retired", report.render())

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_a_port_does_not_guard_root_markdown_it_never_touches(self) -> None:
        bare = Path(self.tmp.name) / "bare"
        fixtures.write(path=bare / "README.md", text="# App\n")
        commit_all(bare)
        (bare / "README.md").write_text("# App, edited\n", encoding="utf-8")

        report = upgrade.port(kit=KIT, project=upgrade.Project(root=bare))

        self.assertEqual((bare / "README.md").read_text(encoding="utf-8"), "# App, edited\n")
        self.assertIn("Doctor is clean.", report.render())

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_an_ignored_kit_file_identical_to_the_kits_is_no_reason_to_refuse(self) -> None:
        bare = Path(self.tmp.name) / "bare"
        fixtures.write(path=bare / "README.md", text="# App\n")
        fixtures.write(path=bare / ".gitignore", text=".claude/\n")
        commit_all(bare)
        project = upgrade.Project(root=bare)
        upgrade.port(kit=KIT, project=project)
        commit_all(bare)

        report = upgrade.upgrade(kit=KIT, project=project)

        self.assertIn("0 written, 0 retired", report.render())

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_a_retired_file_is_named_in_the_report(self) -> None:
        commit_all(self.root)
        upgrade.upgrade(kit=KIT, project=self.project)
        fixtures.write(path=self.root / ".agents" / "skills" / "agent-brain" / "scripts" / "local_hook.py", text="print(1)\n")
        commit_all(self.root)

        report = upgrade.upgrade(kit=KIT, project=self.project)

        self.assertIn("retired .agents/skills/agent-brain/scripts/local_hook.py, the kit no longer ships it", report.render())

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_an_ignored_file_the_run_would_remove_is_refused_too(self) -> None:
        commit_all(self.root)
        upgrade.upgrade(kit=KIT, project=self.project)
        (self.root / ".gitignore").write_bytes((self.root / ".gitignore").read_bytes() + b"scratch.md\r\n")
        commit_all(self.root)
        fixtures.write(path=self.root / ".agents" / "skills" / "agent-brain" / "scratch.md", text="ignored but mine\n")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=KIT, project=self.project)

        self.assertIn(".agents/skills/agent-brain/scratch.md", str(caught.exception))

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_a_repository_whose_git_fails_is_refused_rather_than_unguarded(self) -> None:
        commit_all(self.root)
        with open(self.root / ".git" / "config", "a", encoding="utf-8") as handle:
            handle.write("\nthis is not a config line\n")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=KIT, project=self.project)

        self.assertTrue(str(caught.exception).startswith("git could not read the working tree, so nothing is guarded"))
        self.assertTrue((self.root / ".agent").is_dir())

    def test_outside_git_the_report_says_nothing_is_saved(self) -> None:
        report = upgrade.upgrade(kit=KIT, project=self.project)

        self.assertIn("not a git repository, so nothing is saved before a file is replaced", report.render())


class Copying(ProjectCase):
    def setUp(self) -> None:
        super().setUp()
        self.kit_root = Path(self.tmp.name) / "kit"
        for name in (".agents", ".claude", ".codex", ".gemini"):
            shutil.copytree(KIT.root / name, self.kit_root / name, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("AGENTS.md", "CLAUDE.md", "GEMINI.md", "CONTRIBUTING.md", ".gitignore"):
            shutil.copy(KIT.root / name, self.kit_root / name)
        self.kit = upgrade.Kit(root=self.kit_root)

    def test_a_dotfile_in_the_kit_does_not_ship_and_a_mode_does(self) -> None:
        (self.kit_root / ".agents" / "skills" / "agent-brain" / ".DS_Store").write_bytes(b"\x00\x01\xff")
        (self.kit_root / ".agents" / "skills" / "agent-brain" / "scripts" / "brain.py").chmod(0o755)

        report = upgrade.port(kit=self.kit, project=self.project)

        self.assertFalse((self.root / ".agents" / "skills" / "agent-brain" / ".DS_Store").exists())
        self.assertEqual(os.stat(self.root / ".agents" / "skills" / "agent-brain" / "scripts" / "brain.py").st_mode & 0o777, 0o755)
        self.assertEqual([finding.line() for finding in report.findings], [])

    def test_a_slip_in_a_front_door_the_kit_writes_whole_is_the_kits_and_one_in_section_zero_is_the_projects(self) -> None:
        stub = self.kit_root / "CLAUDE.md"
        stub.write_text(stub.read_text(encoding="utf-8").replace("@AGENTS.md", "@AGENTS.md\n@missing.md", 1), encoding="utf-8")
        agents = self.kit_root / "AGENTS.md"
        agents.write_text(agents.read_text(encoding="utf-8") + "\nSee @.agents/nowhere.md.\n", encoding="utf-8")
        fixtures.write(path=self.root / "AGENTS.md", text="# AGENTS.md\n\n## 0. Project Rules\n\nSee @docs/gone.md.\n")

        report = upgrade.port(kit=self.kit, project=self.project)

        self.assertEqual(
            [(finding.path, finding.message.split(",")[0]) for finding in report.kits_own],
            [("AGENTS.md", "imports @.agents/nowhere.md. with the punctuation glued on"), ("CLAUDE.md", "imports @missing.md")],
        )
        self.assertEqual(
            [(finding.path, finding.message.split(",")[0]) for finding in report.findings],
            [("AGENTS.md", "imports @docs/gone.md. with the punctuation glued on")],
        )

    @unittest.skipIf(not hasattr(os, "geteuid") or os.geteuid() == 0, "a file's mode shuts out only a user who is not root")
    def test_a_front_door_that_cannot_be_read_is_the_projects_to_put_right(self) -> None:
        upgrade.port(kit=self.kit, project=self.project)
        door = self.root / "CLAUDE.md"
        door.chmod(0o200)
        self.addCleanup(door.chmod, 0o644)

        report = upgrade.upgrade(kit=self.kit, project=self.project)

        self.assertEqual([(finding.path, finding.check, finding.message.split(":")[0]) for finding in report.findings], [("CLAUDE.md", "import", "cannot be read")])
        self.assertEqual(report.kits_own, [])

    def test_a_slip_in_section_zero_that_the_kits_words_carry_too_is_counted_once_for_the_project(self) -> None:
        agents = self.kit_root / "AGENTS.md"
        agents.write_text(agents.read_text(encoding="utf-8") + "\nSee @.agents/nowhere.md.\n", encoding="utf-8")
        fixtures.write(path=self.root / "AGENTS.md", text="# AGENTS.md\n\n## 0. Project Rules\n\nSee @.agents/nowhere.md.\n")

        report = upgrade.port(kit=self.kit, project=self.project)

        slip = ("AGENTS.md", "imports @.agents/nowhere.md. with the punctuation glued on")
        self.assertEqual([(finding.path, finding.message.split(",")[0]) for finding in report.kits_own], [slip])
        self.assertEqual([(finding.path, finding.message.split(",")[0]) for finding in report.findings], [slip])

    def test_a_project_file_named_like_a_front_door_inside_the_brain_is_the_projects(self) -> None:
        upgrade.port(kit=self.kit, project=self.project)
        fixtures.write(path=self.root / ".agents" / "CLAUDE.md", text="# CLAUDE.md\n\nNested.\n")

        report = upgrade.upgrade(kit=self.kit, project=self.project)

        self.assertEqual(report.kits_own, [])
        self.assertEqual([(finding.path, finding.check) for finding in report.findings], [("CLAUDE.md", "stray")])

    def test_a_slip_in_the_kits_own_files_is_reported_apart_and_never_counted(self) -> None:
        skill = self.kit_root / ".agents" / "skills" / "agent-brain" / "SKILL.md"
        skill.write_text(skill.read_text(encoding="utf-8").replace("\n\n", "\n\nA dash — the house style bans.\n\n", 1), encoding="utf-8")

        report = upgrade.port(kit=self.kit, project=self.project)

        self.assertEqual(report.findings, [])
        self.assertEqual([finding.path for finding in report.kits_own], ["skills/agent-brain/SKILL.md"])
        self.assertIn("Doctor is clean.", report.render())
        self.assertIn("In the kit's own files, for the kit to put right, not counted here:", report.render())

    def test_a_slip_in_the_brains_own_files_the_install_writes_whole_is_the_kits_and_never_counted(self) -> None:
        for relative in ("house-rules.md", "spec/conventions.md"):
            rules = self.kit_root / ".agents" / relative
            rules.write_text(rules.read_text(encoding="utf-8").replace("\n\n", "\n\nA dash — the house style bans.\n\n", 1), encoding="utf-8")

        report = upgrade.port(kit=self.kit, project=self.project)

        self.assertEqual(report.findings, [])
        self.assertEqual(sorted((finding.path, finding.check) for finding in report.kits_own), [("house-rules.md", "style"), ("spec/conventions.md", "style")])

    def test_a_slip_in_the_chat_style_source_the_install_copies_whole_is_the_kits_and_never_counted(self) -> None:
        source = self.kit_root / ".claude" / "output-styles" / "plain-and-short.md"
        source.write_text(source.read_text(encoding="utf-8").replace("keep-coding-instructions: true\n---\n", "keep-coding-instructions: true\n", 1), encoding="utf-8")

        report = upgrade.port(kit=self.kit, project=self.project)

        self.assertEqual([finding.line() for finding in report.findings], [])
        self.assertIn(".claude/output-styles/plain-and-short.md", [finding.path for finding in report.kits_own])

    def test_a_port_writes_claude_code_the_rules_that_refuse_reading_the_kits_code_by_both_of_its_paths(self) -> None:
        upgrade.port(kit=self.kit, project=self.project)

        settings = json.loads((self.root / ".claude" / "settings.json").read_text(encoding="utf-8"))
        self.assertEqual(
            upgrade.READ_GUARDS,
            ("Read(/.agents/skills/agent-brain/scripts/**)", "Read(/.claude/skills/agent-brain/scripts/**)"),
        )
        self.assertEqual(settings["permissions"], {"deny": list(upgrade.READ_GUARDS)})
        self.assertEqual(settings["env"]["CLAUDE_CODE_SUBAGENT_MODEL"], "opus")

    def test_a_kit_settings_file_that_does_not_parse_is_refused_before_the_first_write(self) -> None:
        fixtures.write(path=self.kit_root / ".claude" / "settings.json", text="{not json")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.port(kit=self.kit, project=self.project)

        self.assertTrue(str(caught.exception).startswith("the kit's .claude/settings.json is not valid JSON"))
        self.assertFalse((self.root / "AGENTS.md").exists())
        self.assertFalse((self.root / ".agents").exists())

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_a_clone_of_the_kit_and_another_spelling_of_it_are_refused_like_the_kit_itself(self) -> None:
        commit_all(self.kit_root)
        clone = Path(self.tmp.name) / "clone"
        git_clone(source=str(self.kit_root), target=clone)
        subprocess.run(["git", "-C", str(self.kit_root), "remote", "add", "origin", "https://forge.example/example-org/agent-setup"], check=True)
        origins = (
            str(self.kit_root),
            f"file://{self.kit_root}",
            str(self.kit_root / ".git"),
            "git@forge.example:example-org/agent-setup.git",
            "ssh://git@forge.example:2222/example-org/agent-setup/",
            "HTTPS://Forge.example/Example-Org/Agent-Setup.git",
        )
        for origin in origins:
            with self.subTest(origin=origin):
                subprocess.run(["git", "-C", str(clone), "remote", "set-url", "origin", origin], check=True)

                with self.assertRaises(upgrade.UpgradeError) as caught:
                    upgrade.upgrade(kit=self.kit, project=upgrade.Project(root=clone))

                self.assertEqual(str(caught.exception), "the project is a worktree or a clone of the kit, there is nothing to install")

        self.assertNotIn("permissions", json.loads((clone / ".claude" / "settings.json").read_text(encoding="utf-8")))
        subprocess.run(["git", "-C", str(clone), "remote", "set-url", "origin", "https://forge.example/example-org/agent-setup-fork"], check=True)
        self.assertIsNone(upgrade.is_the_kit(kit=self.kit, project=upgrade.Project(root=clone)))

        spelled = self.kit_root.parent / self.kit_root.name.upper()
        if not spelled.exists():
            self.skipTest("this filesystem tells case apart, so the other spelling is no folder")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=self.kit, project=upgrade.Project(root=spelled))

        self.assertEqual(str(caught.exception), "the kit is the project, there is nothing to install")

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_a_clone_of_a_clone_a_second_checkout_of_the_origin_and_a_worktree_of_the_kit_are_all_the_kit(self) -> None:
        places = Path(self.tmp.name)
        failing = places / "template" / "hooks" / "post-checkout"
        fixtures.write(path=failing, text="#!/bin/sh\nexit 1\n")
        failing.chmod(0o755)
        template = mock.patch.dict(os.environ, {"GIT_TEMPLATE_DIR": str(failing.parents[1])})
        template.start()
        self.addCleanup(template.stop)
        commit_all(self.kit_root)
        subprocess.run(["git", "-C", str(self.kit_root), "remote", "add", "origin", "https://forge.example/example-org/agent-setup"], check=True)
        git_clone(source=str(self.kit_root), target=places / "clone")
        git_clone(source=str(places / "clone"), target=places / "clone2")
        git_clone(source=str(self.kit_root), target=places / "kit2")
        subprocess.run(["git", "-C", str(places / "kit2"), "remote", "set-url", "origin", "git@forge.example:example-org/agent-setup.git"], check=True)
        subprocess.run(["git", "-C", str(self.kit_root), "worktree", "add", "-q", "--detach", str(places / "kitwt")], check=True)

        for kit, project in ((self.kit_root, places / "clone2"), (places / "kit2", places / "clone"), (places / "kitwt", places / "clone")):
            with self.subTest(kit=kit.name, project=project.name):
                self.assertEqual(
                    upgrade.is_the_kit(kit=upgrade.Kit(root=kit), project=upgrade.Project(root=project)),
                    "the project is a worktree or a clone of the kit, there is nothing to install",
                )

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_the_walk_along_local_origins_reaches_the_kit_in_eight_hops_and_stops_there(self) -> None:
        places = Path(self.tmp.name)
        commit_all(self.kit_root)
        before = self.kit_root
        for hop in range(1, 10):
            here = places / f"hop{hop}"
            git_init(target=here)
            subprocess.run(["git", "-C", str(here), "remote", "add", "origin", str(before)], check=True)
            before = here

        self.assertIsNotNone(upgrade.is_the_kit(kit=self.kit, project=upgrade.Project(root=places / "hop8")))
        self.assertIsNone(upgrade.is_the_kit(kit=self.kit, project=upgrade.Project(root=places / "hop9")))

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_a_clone_by_a_file_url_with_escapes_in_it_is_the_kit(self) -> None:
        places = Path(self.tmp.name)
        spaced = places / "my kit"
        commit_all(self.kit_root)
        git_clone(source=str(self.kit_root), target=spaced)
        git_clone(source=spaced.as_uri(), target=places / "c1")

        self.assertIn("%20", subprocess.run(["git", "-C", str(places / "c1"), "remote", "get-url", "origin"], capture_output=True, text=True, check=True).stdout)
        self.assertEqual(
            upgrade.is_the_kit(kit=upgrade.Kit(root=spaced), project=upgrade.Project(root=places / "c1")),
            "the project is a worktree or a clone of the kit, there is nothing to install",
        )

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_two_checkouts_whose_relative_origins_name_missing_folders_in_different_places_are_not_one(self) -> None:
        places = Path(self.tmp.name)
        for checkout in (places / "a" / "kit", places / "b" / "proj"):
            git_init(target=checkout)
            subprocess.run(["git", "-C", str(checkout), "remote", "add", "origin", "../shared"], check=True)

        self.assertIsNone(upgrade.is_the_kit(kit=upgrade.Kit(root=places / "a" / "kit"), project=upgrade.Project(root=places / "b" / "proj")))

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    @unittest.skipIf(not hasattr(os, "geteuid") or os.geteuid() == 0, "a folder's mode shuts out only a user who is not root")
    def test_two_clones_of_one_shared_folder_git_cannot_open_are_still_one_repository(self) -> None:
        places = Path(self.tmp.name)
        shared = places / "shared.git"
        commit_all(self.kit_root)
        git_clone(source=str(self.kit_root), target=shared, bare=True)
        git_clone(source=str(shared), target=places / "kit2")
        git_clone(source=str(shared), target=places / "proj")
        shared.chmod(0o000)
        self.addCleanup(shared.chmod, 0o755)

        reason = upgrade.is_the_kit(kit=upgrade.Kit(root=places / "kit2"), project=upgrade.Project(root=places / "proj"))

        self.assertEqual(reason, "the project is a worktree or a clone of the kit, there is nothing to install")

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_a_kit_vendored_as_plain_files_inside_a_repository_installs_into_a_sibling_folder(self) -> None:
        mono = Path(self.tmp.name) / "mono"
        shutil.copytree(self.kit_root, mono / "tools" / "kit", symlinks=True)
        fixtures.write(path=mono / "app" / "README.md", text="# App\n")
        commit_all(mono)
        subprocess.run(["git", "-C", str(mono), "remote", "add", "origin", "https://forge.example/acme/mono"], check=True)
        vendored = upgrade.Kit(root=mono / "tools" / "kit")
        project = upgrade.Project(root=mono / "app")

        upgrade.port(kit=vendored, project=project)
        commit_all(mono)
        report = upgrade.upgrade(kit=vendored, project=project)

        self.assertIn("0 written, 0 retired", report.render())

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_a_port_left_uncommitted_takes_an_upgrade_when_the_kits_stamp_has_no_final_newline(self) -> None:
        (self.kit_root / ".agents" / "VERSION").write_text(KIT.version(), encoding="utf-8")
        fixtures.write(path=self.root / "README.md", text="# Shell\n")
        commit_all(self.root)

        upgrade.port(kit=self.kit, project=self.project)
        report = upgrade.upgrade(kit=self.kit, project=self.project)

        self.assertIn("0 written, 0 retired", report.render())

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_files_the_run_wrote_that_hold_windows_line_ends_are_left_as_they_are_by_the_next_upgrade(self) -> None:
        upgrade.port(kit=self.kit, project=self.project)
        written = [*upgrade.ROOT_FILES, str(upgrade.CLAUDE_SETTINGS), f"{upgrade.FOLDER}/{upgrade.VERSION}"]
        crlf = {relative: (self.root / relative).read_bytes().replace(b"\n", b"\r\n") for relative in written}
        for relative, data in crlf.items():
            (self.root / relative).write_bytes(data)
        commit_all(self.root)

        report = upgrade.upgrade(kit=self.kit, project=self.project)

        self.assertEqual({relative: (self.root / relative).read_bytes() for relative in written}, crlf)
        self.assertNotIn("replaced", report.render())
        self.assertNotIn("the old ones are still in git", report.render())

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_a_project_folder_inside_the_kits_own_checkout_with_an_origin_is_no_clone_of_the_kit(self) -> None:
        inner = self.kit_root / "sandbox"
        fixtures.write(path=inner / "README.md", text="# Sandbox\n")
        commit_all(self.kit_root)
        subprocess.run(["git", "-C", str(self.kit_root), "remote", "add", "origin", "https://forge.example/example-org/agent-setup"], check=True)
        project = upgrade.Project(root=inner)

        upgrade.port(kit=self.kit, project=project)
        commit_all(self.kit_root)
        report = upgrade.upgrade(kit=self.kit, project=project)

        self.assertIn("0 written, 0 retired", report.render())

    def test_the_tools_tests_never_ship_and_an_earlier_copy_of_them_is_retired(self) -> None:
        upgrade.port(kit=self.kit, project=self.project)
        stale = self.root / ".agents" / "skills" / "agent-brain" / "scripts" / "tests" / "test_old.py"
        fixtures.write(path=stale, text="print('old')\n")
        fixtures.write(path=stale.parent / "__pycache__" / "test_old.cpython-39.pyc", text="bytecode\n")
        kept = self.root / ".agents" / "skills" / "agent-brain" / "scripts" / "brainlib" / "__pycache__" / "keep.txt"
        fixtures.write(path=kept, text="not bytecode\n")

        report = upgrade.upgrade(kit=self.kit, project=self.project)

        self.assertFalse(stale.exists())
        self.assertFalse(stale.parent.exists())
        self.assertTrue(kept.is_file())
        self.assertIn("retired .agents/skills/agent-brain/scripts/tests/test_old.py", report.render())

    def test_a_read_only_style_copy_is_replaced_by_the_doctor_not_written_through(self) -> None:
        upgrade.port(kit=self.kit, project=self.project)
        copy = self.root / ".codex" / "plain-and-short.md"
        copy.write_text("stale\n", encoding="utf-8")
        copy.chmod(0o444)

        try:
            report = upgrade.upgrade(kit=self.kit, project=self.project)
        finally:
            copy.chmod(0o644)

        self.assertTrue(copy.read_text(encoding="utf-8").startswith("# Plain & Short\n"))
        self.assertNotIn("wrote .codex/plain-and-short.md", report.render())
        self.assertIn("Doctor is clean.", report.render())

    def test_a_kit_whose_agents_md_lacks_the_marker_is_refused_before_any_write(self) -> None:
        agents = self.kit_root / "AGENTS.md"
        agents.write_text(agents.read_text(encoding="utf-8").replace(upgrade.KIT_MARK + "\n\n", ""), encoding="utf-8")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.port(kit=self.kit, project=self.project)

        self.assertEqual(str(caught.exception), "the kit's AGENTS.md has no line saying where its words start, so section 0 has no place to end")
        self.assertFalse((self.root / ".agents").exists())

    def test_a_second_run_writes_nothing_and_says_so(self) -> None:
        upgrade.port(kit=self.kit, project=self.project)
        before = os.stat(self.root / ".agents" / "SYSTEM.md").st_mtime_ns

        report = upgrade.upgrade(kit=self.kit, project=self.project)

        self.assertEqual(os.stat(self.root / ".agents" / "SYSTEM.md").st_mtime_ns, before)
        self.assertIn("0 written, 0 retired", report.render())

    def test_a_file_the_kit_no_longer_ships_is_retired_after_the_new_ones_land(self) -> None:
        upgrade.port(kit=self.kit, project=self.project)
        old = self.root / ".agents" / "skills" / "agent-brain" / "scripts" / "brainlib" / "gone.py"
        fixtures.write(path=old, text="print('old')\n")

        report = upgrade.upgrade(kit=self.kit, project=self.project)

        self.assertFalse(old.exists())
        self.assertTrue((self.root / ".agents" / "skills" / "agent-brain" / "scripts" / "brain.py").is_file())
        self.assertIn("1 retired", report.render())

    def test_a_stale_file_under_the_review_skill_is_retired_too(self) -> None:
        upgrade.port(kit=self.kit, project=self.project)
        old = self.root / ".agents" / "skills" / "branch-review" / "references" / "old.md"
        fixtures.write(path=old, text="old\n")

        report = upgrade.upgrade(kit=self.kit, project=self.project)

        self.assertFalse(old.exists())
        self.assertIn("retired .agents/skills/branch-review/references/old.md", report.render())

    def test_a_symlinked_review_skill_folder_is_refused_too(self) -> None:
        upgrade.port(kit=self.kit, project=self.project)
        shared = Path(self.tmp.name) / "shared-review"
        fixtures.write(path=shared / "SKILL.md", text="shared\n")
        home = self.root / ".agents" / "skills" / "branch-review"
        shutil.rmtree(home)
        home.symlink_to(shared)

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=self.kit, project=self.project)

        self.assertEqual(str(caught.exception), ".agents/skills/branch-review is a symlink, replace it with a real folder first")

    def test_a_slip_under_the_review_skill_is_the_kits_own_too(self) -> None:
        skill = self.kit_root / ".agents" / "skills" / "branch-review" / "SKILL.md"
        skill.write_text(skill.read_text(encoding="utf-8").replace("\n\n", "\n\nA dash — the house style bans.\n\n", 1), encoding="utf-8")

        report = upgrade.port(kit=self.kit, project=self.project)

        self.assertEqual(report.findings, [])
        self.assertEqual([finding.path for finding in report.kits_own], ["skills/branch-review/SKILL.md"])

    def test_a_projects_own_skill_under_a_kit_name_is_refused_on_the_current_layout_too(self) -> None:
        upgrade.port(kit=self.kit, project=self.project)
        own = self.root / ".agents" / "skills" / "branch-review"
        shutil.rmtree(own)
        fixtures.write(path=own / "SKILL.md", text="---\nname: branch-review\ndescription: our own review\n---\n\nOurs.\n")
        fixtures.write(path=own / "checklist.md", text="ours too\n")
        fixtures.write(path=self.root / ".agents" / "VERSION", text="1.1.0\n")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=self.kit, project=self.project)

        self.assertEqual(
            str(caught.exception),
            "the kit now ships a skill named branch-review, and .agents/skills/branch-review is the project's own, "
            "move it aside first, or delete it if an interrupted upgrade left it",
        )
        self.assertEqual((own / "checklist.md").read_text(encoding="utf-8"), "ours too\n")
        self.assertEqual((self.root / ".agents" / "VERSION").read_text(encoding="utf-8"), "1.1.0\n")

    def test_a_projects_own_sub_agent_definition_under_a_kit_name_is_refused_not_replaced(self) -> None:
        upgrade.port(kit=self.kit, project=self.project)
        fixtures.write(path=self.root / ".agents" / "VERSION", text="1.1.0\n")
        shutil.rmtree(self.root / ".agents" / "skills" / "branch-review")
        own = self.root / ".claude" / "agents" / "review-verifier.md"
        fixtures.write(path=own, text="---\nname: review-verifier\ndescription: ours\n---\n\nOur own verifier.\n")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=self.kit, project=self.project)

        self.assertEqual(
            str(caught.exception),
            "the kit now ships a sub agent definition at .claude/agents/review-verifier.md with the branch-review skill, "
            "and the file there is the project's own, move it aside first, or delete it if an interrupted upgrade left it",
        )
        self.assertIn("Our own verifier", own.read_text(encoding="utf-8"))
        self.assertEqual((self.root / ".agents" / "VERSION").read_text(encoding="utf-8"), "1.1.0\n")

    def test_a_projects_own_definition_under_a_name_a_later_release_brought_is_refused_not_replaced(self) -> None:
        upgrade.port(kit=self.kit, project=self.project)
        fixtures.write(path=self.root / ".agents" / "VERSION", text="1.3.0\n")
        own = self.root / ".claude" / "agents" / "review-fixer.md"
        fixtures.write(path=own, text="---\nname: review-fixer\ndescription: ours\n---\n\nOur own fixer.\n")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=self.kit, project=self.project)

        self.assertEqual(
            str(caught.exception),
            "the kit now ships a sub agent definition at .claude/agents/review-fixer.md with the branch-review skill, "
            "and the file there is the project's own, move it aside first, or delete it if an interrupted upgrade left it",
        )
        self.assertIn("Our own fixer", own.read_text(encoding="utf-8"))
        self.assertEqual((self.root / ".agents" / "VERSION").read_text(encoding="utf-8"), "1.3.0\n")

    def test_a_projects_own_folder_where_the_kits_skill_link_goes_is_refused_before_the_first_write(self) -> None:
        upgrade.port(kit=self.kit, project=self.project)
        fixtures.write(path=self.root / ".agents" / "VERSION", text="1.1.0\n")
        shutil.rmtree(self.root / ".agents" / "skills" / "branch-review")
        own = self.root / ".claude" / "skills" / "branch-review"
        own.unlink()
        fixtures.write(path=own / "SKILL.md", text="---\nname: branch-review\ndescription: our own review\n---\n\nOurs.\n")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=self.kit, project=self.project)

        self.assertEqual(
            str(caught.exception),
            "the kit now ships a skill named branch-review, and .claude/skills/branch-review holds a folder of the project's own "
            "where the kit's skill link goes, move it aside first",
        )
        self.assertFalse((self.root / ".agents" / "skills" / "branch-review").exists())
        self.assertEqual((self.root / ".agents" / "VERSION").read_text(encoding="utf-8"), "1.1.0\n")

    def test_every_skill_folder_of_the_kit_checkout_ships_whatever_the_running_tool_knows(self) -> None:
        fixtures.write(path=self.kit_root / ".agents" / "skills" / "extra" / "SKILL.md", text="---\nname: extra\n---\n\nA skill a newer kit ships.\n")
        fixtures.write(path=self.kit_root / ".agents" / "skills" / "extra" / "references" / "notes.md", text="notes\n")
        fixtures.write(path=self.kit_root / ".agents" / "skills" / "scratch" / "draft.md", text="no skill file, not a skill\n")

        self.assertIn(Path("skills/extra/references/notes.md"), self.kit.brain_files())
        self.assertFalse(any("scratch" in str(path) for path in self.kit.brain_files()))

        upgrade.port(kit=self.kit, project=self.project)

        self.assertTrue((self.root / ".agents" / "skills" / "extra" / "SKILL.md").is_file())
        self.assertFalse((self.root / ".agents" / "skills" / "scratch").exists())

    def test_a_projects_own_folder_under_a_skill_the_running_tool_never_heard_of_is_refused_too(self) -> None:
        upgrade.port(kit=self.kit, project=self.project)
        own = self.root / ".agents" / "skills" / "extra"
        fixtures.write(path=own / "SKILL.md", text="---\nname: extra\ndescription: ours\n---\n\nOURS.\n")
        fixtures.write(path=self.kit_root / ".agents" / "skills" / "extra" / "SKILL.md", text="---\nname: extra\n---\n\nKIT.\n")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=self.kit, project=self.project)

        self.assertIn("the kit ships a skill named extra this tool does not know, and .agents/skills/extra differs from it", str(caught.exception))
        self.assertIn("OURS.", (own / "SKILL.md").read_text(encoding="utf-8"))

    def test_a_symlinked_skill_folder_is_refused_before_anything_is_touched(self) -> None:
        upgrade.port(kit=self.kit, project=self.project)
        shared = Path(self.tmp.name) / "shared"
        fixtures.write(path=shared / "SKILL.md", text="shared\n")
        home = self.root / ".agents" / "skills" / "agent-brain"
        shutil.rmtree(home)
        home.symlink_to(shared)
        (self.root / "CLAUDE.md").write_text("MY OWN WORDS\n", encoding="utf-8")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=self.kit, project=self.project)

        self.assertEqual(str(caught.exception), ".agents/skills/agent-brain is a symlink, replace it with a real folder first")
        self.assertEqual((shared / "SKILL.md").read_text(encoding="utf-8"), "shared\n")
        self.assertEqual((self.root / "CLAUDE.md").read_text(encoding="utf-8"), "MY OWN WORDS\n")

    def test_bytecode_under_the_skill_folder_is_neither_guarded_nor_counted(self) -> None:
        upgrade.port(kit=self.kit, project=self.project)
        cache = self.root / ".agents" / "skills" / "agent-brain" / "scripts" / "brainlib" / "__pycache__"
        fixtures.write(path=cache / "store.cpython-39.pyc", text="bytecode\n")

        report = upgrade.upgrade(kit=self.kit, project=self.project)

        self.assertIn("0 written, 0 retired", report.render())

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root ignores file modes")
    def test_a_kit_file_that_cannot_be_read_is_refused_by_name(self) -> None:
        (self.kit_root / ".agents" / "SYSTEM.md").chmod(0)

        try:
            with self.assertRaises(upgrade.UpgradeError) as caught:
                upgrade.port(kit=self.kit, project=self.project)
        finally:
            (self.kit_root / ".agents" / "SYSTEM.md").chmod(0o644)

        self.assertTrue(str(caught.exception).startswith("the kit's .agents/SYSTEM.md cannot be read"))


class Roots(ProjectCase):
    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_a_port_lands_at_the_top_of_the_repository_or_in_the_folder_itself(self) -> None:
        fixtures.write(path=self.root / "README.md", text="# App\n")
        commit_all(self.root)
        deep = self.root / "src" / "app"
        deep.mkdir(parents=True)
        loose = Path(self.tmp.name) / "loose"
        loose.mkdir()

        self.assertEqual(upgrade.port_root(deep), self.root.resolve())
        self.assertEqual(upgrade.port_root(loose), loose.resolve())

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_a_port_never_lands_in_the_home_folder(self) -> None:
        fixtures.write(path=self.root / "README.md", text="# Home\n")
        commit_all(self.root)
        deep = self.root / "notes"
        deep.mkdir()

        loose = Path(self.tmp.name) / "loose"
        loose.mkdir()

        with mock.patch.object(Path, "home", return_value=self.root):
            with self.assertRaises(upgrade.UpgradeError) as caught:
                upgrade.port_root(deep)

        with mock.patch.object(Path, "home", return_value=loose):
            with self.assertRaises(upgrade.UpgradeError):
                upgrade.port_root(loose)

        self.assertEqual(str(caught.exception), f"the project folder found around {deep} is your home folder, name the project with --project")

    def test_the_home_folder_is_known_by_its_identity_not_its_spelling(self) -> None:
        other_case = Path(str(self.root).replace("shell", "SHELL"))
        if not other_case.is_dir():
            self.skipTest("the file system tells the two spellings apart")

        with mock.patch.object(Path, "home", return_value=self.root):
            with self.assertRaises(upgrade.UpgradeError):
                upgrade.port_root(other_case)

    def test_an_upgrade_never_works_on_the_home_folder(self) -> None:
        fixtures.build_old_project(self.root)

        with mock.patch.object(Path, "home", return_value=self.root):
            with self.assertRaises(upgrade.UpgradeError) as caught:
                upgrade.upgrade_root(self.root / ".agent" / "docs")

        self.assertIn("is your home folder", str(caught.exception))

    def test_an_upgrade_works_on_the_nearest_folder_with_a_brain(self) -> None:
        fixtures.build_old_project(self.root)
        loose = Path(self.tmp.name) / "loose"
        loose.mkdir()

        self.assertEqual(upgrade.upgrade_root(self.root / ".agent" / "docs" / "foundation"), self.root.resolve())
        self.assertEqual(upgrade.upgrade_root(loose), loose.resolve())

    def test_a_named_project_has_to_be_there(self) -> None:
        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.named_root(str(self.root / "nope"))

        self.assertEqual(str(caught.exception), f"no folder at {self.root / 'nope'}, so --project names nothing")
        self.assertEqual(upgrade.named_root(str(self.root)), self.root.resolve())

    def test_the_report_opens_with_the_folder_it_works_on(self) -> None:
        fixtures.build_old_project(self.root)

        report = upgrade.upgrade(kit=KIT, project=self.project)

        self.assertEqual(report.lines[0], f"upgrading {self.root.resolve()}")


class Pieces(ProjectCase):
    def test_section_zero_runs_to_the_next_second_level_heading(self) -> None:
        zero = upgrade.section_zero_in(fixtures.OLD_RULES)

        self.assertTrue(zero.startswith("## 0. Project Rules\n"))
        self.assertIn("### 0.1 Gates\n", zero)
        self.assertTrue(zero.endswith("Run `flutter test` before a review.\n"))
        self.assertIsNone(upgrade.section_zero_in("# Rules\n\n## 1. Coding\n"))

    def test_section_zero_takes_the_heading_the_rulebook_names(self) -> None:
        zero = upgrade.section_zero_in("# CLAUDE.md\n\n## 0. Project\n\nMine.\n\n## 1. Coding\n")

        self.assertEqual(zero, "## 0. Project Rules\n\nMine.\n")

    def test_section_zero_ends_at_the_kits_marker_whatever_the_kit_says_after_it(self) -> None:
        zero = "## 0. Project Rules\n\nMine.\n"
        release_one = f"# AGENTS.md\n\n{upgrade.KIT_MARK}\n\nThe rules are in @.agents/SYSTEM.md.\n"
        release_two = f"# AGENTS.md\n\n{upgrade.KIT_MARK}\n\nEvery agent works by @.agents/SYSTEM.md, reworded.\n"

        first = upgrade.with_section_zero(text=release_one, zero=zero)
        second = upgrade.with_section_zero(text=release_two, zero=upgrade.section_zero_in(first))

        self.assertEqual(upgrade.section_zero_in(first), zero)
        self.assertEqual(upgrade.section_zero_in(second), zero)
        self.assertEqual(second, f"# AGENTS.md\n\n{zero}\n{release_two.split(chr(10), 2)[2]}")

    def test_section_zero_keeps_a_heading_inside_a_fence_and_skips_one_in_an_example(self) -> None:
        fenced = "# CLAUDE.md\n\n## 0. Project Rules\n\nGates:\n\n```bash\n## unit\nmake test\n```\n\n## 1. Coding\n"
        example = "# Notes\n\nAn example:\n\n```markdown\n## 0. Project Rules\n\nNot mine.\n```\n"

        self.assertEqual(upgrade.section_zero_in(fenced), "## 0. Project Rules\n\nGates:\n\n```bash\n## unit\nmake test\n```\n")
        self.assertIsNone(upgrade.section_zero_in(example))

    def test_an_empty_section_zero_says_nothing(self) -> None:
        self.assertIsNone(upgrade.section_zero_in("# AGENTS.md\n\n## 0. Project Rules\n\n## 1. Coding\n"))
        self.assertIsNone(upgrade.section_zero_in(f"# AGENTS.md\n\n## 0. Project Rules\n\n{upgrade.KIT_MARK}\n\nThe kit.\n"))

    def test_own_words_leave_out_the_title_the_comments_and_the_imports_but_not_inside_code(self) -> None:
        self.assertEqual(upgrade.own_words(KIT.text(Path("CLAUDE.md"))), "")
        self.assertEqual(upgrade.own_words(KIT.text(Path("GEMINI.md"))), "")
        self.assertEqual(upgrade.own_words(KIT.text(Path("AGENTS.md"))), "")
        self.assertEqual(upgrade.own_words("# CLAUDE.md\n\n<!-- mine -->\n\nRun `make test`.\n\n## Style\n\nTabs.\n"), "Run `make test`.\n\n## Style\n\nTabs.")
        self.assertEqual(
            upgrade.own_words("# CLAUDE.md\n\nStamped.\n\n```python\n@dataclass\nclass Config:\n    pass  # <!-- kept -->\n```\n"),
            "Stamped.\n\n```python\n@dataclass\nclass Config:\n    pass  # <!-- kept -->\n```",
        )
        self.assertEqual(upgrade.own_words("# CLAUDE.md\n\n<!-- run `make test` first -->\n\nOurs.\n"), "Ours.")
        self.assertEqual(upgrade.own_words("# CLAUDE.md\n\nRules: `SYSTEM.md` @.agents/SYSTEM.md\n\nOurs.\n"), "Rules: `SYSTEM.md` @.agents/SYSTEM.md\n\nOurs.")

    def test_a_comment_delimiter_inside_code_neither_cuts_prose_nor_hides_a_comment(self) -> None:
        straddling = "# CLAUDE.md\n\n<!-- note\n\nClose it with:\n\n```text\n-->\n```\n\nRun `make check`.\n"
        lone_opener = "# CLAUDE.md\n\n```markdown\n<!--\n```\n\n<!-- TODO: fill this in -->\n\nOurs.\n"

        self.assertEqual(upgrade.own_words(straddling), "<!-- note\n\nClose it with:\n\n```text\n-->\n```\n\nRun `make check`.")
        self.assertEqual(upgrade.own_words(lone_opener), "```markdown\n<!--\n```\n\n\n\nOurs.")

    def test_a_marker_quoted_inside_code_does_not_end_section_zero(self) -> None:
        text = f"# AGENTS.md\n\n## 0. Project Rules\n\nSee:\n\n```text\n{upgrade.KIT_MARK}\n```\n\nAnd `{upgrade.KIT_MARK}` inline.\n\n{upgrade.KIT_MARK}\n\nThe kit.\n"

        self.assertEqual(
            upgrade.section_zero_in(text),
            f"## 0. Project Rules\n\nSee:\n\n```text\n{upgrade.KIT_MARK}\n```\n\nAnd `{upgrade.KIT_MARK}` inline.\n",
        )

    def test_section_zero_in_a_file_the_first_kit_wrote_ends_where_its_words_stood(self) -> None:
        text = f"# AGENTS.md\n\n## 0. Project Rules\n\nMine.\n\n{upgrade.FIRST_KIT_WORDS}\nRules that belong to this project alone go in a `## 0. Project Rules`\nsection.\n"

        self.assertEqual(upgrade.section_zero_in(text), "## 0. Project Rules\n\nMine.\n")

    def test_the_first_kits_words_quoted_in_section_zero_do_not_end_it(self) -> None:
        quoted = f"## 0. Project Rules\n\nThe kit used to open with:\n\n```text\n{upgrade.FIRST_KIT_WORDS}```\n\nRun `make check` first.\n"
        first_line = "## 0. Project Rules\n\nThe rules for working in this repo are in @.agents/SYSTEM.md and every\nagent works by them, and so do we.\n"

        self.assertEqual(upgrade.section_zero_in(f"# AGENTS.md\n\n{quoted}\n{upgrade.KIT_MARK}\n\nThe kit.\n"), quoted)
        self.assertEqual(upgrade.section_zero_in(f"# AGENTS.md\n\n{quoted}\n## 1. Coding\n"), quoted)
        self.assertEqual(upgrade.section_zero_in(f"# AGENTS.md\n\n{first_line}\n## 1. Coding\n"), first_line)

    def test_section_zero_sits_between_the_title_and_the_kits_marker(self) -> None:
        text = upgrade.with_section_zero(text=f"# AGENTS.md\n\n{upgrade.KIT_MARK}\n\nThe rules.\n", zero="## 0. Project Rules\n\nMine.\n")

        self.assertEqual(text, f"# AGENTS.md\n\n## 0. Project Rules\n\nMine.\n\n{upgrade.KIT_MARK}\n\nThe rules.\n")

    def test_the_indent_is_read_off_a_config_whatever_its_line_breaks(self) -> None:
        kit = '{"hooks": {}}'

        for text in ('{\n    "mine": true\n}\n', '{\r\n    "mine": true\r\n}\r\n', '{\r    "mine": true\r}\r', '{\n\t"mine": true\n}\n'):
            merged = upgrade.merged_config(existing=text, kit=kit, where="x")
            self.assertTrue(merged.startswith("{\n" + ("\t" if "\t" in text else "    ") + '"mine"'), repr(text))

        self.assertTrue(upgrade.merged_config(existing="{}", kit=kit, where="x").startswith('{\n  "hooks"'))

    def test_a_finding_is_the_kits_own_in_a_brain_file_an_agent_file_or_under_a_skill_the_kit_ships_spelled_either_way(self) -> None:
        names = ["agent-brain", "branch-review", "extra"]
        shipped = {Path("SYSTEM.md"), Path("house-rules.md"), Path("spec/conventions.md"), Path("VERSION")}
        copied = {Path(".claude/output-styles/plain-and-short.md"), Path(".claude/agents/review-finder.md"), Path(".codex/plain-and-short.md")}

        def kits(path: str, check: str = "style", message: str = "carries a semicolon") -> bool:
            in_kits_words = {"AGENTS.md": Counter({"imports @gone.md, which is not there": 1}), "CLAUDE.md": Counter({"imports @x.md, which is not there": 1})}
            return upgrade.kits_own(finding=doctor.Finding(path=path, check=check, message=message), names=names, shipped=shipped, copied=copied, in_kits_words=in_kits_words)

        for path in ("skills/agent-brain/SKILL.md", "skills/branch-review/references/angles.md", ".agents/skills/branch-review", ".agent/skills/agent-brain/x.md", "skills/extra/SKILL.md", "house-rules.md", ".agents/spec/conventions.md", ".agent/SYSTEM.md", ".agents/VERSION", ".claude/output-styles/plain-and-short.md", ".claude/agents/review-finder.md", ".codex/plain-and-short.md"):
            with self.subTest(path=path):
                self.assertTrue(kits(path))

        for path in ("skills/deploy/SKILL.md", "docs/x.md", "skills", "spec/mine.md", ".agents/spec", ".claude/skills/branch-review", "CLAUDE.md", ".claude/settings.json", ".claude/agents/mine.md"):
            with self.subTest(path=path):
                self.assertFalse(kits(path))

        self.assertTrue(kits("CLAUDE.md", check="import", message="imports @x.md, which is not there"))
        self.assertFalse(kits("CLAUDE.md", check="import", message="cannot be read: permission denied"))
        self.assertFalse(kits("GEMINI.md", check="import", message="imports @x.md, which is not there"))
        self.assertTrue(kits("AGENTS.md", check="import", message="imports @gone.md, which is not there"))
        self.assertFalse(kits("AGENTS.md", check="import", message="imports @mine.md, which is not there"))

    def test_a_root_file_the_port_wrote_with_windows_line_ends_is_still_the_kits_copy(self) -> None:
        for name in ("CONTRIBUTING.md", ".agents/VERSION", ".codex/plain-and-short.md"):
            fixtures.write(path=self.root / name, text="")
            (self.root / name).write_bytes(KIT.text(Path(name)).replace("\n", "\r\n").encode("utf-8"))
        (self.root / "GEMINI.md").write_bytes(KIT.text(Path("GEMINI.md")).replace("Gemini", "Gemma").encode("utf-8"))

        for name in ("CONTRIBUTING.md", ".agents/VERSION", ".codex/plain-and-short.md"):
            with self.subTest(name=name):
                self.assertTrue(upgrade.same_as_kit(kit=KIT, project=self.project, relative=name))
        self.assertFalse(upgrade.same_as_kit(kit=KIT, project=self.project, relative="GEMINI.md"))
        self.assertFalse(upgrade.same_as_kit(kit=KIT, project=self.project, relative="CLAUDE.md"))

    def test_a_file_that_says_the_text_with_windows_line_ends_is_left_as_it_is_by_put_and_put_config(self) -> None:
        text = '{\n  "env": {}\n}\n'
        said = text.replace("\n", "\r\n").encode("utf-8")
        report = upgrade.Report()
        target = self.root / "said.json"

        for write in (upgrade.put, upgrade.put_config):
            with self.subTest(write=write.__name__):
                target.write_bytes(said)

                write(project=self.project, relative=Path("said.json"), text=text, report=report)

                self.assertEqual(target.read_bytes(), said)

        self.assertEqual(report.lines, [])

    def test_versions_compare_as_numbers(self) -> None:
        self.assertGreater(schema.as_tuple("1.10.0"), schema.as_tuple("1.9.0"))
        self.assertEqual(schema.as_tuple("1.0"), schema.as_tuple("1.0.0"))
        self.assertLess(schema.as_tuple("2.0"), schema.as_tuple("2.0.1"))
        self.assertEqual(schema.as_tuple("1.2.3.4"), (1, 2, 3, 4))
        for text in ("1.x", "", "1.²", "1.٢"):
            with self.assertRaises(ValueError):
                schema.as_tuple(text)

    def test_the_pending_migrations_are_the_ones_the_tree_calls_for(self) -> None:
        first = upgrade.MIGRATIONS[0]
        fixtures.build_old_project(self.root)

        self.assertEqual(upgrade.pending_migrations(project=self.project, want="1.0.0"), [first])
        self.assertEqual(upgrade.pending_migrations(project=self.project, want="2.0.0"), [first])
        self.assertEqual(upgrade.pending_migrations(project=self.project, want="0.5.0"), [])

        (self.root / ".agent").rename(self.root / ".agents")

        self.assertEqual(upgrade.pending_migrations(project=self.project, want="1.0.0"), [])

    @unittest.skipUnless(shutil.which("git"), "git is not on this machine")
    def test_an_ignored_file_that_differs_is_refused_with_a_remedy_git_can_do(self) -> None:
        bare = Path(self.tmp.name) / "bare"
        fixtures.write(path=bare / "README.md", text="# App\n")
        fixtures.write(path=bare / ".gitignore", text="AGENTS.md\n")
        commit_all(bare)
        project = upgrade.Project(root=bare)
        upgrade.port(kit=KIT, project=project)
        commit_all(bare)
        (bare / "AGENTS.md").write_text("# AGENTS.md\n\n## 0. Project Rules\n\nMine.\n", encoding="utf-8")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=KIT, project=project)

        self.assertEqual(
            str(caught.exception),
            "git ignores these and they would be written over or removed, move them out of the way first: AGENTS.md",
        )

    def test_a_migration_runs_while_its_tree_condition_holds_and_the_kit_is_at_least_its_version(self) -> None:
        seen: list[str] = []
        fake = upgrade.Migration(
            to=KIT.version(),
            says="fake",
            needed=lambda project: (project.root / "old-shape").is_file(),
            apply=lambda project, report: seen.append(str(project.root)),
        )
        far = upgrade.Migration(to="99.0.0", says="far", needed=lambda project: True, apply=lambda project, report: seen.append("far"))
        fixtures.build_old_project(self.root)
        fixtures.write(path=self.root / "old-shape", text="")

        with mock.patch.object(upgrade, "MIGRATIONS", (fake, far)):
            upgrade.upgrade(kit=KIT, project=self.project)
            (self.root / "old-shape").unlink()
            upgrade.upgrade(kit=KIT, project=self.project)

        self.assertEqual(seen, [str(self.root)])

    def test_a_stamp_that_is_not_a_version_is_refused(self) -> None:
        fixtures.build_old_project(self.root)
        (self.root / ".agent").rename(self.root / ".agents")
        fixtures.write(path=self.root / ".agents" / "VERSION", text="latest\n")

        with self.assertRaises(upgrade.UpgradeError) as caught:
            upgrade.upgrade(kit=KIT, project=self.project)

        self.assertEqual(str(caught.exception), "VERSION says 'latest', not a version like 1.0.0")

    def test_text_is_read_with_its_line_endings(self) -> None:
        path = self.root / "crlf.txt"
        path.write_bytes(b"one\r\ntwo\r\n")

        self.assertEqual(upgrade.read_as_is(path), "one\r\ntwo\r\n")


if __name__ == "__main__":
    unittest.main()
