"""CHANGELOG.md is the only source of release notes, so the parser that reads it
gates every release.

A malformed file that still parses would publish the empty notes the changelog
was introduced to stop, which is why ``validate`` carries as many cases here as
the extractor does. The committed CHANGELOG.md is checked too: a broken one
fails the same way in CI, but failing here as well names the problem while you
are still editing.

``scripts/`` is deliberately outside the installed package — it runs with no
``uv sync`` — so the module is loaded by path rather than imported.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_release_notes() -> ModuleType:
    path = ROOT / "scripts" / "release_notes.py"
    spec = importlib.util.spec_from_file_location("release_notes", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["release_notes"] = module
    spec.loader.exec_module(module)
    return module


rn = _load_release_notes()


VALID = "\n".join(
    [
        "# Changelog",
        "",
        "Prose above the first heading is not part of any section.",
        "",
        "## Unreleased",
        "",
        "### Added",
        "",
        "- something pending",
        "",
        "## v0.2.0",
        "",
        "### Fixed",
        "",
        "- a released fix",
        "",
        "## v0.1.0",
        "",
        "- the first release",
        "",
    ]
)


class TestParse:
    def test_splits_on_version_headings_in_file_order(self) -> None:
        assert [h for h, _ in rn.parse_changelog(VALID)] == ["Unreleased", "v0.2.0", "v0.1.0"]

    def test_subheadings_are_not_sections(self) -> None:
        headings = [h for h, _ in rn.parse_changelog(VALID)]
        assert "Added" not in headings
        assert "Fixed" not in headings

    def test_subheadings_stay_in_the_body(self) -> None:
        _, body = rn.parse_changelog(VALID)[0]
        assert "### Added" in body
        assert "- something pending" in body

    def test_preamble_above_the_first_heading_is_dropped(self) -> None:
        assert not any("Prose above" in body for _, body in rn.parse_changelog(VALID))


class TestValidate:
    def test_accepts_a_well_formed_file(self) -> None:
        assert rn.validate(VALID) == []

    def test_rejects_a_released_version_with_no_entries(self) -> None:
        text = "# Changelog\n\n## v0.2.0\n\n## v0.1.0\n\n- entry\n"
        assert '"## v0.2.0" has no entries' in rn.validate(text)

    def test_allows_an_empty_unreleased_section(self) -> None:
        # An empty Unreleased just means nothing is pending.
        text = "# Changelog\n\n## Unreleased\n\n## v0.1.0\n\n- entry\n"
        assert rn.validate(text) == []

    def test_rejects_versions_that_are_not_newest_first(self) -> None:
        text = "# Changelog\n\n## v0.1.0\n\n- a\n\n## v0.2.0\n\n- b\n"
        assert '"## v0.2.0" is not below the version above it (newest first)' in rn.validate(text)

    def test_rejects_a_duplicated_version(self) -> None:
        text = "# Changelog\n\n## v0.2.0\n\n- a\n\n## v0.2.0\n\n- b\n"
        assert '"## v0.2.0" appears more than once' in rn.validate(text)

    def test_rejects_a_heading_that_is_neither_unreleased_nor_a_version(self) -> None:
        text = "# Changelog\n\n## Release three\n\n- a\n"
        expected = '"## Release three" is neither "Unreleased" nor a vX.Y.Z version'
        assert expected in rn.validate(text)

    def test_rejects_unreleased_below_a_released_version(self) -> None:
        text = "# Changelog\n\n## v0.1.0\n\n- a\n\n## Unreleased\n\n- b\n"
        assert '"Unreleased" must be the first section' in rn.validate(text)

    def test_rejects_a_file_that_does_not_start_with_the_title(self) -> None:
        text = "# Release notes\n\n## v0.1.0\n\n- a\n"
        assert 'the first line must be "# Changelog"' in rn.validate(text)

    def test_rejects_a_file_with_no_sections(self) -> None:
        assert 'no "## " release sections found' in rn.validate("# Changelog\n\nnothing\n")

    def test_accepts_a_prerelease_version(self) -> None:
        assert rn.validate("# Changelog\n\n## v1.0.0-rc.1\n\n- a\n") == []


class TestSectionFor:
    def test_returns_only_that_version(self) -> None:
        body = rn.section_for(VALID, "v0.2.0")
        assert "- a released fix" in body
        assert "v0.1.0" not in body
        assert "## v0.2.0" not in body

    def test_raises_for_a_missing_version_and_names_the_fix(self) -> None:
        with pytest.raises(rn.GitError, match=r'no "## v0\.3\.0" section'):
            rn.section_for(VALID, "v0.3.0")
        with pytest.raises(rn.GitError, match=r'Rename "## Unreleased"'):
            rn.section_for(VALID, "v0.3.0")

    def test_raises_for_an_empty_section(self) -> None:
        text = "# Changelog\n\n## v0.2.0\n\n## v0.1.0\n\n- entry\n"
        with pytest.raises(rn.GitError, match="is empty"):
            rn.section_for(text, "v0.2.0")


class TestPreviousVersion:
    def test_is_the_next_version_down(self) -> None:
        assert rn.previous_version(VALID, "v0.2.0") == "v0.1.0"

    def test_is_none_for_the_oldest_release(self) -> None:
        assert rn.previous_version(VALID, "v0.1.0") is None

    def test_skips_unreleased(self) -> None:
        assert rn.released_versions(VALID) == ["v0.2.0", "v0.1.0"]

    def test_finds_the_newest_version_below_an_untagged_version(self) -> None:
        assert rn.previous_version(VALID, "v0.3.0") == "v0.2.0"


class TestRender:
    def test_carries_the_pull_commands_for_that_version(self) -> None:
        body = rn.render("v0.2.0", "v0.1.0", "xiphux/openai-api-bridge", "- a fix")
        assert "docker pull ghcr.io/xiphux/openai-api-bridge:0.2.0" in body
        assert "docker pull ghcr.io/xiphux/openai-api-bridge:latest" in body

    def test_carries_the_entries(self) -> None:
        body = rn.render("v0.2.0", "v0.1.0", "xiphux/openai-api-bridge", "- a fix")
        assert "- a fix" in body

    def test_links_a_compare_view_when_there_is_a_previous_release(self) -> None:
        body = rn.render("v0.2.0", "v0.1.0", "xiphux/openai-api-bridge", "- a fix")
        assert "/compare/v0.1.0...v0.2.0" in body

    def test_links_the_commit_list_for_a_first_release(self) -> None:
        body = rn.render("v0.1.0", None, "xiphux/openai-api-bridge", "- first")
        assert "/commits/v0.1.0" in body
        assert "/compare/" not in body


class TestCommittedChangelog:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    def test_is_well_formed(self) -> None:
        assert rn.validate(self.text) == []

    def test_has_an_entry_for_every_released_tag(self) -> None:
        # Release notes are generated per tag, so a tag with no section is a
        # release that would have failed to publish.
        #
        # Skipped rather than silently vacuous where there are no tags to
        # check: CI checks out at depth 1, so this asserts nothing there. It is
        # a local guard, which is where the backfill was written.
        tags = set(rn.git("tag", "--list", "v[0-9]*.[0-9]*.[0-9]*").splitlines())
        if not tags:
            pytest.skip("no tags in this checkout (CI clones at depth 1)")
        documented = set(rn.released_versions(self.text))
        assert tags - documented == set()

    def test_has_an_entry_for_the_current_project_version(self) -> None:
        # Exactly the check docker.yml's release job runs, rather than a weaker
        # restatement of it: section_for also rejects a section that exists but
        # is empty, and raises GitError naming the fix.
        #
        # This holds continuously because the version here moves only at
        # release -- the `Version X.Y.Z` commit IS the tagged commit -- so
        # between releases pyproject.toml names the last RELEASED version,
        # whose section exists. A pending `## Unreleased` above it is
        # irrelevant, since this asks whether the section exists at all, not
        # where it sits. The window where it fails is the one commit that bumps
        # the version without renaming Unreleased, which is the mistake worth
        # catching.
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        version = next(
            line.split("=", 1)[1].strip().strip('"')
            for line in pyproject.splitlines()
            if line.startswith("version =")
        )
        rn.section_for(self.text, f"v{version}")
