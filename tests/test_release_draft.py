"""The commit-subject grouping behind ``release_notes.py --draft``.

Named for the draft, not for release notes: what gets PUBLISHED comes from
CHANGELOG.md and is covered by tests/test_changelog.py. The file kept its old
name for one commit after the change that moved the notes onto the changelog,
which left the obvious place to look when release notes come out wrong holding
only the scaffolding path.

What survives here is that scaffolding: given a tag range, group its commits by
conventional-commit prefix so a changelog entry can be condensed from them by
hand.

It is still worth pinning. A miscategorised or dropped commit is invisible —
nothing else reports it — and a draft that quietly omits a feature is a
changelog entry that quietly omits it too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import release_notes as rn
from release_notes import OTHER, classify, render_draft

REPO = "xiphux/openai-api-bridge"


@pytest.mark.parametrize(
    ("subject", "heading", "text"),
    [
        ("feat(fal): add video", "🚀 Features", "**fal**: add video"),
        ("fix: stop a crash", "🐛 Bug fixes", "stop a crash"),
        ("perf(eviction): bound the scan", "⚡ Performance", "**eviction**: bound the scan"),
        ("docs: say why", "📝 Documentation", "say why"),
        ("test(comfyui): cover it", "🧪 Tests", "**comfyui**: cover it"),
        ("refactor: split it", "🏗️ Internals", "split it"),
        ("types: annotate", "🏗️ Internals", "annotate"),
        ("ci: pin actions", "⚙️ CI & build", "pin actions"),
        ("deps: raise the floor", "📦 Dependencies", "raise the floor"),
        # Dependabot commits every ecosystem as `ci:`; they belong with deps.
        (
            "ci: bump httpx2 from 2.10.0 to 2.12.0",
            "📦 Dependencies",
            "bump httpx2 from 2.10.0 to 2.12.0",
        ),
        # An unknown type, and the older unprefixed style, keep their subject.
        ("wibble: something", OTHER, "wibble: something"),
        ("Add docker image CI", OTHER, "Add docker image CI"),
        ("WIP", OTHER, "WIP"),
    ],
)
def test_classify(subject: str, heading: str, text: str) -> None:
    assert classify(subject) == (heading, text)


def test_a_breaking_change_is_marked() -> None:
    heading, text = classify("feat(api)!: drop the legacy field")
    assert heading == "🚀 Features"
    assert text.startswith("**Breaking** — ")


def test_draft_groups_in_section_order_and_keeps_every_commit() -> None:
    commits = [
        "Version 0.6.0",  # dropped: the tag says this
        "ci: pin actions",
        "feat: one",
        "Older style commit",
        "fix: two",
        "feat(scope): three",
    ]
    body = render_draft("v0.6.0", "v0.5.1", commits)

    assert body.index("🚀 Features") < body.index("🐛 Bug fixes") < body.index("⚙️ CI & build")
    assert body.index("⚙️ CI & build") < body.index(OTHER)
    assert "* one" in body and "* **scope**: three" in body
    assert "* two" in body
    assert "* Older style commit" in body
    assert "Version 0.6.0" not in body
    # No empty headings for sections nothing landed in.
    assert "📝 Documentation" not in body


def test_draft_names_the_range_it_covers() -> None:
    body = render_draft("v0.6.0", "v0.5.1", ["feat: one"])
    assert "v0.5.1..v0.6.0" in body


def test_draft_names_the_tag_alone_for_a_first_release() -> None:
    body = render_draft("v0.1.0", None, ["Initial commit"])
    assert "DRAFT for v0.1.0." in body


def test_draft_is_marked_as_something_to_condense_by_hand() -> None:
    """The draft is scaffolding. Published verbatim it would reintroduce
    exactly the noise the changelog exists to keep out, so the instruction
    travels with the output rather than living only in CLAUDE.md."""
    body = render_draft("v0.6.0", "v0.5.1", ["feat: one"])
    assert body.startswith("<!-- DRAFT")
    assert "condense" in body.lower()
    assert "same unreleased version" in body


def test_draft_says_so_when_a_range_carries_nothing() -> None:
    """A retag with no new commits still needs output; an empty one reads as
    a generation failure."""
    body = render_draft("v0.6.1", "v0.6.0", ["Version 0.6.1"])
    assert "No commits found for this range." in body


class TestDraftCommand:
    """`main()`'s --draft path, which nothing covered before.

    The cases above all call render_draft directly with literal commit lists,
    so main() and subjects() were never exercised and --draft shipped broken
    for the one thing it is for: drafting a version that is not tagged yet.
    """

    def test_drafting_an_untagged_version_reads_the_range_from_head(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The docstring's own example. This raised `fatal: ambiguous argument`
        # from git because previous..tag was handed over with tag not a ref.
        assert rn.main(["v99.0.0", "--draft"]) == 0
        out = capsys.readouterr().out
        assert out.startswith("<!-- DRAFT")
        assert "..HEAD." in out

    def test_drafting_an_already_tagged_version_uses_that_tag(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        tags = rn.git("tag", "--list", "v[0-9]*.[0-9]*.[0-9]*").splitlines()
        if not tags:
            pytest.skip("no tags in this checkout (CI clones at depth 1)")
        newest = tags[-1]
        assert rn.main([newest, "--draft"]) == 0
        out = capsys.readouterr().out
        assert f"..{newest}." in out
        assert "HEAD" not in out.splitlines()[0]
