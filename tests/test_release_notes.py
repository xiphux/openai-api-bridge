"""The release-notes generator: how commit subjects become the published notes.

The notes are what the release page says shipped, and nothing else reports a
mistake here — a miscategorised or dropped commit just isn't mentioned. These
pin the classification and the rendering; the git plumbing around them is
exercised by running the script against this repo's real tags.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from release_notes import OTHER, classify, previous_tag, render

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


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("v0.6.0", "v0.5.1"),
        ("v0.1.0", None),  # the first release compares against nothing
        ("v0.7.0", "v0.6.0"),  # not yet tagged: the newest below it
        ("v0.5.2", "v0.5.1"),  # a patch that skipped a minor still lands right
    ],
)
def test_previous_tag(tag: str, expected: str | None) -> None:
    tags = ["v0.1.0", "v0.2.0", "v0.5.0", "v0.5.1", "v0.6.0"]
    assert previous_tag(tag, tags) == expected


def test_render_groups_in_section_order_and_keeps_every_commit() -> None:
    commits = [
        "Version 0.6.0",  # dropped: the tag says this
        "ci: pin actions",
        "feat: one",
        "Older style commit",
        "fix: two",
        "feat(scope): three",
    ]
    body = render("v0.6.0", "v0.5.1", REPO, commits)

    assert body.index("🚀 Features") < body.index("🐛 Bug fixes") < body.index("⚙️ CI & build")
    assert body.index("⚙️ CI & build") < body.index(OTHER)
    assert "* one" in body and "* **scope**: three" in body
    assert "* two" in body
    assert "* Older style commit" in body
    assert "Version 0.6.0" not in body.split("## What's changed")[1]
    # No empty headings for sections nothing landed in.
    assert "📝 Documentation" not in body


def test_render_names_the_images_and_the_compare_link() -> None:
    body = render("v0.6.0", "v0.5.1", REPO, ["feat: one"])

    assert f"docker pull ghcr.io/{REPO}:0.6.0" in body
    assert f"docker pull ghcr.io/{REPO}:latest" in body
    assert body.rstrip().endswith(f"https://github.com/{REPO}/compare/v0.5.1...v0.6.0")


def test_render_links_to_the_commit_list_for_a_first_release() -> None:
    body = render("v0.1.0", None, REPO, ["Initial commit"])
    assert body.rstrip().endswith(f"https://github.com/{REPO}/commits/v0.1.0")


def test_render_says_so_when_a_release_carries_nothing() -> None:
    """A retag with no new commits still needs a body; an empty one reads as
    a generation failure."""
    body = render("v0.6.1", "v0.6.0", REPO, ["Version 0.6.1"])
    assert "No changes recorded for this release." in body
