"""Release notes for a tag, read from CHANGELOG.md.

    python scripts/release_notes.py v0.7.0 [--repo owner/name] [--previous v0.6.0]
    python scripts/release_notes.py --validate
    python scripts/release_notes.py v0.8.0 --draft

Writes Markdown to stdout: the container pull commands, that version's
CHANGELOG.md section, and a compare link. It EXITS NON-ZERO when the version has
no section, so a release that would say nothing fails in CI instead of
publishing.

``--validate`` checks the file's structure and is what ci.yml runs.

``--draft`` is the old behaviour of this script: group the tag's commit subjects
by their conventional-commit prefix. It is scaffolding for writing a changelog
entry by hand, never the published notes, because commit subjects cannot say
what a release actually delivered. v0.5.0's generated notes listed 17 separate
``fal``-scoped commits -- schema introspection, retry cooldowns, key rejection --
for what a user experienced as one thing: a new fal.ai backend. Commits within a
release also net out against each other, so a fix to a bug introduced two
commits earlier describes something no release ever carried.

Why not GitHub's ``generate_release_notes``: it lists *pull requests* only. This
repo's work lands as direct commits to main -- v0.6.0 held 44 commits and 2
Dependabot PRs, and the generated notes named the 2.

Standard library only, so the release and infra jobs can run it straight from a
checkout with no ``uv sync``.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"

# `## vX.Y.Z`, with an optional prerelease suffix.
_VERSION = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(?:-[0-9A-Za-z.-]+)?$")
_HEADING = re.compile(r"^##\s+(\S.*?)\s*$")
UNRELEASED = "Unreleased"


class GitError(RuntimeError):
    """A git command failed, carrying what git said about it."""


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    if result.returncode != 0:
        # Without this, capture_output swallows git's own message and the
        # caller sees only a CalledProcessError with an exit status — most
        # often for a tag that doesn't exist, which is worth saying plainly.
        raise GitError(f"git {' '.join(args)}: {result.stderr.strip() or result.returncode}")
    return result.stdout.strip()


# ---------------------------------------------------------------- changelog


def parse_changelog(text: str) -> list[tuple[str, str]]:
    """The file's ``## `` sections as (heading, body), in file order."""
    sections: list[tuple[str, list[str]]] = []
    for line in text.split("\n"):
        # `^##\s` cannot match `### `: the third `#` is not whitespace.
        heading = _HEADING.match(line)
        if heading:
            sections.append((heading.group(1), []))
        elif sections:
            sections[-1][1].append(line)
    return [(heading, "\n".join(body).strip()) for heading, body in sections]


def _version_key(heading: str) -> tuple[int, int, int] | None:
    match = _VERSION.match(heading)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def validate(text: str) -> list[str]:
    """Structural problems with the changelog. Empty means valid."""
    problems: list[str] = []
    first_line = text.split("\n")[0] if text else ""
    if not re.match(r"^#\s+Changelog\s*$", first_line):
        problems.append('the first line must be "# Changelog"')

    sections = parse_changelog(text)
    if not sections:
        problems.append('no "## " release sections found')
        return problems

    seen: set[str] = set()
    previous_key: tuple[int, int, int] | None = None
    for index, (heading, body) in enumerate(sections):
        if heading == UNRELEASED:
            # Only ever the top section: an Unreleased below a released
            # version would mean the entries under it had already shipped.
            if index != 0:
                problems.append(f'"{UNRELEASED}" must be the first section')
            continue

        key = _version_key(heading)
        if key is None:
            problems.append(f'"## {heading}" is neither "{UNRELEASED}" nor a vX.Y.Z version')
            continue
        if heading in seen:
            problems.append(f'"## {heading}" appears more than once')
        seen.add(heading)

        # Newest first, so a released section is never appended at the bottom.
        if previous_key is not None and previous_key <= key:
            problems.append(f'"## {heading}" is not below the version above it (newest first)')
        previous_key = key

        # The whole point: a released version with nothing under it publishes
        # the empty notes this file exists to prevent.
        if not body:
            problems.append(f'"## {heading}" has no entries')

    return problems


def section_for(text: str, tag: str) -> str:
    """The entries for one version, without its heading."""
    for heading, body in parse_changelog(text):
        if heading == tag:
            if not body:
                raise GitError(f'CHANGELOG.md\'s "## {tag}" section is empty.')
            return body
    raise GitError(
        f'CHANGELOG.md has no "## {tag}" section. '
        f'Rename "## {UNRELEASED}" to "## {tag}" before tagging.'
    )


def released_versions(text: str) -> list[str]:
    """Released versions in the changelog, in file order (newest first)."""
    return [heading for heading, _ in parse_changelog(text) if _VERSION.match(heading)]


def previous_version(text: str, tag: str) -> str | None:
    """The version released before ``tag``, read from the changelog.

    From the changelog rather than from git tags: the changelog is what this
    command already trusts, which keeps the compare link consistent with the
    entries printed above it.
    """
    versions = released_versions(text)
    if tag in versions:
        index = versions.index(tag)
        return versions[index + 1] if index + 1 < len(versions) else None

    # A tag not in the file yet (a dry run): the newest version below it.
    key = _version_key(tag)
    if key is None:
        return None
    for candidate in versions:
        candidate_key = _version_key(candidate)
        if candidate_key is not None and candidate_key < key:
            return candidate
    return None


def tag_exists(tag: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


# ------------------------------------------------------------------- draft

# Conventional-commit type → heading, in the order sections appear. Types not
# listed here fall into "Other changes" along with the older unprefixed
# commits, so nothing is dropped just because its subject doesn't parse.
SECTIONS: list[tuple[str, tuple[str, ...]]] = [
    ("🚀 Features", ("feat",)),
    ("🐛 Bug fixes", ("fix",)),
    ("⚡ Performance", ("perf",)),
    ("📝 Documentation", ("docs",)),
    ("🧪 Tests", ("test", "tests")),
    ("🏗️ Internals", ("refactor", "style", "types", "chore")),
    ("⚙️ CI & build", ("ci", "build")),
    ("📦 Dependencies", ("deps",)),
]
OTHER = "Other changes"

# "type(scope)!: subject"
_SUBJECT = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]*)\))?(?P<breaking>!)?:\s*(?P<rest>.+)$"
)
# Release chores ("Version 0.6.0"): the tag already says this.
_VERSION_BUMP = re.compile(r"^Version\s+\d+\.\d+", re.IGNORECASE)
# A Dependabot bump is filed under Dependencies whatever prefix it carries: it
# commits as `ci:` for every ecosystem, which would bury it under CI & build.
_BUMP = re.compile(r"^bump\s+\S+\s+from\s+\S+\s+to\s+\S+", re.IGNORECASE)


def subjects(previous: str | None, tag: str) -> list[str]:
    """Commit subjects in ``tag`` and not in ``previous``, newest first.

    Merge commits are left out: a dependency merge repeats the bump commit
    already listed, and says nothing the branch's own commits don't.
    """
    span = f"{previous}..{tag}" if previous else tag
    return [s for s in git("log", "--no-merges", "--format=%s", span).splitlines() if s]


def classify(subject: str) -> tuple[str, str]:
    """(section heading, text to show) for one commit subject."""
    match = _SUBJECT.match(subject)
    if not match:
        return OTHER, subject
    kind = match.group("type")
    rest = match.group("rest")
    scope = match.group("scope")
    text = f"**{scope}**: {rest}" if scope else rest
    if match.group("breaking"):
        text = f"**Breaking** — {text}"
    if _BUMP.match(rest):
        return "📦 Dependencies", text
    for heading, types in SECTIONS:
        if kind in types:
            return heading, text
    return OTHER, subject


def render_draft(tag: str, previous: str | None, commits: list[str]) -> str:
    """A starting point for a hand-written changelog entry, not release notes."""
    grouped: dict[str, list[str]] = {}
    for subject in commits:
        if _VERSION_BUMP.match(subject):
            continue
        heading, text = classify(subject)
        grouped.setdefault(heading, []).append(text)

    span = f"{previous}..{tag}" if previous else tag
    lines = [
        f"<!-- DRAFT for {span}. Condense into CHANGELOG.md by hand: one entry per",
        "     user-visible outcome, nothing for work a user cannot perceive, and no",
        "     entry for a fix to a bug introduced in this same unreleased version. -->",
        "",
    ]
    for heading, _ in [*SECTIONS, (OTHER, ())]:
        entries = grouped.get(heading)
        if not entries:
            continue
        lines.append(f"### {heading}")
        lines.extend(f"* {e}" for e in entries)
        lines.append("")
    if not grouped:
        lines.append("No commits found for this range.")
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------ render


def render(tag: str, previous: str | None, repo: str, entries: str) -> str:
    version = tag.lstrip("v")
    image = f"ghcr.io/{repo}"
    compare = (
        f"https://github.com/{repo}/compare/{previous}...{tag}"
        if previous
        else f"https://github.com/{repo}/commits/{tag}"
    )
    return (
        "\n".join(
            [
                "## Container images",
                "",
                "```bash",
                f"docker pull {image}:{version}",
                f"docker pull {image}:latest",
                "```",
                "",
                "Multi-arch: `linux/amd64`, `linux/arm64`.",
                "",
                "## What's changed",
                "",
                entries,
                "",
                f"**Full Changelog**: {compare}",
            ]
        )
        + "\n"
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", nargs="?", help="the tag being released, e.g. v0.7.0")
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", "xiphux/openai-api-bridge"),
        help="owner/name, for links and the image path",
    )
    parser.add_argument("--previous", help="override the tag to compare against")
    parser.add_argument(
        "--validate", action="store_true", help="check CHANGELOG.md's structure and exit"
    )
    parser.add_argument(
        "--draft",
        action="store_true",
        help="print a commit-derived draft to condense by hand, not release notes",
    )
    args = parser.parse_args(argv)

    try:
        text = CHANGELOG.read_text(encoding="utf-8")
    except OSError:
        print("CHANGELOG.md not found.", file=sys.stderr)
        return 1

    if args.validate:
        problems = validate(text)
        for problem in problems:
            print(f"CHANGELOG.md: {problem}", file=sys.stderr)
        if problems:
            return 1
        print("CHANGELOG.md is well-formed.")
        return 0

    if not args.tag:
        parser.error("a tag is required unless --validate is given")

    previous = args.previous or previous_version(text, args.tag)

    if args.draft:
        # A draft is written BEFORE the version is tagged -- that is the whole
        # point of it, since the entry goes under `## Unreleased` and the
        # version is not named until the bump commit. So `args.tag` is normally
        # not a ref yet, and `previous..tag` handed straight to git failed with
        # a raw `fatal: ambiguous argument` on the exact example the docstring
        # gives. Read the range from HEAD when the target is not a ref, and drop
        # a `previous` that was never tagged, as the notes path below does.
        head = args.tag if tag_exists(args.tag) else "HEAD"
        if previous and not tag_exists(previous):
            previous = None
        try:
            commits = subjects(previous, head)
        except GitError as e:
            print(str(e), file=sys.stderr)
            return 1
        # `head`, not `args.tag`: render_draft rebuilds the span for its own
        # header, so passing the tag would label the output with a range the
        # commits did not come from.
        sys.stdout.write(render_draft(head, previous, commits))
        return 0

    # A version listed in the changelog but never tagged has no compare
    # endpoint; fall back to the commit list rather than link a 404.
    if previous and not tag_exists(previous):
        previous = None
    try:
        entries = section_for(text, args.tag)
    except GitError as e:
        print(str(e), file=sys.stderr)
        return 1
    sys.stdout.write(render(args.tag, previous, args.repo, entries))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
