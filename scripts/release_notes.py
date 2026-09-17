"""Release notes for a tag, built from the commits it contains.

    python scripts/release_notes.py v0.6.0 [--repo owner/name] [--previous v0.5.1]

Writes Markdown to stdout: the container pull commands, the commits since the
previous semver tag grouped by their conventional-commit prefix, and a compare
link.

Why not GitHub's ``generate_release_notes``: it lists *pull requests* only. This
repo's work lands as direct commits to main — v0.6.0 held 44 commits and 2
Dependabot PRs, and the generated notes named the 2 — so the release said
nothing about what shipped. Reading the commits is the only source that sees
everything.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

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


def semver_tags() -> list[str]:
    """Tags of the form vX.Y.Z, oldest first."""
    out = git("tag", "--list", "v[0-9]*.[0-9]*.[0-9]*", "--sort=v:refname")
    return [t for t in out.splitlines() if t]


def previous_tag(tag: str, tags: list[str]) -> str | None:
    """The tag before ``tag``, or None for the first release."""
    if tag not in tags:
        # An unreleased tag (a dry run before pushing): compare against the
        # newest tag that sorts below it.
        earlier = [t for t in tags if _version_key(t) < _version_key(tag)]
        return earlier[-1] if earlier else None
    index = tags.index(tag)
    return tags[index - 1] if index else None


def _version_key(tag: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", tag)[:3])


def subjects(previous: str | None, tag: str) -> list[str]:
    """Commit subjects in ``tag`` and not in ``previous``, newest first.

    Merge commits are left out: a Dependabot merge repeats the bump commit
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


def render(tag: str, previous: str | None, repo: str, commits: list[str]) -> str:
    version = tag.lstrip("v")
    image = f"ghcr.io/{repo}"
    grouped: dict[str, list[str]] = {}
    for subject in commits:
        if _VERSION_BUMP.match(subject):
            continue
        heading, text = classify(subject)
        grouped.setdefault(heading, []).append(text)

    lines = [
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
    ]
    for heading, _ in [*SECTIONS, (OTHER, ())]:
        entries = grouped.get(heading)
        if not entries:
            continue
        lines.append(f"### {heading}")
        lines.extend(f"* {e}" for e in entries)
        lines.append("")
    if not grouped:
        lines.append("No changes recorded for this release.")
        lines.append("")

    if previous:
        lines.append(f"**Full Changelog**: https://github.com/{repo}/compare/{previous}...{tag}")
    else:
        lines.append(f"**Full Changelog**: https://github.com/{repo}/commits/{tag}")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="the tag being released, e.g. v0.6.0")
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", "xiphux/openai-api-bridge"),
        help="owner/name, for links and the image path",
    )
    parser.add_argument("--previous", help="override the tag to compare against")
    args = parser.parse_args(argv)

    try:
        previous = args.previous or previous_tag(args.tag, semver_tags())
        commits = subjects(previous, args.tag)
    except GitError as e:
        sys.exit(str(e))
    sys.stdout.write(render(args.tag, previous, args.repo, commits))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
