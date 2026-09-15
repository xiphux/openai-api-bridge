"""Keep one open GitHub issue per available major upgrade.

    uv pip list --outdated --format json > outdated.json
    uv run --no-project --with pyyaml python scripts/major_upgrade_issues.py outdated.json [--dry-run]

Dependabot is configured to ignore majors, and has to be: it proposes only a
package's newest version, so a pending major would replace the minor and patch
updates that auto-merge. This is how majors are noticed instead — an issue to
act on, not a PR that blocks the others.

Only direct dependencies (pyproject.toml's ``dependencies`` and dependency
groups) are reported, matching what Dependabot proposes; a transitive major is
its parent's concern. Packages Dependabot groups together
(.github/dependabot.yml) share one issue, since they upgrade together. An
issue's body is refreshed when a newer major appears, and it is closed once
nothing in it is behind a major any more. Issues are matched by a marker
comment in the body, and a title edited by hand is kept until the issue's body
next changes (a newer major, or a new locked version in its table).

"Major" is Dependabot's definition, the first version component, so this and
the `ignore` rule cover exactly the same updates. A 0.x minor (0.23 -> 0.24) is
not one: Dependabot proposes it as a PR, and dependabot-automerge.yml either
merges it (allowlisted) or leaves it open, which is notice enough.
"""

from __future__ import annotations

import fnmatch
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

LABEL = "major-upgrade"

# Packages whose major is deliberately not the latest, with why. Listed here
# rather than hidden, so the reason is next to the exclusion. Empty today.
SKIP: dict[str, str] = {}


@dataclass(frozen=True)
class Pending:
    name: str
    current: str
    latest: str


def normalize(name: str) -> str:
    """PEP 503 name normalization, so `Foo_Bar` and `foo-bar` compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def major(version: str) -> int:
    match = re.match(r"\s*v?(\d+)", version)
    return int(match.group(1)) if match else 0


def direct_dependencies(pyproject: Path) -> set[str]:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    specs: list[str] = list(data["project"].get("dependencies", []))
    for group in data.get("dependency-groups", {}).values():
        specs.extend(s for s in group if isinstance(s, str))  # skip {include-group}
    names = set()
    for spec in specs:
        match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", spec)
        if match:
            names.add(normalize(match.group(1)))
    return names


def dependabot_groups(config: Path) -> list[tuple[str, list[str]]]:
    """(group name, glob patterns) from the uv entry in dependabot.yml."""
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    uv: dict[str, Any] = next((u for u in data["updates"] if u["package-ecosystem"] == "uv"), {})
    return [(name, list(g["patterns"])) for name, g in (uv.get("groups") or {}).items()]


def pending_upgrades(
    outdated: list[dict[str, Any]], direct: set[str], groups: list[tuple[str, list[str]]]
) -> dict[str, list[Pending]]:
    pending: dict[str, list[Pending]] = {}
    for info in outdated:
        name = normalize(info["name"])
        if name not in direct or name in SKIP:
            continue
        current, latest = info["version"], info["latest_version"]
        if major(latest) <= major(current):
            continue
        key = next(
            (g for g, patterns in groups if any(fnmatch.fnmatch(name, p) for p in patterns)),
            name,
        )
        pending.setdefault(key, []).append(Pending(name, current, latest))
    return pending


def issue_for(key: str, packages: list[Pending]) -> tuple[str, str]:
    from_major = min(major(p.current) for p in packages)
    to_major = max(major(p.latest) for p in packages)
    title = f"Major upgrade available: {key} {from_major} → {to_major}"
    rows = "\n".join(
        f"| [`{p.name}`](https://pypi.org/project/{p.name}/#history) | {p.current} | {p.latest} |"
        for p in sorted(packages, key=lambda p: p.name)
    )
    body = f"""<!-- {LABEL}:{key} -->
A new major version is out. Dependabot does not propose majors here (see
`.github/dependabot.yml`), so this issue stands in for the PR.

| Package | Locked | Latest |
|---|---|---|
{rows}

Read the release notes for breaking changes, upgrade on a branch
(`uv lock --upgrade-package <name>`, raising the floor in pyproject.toml if
needed), and let CI judge it. This issue closes itself once uv.lock is on the
latest major; the weekly check (`.github/workflows/upgrade-check.yml`) keeps it
current until then."""
    return title, body


def normalized_text(text: str) -> str:
    # GitHub may hand a body back with CRLF line endings or trimmed, which must
    # not count as a change or every run would rewrite every issue.
    return text.replace("\r\n", "\n").strip()


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def read_report(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        report = json.loads(text)
    except json.JSONDecodeError:
        # uv prints a failure (an index error, say) where the JSON would be.
        head = "\n".join(text.strip().splitlines()[:10]) or "(nothing)"
        sys.exit(f"uv pip list did not produce a JSON report. It printed:\n\n{head}")
    if not isinstance(report, list):
        sys.exit(f"Expected a JSON list from uv pip list, got {type(report).__name__}")
    return report


def main(argv: list[str]) -> int:
    paths = [a for a in argv if not a.startswith("--")]
    dry_run = "--dry-run" in argv
    if len(paths) != 1:
        print(__doc__, file=sys.stderr)
        return 2

    outdated = read_report(Path(paths[0]))
    direct = direct_dependencies(Path("pyproject.toml"))
    pending = pending_upgrades(outdated, direct, dependabot_groups(Path(".github/dependabot.yml")))

    if not dry_run:
        label = ["label", "create", LABEL, "--force", "--color", "d4c5f9"]
        gh(*label, "--description", "A major version upgrade is available")

    # Read even on a dry run (listing changes nothing), so the preview shows
    # which issues would be updated or closed, not only which would open.
    by_key: dict[str, dict[str, Any]] = {}
    listing = ["issue", "list", "--label", LABEL, "--state", "open", "--limit", "200"]
    for issue in json.loads(gh(*listing, "--json", "number,title,body")):
        match = re.search(rf"<!-- {LABEL}:(.+?) -->", issue["body"])
        if match:
            by_key[match.group(1)] = issue

    for key, packages in pending.items():
        title, body = issue_for(key, packages)
        existing = by_key.get(key)
        if dry_run:
            print(f"{'update' if existing else 'open'}: {title}")
            for p in packages:
                print(f"  {p.name} {p.current} -> {p.latest}")
        elif existing is None:
            gh("issue", "create", "--title", title, "--body", body, "--label", LABEL)
            print(f"Opened: {title}")
        elif normalized_text(existing["body"]) != normalized_text(body):
            # The title is only rewritten along with a body change (a newer
            # major, or a minor/patch moving the locked version), so a title
            # someone edited stays put until there is news.
            gh("issue", "edit", str(existing["number"]), "--title", title, "--body", body)
            print(f"Updated #{existing['number']}: {title}")
        else:
            print(f"Unchanged #{existing['number']}: {existing['title']}")

    for key, issue in by_key.items():
        if key in pending:
            continue
        if dry_run:
            print(f"close: #{issue['number']} {issue['title']}")
            continue
        comment = "Closing: nothing here is behind a major version any more."
        gh("issue", "close", str(issue["number"]), "--comment", comment)
        print(f"Closed #{issue['number']}: {issue['title']}")

    if not pending:
        print("No direct dependency is behind a major version.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
