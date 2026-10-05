#!/usr/bin/env python3
"""Fail on the advisories this commit adds, not on every one in uv.lock.

pip-audit on its own fails whenever the lock holds any known vulnerability,
so one published against a package already locked -- with or without a fix
-- failed every pull request until someone dealt with it, whatever the pull
request changed. Two Renovate PRs that each fixed one advisory could never
merge, since each still carried the other's. A pull request can answer for
what it changes, so that is what this asks.

It audits HEAD and HEAD's first parent and fails only on advisories HEAD has
that the parent does not. On a pull request, actions/checkout checks out
GitHub's merge commit, whose first parent is the target branch, so the parent
is "the base without this change"; on a push it is the previous commit. The
checkout therefore needs `fetch-depth: 2`. Advisories are compared by id: a
routine update that moves a package already flagged does not count as new.

Each side is audited the same way the job always has: the production
dependencies exported from its own uv.lock (`uv export --no-dev`), checked
hash-pinned with `--disable-pip`, so nothing is resolved or installed.
Advisories already on the parent are printed as warnings and do not fail the
run; Dependabot alerts, Renovate's security PRs and the weekly lock-file
maintenance still surface them.

The pip-audit to run comes from $PIP_AUDIT (e.g. `pip-audit@2.10.1`), and any
arguments are passed to both audits -- `--ignore-vuln <ID>` for an advisory
reviewed and found unreachable, with the reason beside it in ci.yml.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Advisory:
    id: str
    package: str
    versions: set[str] = field(default_factory=set)
    fixes: set[str] = field(default_factory=set)
    description: str = ""


def run(
    *args: str, cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True, check=False)


def git(*args: str) -> str:
    # GIT_LFS_SKIP_SMUDGE: the audit needs the lock, not LFS files, and the
    # checkout drops its credentials, so an LFS download would fail the run.
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    result = run("git", *args, cwd=Path.cwd(), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


def audit(tree: Path, pip_audit: str, extra: list[str]) -> dict[str, Advisory]:
    """Known vulnerabilities in the production dependencies of `tree`, by id."""
    with tempfile.TemporaryDirectory() as tmp:
        requirements = Path(tmp) / "requirements.txt"
        report_path = Path(tmp) / "report.json"
        export = run(
            "uv", "export", "--frozen", "--no-dev", "--no-emit-project",
            "--format", "requirements-txt", "-o", str(requirements),
            cwd=tree,
        )  # fmt: skip
        if export.returncode != 0:
            raise RuntimeError(f"uv export in {tree}: {export.stderr.strip()}")
        result = run(
            "uvx", pip_audit, "-r", str(requirements),
            "--require-hashes", "--disable-pip", "--progress-spinner", "off",
            "--format", "json", "--output", str(report_path), *extra,
            cwd=tree,
        )  # fmt: skip
        # pip-audit exits 1 when it finds anything, so the status says nothing;
        # only a report it could not write -- PyPI or the advisory service
        # down, a bad lock -- is an error, and that must fail, not pass.
        try:
            report = json.loads(report_path.read_text())
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"pip-audit produced no report in {tree}:\n{result.stderr.strip()}"
            ) from error

    advisories: dict[str, Advisory] = {}
    for dependency in report.get("dependencies", []):
        for vuln in dependency.get("vulns", []):
            advisory = advisories.setdefault(
                vuln["id"],
                Advisory(
                    vuln["id"],
                    dependency["name"],
                    description=vuln.get("description", "").split("\n")[0],
                ),
            )
            advisory.versions.add(dependency["version"])
            advisory.fixes.update(vuln.get("fix_versions", []))
    return advisories


def checkout_parent() -> Path | None:
    """The first parent, checked out where it can be audited, or None."""
    try:
        git("rev-parse", "--verify", "--quiet", "HEAD^1^{commit}")
    except RuntimeError:
        if git("rev-parse", "--is-shallow-repository") == "true":
            raise RuntimeError(
                "HEAD's parent is not in this shallow clone: check out with `fetch-depth: 2`"
            ) from None
        return None  # A root commit: there is nothing to compare against.
    tree = Path(tempfile.mkdtemp(prefix="audit-parent-"))
    git("worktree", "add", "--detach", "--quiet", str(tree), "HEAD^1")
    return tree


def describe(advisory: Advisory) -> str:
    fixed = ", ".join(sorted(advisory.fixes)) or "no fixed version"
    installed = ", ".join(sorted(advisory.versions))
    text = f"{advisory.id} {advisory.package} {installed} (fixed in: {fixed})"
    return f"{text}: {advisory.description}" if advisory.description else text


def main() -> int:
    pip_audit = os.environ.get("PIP_AUDIT")
    if not pip_audit:
        raise RuntimeError("set PIP_AUDIT to the pip-audit to run, e.g. pip-audit@2.10.1")
    extra = sys.argv[1:]

    head = audit(Path.cwd(), pip_audit, extra)
    parent: dict[str, Advisory] = {}
    parent_tree = checkout_parent()
    if parent_tree:
        try:
            parent = audit(parent_tree, pip_audit, extra)
        finally:
            git("worktree", "remove", "--force", str(parent_tree))
            shutil.rmtree(parent_tree, ignore_errors=True)

    added = [a for key, a in head.items() if key not in parent]
    existing = [a for key, a in head.items() if key in parent]
    for advisory in existing:
        print(
            f"::warning title=Existing advisory::{describe(advisory)} "
            "-- already on the parent commit, so not failing this run"
        )
    for advisory in added:
        print(f"::error title=New advisory::{describe(advisory)}")
    note = "" if parent_tree else " (no parent commit: every advisory counts as new)"
    print(f"{len(added)} new and {len(existing)} existing advisories{note}")
    return 1 if added else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as error:
        # Every failure fails the gate: a report that could not be produced
        # proves nothing about the lock.
        print(f"::error title=Audit failed::{error}")
        sys.exit(1)
