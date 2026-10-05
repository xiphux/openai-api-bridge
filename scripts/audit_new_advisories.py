#!/usr/bin/env python3
"""Fail on the advisories this change adds, not on every one in uv.lock.

pip-audit on its own fails whenever the lock holds any known vulnerability,
so one published against a package already locked -- with or without a fix
-- failed every pull request until someone dealt with it, whatever the pull
request changed. Two Renovate PRs that each fixed one advisory could never
merge, since each still carried the other's. A pull request can answer for
what it changes, so that is what this asks.

It audits HEAD and a baseline commit and fails only on advisories HEAD has
that the baseline does not. The baseline is $AUDIT_BASE, which CI sets from
scripts/audit-baseline.sh: on a pull request, the merge commit's first parent
(the target branch); on a push, the last commit a successful CI run passed on
that branch, so a push of several commits, or a run that failed and was
followed by another, cannot pass an advisory nothing compared against a state
without it. Unset (a local run), the baseline is HEAD's first parent; set but
empty, there is none and every advisory counts as new. Advisories are compared
by id: a routine update that moves a package already flagged does not count
as new. (An id that covered two packages would be masked on the second; no
such advisory has been seen in this lock.)

Each side is audited the same way the job always has: the production
dependencies exported from its own uv.lock (`uv export --no-dev`), checked
hash-pinned with `--disable-pip`, so nothing is resolved or installed.
Advisories already on the baseline are printed as warnings and do not fail
the run; Dependabot alerts, Renovate's security PRs and the weekly lock-file
maintenance still surface them. Dependencies pip-audit could not audit at all
are printed as warnings too, as its own table used to show them.

The report has to be what this reads: a `dependencies` list whose entries each
carry `vulns` or a `skip_reason`. Anything else -- a newer pip-audit that
renamed a field, a format override -- fails, rather than reading as clean.

The pip-audit to run comes from $PIP_AUDIT (e.g. `pip-audit@2.10.1`), and any
arguments are passed to both audits -- `--ignore-vuln <ID>` for an advisory
reviewed and found unreachable, with the reason beside it in ci.yml. Arguments
that change the report's format or destination are refused.
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
from typing import Any

# Flags that would change the report's format or where it is written, which
# would leave this reading something it does not understand.
REFUSED = ("-f", "--format", "-o", "--output")


@dataclass
class Advisory:
    id: str
    package: str
    versions: set[str] = field(default_factory=set)
    fixes: set[str] = field(default_factory=set)
    description: str = ""


@dataclass
class Report:
    advisories: dict[str, Advisory] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)


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


def parse(report: Any, tree: Path) -> Report:
    """The advisories in a pip-audit JSON report, refusing any other shape."""
    if not isinstance(report, dict) or not isinstance(report.get("dependencies"), list):
        raise RuntimeError(f"pip-audit's report in {tree} has no `dependencies` list")
    parsed = Report()
    for dependency in report["dependencies"]:
        if not isinstance(dependency, dict) or "name" not in dependency:
            raise RuntimeError(f"pip-audit's report in {tree} has an unreadable dependency")
        if "skip_reason" in dependency:
            parsed.skipped.append(f"{dependency['name']}: {dependency['skip_reason']}")
            continue
        vulns = dependency.get("vulns")
        if not isinstance(vulns, list):
            raise RuntimeError(
                f"pip-audit's report in {tree} has no `vulns` for {dependency['name']}"
            )
        for vuln in vulns:
            advisory = parsed.advisories.setdefault(
                vuln["id"],
                Advisory(
                    vuln["id"],
                    dependency["name"],
                    description=vuln.get("description", "").split("\n")[0],
                ),
            )
            advisory.versions.add(dependency["version"])
            advisory.fixes.update(vuln.get("fix_versions", []))
    return parsed


def audit(tree: Path, pip_audit: str, extra: list[str]) -> Report:
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
            *extra, "--format", "json", "--output", str(report_path),
            cwd=tree,
        )  # fmt: skip
        # pip-audit exits 1 when it finds anything and 0 when it does not.
        # Anything else, or a report it could not write -- PyPI or the
        # advisory service down, a bad lock -- is an error, and must fail.
        try:
            report = parse(json.loads(report_path.read_text()), tree)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise RuntimeError(
                f"pip-audit produced no report in {tree}:\n{result.stderr.strip()}"
            ) from error
    if result.returncode not in (0, 1) or (result.returncode == 1) != bool(report.advisories):
        raise RuntimeError(
            f"pip-audit exited {result.returncode} with {len(report.advisories)} "
            f"advisories read in {tree}:\n{result.stderr.strip()}"
        )
    return report


def baseline() -> str:
    """The commit to compare against, or '' for none."""
    if "AUDIT_BASE" in os.environ:
        base = os.environ["AUDIT_BASE"].strip()
        if base:
            git("rev-parse", "--verify", "--quiet", f"{base}^{{commit}}")
        return base
    try:
        return git("rev-parse", "--verify", "--quiet", "HEAD^1^{commit}")
    except RuntimeError:
        if git("rev-parse", "--is-shallow-repository") == "true":
            raise RuntimeError(
                "HEAD's parent is not in this shallow clone: check out with more history"
            ) from None
        return ""  # A root commit: there is nothing to compare against.


def audit_baseline(base: str, pip_audit: str, extra: list[str]) -> Report:
    tree = Path(tempfile.mkdtemp(prefix="audit-baseline-"))
    try:
        git("worktree", "add", "--detach", "--quiet", str(tree), base)
        try:
            return audit(tree, pip_audit, extra)
        finally:
            git("worktree", "remove", "--force", str(tree))
    finally:
        shutil.rmtree(tree, ignore_errors=True)


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
    refused = [a for a in extra if a.split("=", 1)[0] in REFUSED]
    if refused:
        raise RuntimeError(f"refusing {' '.join(refused)}: this reads pip-audit's JSON report")

    head = audit(Path.cwd(), pip_audit, extra)
    base = baseline()
    parent = audit_baseline(base, pip_audit, extra).advisories if base else {}

    added = [a for key, a in head.advisories.items() if key not in parent]
    existing = [a for key, a in head.advisories.items() if key in parent]
    for skipped in head.skipped:
        print(f"::warning title=Not audited::{skipped}")
    for advisory in existing:
        print(
            f"::warning title=Existing advisory::{describe(advisory)} "
            "-- already on the baseline commit, so not failing this run"
        )
    for advisory in added:
        print(f"::error title=New advisory::{describe(advisory)}")
    against = (
        f" against {base[:12]}" if base else " (no baseline commit: every advisory counts as new)"
    )
    print(f"{len(added)} new and {len(existing)} existing advisories{against}")
    return 1 if added else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as error:
        # Every failure fails the gate: a report that could not be produced
        # proves nothing about the lock.
        print(f"::error title=Audit failed::{error}")
        sys.exit(1)
