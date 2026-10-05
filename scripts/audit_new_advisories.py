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
scripts/audit-baseline.sh: the last commit a successful CI run passed on the
target branch, on a pull request as well as a push, so a push of several
commits, a run that failed and was followed by another, or a pull request
onto a tip that failed, cannot pass an advisory nothing compared against a
state without it. Unset (a local run), the baseline is HEAD's first parent --
outside CI only: in CI an unset $AUDIT_BASE means the step lost it, and fails
rather than quietly comparing against the parent. Set but empty, there is no
baseline and every advisory counts as new. Advisories are compared by package
and id: a routine update that moves a package already flagged does not count
as new, and an advisory the baseline had against one package does not vouch
for another it now reaches.

Each side is audited the same way the job always has: the production
dependencies exported from its own uv.lock (`uv export --no-dev`), checked
hash-pinned with `--disable-pip`, so nothing is resolved or installed.
Advisories already on the baseline are printed as warnings and do not fail
the run; Dependabot alerts, Renovate's security PRs and the weekly lock-file
maintenance still surface them. Dependencies pip-audit could not audit at all
are printed as warnings too, as its own table used to show them.

The report has to be what this reads: a `dependencies` list whose entries each
carry `vulns` or a `skip_reason`, and that covers every requirement exported.
Environment markers are stripped from the export first: pip-audit drops a
requirement whose marker is false where it runs, and it runs on the CI
runner's Python and architecture, not the image's, so a dependency only for
Python 3.13+ or only for aarch64 would otherwise ship unaudited. Anything
that still leaves a requirement out -- a newer pip-audit that renamed a
field, a format override, `--dry-run`, which writes an empty report and
exits 0 -- fails, rather than reading as clean.

The pip-audit to run comes from $PIP_AUDIT (e.g. `pip-audit@2.10.1`), and any
arguments are passed to both audits -- `--ignore-vuln <ID>` for an advisory
reviewed and found unreachable, with the reason beside it in ci.yml. They go
before this script's own `--format json --output <file>`, and argparse keeps
the last value given, so it is that order which guarantees the report this
reads; the refusal of -f/--format/-o/--output below is a clearer error for the
obvious spellings, not the guarantee (an abbreviation such as `--form` would
get past it, and lose to the later flag all the same).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Flags that would change the report's format or where it is written, which
# would leave this reading something it does not understand, or (--dry-run)
# leave it empty. The coverage check in audit() is what catches a report that
# skips requirements by any other route.
REFUSED = ("-f", "--format", "-o", "--output", "--dry-run", "-d")


@dataclass
class Advisory:
    id: str
    package: str
    versions: set[str] = field(default_factory=set)
    fixes: set[str] = field(default_factory=set)
    description: str = ""


@dataclass
class Report:
    # Keyed by (package, advisory id): an advisory against one package does
    # not vouch for another it reaches.
    advisories: dict[tuple[str, str], Advisory] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    packages: set[str] = field(default_factory=set)


def canonical(name: str) -> str:
    """A package name as PEP 503 compares them."""
    return re.sub(r"[-_.]+", "-", name).lower()


REQUIREMENT = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+?)(\s*;[^\\]*)?(\s*\\)?$")


def without_markers(requirements: str) -> str:
    """A `uv export` with every environment marker removed.

    pip-audit drops a requirement whose marker is false for the interpreter
    it runs on, and it runs on the CI runner's Python and architecture, not
    the image's: a dependency only for Python 3.13+, or only for aarch64,
    would ship unaudited. Auditing every locked package, wherever it would
    install, can only add findings.
    """
    lines = []
    for line in requirements.splitlines():
        match = REQUIREMENT.match(line)
        if match and match[3]:
            line = f"{match[1]}=={match[2]}{' ' + match[4].strip() if match[4] else ''}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def exported_requirements(requirements: str) -> set[str]:
    """Every package a `uv export` lists."""
    return {canonical(m[1]) for line in requirements.splitlines() if (m := REQUIREMENT.match(line))}


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
        parsed.packages.add(canonical(dependency["name"]))
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
                (canonical(dependency["name"]), vuln["id"]),
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
        requirements.write_text(without_markers(requirements.read_text()))
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
        missing = exported_requirements(requirements.read_text()) - report.packages
    if result.returncode not in (0, 1) or (result.returncode == 1) != bool(report.advisories):
        raise RuntimeError(
            f"pip-audit exited {result.returncode} with {len(report.advisories)} "
            f"advisories read in {tree}:\n{result.stderr.strip()}"
        )
    if missing:
        raise RuntimeError(
            f"pip-audit's report in {tree} leaves out {len(missing)} exported "
            f"requirement(s): {', '.join(sorted(missing))}"
        )
    return report


def baseline() -> str:
    """The commit to compare against, or '' for none."""
    if "AUDIT_BASE" in os.environ:
        base = os.environ["AUDIT_BASE"].strip()
        if base:
            git("rev-parse", "--verify", "--quiet", f"{base}^{{commit}}")
        return base
    # In CI the baseline comes from scripts/audit-baseline.sh, always, even
    # when it is empty. Unset there means the step lost it -- a rename, a
    # dropped `env:` -- and the parent is the comparison that let a push of
    # several commits through, so refuse rather than fall back to it.
    if os.environ.get("GITHUB_ACTIONS") == "true":
        raise RuntimeError(
            "AUDIT_BASE is not set: in CI it must come from scripts/audit-baseline.sh"
        )
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
