"""The dependency audit's scripts, and the workflow that runs them.

CI fails a change only for vulnerabilities it adds over a baseline commit, so
what decides the baseline (scripts/audit-baseline.sh), what counts as added
(scripts/audit_new_advisories.py) and how the workflow wires the two together
are the whole of the gate. Earlier versions compared a push only with its
parent, compared a pull request with the target branch's tip, and read a
pip-audit report of the wrong shape as clean; each let a vulnerability
through, and each is pinned below.

The scripts run for real against throwaway repositories. `uv`, `uvx` and
`gh` are fakes first on PATH: `uvx pip-audit` writes the audit-report.json
committed in whichever tree it runs in (so the baseline's worktree reports
its own) and exits as pip-audit would, and `gh` serves canned run lists per
branch.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "scripts" / "audit_new_advisories.py"
BASELINE = ROOT / "scripts" / "audit-baseline.sh"
WORKFLOWS = ROOT / ".github" / "workflows"

FAKES = {
    # uv export ... -o <file>: an empty requirements file is all the fake
    # pip-audit needs.
    "uv": 'for a; do [ "$prev" = -o ] && : > "$a"; prev=$a; done',
    # uvx pip-audit ... --output <file>: this tree's report, exit 1 if it
    # holds a vulnerability; FAKE_RC overrides the exit status.
    "uvx": "\n".join(
        [
            'for a; do [ "$prev" = --output ] && out=$a; prev=$a; done',
            'cp audit-report.json "$out"',
            'if [ -n "${FAKE_RC:-}" ]; then exit "$FAKE_RC"; fi',
            "grep -q '\"id\"' audit-report.json && exit 1",
            "exit 0",
        ]
    ),
    # gh api ... -f branch=<b> ...: the lines in $FAKE_RUNS/<b>.
    "gh": "\n".join(
        [
            '[ -n "${FAKE_GH_FAIL:-}" ] && { echo "gh: HTTP 500" >&2; exit 1; }',
            "for a; do case $a in branch=*) b=${a#branch=};; esac; done",
            'cat "$FAKE_RUNS/$b" 2>/dev/null || true',
        ]
    ),
}


@pytest.fixture(scope="module")
def fake_bin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("bin")
    for name, body in FAKES.items():
        path = directory / name
        path.write_text(f"#!/usr/bin/env bash\n{body}\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return directory


def report(*vulns: str, skipped: str | None = None) -> dict[str, Any]:
    dependencies: list[dict[str, Any]] = [
        {
            "name": "pkg",
            "version": "1.0",
            "vulns": [
                {"id": v, "fix_versions": ["2.0"], "description": f"{v} desc"} for v in vulns
            ],
        }
    ]
    if skipped:
        dependencies.append({"name": skipped, "skip_reason": "not on PyPI"})
    return {"dependencies": dependencies}


class Repo:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "test")
        self.git("config", "commit.gpgsign", "false")

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=self.path, check=True, text=True, capture_output=True
        ).stdout.strip()

    def commit(self, content: dict[str, Any], message: str) -> str:
        (self.path / "audit-report.json").write_text(json.dumps(content))
        (self.path / "uv.lock").write_text(f"# {message}\n")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    (tmp_path / "repo").mkdir()
    return Repo(tmp_path / "repo")


def run_audit(repo: Repo, fake_bin: Path, *args: str, **env: str) -> tuple[int, str]:
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PIP_AUDIT": "pip-audit@test",
        **env,
    }
    if "AUDIT_BASE" not in env:  # A CI-set one in the caller must not leak in.
        environment.pop("AUDIT_BASE", None)
    result = subprocess.run(
        ["python3", str(AUDIT), *args],
        cwd=repo.path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode, result.stdout + result.stderr


class TestAudit:
    def test_fails_on_a_vulnerability_the_change_adds(self, repo: Repo, fake_bin: Path) -> None:
        repo.commit(report(), "base")
        repo.commit(report("PYSEC-new"), "adds one")
        status, out = run_audit(repo, fake_bin)
        assert status == 1
        assert "::error title=New advisory::PYSEC-new" in out

    def test_only_warns_on_one_the_baseline_had(self, repo: Repo, fake_bin: Path) -> None:
        repo.commit(report("PYSEC-old"), "base")
        repo.commit(report("PYSEC-old"), "unrelated")
        status, out = run_audit(repo, fake_bin)
        assert status == 0
        assert "::warning title=Existing advisory::PYSEC-old" in out

    def test_compares_against_audit_base_when_set(self, repo: Repo, fake_bin: Path) -> None:
        green = repo.commit(report(), "green")
        repo.commit(report("PYSEC-early"), "adds one")
        repo.commit(report("PYSEC-early"), "unrelated tip")
        assert run_audit(repo, fake_bin)[0] == 0  # Against the parent it only warns.
        assert run_audit(repo, fake_bin, AUDIT_BASE=green)[0] == 1

    def test_counts_everything_as_new_without_a_baseline(self, repo: Repo, fake_bin: Path) -> None:
        repo.commit(report("PYSEC-old"), "base")
        repo.commit(report("PYSEC-old"), "tip")
        assert run_audit(repo, fake_bin, AUDIT_BASE="")[0] == 1

    def test_warns_about_dependencies_it_could_not_audit(self, repo: Repo, fake_bin: Path) -> None:
        repo.commit(report(), "base")
        repo.commit(report(skipped="private-pkg"), "tip")
        status, out = run_audit(repo, fake_bin)
        assert status == 0
        assert "::warning title=Not audited::private-pkg: not on PyPI" in out

    @pytest.mark.parametrize(
        "content",
        [{}, {"deps": []}, {"dependencies": {}}, {"dependencies": [{"name": "x", "version": "1"}]}],
        ids=["empty", "renamed key", "not a list", "entry without vulns"],
    )
    def test_fails_closed_on_a_report_of_the_wrong_shape(
        self, repo: Repo, fake_bin: Path, content: dict[str, Any]
    ) -> None:
        repo.commit(content, "base")
        repo.commit(content, "tip")
        status, out = run_audit(repo, fake_bin, FAKE_RC="0")
        assert status == 1
        assert "::error title=Audit failed::" in out

    def test_fails_closed_when_the_exit_status_disagrees(self, repo: Repo, fake_bin: Path) -> None:
        repo.commit(report(), "base")
        repo.commit(report(), "tip")
        # pip-audit said it found something; the report says nothing: unread.
        assert run_audit(repo, fake_bin, FAKE_RC="1")[0] == 1
        assert run_audit(repo, fake_bin, FAKE_RC="2")[0] == 1

    @pytest.mark.parametrize("flag", ["--format", "--format=markdown", "-f", "--output=x", "-o"])
    def test_refuses_flags_that_would_move_or_reshape_the_report(
        self, repo: Repo, fake_bin: Path, flag: str
    ) -> None:
        repo.commit(report(), "base")
        repo.commit(report(), "tip")
        status, out = run_audit(repo, fake_bin, flag)
        assert status == 1
        assert "refusing" in out

    def test_fails_closed_on_a_baseline_it_cannot_find(self, repo: Repo, fake_bin: Path) -> None:
        repo.commit(report(), "base")
        repo.commit(report(), "tip")
        assert run_audit(repo, fake_bin, AUDIT_BASE="deadbeef")[0] == 1


def run_baseline(
    repo: Repo, fake_bin: Path, tmp_path: Path, runs: dict[str, list[str]], **env: str
) -> tuple[int, str, str | None]:
    # A fresh directory per call: a second call in one test must not see the
    # first one's runs.
    runs_dir = Path(tempfile.mkdtemp(dir=tmp_path, prefix="runs-"))
    for branch, shas in runs.items():
        lines = [
            f"2026-01-{28 - i:02d}T00:00:00Z {sha} https://example.test/run/{i}"
            for i, sha in enumerate(shas)
        ]
        (runs_dir / branch).write_text("\n".join(lines) + "\n")
    output = tmp_path / "github-output"
    output.write_text("")
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "GITHUB_ACTIONS": "true",
        "GITHUB_OUTPUT": str(output),
        "GITHUB_REPOSITORY": "owner/repo",
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF_NAME": "main",
        "GITHUB_REF_TYPE": "branch",
        "GH_TOKEN": "token",
        "AUDIT_WORKFLOWS": "ci.yml",
        "AUDIT_DEFAULT_BRANCH": "main",
        "FAKE_RUNS": str(runs_dir),
        **env,
    }
    result = subprocess.run(
        ["bash", str(BASELINE)],
        cwd=repo.path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    match = re.search(r"^sha=(.*)$", output.read_text(), re.M)
    return result.returncode, result.stdout + result.stderr, match[1] if match else None


class TestBaseline:
    def test_push_uses_the_last_green_run_however_many_commits_came_since(
        self, repo: Repo, fake_bin: Path, tmp_path: Path
    ) -> None:
        green = repo.commit(report(), "green")
        repo.commit(report(), "unverified 1")
        repo.commit(report(), "unverified 2")
        assert run_baseline(repo, fake_bin, tmp_path, {"main": [green]})[2] == green

    def test_push_never_compares_a_commit_with_itself(
        self, repo: Repo, fake_bin: Path, tmp_path: Path
    ) -> None:
        earlier = repo.commit(report(), "earlier green")
        head = repo.commit(report(), "re-run of a green head")
        assert run_baseline(repo, fake_bin, tmp_path, {"main": [head, earlier]})[2] == earlier

    def test_pull_request_uses_the_last_green_run_not_the_tip(
        self, repo: Repo, fake_bin: Path, tmp_path: Path
    ) -> None:
        green = repo.commit(report(), "green main")
        repo.commit(report("PYSEC-direct"), "red direct push to main")
        repo.git("checkout", "-q", "-b", "pr")
        repo.commit(report("PYSEC-direct"), "pr change")
        repo.git("checkout", "-q", "main")
        repo.git("merge", "-q", "--no-ff", "-m", "merge ref", "pr")
        sha = run_baseline(
            repo,
            fake_bin,
            tmp_path,
            {"main": [green]},
            GITHUB_EVENT_NAME="pull_request",
            GITHUB_BASE_REF="main",
            GITHUB_REF_NAME="1/merge",
        )[2]
        assert sha == green
        assert run_audit(repo, fake_bin, AUDIT_BASE=green)[0] == 1

    def test_falls_back_to_the_default_branch_then_to_none(
        self, repo: Repo, fake_bin: Path, tmp_path: Path
    ) -> None:
        green = repo.commit(report(), "green main")
        repo.git("checkout", "-q", "-b", "feature")
        repo.commit(report(), "feature work")
        on_default = run_baseline(
            repo, fake_bin, tmp_path, {"main": [green]}, GITHUB_REF_NAME="feature"
        )
        assert on_default[2] == green
        status, out, sha = run_baseline(repo, fake_bin, tmp_path, {}, GITHUB_REF_NAME="feature")
        assert (status, sha) == (0, "")
        assert "every high or critical advisory counts as new" in out

    def test_fails_when_the_lookup_fails(self, repo: Repo, fake_bin: Path, tmp_path: Path) -> None:
        green = repo.commit(report(), "green")
        repo.commit(report(), "tip")
        status, _, sha = run_baseline(repo, fake_bin, tmp_path, {"main": [green]}, FAKE_GH_FAIL="1")
        assert status != 0
        assert sha is None


def jobs(workflow: str) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load((WORKFLOWS / workflow).read_text())
    return loaded["jobs"]


class TestWiring:
    """What the workflow has to give scripts/audit-baseline.sh, and pass on.

    None of these fails visibly when it goes missing -- a dropped AUDIT_BASE
    quietly falls back to comparing against HEAD's parent -- so each is
    pinned.
    """

    job = jobs("ci.yml")["audit"]
    steps: list[dict[str, Any]] = job["steps"]
    baseline = next(step for step in steps if step.get("id") == "audit-baseline")
    audit = next(step for step in steps if step.get("name") == "pip-audit")

    def test_checks_out_the_full_history(self) -> None:
        checkout = next(
            s for s in self.steps if str(s.get("uses", "")).startswith("actions/checkout@")
        )
        assert checkout["with"]["fetch-depth"] == 0

    def test_may_look_up_earlier_runs(self) -> None:
        assert self.job["permissions"]["actions"] == "read"
        assert jobs("docker.yml")["tests"]["permissions"]["actions"] == "read"

    def test_finds_the_baseline_first_from_workflows_that_exist(self) -> None:
        assert self.baseline["run"] == "scripts/audit-baseline.sh"
        assert self.steps.index(self.baseline) < self.steps.index(self.audit)
        for workflow in self.baseline["env"]["AUDIT_WORKFLOWS"].split(","):
            assert (WORKFLOWS / workflow).exists()

    def test_hands_the_audit_that_baseline(self) -> None:
        assert self.audit["env"]["AUDIT_BASE"] == "${{ steps.audit-baseline.outputs.sha }}"
        assert "scripts/audit_new_advisories.py" in self.audit["run"]
