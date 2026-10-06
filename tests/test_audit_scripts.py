"""The dependency audit's scripts, and the workflow that runs them.

CI fails a change only for vulnerabilities it adds over a baseline commit, so
what decides the baseline (scripts/audit-baseline.sh), what counts as added
(scripts/audit_new_advisories.py) and how the workflow wires the two together
are the whole of the gate. Earlier versions compared a push only with its
parent, compared a pull request with the target branch's tip, and read a
pip-audit report of the wrong shape as clean; each let a vulnerability
through, and each is pinned below.

The scripts run for real against throwaway repositories. `uv`, `uvx` and
`gh` are fakes first on PATH:

- `uv export` writes the requirements.txt committed in whichever tree it runs
  in (empty if there is none), and `uvx pip-audit` writes that tree's
  audit-report.json and exits as pip-audit would -- so the baseline's worktree
  reports its own. Both log their arguments.
- `gh` serves a list of workflow runs the way the API does -- filtered by
  workflow, branch and `status` -- through the script's own `--jq` program
  with the real jq, so the filters that keep a failed run or a pull request's
  run from becoming a baseline are exercised rather than assumed.

Every child process gets an environment with no GITHUB_*, GIT_* or AUDIT_*
variables and no global git config, so neither a CI runner nor a developer's
shell (or a git hook's GIT_DIR) can change what these find.
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
    # uv export ... -o <file>: this tree's requirements.txt, if it has one.
    "uv": "\n".join(
        [
            'echo "uv $*" >> "${FAKE_LOG:-/dev/null}"',
            'for a; do [ "$prev" = -o ] && out=$a; prev=$a; done',
            'if [ -f requirements.txt ]; then cp requirements.txt "$out"; else : > "$out"; fi',
        ]
    ),
    # uvx pip-audit ... --output <file>: this tree's report, exit 1 if it
    # holds a vulnerability; FAKE_RC overrides the exit status.
    "uvx": "\n".join(
        [
            'echo "uvx $*" >> "${FAKE_LOG:-/dev/null}"',
            'for a; do [ "$prev" = --output ] && out=$a; [ "$prev" = -r ] && req=$a; prev=$a; done',
            '[ -n "${FAKE_REQS:-}" ] && { cat "$req"; echo "--- end of pass"; } >> "$FAKE_REQS"',
            'cp audit-report.json "$out"',
            'if [ -n "${FAKE_RC:-}" ]; then exit "$FAKE_RC"; fi',
            "grep -q '\"id\"' audit-report.json && exit 1",
            "exit 0",
        ]
    ),
    # gh api -X GET repos/<repo>/actions/workflows/<file>/runs -f branch=<b>
    # [-f status=<s>] ... --jq <program>: the runs in $FAKE_RUNS/runs.json for
    # that workflow and branch -- and conclusion, if asked -- as
    # {workflow_runs: [...]}, through <program>.
    "gh": "\n".join(
        [
            '[ -n "${FAKE_GH_FAIL:-}" ] && { echo "gh: HTTP 500" >&2; exit 1; }',
            'workflow="" branch="" status="" program=""',
            "while [ $# -gt 0 ]; do",
            "  case $1 in",
            "    */actions/workflows/*/runs) workflow=${1%/runs}; workflow=${workflow##*/} ;;",
            "    -f) case $2 in branch=*) branch=${2#branch=} ;;"
            " status=*) status=${2#status=} ;; esac; shift ;;",
            "    --jq) program=$2; shift ;;",
            "  esac",
            "  shift",
            "done",
            'jq --arg w "$workflow" --arg b "$branch" --arg s "$status" \\',
            "  '{workflow_runs: [.[] | select(.workflow == $w and .head_branch == $b"
            ' and ($s == "" or .conclusion == $s))]}\' \\',
            '  "$FAKE_RUNS/runs.json" | jq -r "$program"',
        ]
    ),
}


def clean_env(fake_bin: Path, **extra: str) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GITHUB_", "GIT_", "AUDIT_"))
    }
    return {
        **environment,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        **extra,
    }


@pytest.fixture(scope="module")
def fake_bin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("bin")
    for name, body in FAKES.items():
        path = directory / name
        path.write_text(f"#!/usr/bin/env bash\n{body}\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return directory


def report(
    *vulns: str, packages: tuple[str, ...] = ("pkg",), skipped: str | None = None
) -> dict[str, Any]:
    """A pip-audit report: each of `packages` -- `name`, at 1.0, or
    `name==version` -- with each of `vulns`."""
    dependencies: list[dict[str, Any]] = [
        {
            "name": package.split("==")[0],
            "version": package.split("==")[1] if "==" in package else "1.0",
            "vulns": [
                {"id": v, "fix_versions": ["2.0"], "description": f"{v} desc"} for v in vulns
            ],
        }
        for package in packages
    ]
    if skipped:
        dependencies.append({"name": skipped, "skip_reason": "not on PyPI"})
    return {"dependencies": dependencies}


class Repo:
    def __init__(self, path: Path, fake_bin: Path, *, init: bool = True) -> None:
        self.path = path
        self.env = clean_env(fake_bin)
        if init:
            self.git("init", "-q", "-b", "main")
            self.git("config", "user.email", "test@example.com")
            self.git("config", "user.name", "test")
            self.git("config", "commit.gpgsign", "false")

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=self.path, env=self.env, check=True, text=True, capture_output=True
        ).stdout.strip()

    def commit(self, content: dict[str, Any], message: str, requirements: str | None = None) -> str:
        (self.path / "audit-report.json").write_text(json.dumps(content))
        (self.path / "uv.lock").write_text(f"# {message}\n")
        if requirements is not None:
            (self.path / "requirements.txt").write_text(requirements)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path, fake_bin: Path) -> Repo:
    (tmp_path / "repo").mkdir()
    return Repo(tmp_path / "repo", fake_bin)


def run_audit(repo: Repo, fake_bin: Path, *args: str, **env: str) -> tuple[int, str]:
    result = subprocess.run(
        ["python3", str(AUDIT), *args],
        cwd=repo.path,
        env=clean_env(fake_bin, PIP_AUDIT="pip-audit@test", **env),
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

    def test_counts_one_as_new_once_it_reaches_another_package(
        self, repo: Repo, fake_bin: Path
    ) -> None:
        repo.commit(report("GHSA-shared", packages=("one",)), "base")
        repo.commit(report("GHSA-shared", packages=("one", "two")), "reaches a second package")
        status, out = run_audit(repo, fake_bin)
        assert status == 1
        assert "::error title=New advisory::GHSA-shared two 1.0" in out
        assert "::warning title=Existing advisory::GHSA-shared one 1.0" in out

    def test_compares_against_audit_base_when_set(self, repo: Repo, fake_bin: Path) -> None:
        green = repo.commit(report(), "green")
        repo.commit(report("PYSEC-early"), "adds one")
        repo.commit(report("PYSEC-early"), "unrelated tip")
        assert run_audit(repo, fake_bin)[0] == 0  # Against the parent it only warns.
        status, out = run_audit(repo, fake_bin, AUDIT_BASE=green)
        assert status == 1
        assert "::error title=New advisory::PYSEC-early" in out

    def test_counts_everything_as_new_without_a_baseline(self, repo: Repo, fake_bin: Path) -> None:
        repo.commit(report("PYSEC-old"), "base")
        repo.commit(report("PYSEC-old"), "tip")
        status, out = run_audit(repo, fake_bin, AUDIT_BASE="")
        assert status == 1
        assert "::error title=New advisory::PYSEC-old" in out

    def test_refuses_to_fall_back_to_the_parent_in_ci(self, repo: Repo, fake_bin: Path) -> None:
        repo.commit(report(), "base")
        repo.commit(report(), "tip")
        status, out = run_audit(repo, fake_bin, GITHUB_ACTIONS="true")
        assert status == 1
        assert "::error title=Audit failed::AUDIT_BASE is not set" in out

    def test_warns_about_dependencies_it_could_not_audit(self, repo: Repo, fake_bin: Path) -> None:
        repo.commit(report(), "base")
        repo.commit(report(skipped="private-pkg"), "tip")
        status, out = run_audit(repo, fake_bin)
        assert status == 0
        assert "::warning title=Not audited::private-pkg: not on PyPI" in out

    def test_audits_the_exported_lock_hash_pinned_and_unresolved(
        self, repo: Repo, fake_bin: Path, tmp_path: Path
    ) -> None:
        repo.commit(report(), "base")
        repo.commit(report(), "tip")
        log = tmp_path / "calls.log"
        assert run_audit(repo, fake_bin, "--ignore-vuln", "X", FAKE_LOG=str(log))[0] == 0
        calls = log.read_text().splitlines()
        exports = [c for c in calls if c.startswith("uv ")]
        audits = [c for c in calls if c.startswith("uvx ")]
        assert len(exports) == len(audits) == 2  # HEAD's and the baseline's.
        for export in exports:
            assert export.startswith("uv export --frozen --no-dev --no-emit-project ")
        for audit in audits:
            assert audit.startswith("uvx pip-audit@test -r ")
            assert "--require-hashes --disable-pip" in audit
            # The caller's arguments first, so the report's format and place,
            # last, win.
            assert re.search(r"--ignore-vuln X --format json --output \S+$", audit), audit

    def test_fails_closed_on_a_report_that_leaves_out_a_requirement(
        self, repo: Repo, fake_bin: Path
    ) -> None:
        requirements = (
            "pkg==1.0 \\\n    --hash=sha256:aa\n"
            "Left_Out==2.0 \\\n    --hash=sha256:bb\n"
            "colorama==0.4.6 ; sys_platform == 'win32' \\\n    --hash=sha256:cc\n"
        )
        repo.commit(report(), "base", requirements)
        repo.commit(report(), "tip")
        status, out = run_audit(repo, fake_bin)
        assert status == 1
        # Markers are stripped, so the win32-only one must be audited too.
        assert "leaves out 2 exported requirement(s): colorama==0.4.6, left-out==2.0" in out

    def test_audits_every_locked_package_whatever_its_marker(
        self, repo: Repo, fake_bin: Path, tmp_path: Path
    ) -> None:
        # pip-audit drops a requirement whose marker is false where it runs:
        # on the runner's Python and architecture, not the image's.
        requirements = (
            "pkg==1.0 ; python_full_version >= '3.13' \\\n    --hash=sha256:aa\n"
            "other==2.0 ; platform_machine == 'aarch64'\n"
        )
        repo.commit(report(packages=("pkg", "other==2.0")), "base", requirements)
        repo.commit(report(packages=("pkg", "other==2.0")), "tip")
        seen = tmp_path / "requirements.txt"
        assert run_audit(repo, fake_bin, FAKE_REQS=str(seen))[0] == 0
        head = seen.read_text().split("--- end of pass\n")[0]
        assert head == "pkg==1.0 \\\n    --hash=sha256:aa\nother==2.0\n"

    def test_audits_a_package_locked_at_two_versions_in_separate_passes(
        self, repo: Repo, fake_bin: Path, tmp_path: Path
    ) -> None:
        # uv locks a package twice when the lock forks by Python version; with
        # the markers gone, pip-audit refuses a file that names it twice.
        requirements = (
            "urllib3==1.26.0 ; python_full_version < '3.13' \\\n    --hash=sha256:aa\n"
            "urllib3==2.6.3 ; python_full_version >= '3.13' \\\n    --hash=sha256:bb\n"
            "pkg==1.0\n"
        )
        both = report(packages=("urllib3==1.26.0", "urllib3==2.6.3", "pkg"))
        repo.commit(both, "base", requirements)
        repo.commit(both, "tip")
        seen = tmp_path / "requirements.txt"
        assert run_audit(repo, fake_bin, FAKE_REQS=str(seen))[0] == 0
        first, second = seen.read_text().split("--- end of pass\n")[:2]
        assert first == "urllib3==1.26.0 \\\n    --hash=sha256:aa\npkg==1.0\n"
        assert second == "urllib3==2.6.3 \\\n    --hash=sha256:bb\n"

    def test_requires_each_version_of_a_package_to_be_covered(
        self, repo: Repo, fake_bin: Path
    ) -> None:
        requirements = "urllib3==1.26.0 ; python_full_version < '3.13'\nurllib3==2.6.3\n"
        one = report(packages=("urllib3==2.6.3",))
        repo.commit(one, "base", requirements)
        repo.commit(one, "tip")
        status, out = run_audit(repo, fake_bin)
        assert status == 1
        assert "leaves out 1 exported requirement(s): urllib3==1.26.0" in out

    def test_counts_a_requirement_pinned_by_url(self, repo: Repo, fake_bin: Path) -> None:
        # pip-audit cannot audit one, but has to say so: left out, it fails.
        requirements = "pkg @ https://example.test/pkg-1.0.tar.gz ; sys_platform == 'win32'\n"
        repo.commit(report(packages=()), "base", requirements)
        repo.commit(report(packages=()), "tip")
        status, out = run_audit(repo, fake_bin)
        assert status == 1
        assert "leaves out 1 exported requirement(s): pkg" in out
        repo.commit(report(packages=(), skipped="pkg"), "now reported as skipped")
        status, out = run_audit(repo, fake_bin, AUDIT_BASE="")
        assert status == 0, out
        assert "::warning title=Not audited::pkg: not on PyPI" in out

    def test_fails_closed_on_a_requirement_it_cannot_read(self, repo: Repo, fake_bin: Path) -> None:
        repo.commit(report(), "base", "@@ not a requirement\n")
        repo.commit(report(), "tip")
        status, out = run_audit(repo, fake_bin)
        assert status == 1
        assert "cannot read the exported requirement '@@ not a requirement'" in out

    def test_escapes_what_it_prints_into_workflow_commands(
        self, repo: Repo, fake_bin: Path
    ) -> None:
        repo.commit(report(), "base")
        repo.commit(report("PYSEC-100%\r::error::forged"), "adds one")
        status, out = run_audit(repo, fake_bin)
        assert status == 1
        assert "::error title=New advisory::PYSEC-100%25%0D::error::forged" in out

    def test_accepts_a_report_that_covers_every_requirement(
        self, repo: Repo, fake_bin: Path
    ) -> None:
        requirements = "pkg==1.0 \\\n    --hash=sha256:aa\nother==2.0\n"
        repo.commit(report(packages=("pkg", "other==2.0")), "base", requirements)
        repo.commit(report(packages=("pkg", "other==2.0")), "tip")
        assert run_audit(repo, fake_bin)[0] == 0

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
        for rc in ("1", "2"):
            status, out = run_audit(repo, fake_bin, FAKE_RC=rc)
            assert status == 1
            assert f"pip-audit exited {rc}" in out

    @pytest.mark.parametrize(
        "flag", ["--format", "--format=markdown", "-f", "--output=x", "-o", "--dry-run", "-d"]
    )
    def test_refuses_flags_that_would_move_reshape_or_empty_the_report(
        self, repo: Repo, fake_bin: Path, flag: str
    ) -> None:
        repo.commit(report(), "base")
        repo.commit(report(), "tip")
        status, out = run_audit(repo, fake_bin, flag)
        assert status == 1
        assert f"refusing {flag}" in out

    def test_fails_closed_on_a_baseline_it_cannot_find(self, repo: Repo, fake_bin: Path) -> None:
        repo.commit(report(), "base")
        repo.commit(report(), "tip")
        status, out = run_audit(repo, fake_bin, AUDIT_BASE="deadbeef")
        assert status == 1
        assert "::error title=Audit failed::" in out


def run_baseline(
    repo: Repo, fake_bin: Path, tmp_path: Path, runs: list[dict[str, str]], **env: str
) -> tuple[int, str, str | None]:
    """Runs the script as CI would; returns its status, output and chosen sha.

    `runs` are newest first unless they say when they ran; each needs a `sha`
    and may give `branch`, `workflow`, `event`, `conclusion` and `at`.
    """
    # A fresh directory per call: a second call in one test must not see the
    # first one's runs.
    runs_dir = Path(tempfile.mkdtemp(dir=tmp_path, prefix="runs-"))
    api = [
        {
            "workflow": run.get("workflow", "ci.yml"),
            "head_branch": run.get("branch", "main"),
            "head_sha": run["sha"],
            "event": run.get("event", "push"),
            "status": "completed",
            "conclusion": run.get("conclusion", "success"),
            "created_at": run.get("at", f"2026-01-{28 - i:02d}T00:00:00Z"),
            "html_url": f"https://example.test/run/{i}",
        }
        for i, run in enumerate(runs)
    ]
    (runs_dir / "runs.json").write_text(json.dumps(api))
    output = runs_dir / "github-output"
    output.write_text("")
    environment = clean_env(
        fake_bin,
        **{
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
        },
    )
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


PULL_REQUEST = {
    "GITHUB_EVENT_NAME": "pull_request",
    "GITHUB_BASE_REF": "main",
    "GITHUB_REF_NAME": "1/merge",
}


class TestBaseline:
    def test_prints_the_first_parent_outside_actions(self, repo: Repo, fake_bin: Path) -> None:
        first = repo.commit(report(), "one")
        repo.commit(report(), "two")
        result = subprocess.run(
            ["bash", str(BASELINE)],
            cwd=repo.path,
            env=clean_env(fake_bin),
            text=True,
            capture_output=True,
            check=True,
        )
        assert result.stdout.strip() == first

    def test_push_uses_the_last_green_run_however_many_commits_came_since(
        self, repo: Repo, fake_bin: Path, tmp_path: Path
    ) -> None:
        green = repo.commit(report(), "green")
        repo.commit(report(), "unverified 1")
        repo.commit(report(), "unverified 2")
        assert run_baseline(repo, fake_bin, tmp_path, [{"sha": green}])[2] == green

    def test_skips_a_run_that_failed_however_recent(
        self, repo: Repo, fake_bin: Path, tmp_path: Path
    ) -> None:
        green = repo.commit(report(), "green")
        red = repo.commit(report("PYSEC-direct"), "red direct push")
        repo.commit(report("PYSEC-direct"), "tip")
        runs = [{"sha": red, "conclusion": "failure"}, {"sha": green}]
        assert run_baseline(repo, fake_bin, tmp_path, runs)[2] == green

    def test_skips_a_pull_requests_run_even_on_a_branch_of_the_same_name(
        self, repo: Repo, fake_bin: Path, tmp_path: Path
    ) -> None:
        # A fork's branch can be called main; its run tested a merge commit,
        # not the head it reports.
        green = repo.commit(report(), "green")
        pr = repo.commit(report(), "a pull request head, later merged")
        repo.commit(report(), "tip")
        runs = [{"sha": pr, "event": "pull_request"}, {"sha": green}]
        assert run_baseline(repo, fake_bin, tmp_path, runs)[2] == green

    def test_takes_the_newest_green_run_across_every_listed_workflow(
        self, repo: Repo, fake_bin: Path, tmp_path: Path
    ) -> None:
        older = repo.commit(report(), "green in ci.yml")
        newer = repo.commit(report(), "green in docker.yml")
        repo.commit(report(), "tip")
        runs = [
            {"sha": older, "workflow": "ci.yml", "at": "2026-02-01T00:00:00Z"},
            {"sha": newer, "workflow": "docker.yml", "at": "2026-02-02T00:00:00Z"},
        ]
        sha = run_baseline(repo, fake_bin, tmp_path, runs, AUDIT_WORKFLOWS="ci.yml,docker.yml")[2]
        assert sha == newer

    def test_push_never_compares_a_commit_with_itself(
        self, repo: Repo, fake_bin: Path, tmp_path: Path
    ) -> None:
        earlier = repo.commit(report(), "earlier green")
        head = repo.commit(report(), "re-run of a green head")
        runs = [{"sha": head}, {"sha": earlier}]
        assert run_baseline(repo, fake_bin, tmp_path, runs)[2] == earlier

    def test_skips_runs_that_are_not_ancestors_or_no_longer_exist(
        self, repo: Repo, fake_bin: Path, tmp_path: Path
    ) -> None:
        green = repo.commit(report(), "green")
        repo.git("checkout", "-q", "-b", "elsewhere")
        elsewhere = repo.commit(report(), "not an ancestor of main")
        repo.git("checkout", "-q", "main")
        repo.commit(report(), "tip")
        runs = [{"sha": "f" * 40}, {"sha": elsewhere}, {"sha": green}]
        assert run_baseline(repo, fake_bin, tmp_path, runs)[2] == green

    def test_pull_request_uses_the_last_green_run_not_the_tip(
        self, repo: Repo, fake_bin: Path, tmp_path: Path
    ) -> None:
        green = repo.commit(report(), "green main")
        repo.commit(report("PYSEC-direct"), "red direct push to main")
        repo.git("checkout", "-q", "-b", "pr")
        repo.commit(report("PYSEC-direct"), "pr change")
        repo.git("checkout", "-q", "main")
        repo.git("merge", "-q", "--no-ff", "-m", "merge ref", "pr")
        sha = run_baseline(repo, fake_bin, tmp_path, [{"sha": green}], **PULL_REQUEST)[2]
        assert sha == green
        status, out = run_audit(repo, fake_bin, AUDIT_BASE=green)
        assert status == 1
        assert "::error title=New advisory::PYSEC-direct" in out

    def test_pull_request_reads_ancestry_from_the_merges_first_parent(
        self, repo: Repo, fake_bin: Path, tmp_path: Path
    ) -> None:
        # main was force-pushed back past a green commit the PR still
        # contains: that commit is in the merge, but no longer on the branch
        # it targets.
        kept = repo.commit(report(), "green, still on main")
        dropped = repo.commit(report(), "green, later dropped from main")
        repo.git("checkout", "-q", "-b", "pr")
        repo.commit(report(), "pr change")
        repo.git("checkout", "-q", "main")
        repo.git("reset", "-q", "--hard", kept)
        repo.git("merge", "-q", "--no-ff", "-m", "merge ref", "pr")
        runs = [{"sha": dropped}, {"sha": kept}]
        assert run_baseline(repo, fake_bin, tmp_path, runs, **PULL_REQUEST)[2] == kept

    def test_pull_request_into_another_branch_uses_that_branch(
        self, repo: Repo, fake_bin: Path, tmp_path: Path
    ) -> None:
        on_main = repo.commit(report(), "green main")
        repo.git("checkout", "-q", "-b", "release")
        on_release = repo.commit(report(), "green release")
        repo.git("checkout", "-q", "-b", "pr")
        repo.commit(report(), "pr change")
        repo.git("checkout", "-q", "release")
        repo.git("merge", "-q", "--no-ff", "-m", "merge ref", "pr")
        runs = [{"sha": on_release, "branch": "release"}, {"sha": on_main}]
        env = {**PULL_REQUEST, "GITHUB_BASE_REF": "release"}
        assert run_baseline(repo, fake_bin, tmp_path, runs, **env)[2] == on_release

    def test_tag_uses_the_default_branch_and_not_the_tagged_commit(
        self, repo: Repo, fake_bin: Path, tmp_path: Path
    ) -> None:
        green = repo.commit(report(), "green main")
        decoy = repo.commit(report(), "green on a branch named like the tag")
        tagged = repo.commit(report(), "tagged, and green on main")
        runs = [{"sha": tagged}, {"sha": decoy, "branch": "v1.0.0"}, {"sha": green}]
        sha = run_baseline(
            repo, fake_bin, tmp_path, runs, GITHUB_REF_TYPE="tag", GITHUB_REF_NAME="v1.0.0"
        )[2]
        assert sha == green

    def test_falls_back_to_the_default_branch_then_to_none(
        self, repo: Repo, fake_bin: Path, tmp_path: Path
    ) -> None:
        green = repo.commit(report(), "green main")
        repo.git("checkout", "-q", "-b", "feature")
        repo.commit(report(), "feature work")
        on_default = run_baseline(
            repo, fake_bin, tmp_path, [{"sha": green}], GITHUB_REF_NAME="feature"
        )
        assert on_default[2] == green
        status, out, sha = run_baseline(repo, fake_bin, tmp_path, [], GITHUB_REF_NAME="feature")
        assert (status, sha) == (0, "")
        assert "everything the audit finds counts as new" in out

    def test_fails_when_the_lookup_fails(self, repo: Repo, fake_bin: Path, tmp_path: Path) -> None:
        green = repo.commit(report(), "green")
        repo.commit(report(), "tip")
        status, _, sha = run_baseline(repo, fake_bin, tmp_path, [{"sha": green}], FAKE_GH_FAIL="1")
        assert status != 0
        assert sha is None

    def test_fails_in_a_shallow_clone(self, repo: Repo, fake_bin: Path, tmp_path: Path) -> None:
        repo.commit(report(), "one")
        repo.commit(report(), "two")
        shallow = tmp_path / "shallow"
        subprocess.run(
            ["git", "clone", "-q", "--depth", "1", f"file://{repo.path}", str(shallow)],
            env=clean_env(fake_bin),
            check=True,
        )
        clone = Repo(shallow, fake_bin, init=False)
        status, out, sha = run_baseline(clone, fake_bin, tmp_path, [])
        assert status != 0
        assert "shallow" in out
        assert sha is None


def jobs(workflow: str) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load((WORKFLOWS / workflow).read_text())
    found: dict[str, Any] = loaded["jobs"]
    return found


def callers_of_ci() -> dict[str, list[dict[str, Any]]]:
    """Every workflow with a job that calls ci.yml, with those jobs."""
    found = {}
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        callers = [
            job
            for job in jobs(path.name).values()
            if job.get("uses") == "./.github/workflows/ci.yml"
        ]
        if callers:
            found[path.name] = callers
    return found


class TestWiring:
    """What the workflow has to give scripts/audit-baseline.sh, and pass on.

    None of these fails visibly when it goes missing -- a dropped AUDIT_BASE
    quietly falls back to comparing against HEAD's parent, and a workflow in
    AUDIT_WORKFLOWS that goes green without auditing makes every commit it
    passes a baseline -- so each is pinned. So is the audit job's
    independence from `gate`: advisories are published while content stands
    still, and a merge that inherited a pull request's pass would become a
    baseline nothing audited.
    """

    job = jobs("ci.yml")["audit"]
    steps: list[dict[str, Any]] = job["steps"]
    baseline = next(step for step in steps if step.get("id") == "audit-baseline")
    audit = next(step for step in steps if step.get("name") == "pip-audit")
    callers = callers_of_ci()

    def test_reports_under_the_name_branch_protection_requires(self) -> None:
        # Branch protection requires the audit by this name for a red audit to
        # block a merge, and a rename has to be matched there: a required check
        # that is never reported blocks every merge until it is.
        assert self.job["name"] == "Dependency audit"

    def test_never_skips_and_never_fails_quietly(self) -> None:
        for key in ("if", "needs", "continue-on-error"):
            assert key not in self.job, key
        for step in (self.baseline, self.audit):
            assert "if" not in step, step["name"]
            assert "continue-on-error" not in step, step["name"]

    def test_checks_out_the_full_history(self) -> None:
        checkout = next(
            s for s in self.steps if str(s.get("uses", "")).startswith("actions/checkout@")
        )
        assert checkout["with"]["fetch-depth"] == 0

    def test_may_look_up_earlier_runs(self) -> None:
        assert self.job["permissions"]["actions"] == "read"

    def test_is_granted_that_lookup_by_every_workflow_that_calls_it(self) -> None:
        assert self.callers  # docker.yml, today.
        for name, callers in self.callers.items():
            for caller in callers:
                assert caller["permissions"]["actions"] == "read", name

    def test_finds_the_baseline_first_only_in_workflows_that_run_the_audit(self) -> None:
        assert self.baseline["run"] == "scripts/audit-baseline.sh"
        assert self.steps.index(self.baseline) < self.steps.index(self.audit)
        workflows = self.baseline["env"]["AUDIT_WORKFLOWS"].split(",")
        assert workflows
        for workflow in workflows:
            assert workflow == "ci.yml" or workflow in self.callers, workflow
        assert self.baseline["env"]["GH_TOKEN"] == "${{ github.token }}"
        assert (
            self.baseline["env"]["AUDIT_DEFAULT_BRANCH"]
            == "${{ github.event.repository.default_branch }}"
        )

    def test_hands_the_audit_that_baseline_and_runs_it_as_written(self) -> None:
        assert self.audit["env"]["AUDIT_BASE"] == "${{ steps.audit-baseline.outputs.sha }}"
        assert self.audit["run"].strip() == "python3 scripts/audit_new_advisories.py"
