"""The CI gate cannot report a pass it did not earn.

On `main` and on release tags, ci.yml's check jobs may skip: `gate` inherits
a pass this content has already earned, either because an earlier run checked
this exact commit or because the merge commit's tree is identical to the
branch that merged. So `gate` is the only thing standing between a commit and
an image that verified nothing -- docker.yml runs `build-and-push` behind
`needs: tests`, and nothing else re-checks.

Two properties keep that honest, and both are asserted here rather than left
to the comments.

First, nothing but `gate` decides to skip: every check job's condition defers
to it and says nothing of its own about refs, events or paths.

Second, a red check must stop the publish. `needs: tests` does that -- any job
in the called workflow going red fails it, including one nobody remembered to
wire up. But a workflow is not failed by a job that SKIPS, nor by one that
goes red under `continue-on-error`, so a check of either shape would sail
straight past that guarantee.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"

# The one condition a check job may carry: skip iff the gate reported a pass.
GATE_SKIP = "${{ !cancelled() && needs.gate.outputs.passed != 'true' }}"


def load(name: str) -> dict[str, Any]:
    jobs: dict[str, Any] = yaml.safe_load((WORKFLOWS / name).read_text())["jobs"]
    return jobs


CI_JOBS = load("ci.yml")
CHECKS = [name for name in CI_JOBS if name != "gate"]
GATE = CI_JOBS["gate"]
SCRIPT = next(s for s in GATE["steps"] if s.get("id") == "inherit")["run"]


def test_has_check_jobs_to_guard() -> None:
    # Without this the suite passes vacuously if the jobs are renamed or the
    # `jobs:` key changes shape -- green from a rule matching nothing.
    assert CHECKS


@pytest.mark.parametrize("name", CHECKS)
def test_check_cannot_pass_without_running(name: str) -> None:
    assert CI_JOBS[name].get("if") == GATE_SKIP
    assert CI_JOBS[name].get("continue-on-error", False) is False


@pytest.mark.parametrize("name", CHECKS)
def test_no_step_fails_quietly(name: str) -> None:
    # The job-level key is the obvious way to report an unearned pass; a
    # step-level one is the easy way, and leaves the job green over a red
    # check. `gate` is the one legitimate use here, and is outside CHECKS.
    offenders = [
        step.get("name") or step.get("uses") or f"step {i}"
        for i, step in enumerate(CI_JOBS[name].get("steps") or [])
        if step.get("continue-on-error")
    ]
    assert offenders == []


def test_gate_runs_on_main_and_tags_only() -> None:
    # Both refs matter and for different reasons: `main` is the merge case, a
    # tag is the release case. Pull requests are excluded deliberately -- the
    # checkout there is the merge ref, whose second parent is the branch head,
    # so the tree lookup would compare against the very branch under test.
    condition = GATE["if"]
    assert "github.ref == 'refs/heads/main'" in condition
    assert "startsWith(github.ref, 'refs/tags/')" in condition
    assert "pull_request" not in condition


def test_gate_keeps_both_lookups() -> None:
    # They cover different cases and neither subsumes the other: the SHA
    # lookup is what makes a tagged release cheap, the tree lookup is what
    # makes a merge cheap. Losing one is a silent halving -- the other still
    # answers, and nothing goes red.
    assert "GITHUB_SHA" in SCRIPT
    assert "HEAD^2" in SCRIPT


def test_lookups_name_different_workflows() -> None:
    # ci.yml never has a run of its own on main or a tag: docker.yml calls it,
    # and a called workflow runs inside the caller's run. So the branch/tag
    # evidence is a docker.yml run while the pull-request evidence is a ci.yml
    # run, and collapsing them onto one name would break whichever case lost
    # its workflow, silently.
    env = next(s for s in GATE["steps"] if s.get("id") == "inherit")["env"]
    assert env["SAME_SHA_WORKFLOW"] == ".github/workflows/docker.yml"
    assert env["PR_WORKFLOW"] == ".github/workflows/ci.yml"


def test_gate_never_treats_this_run_as_evidence() -> None:
    # The SHA lookup lists runs for this commit, and this run is in that list.
    # Without the exclusion it would find itself.
    assert "GITHUB_RUN_ID" in SCRIPT


def test_gate_rechecks_on_rerun() -> None:
    # "Re-run all jobs" is what you reach for when you doubt a result, so
    # inheriting on a retry would hand back the same answer and leave no way
    # to force a genuine check short of pushing an empty commit.
    assert "GITHUB_RUN_ATTEMPT" in SCRIPT


def test_gate_fails_closed() -> None:
    assert GATE.get("continue-on-error") is True


def test_gate_output_has_one_source() -> None:
    # Never a literal, and never a fallback whose default is "verified".
    assert GATE["outputs"]["passed"] == "${{ steps.inherit.outputs.passed }}"


def test_pass_is_claimed_once_and_last() -> None:
    # Both lookups exit BEFORE the claim on every path that does not inherit,
    # so a single occurrence at the end is the shape of "nothing objected".
    assert SCRIPT.count("passed=true") == 1
    assert "passed=true" in SCRIPT.rstrip().splitlines()[-1]


def test_gate_is_granted_the_scope_its_lookups_need() -> None:
    # The one failure here that is invisible rather than loud: without
    # `actions: read` both lookups fail, the gate reports "not verified", and
    # every merge and every tag quietly pays full price with nothing red.
    assert GATE["permissions"]["actions"] == "read"


def test_publish_is_gated_on_checks_concluding_success() -> None:
    # Deliberately bare. A called workflow whose jobs ALL skip concludes
    # `skipped`, not success, so skipping the publish is the correct response
    # to the checks having verified nothing -- adding `!cancelled()` or
    # `always()` here would push an image whose checks went red.
    #
    # That cannot arise while `gate` runs on main and tags: it is itself a job
    # that concluded, so the called workflow resolves to success rather than
    # skipped even when every check skips.
    docker = load("docker.yml")
    assert docker["build-and-push"]["needs"] == "tests"
    assert "if" not in docker["build-and-push"]
