#!/usr/bin/env bash
# Prints the commit the dependency audit compares HEAD against: the last state
# of the target branch that CI passed. The audit fails only on advisories HEAD
# has that this commit did not, so what it is matters as much as the audit
# does.
#
# It is the head of the last SUCCESSFUL run, on the target branch, of one of
# the workflows in $AUDIT_WORKFLOWS (comma-separated workflow file names: the
# ones whose push runs include the audit), excluding pull-request runs. The
# target branch is the branch pushed to; on a pull request, the branch it
# merges into; for a tag, the default branch. The same rule for every event is
# what makes a pass mean the same thing everywhere:
#
# - Not HEAD's first parent on a push. A push of several commits runs CI once,
#   on the tip, so an advisory from an earlier commit in it would already be
#   on the tip's parent and only warn.
# - Not the target branch's tip on a pull request. That tip can be a direct
#   push whose run failed on an advisory; comparing against it, the PR would
#   pass, its merge would inherit that pass through `gate` without auditing,
#   and the merge would become the baseline. Against the last green state,
#   every PR fails while the branch carries an advisory nothing has passed,
#   until it is fixed or ignored -- which is also why a merge that inherits a
#   PR's pass is a sound baseline in turn.
#
# A candidate counts only if it is an ancestor of the target branch's tip as
# this run sees it (HEAD^1, the merge commit's first parent, on a pull
# request) and, on a push, is not HEAD itself: a re-run of a green commit must
# not compare against itself, since that audit could never fail. A candidate
# no longer in this clone -- a force-push rewrote it -- is skipped.
#
# With none on the target branch, the default branch's runs are searched the
# same way: where a new branch forked off is a sound baseline. With none
# there either, there is no baseline, and the audit counts every advisory as
# new -- the old fail-on-any gate, for the case where nothing can vouch for
# the starting point. A failed API call or git error fails the script: an
# unknown baseline is not a reason to assume a recent one.
#
# Outside GitHub Actions, for a local run: HEAD's first parent.
#
# In Actions it writes `sha=<commit>` (empty for none) to $GITHUB_OUTPUT and
# reports its choice as a workflow notice; otherwise it prints the commit. It
# needs a full-history checkout (`fetch-depth: 0`) and `actions: read`, with
# GH_TOKEN, AUDIT_WORKFLOWS and AUDIT_DEFAULT_BRANCH
# (github.event.repository.default_branch) set, and GITHUB_REPOSITORY,
# GITHUB_EVENT_NAME, GITHUB_REF_NAME, GITHUB_REF_TYPE and GITHUB_BASE_REF as
# Actions sets them.
set -euo pipefail

if [ "${GITHUB_ACTIONS:-}" != true ]; then
  if git rev-parse --verify --quiet 'HEAD^1^{commit}'; then
    exit 0
  fi
  if [ "$(git rev-parse --is-shallow-repository)" = true ]; then
    echo "audit baseline: HEAD's parent is not in this shallow clone" >&2
    exit 1
  fi
  exit 0 # A root commit: there is nothing to compare against.
fi

: "${GH_TOKEN:?}" "${GITHUB_REPOSITORY:?}" "${GITHUB_EVENT_NAME:?}" "${AUDIT_WORKFLOWS:?}" "${AUDIT_DEFAULT_BRANCH:?}"

result() { # sha, then the notice saying how it was chosen
  echo "::notice title=Audit baseline::$2"
  echo "sha=$1" >> "${GITHUB_OUTPUT:?}"
  exit 0
}

if [ "$(git rev-parse --is-shallow-repository)" = true ]; then
  echo "::error title=Audit baseline::this clone is shallow: the audit job needs \`fetch-depth: 0\` to reach older commits"
  exit 1
fi

if [ "$GITHUB_EVENT_NAME" = pull_request ]; then
  target=${GITHUB_BASE_REF:?}
  tip=$(git rev-parse --verify 'HEAD^1^{commit}')
  head=""
elif [ "${GITHUB_REF_TYPE:-branch}" = tag ]; then
  target=$AUDIT_DEFAULT_BRANCH
  tip=$(git rev-parse --verify 'HEAD^{commit}')
  head=$tip
else
  target=${GITHUB_REF_NAME:?}
  tip=$(git rev-parse --verify 'HEAD^{commit}')
  head=$tip
fi

# Sets found_sha and found_url to the newest successful non-PR run on $1
# whose head is an ancestor of $tip (and is not HEAD itself, on a push), or
# leaves them empty. Not called through $(...): a subshell there does not
# inherit `set -e`, and a failed lookup has to stop the script.
found_sha="" found_url=""
last_green() {
  local branch=$1 runs="" workflow sha url status
  local -a workflows
  # Collected before the loop below breaks out of it, so a failed lookup fails
  # the script rather than being lost in a pipeline.
  IFS=',' read -r -a workflows <<< "$AUDIT_WORKFLOWS"
  for workflow in "${workflows[@]}"; do
    runs+=$(gh api -X GET "repos/$GITHUB_REPOSITORY/actions/workflows/$workflow/runs" \
      -f branch="$branch" -f status=success -f per_page=100 \
      --jq '.workflow_runs[] | select(.event != "pull_request") | "\(.created_at) \(.head_sha) \(.html_url)"')
    runs+=$'\n'
  done
  while read -r _ sha url; do
    [ -n "$sha" ] && [ "$sha" != "$head" ] || continue
    git cat-file -e "$sha^{commit}" 2> /dev/null || continue
    status=0
    git merge-base --is-ancestor "$sha" "$tip" || status=$?
    case $status in
      0) found_sha=$sha found_url=$url; return ;;
      1) ;;
      *) echo "::error title=Audit baseline::git merge-base failed for $sha"; exit 1 ;;
    esac
  done < <(printf '%s' "$runs" | sort -r)
}

last_green "$target"
if [ -n "$found_sha" ]; then
  result "$found_sha" "comparing against the last successful run on $target: $found_url ($found_sha)"
fi
if [ "$target" != "$AUDIT_DEFAULT_BRANCH" ]; then
  last_green "$AUDIT_DEFAULT_BRANCH"
  if [ -n "$found_sha" ]; then
    result "$found_sha" "no successful run on $target is an ancestor; comparing against the last successful run on $AUDIT_DEFAULT_BRANCH that is: $found_url ($found_sha)"
  fi
fi
echo "::warning title=Audit baseline::no successful run on $target or $AUDIT_DEFAULT_BRANCH is an ancestor of this commit, so there is no baseline: every high or critical advisory counts as new"
echo "sha=" >> "${GITHUB_OUTPUT:?}"
