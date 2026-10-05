#!/usr/bin/env bash
# Prints the commit the dependency audit compares HEAD against: the last state
# of this branch that CI is known to have passed. The audit fails only on
# advisories HEAD has that this commit did not, so what it is matters as much
# as the audit does.
#
# - On a pull request: HEAD's first parent. actions/checkout checks out
#   GitHub's merge commit, whose first parent is the target branch, so that
#   is "the base without this change", and branch protection keeps it current.
#
# - Otherwise -- a push, a dispatch, a release tag: the head commit of the
#   last SUCCESSFUL run, on this branch, of one of the workflows in
#   $AUDIT_WORKFLOWS (comma-separated workflow file names: the ones whose push
#   runs include the audit). For a tag, the default branch's. Not HEAD's first
#   parent: a push of several commits runs CI once, on the tip, so an advisory
#   from an earlier commit in it would already be on the tip's parent and only
#   warn. Not the commit before the push either: a run that failed on an
#   advisory, or was cancelled by the next push, would have the next push
#   compare against its commits and pass them. Against the last green state,
#   an advisory introduced since keeps failing until it is fixed or ignored.
#
#   Only a run whose head commit is HEAD or an ancestor of it counts, so a
#   re-run of an old commit cannot stand in. If none is found -- a new
#   branch, runs past their retention, a force-push that rewrote what passed
#   -- it falls back to the commit before the push, then to HEAD's first
#   parent, and says so.
#
# - Outside GitHub Actions: HEAD's first parent, for a local run.
#
# Prints nothing for a root commit, which has nothing to compare against.
# How it got there goes to stderr. A failed API call fails the script: an
# unknown baseline is not a reason to assume a recent one.
#
# Needs a full-history checkout (`fetch-depth: 0`) to find older commits, and
# `actions: read` for the run lookup, with GH_TOKEN, GITHUB_REPOSITORY,
# GITHUB_REF_NAME and GITHUB_REF_TYPE (set by Actions), AUDIT_WORKFLOWS,
# AUDIT_BEFORE (github.event.before) and AUDIT_DEFAULT_BRANCH
# (github.event.repository.default_branch).
set -euo pipefail

say() { echo "audit baseline: $1" >&2; }

first_parent() {
  if git rev-parse --verify --quiet 'HEAD^1^{commit}'; then
    return
  fi
  if [ "$(git rev-parse --is-shallow-repository)" = true ]; then
    say "HEAD's parent is not in this shallow clone"
    exit 1
  fi
  say "HEAD is a root commit; there is nothing to compare against"
}

if [ "${GITHUB_ACTIONS:-}" != true ] || [ "${GITHUB_EVENT_NAME:-}" = pull_request ]; then
  say "HEAD's first parent (${GITHUB_EVENT_NAME:-a local run})"
  first_parent
  exit 0
fi

: "${GH_TOKEN:?}" "${GITHUB_REPOSITORY:?}" "${GITHUB_REF_NAME:?}" "${AUDIT_WORKFLOWS:?}"
if [ "$(git rev-parse --is-shallow-repository)" = true ]; then
  say "this clone is shallow: the audit job needs \`fetch-depth: 0\` to reach older commits"
  exit 1
fi

branch=$GITHUB_REF_NAME
if [ "${GITHUB_REF_TYPE:-branch}" = tag ]; then
  branch=${AUDIT_DEFAULT_BRANCH:?}
fi

# Collected before the loop below breaks out of it, so a failed lookup fails
# the script rather than being lost in a pipeline.
runs=""
IFS=',' read -r -a workflows <<< "$AUDIT_WORKFLOWS"
for workflow in "${workflows[@]}"; do
  runs+=$(gh api -X GET "repos/$GITHUB_REPOSITORY/actions/workflows/$workflow/runs" \
    -f branch="$branch" -f status=success -f per_page=50 \
    --jq '.workflow_runs[] | select(.event != "pull_request") | "\(.created_at) \(.head_sha) \(.html_url)"')
  runs+=$'\n'
done

while read -r _ sha url; do
  [ -n "$sha" ] || continue
  if git merge-base --is-ancestor "$sha" HEAD 2> /dev/null; then
    say "the last successful run on $branch, $url ($sha)"
    echo "$sha"
    exit 0
  fi
done < <(printf '%s' "$runs" | sort -r)

before=${AUDIT_BEFORE:-}
if [ -n "$before" ] && [ "$before" != 0000000000000000000000000000000000000000 ] &&
  git cat-file -e "$before^{commit}" 2> /dev/null; then
  say "no earlier successful run on $branch is an ancestor of HEAD; using the commit before this push ($before)"
  echo "$before"
  exit 0
fi

say "no earlier successful run on $branch, and no commit before this push; using HEAD's first parent"
first_parent
