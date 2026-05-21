#!/usr/bin/env bash
# Branch protection: block direct commits to protected branches.
set -euo pipefail

if [ -n "${CSA_SESSION_ID:-}" ]; then
    echo "[branch-protection] Inside sandbox -- skipping."
    exit 0
fi

branch="$(git symbolic-ref --short HEAD 2>/dev/null || true)"
[ -z "${branch}" ] && exit 0

protected_branches="main dev master"
for protected in ${protected_branches}; do
    if [ "${branch}" = "${protected}" ]; then
        echo "BLOCKED: Cannot commit directly to '${branch}'." >&2
        echo "Fix: create a feature branch with: git checkout -b feat/<description>" >&2
        echo "Why: protected branches must receive reviewed PR merges only." >&2
        exit 1
    fi
done
