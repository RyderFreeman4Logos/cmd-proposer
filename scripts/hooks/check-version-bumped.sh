#!/usr/bin/env bash
# Version guard: require workspace version changes before feature branch pushes.
set -euo pipefail

if [ -n "${CSA_SESSION_ID:-}" ]; then
    echo "[check-version-bumped] Inside sandbox -- skipping."
    exit 0
fi

branch="$(git branch --show-current)"
case "${branch}" in
    ""|main|dev|master)
        exit 0
        ;;
esac

base_ref="${CPS_VERSION_BASE_REF:-origin/main}"
if ! git rev-parse --verify "${base_ref}" >/dev/null 2>&1; then
    base_ref="main"
fi

ahead_count="$(git rev-list --count "${base_ref}..HEAD")"
if [ "${ahead_count}" = "0" ]; then
    echo "[check-version-bumped] Branch has no commits ahead of ${base_ref} -- skipping."
    exit 0
fi

current_version="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -n 1)"
base_version="$(git show "${base_ref}:Cargo.toml" | sed -n 's/^version = "\(.*\)"/\1/p' | head -n 1)"

if [ -z "${current_version}" ] || [ -z "${base_version}" ]; then
    echo "BLOCKED: could not read workspace versions from Cargo.toml." >&2
    echo "Fix: ensure [workspace.package] version exists in Cargo.toml." >&2
    echo "Why: release metadata must be mechanically checkable." >&2
    exit 1
fi

if [ "${current_version}" = "${base_version}" ]; then
    echo "BLOCKED: workspace version is still ${current_version}, matching ${base_ref}." >&2
    echo "Fix: run: just bump-patch" >&2
    echo "Why: feature branch pushes must carry an explicit release version change." >&2
    exit 1
fi
