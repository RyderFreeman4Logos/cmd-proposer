#!/usr/bin/env bash
# Rust formatter: format only staged Rust files and re-stage those files.
set -euo pipefail

if [ -n "${CSA_SESSION_ID:-}" ]; then
    echo "[format-staged-rust] Inside sandbox -- skipping."
    exit 0
fi

mapfile -d '' rust_files < <(git diff --cached --name-only --diff-filter=ACMR -z -- '*.rs')

if [ "${#rust_files[@]}" -eq 0 ]; then
    echo "[format-staged-rust] No staged Rust files -- skipping."
    exit 0
fi

if ! command -v rustfmt >/dev/null 2>&1; then
    echo "BLOCKED: rustfmt is not installed." >&2
    echo "Fix: install the Rust formatter component with: rustup component add rustfmt" >&2
    echo "Why: staged Rust files must be formatted deterministically before commit." >&2
    exit 1
fi

mixed_files=()
for file in "${rust_files[@]}"; do
    if ! git diff --quiet -- "${file}"; then
        mixed_files+=("${file}")
    fi
done

if [ "${#mixed_files[@]}" -ne 0 ]; then
    echo "BLOCKED: staged Rust files also have unstaged edits:" >&2
    printf '  %s\n' "${mixed_files[@]}" >&2
    echo "Fix: stage or revert the unstaged edits before committing." >&2
    echo "Why: auto-formatting would otherwise stage content that was not part of the commit." >&2
    exit 1
fi

rustfmt --edition 2021 "${rust_files[@]}"
git add "${rust_files[@]}"
