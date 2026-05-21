#!/usr/bin/env bash
# Artifact guard: block generated and scratch files from being staged.
set -euo pipefail

if [ -n "${CSA_SESSION_ID:-}" ]; then
    echo "[check-generated-artifacts] Inside sandbox -- skipping."
    exit 0
fi

blocked=()
while IFS= read -r -d '' file; do
    case "${file}" in
        .test-target/*|.tmp/*|target/*|dist/*|node_modules/*|__pycache__/*|*.pyc|*.log|*_output.*|screenshots/*|screenshot/*)
            blocked+=("${file}")
            ;;
    esac
done < <(git diff --cached --name-only --diff-filter=ACMR -z)

if [ "${#blocked[@]}" -ne 0 ]; then
    echo "BLOCKED: generated or scratch artifacts are staged:" >&2
    printf '  %s\n' "${blocked[@]}" >&2
    echo "Fix: unstage artifacts with: git restore --staged <path>" >&2
    echo "Why: generated outputs make reviews noisy and are not source-of-truth inputs." >&2
    exit 1
fi
