#!/usr/bin/env bash
# Charset guard: block newly staged Han characters outside allowed agent files.
set -euo pipefail

if [ -n "${CSA_SESSION_ID:-}" ]; then
    echo "[check-charset] Inside sandbox -- skipping."
    exit 0
fi

has_violation=0

while IFS= read -r -d '' file; do
    case "${file}" in
        .agents/*|.claude/*|.csa/*|AGENTS.md|CLAUDE.md)
            continue
            ;;
    esac

    if git diff --cached --unified=0 -- "${file}" | rg -n '^\+[^+].*\p{Han}' >/dev/null; then
        echo "BLOCKED: staged content adds non-English text in ${file}." >&2
        git diff --cached --unified=0 -- "${file}" | rg -n '^\+[^+].*\p{Han}' >&2
        has_violation=1
    fi
done < <(git diff --cached --name-only --diff-filter=ACMR -z)

if [ "${has_violation}" -ne 0 ]; then
    echo "Fix: rewrite code, comments, and committed docs in English, or add a narrow allowlist for deliberate i18n/test data." >&2
    echo "Why: committed project artifacts should stay readable to the full toolchain and review workflow." >&2
    exit 1
fi
