#!/usr/bin/env bash
# Monolith guard: block oversized staged files that hurt review and LLM context.
set -euo pipefail

if [ -n "${CSA_SESSION_ID:-}" ]; then
    echo "[find-monolith-files] Inside sandbox -- skipping."
    exit 0
fi

token_threshold="${MONOLITH_TOKEN_THRESHOLD:-8000}"
line_threshold="${MONOLITH_LINE_THRESHOLD:-800}"
has_violation=0

should_skip() {
    case "$1" in
        *.lock|Cargo.lock|weave.lock|.github/workflows/*|.agents/*|.claude/*|.csa/*|.gitnexus/*|crates/cps-agent/src/agent_loop.rs|crates/cps-policy/src/lib.rs|crates/cps-doc-runner/src/lib.rs)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

while IFS= read -r -d '' file; do
    [ -f "${file}" ] || continue
    if should_skip "${file}"; then
        continue
    fi

    line_count="$(wc -l < "${file}" | tr -d ' ')"
    token_count=""
    if command -v tokuin >/dev/null 2>&1; then
        if token_json="$(tokuin "${file}" --model gpt-4 --format json 2>/dev/null)"; then
            token_count="$(
                printf '%s\n' "${token_json}" \
                    | sed -n 's/.*"total_tokens"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' \
                    | head -n 1
            )"
        fi
    fi

    if [ -n "${token_count}" ] && [ "${token_count}" -gt "${token_threshold}" ]; then
        echo "BLOCKED: ${file} has ${token_count} tokens, above threshold ${token_threshold}." >&2
        has_violation=1
        continue
    fi

    if [ "${line_count}" -gt "${line_threshold}" ]; then
        echo "BLOCKED: ${file} has ${line_count} lines, above threshold ${line_threshold}." >&2
        has_violation=1
    fi
done < <(git diff --cached --name-only --diff-filter=ACMR -z)

if [ "${has_violation}" -ne 0 ]; then
    echo "Fix: split the staged file before committing, then re-run the commit." >&2
    echo "Why: oversized files degrade review precision and context-aware tooling." >&2
    exit 1
fi
