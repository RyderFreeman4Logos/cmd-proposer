#!/usr/bin/env bash
# Post-merge rebuild: keep the local cps binary aligned after merges.
set -euo pipefail

if [ -n "${CSA_SESSION_ID:-}" ]; then
    echo "[post-merge] Inside sandbox -- skipping."
    exit 0
fi

install_root="${CARGO_INSTALL_ROOT:-${HOME}/.cargo}"
install_bin="${install_root}/bin"

if [ ! -d "${install_bin}" ] || [ ! -w "${install_bin}" ]; then
    echo "[post-merge] Install target ${install_bin} is not writable -- skipping."
    exit 0
fi

echo "[post-merge] Rebuilding cps..."
if just install; then
    echo "[post-merge] Installed cps successfully."
else
    status="$?"
    echo "[post-merge] WARNING: install failed with exit ${status}." >&2
    exit 0
fi
