#!/usr/bin/env bash
# Fast repo search using ripgrep with sensible defaults for this workspace.
# Usage: scripts/fastgrep.sh PATTERN [ADDITIONAL_RG_ARGS]
set -euo pipefail
if ! command -v rg >/dev/null 2>&1; then
  echo "ripgrep (rg) is not installed. Please install it first." >&2
  exit 1
fi
root_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root_dir"
RG_ARGS=("-n" "--hidden" "-S" "-g" "!target" "-g" "!.git" "-g" "!**/*.ipynb")
rg "${RG_ARGS[@]}" "$@"
