#!/usr/bin/env bash
# Workspace-local convenience aliases that prefer ripgrep without overriding system grep.
# Source this from VS Code terminal profile or manually: source scripts/aliases.sh

# Prefer ripgrep when available
if command -v rg >/dev/null 2>&1; then
  # Safe alias: use gg for project searches (don’t shadow system grep)
  alias gg='rg -n --hidden -S -g "!target" -g "!.git" -g "!**/*.ipynb"'
fi
