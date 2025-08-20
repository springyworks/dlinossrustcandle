#!/usr/bin/env bash
set -euo pipefail
PID_FILE=/tmp/dlinoss_mcp_server.pid
READY_FILE=/tmp/dlinoss_mcp_ready
if [[ -f "$PID_FILE" ]]; then
  PID=$(cat "$PID_FILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo "[mcp-server] Stopping pid $PID";
    kill "$PID" 2>/dev/null || true
    sleep 0.5
    if kill -0 "$PID" 2>/dev/null; then
      echo "[mcp-server] Force killing pid $PID";
      kill -9 "$PID" 2>/dev/null || true
    fi
  else
    echo "[mcp-server] Stale pid file (process not running)"
  fi
  rm -f "$PID_FILE"
fi
rm -f "$READY_FILE"
echo "[mcp-server] Stopped"
