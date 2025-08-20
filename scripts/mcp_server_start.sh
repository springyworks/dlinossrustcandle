#!/usr/bin/env bash
# Starts the dlinoss MCP server keeping stdin open via a FIFO so other terminals can send JSON-RPC
# requests without the process exiting. Replaces any existing instance.
set -euo pipefail

PID_FILE=/tmp/dlinoss_mcp_server.pid
READY_FILE=${READY_FILE:-/tmp/dlinoss_mcp_ready}
STDOUT_LOG=/tmp/dlinoss_mcp_stdout.log
STDERR_LOG=/tmp/dlinoss_mcp_stderr.log
JSON_PIPE=/tmp/dlinoss_mcp_stdin
TAIL_PID_FILE=/tmp/dlinoss_mcp_tail.pid

echo "[mcp-server] Preparing environment"

# Stop existing instance (if any)
if [[ -f "$PID_FILE" ]]; then
  oldpid=$(cat "$PID_FILE")
  if kill -0 "$oldpid" 2>/dev/null; then
    echo "[mcp-server] Stopping previous instance $oldpid"
    kill "$oldpid" 2>/dev/null || true
    sleep 0.3
    if kill -0 "$oldpid" 2>/dev/null; then
      kill -9 "$oldpid" 2>/dev/null || true
    fi
  fi
  rm -f "$PID_FILE"
fi

if [[ -f "$TAIL_PID_FILE" ]]; then
  tpid=$(cat "$TAIL_PID_FILE") || true
  if [[ -n "${tpid:-}" ]] && kill -0 "$tpid" 2>/dev/null; then
    kill "$tpid" 2>/dev/null || true
  fi
  rm -f "$TAIL_PID_FILE"
fi

rm -f "$READY_FILE" "$STDOUT_LOG" "$STDERR_LOG"
[[ -p "$JSON_PIPE" ]] && rm -f "$JSON_PIPE"
mkfifo "$JSON_PIPE"

echo "[mcp-server] Launching server (fifo=$JSON_PIPE)"

stdbuf -oL -eL cargo run -q -p xtask -- mcp-serve --fft --ready-file "$READY_FILE" <"$JSON_PIPE" \
  >>"$STDOUT_LOG" 2>>"$STDERR_LOG" &
SPID=$!
echo $SPID > "$PID_FILE"

# Tail stdout for user (separate process we can terminate independently)
tail -f "$STDOUT_LOG" & echo $! > "$TAIL_PID_FILE"

printf "[mcp-server] Waiting for readiness"
for i in {1..120}; do
  if [[ -f "$READY_FILE" ]] && grep -q "ready" "$READY_FILE" 2>/dev/null; then
    printf "\n[mcp-server] Ready."
    break
  fi
  sleep 0.25
  printf "."
done
echo

if ! kill -0 "$SPID" 2>/dev/null; then
  echo "[mcp-server] ERROR: server exited early (pid $SPID)" >&2
  echo "--- STDERR (last 40 lines) ---" >&2
  tail -n 40 "$STDERR_LOG" >&2 || true
  exit 1
fi

echo "[mcp-server] PID: $SPID"
echo "[mcp-server] Send a request from another terminal, e.g.:"
echo "  echo '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"dlinoss.ping\",\"params\":{}}' > $JSON_PIPE"
echo "[mcp-server] Or start interactive client: scripts/mcp_client_interactive.sh"
echo "[mcp-server] Stdout log: $STDOUT_LOG"
echo "[mcp-server] Stderr log: $STDERR_LOG"
echo "[mcp-server] To stop: scripts/mcp_server_stop.sh"
