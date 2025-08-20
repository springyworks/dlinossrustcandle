#!/usr/bin/env bash
# Simple persistent interactive client for the dlinoss MCP server running in Terminal 1.
# Opens a fifo so we can keep stdin of the server open while you type or send scripted JSON lines.
set -euo pipefail
PID_FILE=/tmp/dlinoss_mcp_server.pid
if [[ ! -f "$PID_FILE" ]]; then
  echo "[mcp-client] Server pid file not found: $PID_FILE" >&2
  echo "Start the server first: scripts/mcp_server_start.sh" >&2
  exit 1
fi
PID=$(cat "$PID_FILE")
if ! kill -0 "$PID" 2>/dev/null; then
  echo "[mcp-client] Server process $PID not running" >&2
  exit 1
fi
JSON_PIPE=${JSON_PIPE:-/tmp/dlinoss_mcp_stdin}
if [[ ! -p "$JSON_PIPE" ]]; then
  echo "[mcp-client] Named pipe $JSON_PIPE not found. (Server script creates it)" >&2
  echo "Re-run server with latest start script that exposes a fifo for stdin." >&2
  exit 1
fi
cat <<'HELP'
Interactive MCP client (writes to server FIFO, read responses in the server terminal tail).
Usage:
  Type a single-line JSON-RPC request then Enter, OR use a shortcut command:
    :ping         -> dlinoss.ping
    :methods      -> dlinoss.listMethods
    :init         -> dlinoss.init (defaults)
    :init <json>  -> dlinoss.init with custom params object (e.g. :init {"state_dim":8})
    :step         -> dlinoss.step {"steps":1}
    :step <n>     -> dlinoss.step {"steps":n}
    :status       -> dlinoss.status
    :getstate     -> dlinoss.getState {"which":"rings","limit":64}
    :getstate <n> -> dlinoss.getState {"which":"rings","limit":n}
    :fft          -> dlinoss.getFft {"size":256}
    :fft <n>      -> dlinoss.getFft {"size":n}
    :file <path>  -> send raw file contents (minified) as request
    :help         -> show this help
    :quit / :q    -> exit client

Examples:
  {"jsonrpc":"2.0","id":1,"method":"dlinoss.ping","params":{}}
  :init {"state_dim":12,"input_dim":3,"output_dim":2,"t_len":5000}
  :step 100
  :getstate 200
  :fft 512
HELP
next_id=1
send_json(){
  local line="$1"
  echo "$line" > "$JSON_PIPE"
  echo "[sent] $line"
}
minify_file(){
  tr -d '\n' < "$1" | sed 's/[[:space:]]\+/ /g'
}
while IFS= read -r -p "> " cmd; do
  case "$cmd" in
    :quit|:q|:exit) break ;;
  :help) cat <<'HLP'
Shortcuts:
  :ping | :methods | :init [json] | :step [n] | :status | :getstate [limit] | :fft [size] | :file <path> | :quit
HLP
    ;;
  :ping) send_json "{\"jsonrpc\":\"2.0\",\"id\":$next_id,\"method\":\"dlinoss.ping\",\"params\":{}}" ;;
  :methods) send_json "{\"jsonrpc\":\"2.0\",\"id\":$next_id,\"method\":\"dlinoss.listMethods\",\"params\":{}}" ;;
  :init) send_json "{\"jsonrpc\":\"2.0\",\"id\":$next_id,\"method\":\"dlinoss.init\",\"params\":{}}" ;;
  :init\ *)
    params="${cmd#:init }"
    send_json "{\"jsonrpc\":\"2.0\",\"id\":$next_id,\"method\":\"dlinoss.init\",\"params\":$params}" ;;
  :step) send_json "{\"jsonrpc\":\"2.0\",\"id\":$next_id,\"method\":\"dlinoss.step\",\"params\":{}}" ;;
  :step\ *)
    n="${cmd#:step }"
    send_json "{\"jsonrpc\":\"2.0\",\"id\":$next_id,\"method\":\"dlinoss.step\",\"params\":{\"steps\":$n}}" ;;
  :status) send_json "{\"jsonrpc\":\"2.0\",\"id\":$next_id,\"method\":\"dlinoss.status\",\"params\":{}}" ;;
  :getstate) send_json "{\"jsonrpc\":\"2.0\",\"id\":$next_id,\"method\":\"dlinoss.getState\",\"params\":{\"which\":\"rings\",\"limit\":64}}" ;;
  :getstate\ *)
    lim="${cmd#:getstate }"
    send_json "{\"jsonrpc\":\"2.0\",\"id\":$next_id,\"method\":\"dlinoss.getState\",\"params\":{\"which\":\"rings\",\"limit\":$lim}}" ;;
  :fft) send_json "{\"jsonrpc\":\"2.0\",\"id\":$next_id,\"method\":\"dlinoss.getFft\",\"params\":{\"size\":256}}" ;;
  :fft\ *)
    sz="${cmd#:fft }"
    send_json "{\"jsonrpc\":\"2.0\",\"id\":$next_id,\"method\":\"dlinoss.getFft\",\"params\":{\"size\":$sz}}" ;;
    :file*)
       fpath="${cmd#:file }"
       if [[ -f "$fpath" ]]; then
         send_json "$(minify_file "$fpath")"
       else
         echo "[mcp-client] File not found: $fpath" >&2
       fi
       ;;
    *)
       if [[ "$cmd" =~ ^\{.*\}$ ]]; then
         send_json "$cmd"
       elif [[ -z "$cmd" ]]; then
         continue
       else
         echo "[mcp-client] Unknown command. Use :help keywords (:ping :init :step :status :getstate :fft :methods :quit) or raw JSON." >&2
       fi
       ;;
  esac
  next_id=$((next_id+1))
done
echo "[mcp-client] Bye."