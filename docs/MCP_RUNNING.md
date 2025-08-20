# D-LinOSS MCP Server – Running & Integration Guide

This document explains how to run the embedded D-LinOSS MCP (Model Context Protocol style JSON-RPC over stdio) server, interact with it manually, and (prototype) connect an editor/extension.

## Overview

The crate `dlinoss-mcp` exposes a lightweight JSON-RPC 2.0 server over stdin/stdout. It simulates a streaming D-LinOSS layer and exposes methods for stepping the simulation and retrieving recent outputs plus a simple FFT spectrum. Communication is strictly line-delimited JSON objects.

### Transport / Mode
- Mode: stdio (one JSON object per line)
- Readiness: server prints a single line: `{ "mcp_ready": true, "mode": "stdio" }` early during startup. Start scripts also optionally write a readiness file containing `ready` if `DLINOSS_MCP_READY_FILE` is set.

### Implemented Methods (`method` field)
| Method | Description | Params | Result Shape |
|--------|-------------|--------|--------------|
| `dlinoss.ping` | Health & timestamp | `{}` | `{ ok: true, ts: <ms> }` |
| `dlinoss.listMethods` | Enumerate supported methods | `{}` | `string[]` |
| `dlinoss.init` | Initialize simulation (alloc layer + buffers) | `{ state_dim?, input_dim?, output_dim?, delta_t?, t_len? }` | `{ ok: true }` |
| `dlinoss.step` | Advance simulation synthetic input & produce outputs | `{ steps? }` | `{ current_t, last_output }` |
| `dlinoss.status` | Current cursor & config | `{}` | `{ current_t, config: {...} }` |
| `dlinoss.pause` | (No-op placeholder) | `{}` | `{ paused: true }` |
| `dlinoss.resume` | (No-op placeholder) | `{}` | `{ paused: false }` |
| `dlinoss.getState` | Fetch ring buffer slice | `{ which: "rings", limit? }` | `{ data: f32[] }` |
| `dlinoss.getFft` | FFT (feature or naive) of latest samples | `{ size? }` | `{ spectrum: [{ f, mag }, ...] }` |

Error responses follow JSON-RPC 2.0 with `error: { code, message, data? }`.

## 1. Build Everything

Fast path (recommended):
```
cargo run -p xtask -- ci
```
Naive path:
```
cargo build
cargo test
```

## 2. Start the MCP Server (Two Options)

### A) Direct via xtask (foreground)
```
# CPU only
cargo run -p xtask -- mcp-serve
# With FFT feature
cargo run -p xtask -- mcp-serve --fft
```
The process prints readiness JSON then streams JSON responses per line.

Optional readiness file:
```
DLINOSS_MCP_READY_FILE=/tmp/dlinoss_mcp_ready cargo run -p xtask -- mcp-serve --fft
```

### B) Managed Background Script (FIFO stdin)
Use the helper script which creates a named pipe so you can push JSON from other shells without closing stdin:
```
./scripts/mcp_server_start.sh
```
Logs:
- Stdout: `/tmp/dlinoss_mcp_stdout.log`
- Stderr: `/tmp/dlinoss_mcp_stderr.log`
- PID file: `/tmp/dlinoss_mcp_server.pid`
- Readiness file: `/tmp/dlinoss_mcp_ready`
- Write requests to the pipe: `/tmp/dlinoss_mcp_stdin`

Stop:
```
./scripts/mcp_server_stop.sh
```

## 3. Manual Interaction Examples

Assuming FIFO mode:
```
echo '{"jsonrpc":"2.0","id":1,"method":"dlinoss.ping","params":{}}' > /tmp/dlinoss_mcp_stdin

echo '{"jsonrpc":"2.0","id":2,"method":"dlinoss.init","params":{"state_dim":16,"input_dim":2,"output_dim":3,"t_len":5000}}' > /tmp/dlinoss_mcp_stdin

echo '{"jsonrpc":"2.0","id":3,"method":"dlinoss.step","params":{"steps":256}}' > /tmp/dlinoss_mcp_stdin

echo '{"jsonrpc":"2.0","id":4,"method":"dlinoss.getState","params":{"which":"rings","limit":64}}' > /tmp/dlinoss_mcp_stdin

echo '{"jsonrpc":"2.0","id":5,"method":"dlinoss.getFft","params":{"size":128}}' > /tmp/dlinoss_mcp_stdin
```
Tail the stdout log to view responses:
```
tail -f /tmp/dlinoss_mcp_stdout.log
```

## 4. Run the Built-In Demo Flow
```
# Quick scripted roundtrip (ping, init, step, state, fft)
cargo run -p xtask -- mcp-demo --steps 300 --fft-size 128 --fft
```

## 5. Prototype VS Code Extension (Stdio Client)

A minimal extension can spawn the server (or attach to an already running `xtask mcp-serve`) and implement a line-oriented JSON-RPC client.

Key steps implemented in the provided scaffold (see `vscode-extension/` if created):
1. Activate command runs `cargo run -p xtask -- mcp-serve --fft` in a pseudo-terminal.
2. Collect readiness line; after that send `dlinoss.ping` and display result in an output channel.
3. Provide commands: `D-LinOSS: Step Simulation` and `D-LinOSS: FFT` which send `dlinoss.step` & `dlinoss.getFft` respectively.

If you prefer attaching to the FIFO script:
- Start `./scripts/mcp_server_start.sh` first.
- Extension instead opens write stream to `/tmp/dlinoss_mcp_stdin` and tails `/tmp/dlinoss_mcp_stdout.log`.

## 6. Data Flow Notes
- The server currently generates synthetic sinusoidal inputs each call to `step` (no external input injection yet).
- `output_ring` holds the last N (≈4096) scalar outputs (averaging multi-output dim).
- FFT: uses Candle `rfft` when compiled with `--features fft`; otherwise falls back to `naive_fft` (O(N^2)).

## 7. Extending the Protocol
Potential next methods:
- `dlinoss.injectInput` with explicit input window
- `dlinoss.reset`
- `dlinoss.subscribe` / push notifications (would require multiplexed channel or embedding event messages)
- Structured error codes registry

## 8. Observability
Add flags/env to increase diagnostic verbosity (log perf counters, layer parameters). Could add a `dlinoss.status.verbose` returning internal state norms.

## 9. Testing & CI
- `crates/dlinoss-mcp/tests/roundtrip.rs` already validates a full basic flow.
- Add more tests for error handling & FFT gating.

## 10. Quick Troubleshooting
| Symptom | Cause | Fix |
|---------|-------|-----|
| No readiness line | Build failure or crash | Check `/tmp/dlinoss_mcp_stderr.log` |
| `method not found` | Misspelled method | Use `dlinoss.listMethods` |
| FFT error `not enough samples` | Requested size > ring length | Perform more `dlinoss.step` first |
| Hang on FIFO write | Writer closed / server exited | Verify PID in `/tmp/dlinoss_mcp_server.pid` |

## 11. Assistant / Copilot Attachment
To let this assistant inspect the live process:
- Run with logs & readiness file via script.
- The assistant can read `/tmp/dlinoss_mcp_stdout.log` and `/tmp/dlinoss_mcp_stdin` (write) in future enhanced tooling scenarios (conceptual; current session is read-only for those paths unless explicitly opened).

## 12. Security Considerations
Currently no authentication. If exposing beyond local dev, restrict via:
- Named pipe permissions (chmod 600)
- Add shared secret param required in each request
- Optionally move to TCP with loopback bind only.

---
This is a prototype; iterate as protocol requirements mature.
