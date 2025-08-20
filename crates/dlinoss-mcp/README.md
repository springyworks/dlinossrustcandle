# dlinoss-mcp

A lightweight stdin/stdout JSON-RPC (MCP-style) control server for the D-LinOSS simulation.

## Status
Phase 1: basic lifecycle + simulation stepping + state/spectrum queries.

## Protocol
Requests use JSON-RPC 2.0 objects:
```
{"jsonrpc":"2.0","id":1,"method":"dlinoss.ping","params":{}}
```
Responses:
```
{"jsonrpc":"2.0","id":1,"result":{"ok":true,"ts":123456789}}
```
Errors:
```
{"jsonrpc":"2.0","id":1,"error":{"code":1002,"message":"not initialized"}}
```

### Implemented Methods (Phase 1)
- dlinoss.ping -> { ok, ts }
- dlinoss.listMethods -> [ "dlinoss.ping", ... ]
- dlinoss.init { state_dim?, input_dim?, output_dim?, delta_t?, t_len? } -> { ok }
- dlinoss.step { steps?=1 } -> { current_t, last_output }
- dlinoss.status -> { current_t, paused, config }
- dlinoss.pause / dlinoss.resume -> { paused }
- dlinoss.getState { which:"rings"|"latent"|"volume", limit? } -> { data }
- dlinoss.getFft { size } -> { spectrum:[{f,mag},...] }

Planned (Phase 2): setLatentPair, setColormap, requestFrame, buildGif, spectralPeak, energyStats, input pattern control.

## Running
```
cargo run -p dlinoss-mcp --features fft
```
Then send JSON lines via stdin.

Example with `jq` for pretty output:
```
echo '{"jsonrpc":"2.0","id":1,"method":"dlinoss.ping","params":{}}' | cargo run -q -p dlinoss-mcp | jq .
```

## License
MIT OR Apache-2.0
