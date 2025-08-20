# D-LinOSS MCP VS Code Extension (Prototype)

This is a minimal client to interact with the embedded `dlinoss-mcp` JSON-RPC server.

## Features
- Start a local MCP server (`xtask mcp-serve`) with optional FFT.
- Send step and FFT requests.
- View raw JSON lines and parsed results in an output channel.

## Develop
```
cd vscode-extension
npm install
npm run watch
```
Press F5 to launch an Extension Development Host, then run the command:
- `D-LinOSS: Start MCP Session`

## Commands
| Command | Action |
|---------|--------|
| D-LinOSS: Start MCP Session | Spawns server and sends `dlinoss.ping` when ready |
| D-LinOSS: Step Simulation | Sends `dlinoss.step {steps}` |
| D-LinOSS: Fetch FFT | Sends `dlinoss.getFft {size}` |

## Notes
- This is a prototype; no reconnection or advanced error handling.
- When you close the VS Code window the spawned process is killed.
