# MCP Server Documentation for D-LinOSS

## Overview
This document explains how to use the D-LinOSS MCP (Model Context Protocol) server within VS Code to control and interact with the simulation through AI assistants.

## Setup

### 1. VS Code Configuration
The `.vscode/settings.json` file has been configured with:

```json
{
  "mcp.servers": {
    "dlinoss-simulation": {
      "command": "cargo",
      "args": ["run", "-p", "xtask", "--", "mcp-serve", "--fft"],
      "cwd": "${workspaceFolder}",
      "env": {
        "DLINOSS_MCP_READY_FILE": "/tmp/dlinoss_mcp_ready"
      }
    }
  },
  "languageModel.mcp.enable": true,
  "languageModel.mcp.servers": ["dlinoss-simulation"]
}
```

### 2. Required Extensions
- `zebradev.mcp-server-runner` - Installed ✅
- Optional: `automatalabs.copilot-mcp` - For enhanced MCP management

### 3. Server Capabilities
The D-LinOSS MCP server provides these methods:

- **`dlinoss.ping`** - Health check
- **`dlinoss.listMethods`** - Discover capabilities
- **`dlinoss.init`** - Initialize simulation with parameters
- **`dlinoss.step`** - Advance simulation by N steps
- **`dlinoss.status`** - Get current simulation state
- **`dlinoss.getState`** - Retrieve time-series data from ring buffers
- **`dlinoss.getFft`** - Compute FFT spectrum of outputs
- **`dlinoss.pause`/`dlinoss.resume`** - Control simulation flow

## Usage Examples

### Via AI Assistant (Copilot/Chat)
Once configured, you can interact with your simulation through natural language:

```
@workspace "Start a D-LinOSS simulation with 16 state dimensions, run 500 steps, and show me the FFT spectrum"
```

The AI assistant will:
1. Initialize the simulation via `dlinoss.init`
2. Run steps via `dlinoss.step`
3. Extract data via `dlinoss.getState`
4. Compute spectrum via `dlinoss.getFft`
5. Analyze and present results

### Via MCP Server Runner Extension
1. Open Command Palette (`Ctrl+Shift+P`)
2. Search for "MCP Server Runner"
3. Select "dlinoss-simulation" server
4. Use the UI to send JSON-RPC requests

### Manual Testing
Run the server directly:

```bash
# In terminal
cargo run -p xtask -- mcp-serve --fft

# In another terminal, send requests:
echo '{"jsonrpc":"2.0","id":1,"method":"dlinoss.ping","params":{}}' | nc localhost stdin
```

## Integration with VS Code Language Models

The MCP server integrates with VS Code's language model API, allowing:

1. **Contextual Code Generation**: AI can analyze simulation data and generate relevant code
2. **Intelligent Documentation**: Auto-generate docs based on simulation behavior
3. **Adaptive Debugging**: AI can modify simulation parameters based on output analysis
4. **Interactive Exploration**: Natural language queries about simulation state

## Example AI Workflows

### 1. Simulation Health Check
```
"Check if the D-LinOSS simulation is running properly"
```
→ AI calls `dlinoss.ping` and `dlinoss.status`

### 2. Parameter Optimization
```
"Run the simulation with different damping parameters and find the most stable configuration"
```
→ AI iteratively calls `dlinoss.init` with varying parameters

### 3. Spectral Analysis
```
"Analyze the frequency response of the current simulation"
```
→ AI calls `dlinoss.getFft` and interprets the spectrum

### 4. Data Export
```
"Export the last 1000 simulation data points to CSV format"
```
→ AI calls `dlinoss.getState` and formats data

## Troubleshooting

### Server Not Starting
- Check that Rust/Cargo is in PATH
- Verify the workspace folder is correct
- Look for compilation errors in the Output panel

### MCP Not Available in Chat
- Ensure `languageModel.mcp.enable` is true
- Restart VS Code after configuration changes
- Check that the MCP Server Runner extension is enabled

### Connection Issues
- Check `/tmp/dlinoss_mcp_ready` file exists when server starts
- Verify no port conflicts (server uses stdin/stdout)
- Look for error messages in Terminal output

## Advanced Configuration

### Custom Server Parameters
Modify `.vscode/settings.json` to add custom arguments:

```json
{
  "mcp.servers": {
    "dlinoss-simulation": {
      "args": ["run", "-p", "xtask", "--", "mcp-serve", "--fft", "--ready-file", "/custom/path"]
    }
  }
}
```

### Environment Variables
```json
{
  "mcp.servers": {
    "dlinoss-simulation": {
      "env": {
        "DLINOSS_MCP_READY_FILE": "/tmp/dlinoss_ready",
        "RUST_LOG": "debug"
      }
    }
  }
}
```

## Benefits

✅ **Natural Language Control**: Use plain English to control complex simulations
✅ **Intelligent Analysis**: AI interprets simulation data and provides insights  
✅ **Automated Workflows**: Chain multiple simulation operations together
✅ **Real-time Interaction**: Dynamic parameter adjustment based on results
✅ **Documentation Integration**: Auto-generate reports and explanations
✅ **Error Handling**: AI can detect and respond to simulation issues

This setup transforms VS Code into an intelligent simulation control interface!