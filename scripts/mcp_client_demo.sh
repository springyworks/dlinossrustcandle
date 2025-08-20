#!/bin/bash

# MCP Client Demo - Shows how an AI assistant can interact with D-LinOSS
echo "🤖 AI Assistant connecting to D-LinOSS MCP server..."

# Start the server in background
cargo run -p xtask -- mcp-serve --fft --ready-file /tmp/dlinoss_ready &
SERVER_PID=$!

# Wait for readiness
echo "⏳ Waiting for server to be ready..."
for i in {1..10}; do
    if [ -f /tmp/dlinoss_ready ]; then
        echo "✅ Server ready!"
        break
    fi
    sleep 0.5
done

if [ ! -f /tmp/dlinoss_ready ]; then
    echo "❌ Server failed to start"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Create named pipes for communication
PIPE_IN="/tmp/mcp_in_$$"
PIPE_OUT="/tmp/mcp_out_$$"
mkfifo "$PIPE_IN" "$PIPE_OUT"

# Start client connection
(
    echo "🔗 Establishing MCP connection..."
    
    # 1. List available methods
    echo '{"jsonrpc":"2.0","id":1,"method":"dlinoss.listMethods","params":{}}'
    
    # 2. Ping the server
    echo '{"jsonrpc":"2.0","id":2,"method":"dlinoss.ping","params":{}}'
    
    # 3. Initialize simulation
    echo '{"jsonrpc":"2.0","id":3,"method":"dlinoss.init","params":{"state_dim":16,"input_dim":2,"output_dim":3,"t_len":10000}}'
    
    # 4. Run simulation steps
    echo '{"jsonrpc":"2.0","id":4,"method":"dlinoss.step","params":{"steps":300}}'
    
    # 5. Get simulation state
    echo '{"jsonrpc":"2.0","id":5,"method":"dlinoss.getState","params":{"which":"rings","limit":256}}'
    
    # 6. Compute FFT spectrum
    echo '{"jsonrpc":"2.0","id":6,"method":"dlinoss.getFft","params":{"size":256}}'
    
) | while IFS= read -r request; do
    echo "📤 Sending: $request"
    echo "$request"
    sleep 0.2
done | cargo run -p dlinoss-mcp --features fft | while IFS= read -r response; do
    echo "📥 Received: $response"
done

# Cleanup
rm -f "$PIPE_IN" "$PIPE_OUT" /tmp/dlinoss_ready
kill $SERVER_PID 2>/dev/null
echo "🏁 Demo complete"