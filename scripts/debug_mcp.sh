#!/bin/bash

# Debug MCP Demo - Let's see what's happening with the outputs
echo "🔍 Debugging MCP server output collection..."

cat > /tmp/debug_mcp.txt << 'EOF'
{"jsonrpc":"2.0","id":1,"method":"dlinoss.init","params":{"state_dim":4,"input_dim":1,"output_dim":1,"t_len":1000}}
{"jsonrpc":"2.0","id":2,"method":"dlinoss.step","params":{"steps":50}}
{"jsonrpc":"2.0","id":3,"method":"dlinoss.status","params":{}}
{"jsonrpc":"2.0","id":4,"method":"dlinoss.getState","params":{"which":"rings","limit":100}}
EOF

echo "📡 Sending debug requests..."

cargo run -p dlinoss-mcp --features fft < /tmp/debug_mcp.txt 2>&1 | while IFS= read -r line; do
    if [[ "$line" == *"mcp_ready"* ]]; then
        echo "✅ Server ready"
    elif [[ "$line" == *"result"* ]]; then
        echo "📊 $line"
    elif [[ "$line" == *"error"* ]]; then
        echo "❌ $line"
    elif [[ "$line" != "" ]]; then
        echo "🔍 $line"
    fi
done

rm -f /tmp/debug_mcp.txt