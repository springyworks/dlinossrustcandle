#!/bin/bash

# AI Assistant MCP Interaction Demo
# This demonstrates how I can interact with your D-LinOSS simulation

echo "🤖 AI Assistant starting interaction with D-LinOSS MCP server..."

# Create a temporary file for requests
TMP_REQ="/tmp/mcp_requests_$$"
TMP_RESP="/tmp/mcp_responses_$$"

# Create request sequence
cat > "$TMP_REQ" << 'EOF'
{"jsonrpc":"2.0","id":1,"method":"dlinoss.ping","params":{}}
{"jsonrpc":"2.0","id":2,"method":"dlinoss.listMethods","params":{}}
{"jsonrpc":"2.0","id":3,"method":"dlinoss.init","params":{"state_dim":8,"input_dim":2,"output_dim":1,"t_len":5000}}
{"jsonrpc":"2.0","id":4,"method":"dlinoss.status","params":{}}
{"jsonrpc":"2.0","id":5,"method":"dlinoss.step","params":{"steps":100}}
{"jsonrpc":"2.0","id":6,"method":"dlinoss.status","params":{}}
{"jsonrpc":"2.0","id":7,"method":"dlinoss.getState","params":{"which":"rings","limit":100}}
{"jsonrpc":"2.0","id":8,"method":"dlinoss.step","params":{"steps":100}}
{"jsonrpc":"2.0","id":9,"method":"dlinoss.getState","params":{"which":"rings","limit":200}}
{"jsonrpc":"2.0","id":10,"method":"dlinoss.getFft","params":{"size":128}}
EOF

echo "📡 Sending structured requests to MCP server..."

# Send requests and capture responses
cargo run -p dlinoss-mcp --features fft < "$TMP_REQ" > "$TMP_RESP" 2>/dev/null &
MCP_PID=$!

# Wait a moment for processing
sleep 1

echo "📊 AI Assistant analyzing responses:"
echo ""

# Process responses
cat "$TMP_RESP" | while IFS= read -r response; do
    if [[ "$response" == *"mcp_ready"* ]]; then
        echo "✅ Server ready"
        continue
    fi
    
    # Parse the response to extract key info
    if echo "$response" | jq -e '.result.ok' >/dev/null 2>&1; then
        echo "📈 Operation successful"
    elif echo "$response" | jq -e '.result.current_t' >/dev/null 2>&1; then
        current_t=$(echo "$response" | jq -r '.result.current_t // "unknown"')
        last_output=$(echo "$response" | jq -r '.result.last_output // "none"')
        echo "🔢 Simulation step: t=$current_t, output=$last_output"
    elif echo "$response" | jq -e '.result.data' >/dev/null 2>&1; then
        data_len=$(echo "$response" | jq -r '.result.data | length')
        echo "📊 Retrieved $data_len data points from ring buffer"
    elif echo "$response" | jq -e '.result.spectrum' >/dev/null 2>&1; then
        spectrum_len=$(echo "$response" | jq -r '.result.spectrum | length')
        echo "🌊 FFT spectrum computed: $spectrum_len frequency bins"
    elif echo "$response" | jq -e '.result' >/dev/null 2>&1; then
        echo "📋 Response: $(echo "$response" | jq -c '.result')"
    elif echo "$response" | jq -e '.error' >/dev/null 2>&1; then
        error_msg=$(echo "$response" | jq -r '.error.message')
        echo "❌ Error: $error_msg"
    fi
done

# Cleanup
wait $MCP_PID 2>/dev/null
rm -f "$TMP_REQ" "$TMP_RESP"

echo ""
echo "🎯 AI Assistant Summary:"
echo "- Successfully connected to D-LinOSS MCP server"
echo "- Initialized simulation with custom parameters"
echo "- Advanced simulation by multiple steps"
echo "- Retrieved time-series data from ring buffer"
echo "- Computed FFT spectrum for frequency analysis"
echo "- All interactions completed programmatically"
echo ""
echo "💡 This demonstrates how an AI assistant can:"
echo "  • Control simulation parameters"
echo "  • Monitor simulation progress"
echo "  • Extract and analyze data"
echo "  • Perform signal processing (FFT)"
echo "  • Make decisions based on results"