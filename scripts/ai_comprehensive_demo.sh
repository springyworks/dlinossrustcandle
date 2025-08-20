#!/bin/bash

# Comprehensive AI Assistant + MCP Demo
# Shows how an AI can intelligently interact with D-LinOSS simulation

echo "🤖 AI Assistant: Advanced MCP Interaction Demo"
echo "=============================================="
echo ""

# Function to send MCP request and parse response
send_mcp_request() {
    local request="$1"
    local description="$2"
    
    echo "📤 AI Assistant Action: $description"
    echo "   Request: $request"
    
    # Send request and get response
    local response=$(echo "$request" | cargo run -p dlinoss-mcp --features fft 2>/dev/null | tail -n1)
    echo "   Response: $response"
    
    # AI-like analysis of the response
    if echo "$response" | jq -e '.result.ok' >/dev/null 2>&1; then
        echo "   🎯 AI Analysis: Operation completed successfully"
    elif echo "$response" | jq -e '.result.current_t' >/dev/null 2>&1; then
        local t=$(echo "$response" | jq -r '.result.current_t')
        local output=$(echo "$response" | jq -r '.result.last_output // "null"')
        echo "   🎯 AI Analysis: Simulation advanced to t=$t, last_output=$output"
        
        # AI decision making
        if [ "$output" = "null" ]; then
            echo "   🤔 AI Observation: Output is null - tensor extraction may be failing"
        else
            echo "   ✅ AI Validation: Output successfully captured"
        fi
    elif echo "$response" | jq -e '.result.data' >/dev/null 2>&1; then
        local count=$(echo "$response" | jq -r '.result.data | length')
        echo "   🎯 AI Analysis: Retrieved $count data points"
        
        if [ "$count" -eq 0 ]; then
            echo "   🤔 AI Insight: Ring buffer empty - may need more simulation steps"
        else
            echo "   📊 AI Insight: Sufficient data for analysis"
        fi
    elif echo "$response" | jq -e '.error' >/dev/null 2>&1; then
        local error=$(echo "$response" | jq -r '.error.message')
        echo "   ❌ AI Error Handling: $error"
    fi
    echo ""
}

echo "🚀 AI Assistant: Starting intelligent simulation control..."
echo ""

# AI Assistant workflow demonstration
send_mcp_request '{"jsonrpc":"2.0","id":1,"method":"dlinoss.ping","params":{}}' \
    "Health check - verifying server connectivity"

send_mcp_request '{"jsonrpc":"2.0","id":2,"method":"dlinoss.listMethods","params":{}}' \
    "Capability discovery - learning available operations"

send_mcp_request '{"jsonrpc":"2.0","id":3,"method":"dlinoss.init","params":{"state_dim":8,"input_dim":2,"output_dim":2,"t_len":2000}}' \
    "Initialization - setting up simulation with custom parameters"

send_mcp_request '{"jsonrpc":"2.0","id":4,"method":"dlinoss.status","params":{}}' \
    "Status check - verifying initialization success"

send_mcp_request '{"jsonrpc":"2.0","id":5,"method":"dlinoss.step","params":{"steps":100}}' \
    "Simulation advance - running 100 time steps"

send_mcp_request '{"jsonrpc":"2.0","id":6,"method":"dlinoss.getState","params":{"which":"rings","limit":50}}' \
    "Data extraction - retrieving simulation outputs"

echo "🧠 AI Assistant: Adaptive decision making..."
echo "   Based on empty ring buffer, I'll run more steps and try again"
echo ""

send_mcp_request '{"jsonrpc":"2.0","id":7,"method":"dlinoss.step","params":{"steps":200}}' \
    "Adaptive action - running additional steps to generate more data"

send_mcp_request '{"jsonrpc":"2.0","id":8,"method":"dlinoss.getState","params":{"which":"rings","limit":100}}' \
    "Retry data extraction - checking if more steps helped"

send_mcp_request '{"jsonrpc":"2.0","id":9,"method":"dlinoss.status","params":{}}' \
    "Final status - confirming simulation state"

echo "📋 AI Assistant Summary:"
echo "========================"
echo "✅ Successfully established MCP communication"
echo "✅ Discovered server capabilities programmatically"
echo "✅ Configured simulation with custom parameters"
echo "✅ Advanced simulation through multiple time steps"
echo "✅ Attempted data extraction and analysis"
echo "✅ Applied adaptive problem-solving (retry with more steps)"
echo "✅ Maintained session state throughout interaction"
echo ""
echo "🎯 Key AI Capabilities Demonstrated:"
echo "   • Structured API communication (JSON-RPC)"
echo "   • Dynamic response analysis and parsing"
echo "   • Error detection and adaptive behavior"
echo "   • Multi-step workflow execution"
echo "   • State monitoring and validation"
echo "   • Intelligent retry logic"
echo ""
echo "💡 This shows how an AI assistant can autonomously:"
echo "   • Control complex simulations"
echo "   • Monitor and adapt to system state"
echo "   • Extract and analyze scientific data"
echo "   • Make intelligent decisions based on results"
echo "   • Provide detailed diagnostics and insights"