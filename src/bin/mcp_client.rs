use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

fn print_help() {
    println!("Commands:");
    println!("  send <json>    - Send JSON string to MCP server");
    println!(
        "  commcheck-instructionstest - Run a test sequence of JSON-RPC instructions against the server"
    );
    println!("  status         - Print connection and server status details");
    println!("  help           - Show this help message");
    println!("  exit           - Quit the client");
    println!(
        "\nNote: This client auto-starts an MCP server (subprocess). If you want to connect to an existing server, run it separately and modify the client."
    );
}

fn main() {
    // Launch MCP server as child process
    let mut child = Command::new("cargo")
        .args(&["run", "-p", "xtask", "--", "mcp-serve", "--fft"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to start MCP server");

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    let child_id = child.id();
    ctrlc::set_handler(move || {
        println!(
            "\nCtrl+C detected. Killing MCP server (PID {})...",
            child_id
        );
        r.store(false, Ordering::SeqCst);
        // Kill the child process
        let _ = Command::new("kill").arg(format!("{}", child_id)).status();
        std::process::exit(0);
    })
    .expect("Error setting Ctrl-C handler");

    let stdin = child
        .stdin
        .as_mut()
        .expect("Failed to open MCP server stdin");
    let stdout = child
        .stdout
        .as_mut()
        .expect("Failed to open MCP server stdout");
    let mut reader = BufReader::new(stdout);

    println!("MCP Client CLI. Type 'help' for commands.");
    let input = io::stdin();
    // Wait for MCP server ready message before showing prompt
    let mut startup_msg = String::new();
    reader
        .read_line(&mut startup_msg)
        .expect("Failed to read MCP server startup message");
    println!("{}", startup_msg.trim());
    println!("Autostarted server PID: {}", child_id);
    println!("Autostarted server command: cargo run -p xtask -- mcp-serve --fft");
    // Print connection status details
    println!("Connection: stdio (stdin/stdout JSON-RPC)");
    println!("Connected to server PID {} via stdio", child_id);
    print!("\n\x1b[1;32mmcp> \x1b[0m");
    io::stdout().flush().unwrap();
    // Show help on first prompt
    print_help();
    loop {
        let mut line = String::new();
        if input.read_line(&mut line).is_err() {
            println!("Error reading input.");
            continue;
        }
        let line = line.trim();
        match line {
            "status" => {
                let ready = fs::metadata("/tmp/dlinoss_mcp_ready").is_ok();
                println!("Server PID: {}", child_id);
                println!("Server command: cargo run -p xtask -- mcp-serve --fft");
                println!("Connection: stdio (stdin/stdout JSON-RPC)");
                println!("Ready file present: {}", ready);
                println!("Startup message: {}", startup_msg.trim());
            }
            "exit" => {
                println!("Exiting MCP client. Killing MCP server...");
                let _ = child.kill();
                break;
            }
            "help" => {
                print_help();
            }
            "commcheck-instructionstest" => {
                let test_cmds = vec![
                    "{\"jsonrpc\":\"2.0\",\"method\":\"ping\",\"id\":1}",
                    "{\"jsonrpc\":\"2.0\",\"method\":\"get_capabilities\",\"id\":2}",
                    "{\"jsonrpc\":\"2.0\",\"method\":\"help\",\"id\":3}",
                ];
                for cmd in test_cmds {
                    writeln!(stdin, "{}", cmd).expect("Failed to write to MCP server");
                    stdin.flush().expect("Failed to flush MCP server stdin");
                    let mut response = String::new();
                    reader
                        .read_line(&mut response)
                        .expect("Failed to read response");
                    println!("Response: {}", response.trim());
                }
            }
            _ if line.starts_with("send ") => {
                let json = line[5..].trim();
                writeln!(stdin, "{}", json).expect("Failed to write to MCP server");
                stdin.flush().expect("Failed to flush MCP server stdin");
                let mut response = String::new();
                reader
                    .read_line(&mut response)
                    .expect("Failed to read response");
                println!("Server: {}", response.trim());
            }
            _ => {
                println!("Unknown command. Type 'help' for a list of commands.");
            }
        }
        // Always print the prompt after each command
        print!("\n\x1b[1;32mmcp> \x1b[0m");
        io::stdout().flush().unwrap();
    }
}
