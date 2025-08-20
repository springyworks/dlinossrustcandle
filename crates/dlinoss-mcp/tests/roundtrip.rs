use serde_json::Value;
use serde_json::json;
use std::io::{BufRead, Write};
use std::process::{Command, Stdio};

/// Send a JSON-RPC request and return parsed response.
fn send(
    req: &Value,
    stdin: &mut std::process::ChildStdin,
    stdout: &mut std::io::Lines<std::io::BufReader<std::process::ChildStdout>>,
) -> Value {
    let line = serde_json::to_string(req).expect("serialize request");
    writeln!(stdin, "{}", line).expect("write request");
    stdin.flush().ok();
    let resp_line = stdout.next().expect("response line").expect("read line");
    serde_json::from_str(&resp_line).expect("parse response")
}

fn extract_result(resp: &Value) -> &Value {
    resp.get("result").expect("result field present")
}

#[test]
fn mcp_basic_roundtrip() {
    // Use compiled binary path provided by Cargo; ensures this runs after build.
    let bin = env!("CARGO_BIN_EXE_dlinoss-mcp");
    let mut child = Command::new(bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn mcp server");
    let stdin = child.stdin.as_mut().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let mut lines = std::io::BufReader::new(stdout).lines();

    // 1. ping
    let ping_req = json!({"jsonrpc":"2.0","id":1,"method":"dlinoss.ping","params":{}});
    let ping_resp = send(&ping_req, stdin, &mut lines);
    assert!(
        ping_resp.get("result").is_some(),
        "ping should return result: {ping_resp}"
    );

    // 2. init
    let init_req = json!({"jsonrpc":"2.0","id":2,"method":"dlinoss.init","params":{"state_dim":16,"input_dim":2,"output_dim":3,"t_len":2000}});
    let init_resp = send(&init_req, stdin, &mut lines);
    assert!(
        extract_result(&init_resp)
            .get("ok")
            .and_then(Value::as_bool)
            == Some(true),
        "init ok"
    );

    // 3. step (enough samples for FFT)
    let step_req = json!({"jsonrpc":"2.0","id":3,"method":"dlinoss.step","params":{"steps":300}});
    let step_resp = send(&step_req, stdin, &mut lines);
    let cur_t = extract_result(&step_resp)
        .get("current_t")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    assert!(cur_t >= 300, "current_t after step >=300, got {cur_t}");

    // 4. getState rings (limit 32)
    let state_req = json!({"jsonrpc":"2.0","id":4,"method":"dlinoss.getState","params":{"which":"rings","limit":32}});
    let state_resp = send(&state_req, stdin, &mut lines);
    let data = extract_result(&state_resp)
        .get("data")
        .and_then(Value::as_array)
        .expect("data array");
    assert!(!data.is_empty(), "ring data not empty");
    assert!(data.len() <= 32, "ring data length <= 32");

    // 5. FFT (size 128)
    let fft_req = json!({"jsonrpc":"2.0","id":5,"method":"dlinoss.getFft","params":{"size":128}});
    let fft_resp = send(&fft_req, stdin, &mut lines);
    let spec = extract_result(&fft_resp)
        .get("spectrum")
        .and_then(Value::as_array)
        .expect("spectrum array");
    assert!(!spec.is_empty(), "spectrum not empty");
    let first = &spec[0];
    assert!(
        first.get("f").is_some() && first.get("mag").is_some(),
        "spectrum element shape"
    );

    // Explicitly close stdin by dropping the owned handle (clone ownership via take in earlier setup would simplify; here just let scope end)
    // (No action needed; we don't have ownership separate from &mut reference)
    let _ = child.wait();
}
