use anyhow::Result;
use candlekos::{DType, Device, Tensor};
use dlinossrustcandle::{DLinOssLayer, DLinOssLayerConfig};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::io::{self, BufRead, Write};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug)]
struct SimState {
    layer: DLinOssLayer,
    device: Device,
    input_dim: usize,
    output_dim: usize,
    state_dim: usize,
    delta_t: f64,
    t_len: usize,
    current_t: usize,
    input_cache: Vec<f32>,
    output_ring: Vec<f32>,
    ring_cap: usize,
    last_output: Option<f32>,
    last_state: Option<Tensor>,
}

impl SimState {
    fn new(cfg: &DLinOssLayerConfig, t_len: usize, device: Device) -> Result<Self> {
        let layer = DLinOssLayer::deterministic(cfg.clone(), &device)?;
        Ok(Self {
            layer,
            device,
            input_dim: cfg.input_dim,
            output_dim: cfg.output_dim,
            state_dim: cfg.state_dim,
            delta_t: cfg.delta_t,
            t_len,
            current_t: 0,
            input_cache: vec![0.0; t_len * cfg.input_dim],
            output_ring: Vec::with_capacity(4096),
            ring_cap: 4096,
            last_output: None,
            last_state: None,
        })
    }

    fn step(&mut self, steps: usize) -> Result<()> {
        if self.current_t >= self.t_len {
            return Ok(());
        }
        let actual = steps.min(self.t_len - self.current_t);
        let start = self.current_t;
        // Fill synthetic input
        for t in start..start + actual {
            let phase = t as f32 * 0.07;
            let base = t * self.input_dim;
            if let Some(slot) = self.input_cache.get_mut(base) {
                *slot = phase.sin();
            }
            if self.input_dim > 1 {
                if let Some(slot) = self.input_cache.get_mut(base + 1) {
                    *slot = (0.5 * phase).cos();
                }
            }
        }
        let slice = &self.input_cache[start * self.input_dim..(start + actual) * self.input_dim];
        let u = Tensor::from_slice(slice, (1, actual, self.input_dim), &self.device)?;
        let (w_seq, y) = self.layer.forward_with_state(&u, None)?;
        // Keep last state snapshot
        if let Ok(last) = w_seq.narrow(1, w_seq.dims()[1] - 1, 1) {
            self.last_state = last.squeeze(1).ok();
        }
        // Flatten outputs across q if q>1 via mean BEFORE extracting last scalar
        let mut y_flat = y.squeeze(0)?; // [actual, q]
        if y_flat.dims().len() == 2 && y_flat.dims()[1] > 1 {
            y_flat = y_flat.mean(1)?;
        } // now [actual]
        if let Ok(vals) = y_flat.to_vec1::<f32>() {
            if let Some(last) = vals.last() {
                self.last_output = Some(*last);
            }
            for v in vals {
                push_ring(&mut self.output_ring, v, self.ring_cap);
            }
        }
        self.current_t += actual;
        Ok(())
    }
}

fn push_ring(buf: &mut Vec<f32>, v: f32, cap: usize) {
    if cap == 0 {
        return;
    }
    if buf.len() >= cap {
        let drop = (cap / 4).max(1);
        buf.drain(0..drop);
    }
    buf.push(v);
}

// JSON-RPC core types
#[derive(Deserialize)]
struct RpcEnvelope {
    jsonrpc: String,
    id: serde_json::Value,
    method: String,
    #[serde(default)]
    params: serde_json::Value,
}
#[derive(Serialize)]
#[serde(untagged)]
enum RpcResponse {
    Ok {
        jsonrpc: &'static str,
        id: serde_json::Value,
        result: serde_json::Value,
    },
    Err {
        jsonrpc: &'static str,
        id: serde_json::Value,
        error: RpcError,
    },
}
#[derive(Serialize)]
struct RpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<serde_json::Value>,
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

fn main() -> Result<()> {
    // Optional readiness signal: if DLINOSS_MCP_READY_FILE is set, create the file after startup.
    if let Ok(path) = std::env::var("DLINOSS_MCP_READY_FILE") {
        let _ = std::fs::write(path, b"ready");
    }
    // Emit a simple readiness line (not a JSON-RPC response) so launchers can wait for this exact string.
    println!("{{\"mcp_ready\":true,\"mode\":\"stdio\"}}");
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut sim: Option<SimState> = None;
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let env: RpcEnvelope = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                writeln!(
                    stdout,
                    "{}",
                    serde_json::to_string(&RpcResponse::Err {
                        jsonrpc: "2.0",
                        id: serde_json::Value::Null,
                        error: RpcError {
                            code: -32700,
                            message: format!("parse error: {e}"),
                            data: None
                        }
                    })?
                )?;
                stdout.flush()?;
                continue;
            }
        };
        let result = handle(env, &mut sim);
        if let Err(resp) = result {
            writeln!(stdout, "{}", serde_json::to_string(&resp)?)?;
        } else if let Ok(resp) = result {
            writeln!(stdout, "{}", serde_json::to_string(&resp)?)?;
        }
        stdout.flush()?;
    }
    Ok(())
}

fn handle(
    env: RpcEnvelope,
    sim: &mut Option<SimState>,
) -> std::result::Result<RpcResponse, RpcResponse> {
    let RpcEnvelope {
        jsonrpc,
        id,
        method,
        params,
    } = env;
    if jsonrpc != "2.0" {
        return Err(err_resp(id, -32600, "invalid jsonrpc version", None));
    }
    macro_rules! ok {
        ($val:expr) => {
            RpcResponse::Ok {
                jsonrpc: "2.0",
                id: id.clone(),
                result: $val,
            }
        };
    }
    match method.as_str() {
        // Generic MCP-style initialization (some runners send this before custom names)
        "initialize" => {
            // Provide minimal server info & empty capabilities per emerging MCP expectations.
            Ok(ok!(json!({
                "serverInfo": {"name": "dlinoss-mcp", "version": env!("CARGO_PKG_VERSION")},
                "capabilities": {}
            })))
        }
        // Generic shorthand ping some clients may attempt.
        "ping" => Ok(ok!(json!({"ok": true, "ts": now_ms()}))),
        "dlinoss.ping" => Ok(ok!(json!({"ok": true, "ts": now_ms()}))),
        "dlinoss.listMethods" => Ok(ok!(json!([
            "dlinoss.ping",
            "dlinoss.listMethods",
            "dlinoss.init",
            "dlinoss.step",
            "dlinoss.status",
            "dlinoss.pause",
            "dlinoss.resume",
            "dlinoss.getState",
            "dlinoss.getFft"
        ]))),
        "dlinoss.init" => {
            let cfgp: InitParams = match serde_json::from_value(params) {
                Ok(p) => p,
                Err(e) => return Err(err_resp(id, 1001, &format!("bad params: {e}"), None)),
            };
            let cfg = DLinOssLayerConfig {
                state_dim: cfgp.state_dim.unwrap_or(16),
                input_dim: cfgp.input_dim.unwrap_or(2),
                output_dim: cfgp.output_dim.unwrap_or(3),
                delta_t: cfgp.delta_t.unwrap_or(1e-2),
                dtype: DType::F32,
            };
            match SimState::new(&cfg, cfgp.t_len.unwrap_or(10_000), Device::Cpu) {
                Ok(s) => {
                    *sim = Some(s);
                    Ok(ok!(json!({"ok": true})))
                }
                Err(e) => Err(err_resp(id, -32000, &format!("init failed: {e}"), None)),
            }
        }
        "dlinoss.step" => {
            let p: StepParams = match serde_json::from_value(params) {
                Ok(p) => p,
                Err(e) => return Err(err_resp(id, 1001, &format!("bad params: {e}"), None)),
            };
            let s = match sim.as_mut() {
                Some(s) => s,
                None => return Err(err_resp(id, 1002, "not initialized", None)),
            };
            if let Err(e) = s.step(p.steps.unwrap_or(1)) {
                return Err(err_resp(id, -32000, &format!("step error: {e}"), None));
            }
            Ok(ok!(
                json!({"current_t": s.current_t, "last_output": s.last_output })
            ))
        }
        "dlinoss.status" => {
            let s = match sim.as_ref() {
                Some(s) => s,
                None => return Err(err_resp(id, 1002, "not initialized", None)),
            };
            Ok(ok!(
                json!({"current_t": s.current_t, "config": {"state_dim": s.state_dim, "input_dim": s.input_dim, "output_dim": s.output_dim, "delta_t": s.delta_t, "t_len": s.t_len }})
            ))
        }
        "dlinoss.pause" => Ok(ok!(json!({"paused": true }))),
        "dlinoss.resume" => Ok(ok!(json!({"paused": false }))),
        "dlinoss.getState" => {
            let p: StateParams = match serde_json::from_value(params) {
                Ok(p) => p,
                Err(e) => return Err(err_resp(id, 1001, &format!("bad params: {e}"), None)),
            };
            let s = match sim.as_ref() {
                Some(s) => s,
                None => return Err(err_resp(id, 1002, "not initialized", None)),
            };
            match p.which.as_str() {
                "rings" => {
                    let limit = p.limit.unwrap_or(256);
                    let start = s.output_ring.len().saturating_sub(limit);
                    let slice = &s.output_ring[start..];
                    Ok(ok!(json!({"data": slice})))
                }
                _ => Err(err_resp(id, 1001, "unsupported state kind", None)),
            }
        }
        "dlinoss.getFft" => {
            let p: FftParams = match serde_json::from_value(params) {
                Ok(p) => p,
                Err(e) => return Err(err_resp(id, 1001, &format!("bad params: {e}"), None)),
            };
            let s = match sim.as_ref() {
                Some(s) => s,
                None => return Err(err_resp(id, 1002, "not initialized", None)),
            };
            let size = p.size.unwrap_or(256);
            if s.output_ring.len() < size {
                return Err(err_resp(id, 1003, "not enough samples", None));
            }
            let window = &s.output_ring[s.output_ring.len() - size..];
            #[cfg(feature = "fft")]
            {
                if let Some(spec) = candle_fft(window) {
                    if spec
                        .get("spectrum")
                        .and_then(|v| v.as_array())
                        .map(|a| !a.is_empty())
                        .unwrap_or(false)
                    {
                        return Ok(ok!(spec));
                    }
                }
            }
            let spec = naive_fft(window);
            Ok(ok!(spec))
        }
        _ => Err(err_resp(id, -32601, "method not found", None)),
    }
}

#[derive(Deserialize)]
struct InitParams {
    state_dim: Option<usize>,
    input_dim: Option<usize>,
    output_dim: Option<usize>,
    delta_t: Option<f64>,
    t_len: Option<usize>,
}
#[derive(Deserialize)]
struct StepParams {
    steps: Option<usize>,
}
#[derive(Deserialize)]
struct StateParams {
    which: String,
    limit: Option<usize>,
}
#[derive(Deserialize)]
struct FftParams {
    size: Option<usize>,
}

fn err_resp(
    id: serde_json::Value,
    code: i32,
    msg: &str,
    data: Option<serde_json::Value>,
) -> RpcResponse {
    RpcResponse::Err {
        jsonrpc: "2.0",
        id,
        error: RpcError {
            code,
            message: msg.to_string(),
            data,
        },
    }
}

fn naive_fft(window: &[f32]) -> serde_json::Value {
    let n = window.len();
    let half = n / 2;
    let mut out = Vec::with_capacity(half);
    for k in 0..half {
        let mut re = 0f32;
        let mut im = 0f32;
        let kk = k as f32;
        for (t, &v) in window.iter().enumerate() {
            let th = 2.0 * std::f32::consts::PI * kk * (t as f32) / (n as f32);
            re += v * th.cos();
            im -= v * th.sin();
        }
        let mag = (re * re + im * im).sqrt() / (n as f32);
        out.push(serde_json::json!({"f":k, "mag":mag}));
    }
    serde_json::json!({"spectrum": out})
}

#[cfg(feature = "fft")]
fn candle_fft(window: &[f32]) -> Option<serde_json::Value> {
    use candlekos::Tensor;
    let dev = Device::Cpu;
    let t = Tensor::from_slice(window, (window.len(),), &dev).ok()?;
    let len = window.len();
    let f = t.rfft(0usize, true).ok()?;
    let dims = f.dims();
    if dims.len() != 2 || dims[1] != 2 {
        return None;
    }
    let rows = dims[0];
    let data = f.to_vec2::<f32>().ok()?;
    let mut out = Vec::with_capacity(rows);
    for k in 0..rows {
        let re = data[k][0];
        let im = data[k][1];
        let mag = (re * re + im * im).sqrt() / (len as f32);
        out.push(serde_json::json!({"f":k, "mag":mag}));
    }
    Some(serde_json::json!({"spectrum": out}))
}
