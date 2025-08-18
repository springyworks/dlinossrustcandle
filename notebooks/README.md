# D-LinOSS Notebooks

This folder contains a small glue crate (`dlinoss-notebooks`) designed for a smooth evcxr/Jupyter experience with a single dependency per notebook. The crate re-exports:

- Candle's notebook helpers (`candle_notebooks`)
- The D‑LinOSS API (`DLinOssLayer`, `DLinOssLayerConfig`, and helpers)
- Convenience utilities for research (e.g., `SignalGen`), and optional FFT helpers when the `fft` feature is enabled

The policy is strict on purpose: each notebook must declare exactly one `:dep` line. This keeps notebooks portable, avoids dependency drift, and matches upstream Candle notebook practices.

## Quick start (from this `notebooks/` directory)

- CPU only (no FFT):

```
:dep dlinoss-notebooks = { path = "." }
```

- With FFT helpers enabled:

```
:dep dlinoss-notebooks = { path = ".", features = ["fft"] }
```

After that single `:dep`, import what you need:

```
// Evcxr code cell (Rust)
use dlinoss_notebooks::*; // brings in Result, Candle notebook utils, DLinOSS API, SignalGen
```

## Minimal end‑to‑end example

```
// 1) Single dependency
:dep dlinoss-notebooks = { path = ".", features = ["fft"] }

// 2) Bring symbols into scope
use dlinoss_notebooks::*;

// 3) Build a tiny DLinOSS layer on CPU
let device = candle_core::Device::Cpu;
let cfg = DLinOssLayerConfig { state_dim: 16, input_dim: 1, output_dim: 1, delta_t: 1e-2, dtype: candle_core::DType::F32 };
let layer = DLinOssLayer::new(cfg, &device)?;

// 4) Create an input signal and run a forward pass
let x = SignalGen::sine(256, 1.5, 1e-2)?;           // shape [1, 256, 1]
let y = layer.forward(&x, None)?;                   // shape [1, 256, 1]
println!("y shape: {:?}", y.dims());

// 5) Optional: quick FFT magnitude for visualization (requires features=["fft"]))
let x_host: Vec<f32> = x.squeeze(0)?.squeeze(1)?.to_vec1::<f32>()?;
let mag = rfft_magnitude(&x_host)?;                 // Vec<f32> magnitudes
println!("first 8 magnitudes: {:?}", &mag[..8.min(mag.len())]);
```

Notes:
- End cells with a concrete value or a println! to keep evcxr happy.
- Avoid using `?` at the top level; prefer helper closures or return a `Result` value.
- Keep shapes explicit; Candle expects `[B, T, In]` and returns `[B, T, Out]`.

## Running a notebook from elsewhere

If you're not running the notebook from this `notebooks/` folder, use an absolute path to the crate:

```
:dep dlinoss-notebooks = { path = "/home/rustuser/projects/rust/active/dlinossrustcandle/notebooks", features = ["fft"] }
```

The rest of the code remains the same (`use dlinoss_notebooks::*;`).

## Why a single :dep?

- Guarantees one source of truth for all notebook dependencies.
- Ensures consistent versions and features across notebooks.
- Mirrors Candle's `candle_notebooks` best practices.

## Troubleshooting

- "crate not found": make sure the `path` points to this `notebooks/` directory.
- FFT helpers are empty without the `fft` feature; enable with `features = ["fft"]`.
- If your kernel restarted or evcxr state reset, just re-run the single `:dep` cell then the rest.
