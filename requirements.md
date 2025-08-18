# D-LinOSS Requirements

Crate: dlinossrustcandle  
Workspace: /home/rustuser/projects/rust/active/dlinossrustcandle

## Quick guide for contributors and Copilot

- Don’t reimplement scan/FFT. Use Candle’s built-in `cumsum`/`exclusive_scan` and FFT APIs. The `crates/dlinoss-augment` sub-crate provides traits that call these operations directly.
- Use VS Code search wisely; avoid heavy disk churn. Prefer targeted listings and searches.
- Add useful Rust doc comments at the top of files for quick overviews.
- When targeting a Windows 11 executable from the Ubuntu 24 environment, place the generated `.exe` in `/media/rustuser/onSSD/FROMUBUNTU24` so it’s accessible from your Windows 11 native boot.
- Don’t suppress warnings. Treat warnings as signals to improve quality.
- Notebooks (evcxr) single-dep rule: each notebook must declare exactly one `:dep`, pointing to the glue crate `dlinoss-notebooks = { path = "." }`. Do not add other `:dep`s; the glue crate re-exports Candle notebook utilities, the D‑LinOSS API, and optional FFT helpers.

## Goals

- Implement a D-LinOSS (Damped Linear Oscillatory State-Space) layer/module on Candle tensors using the local Candle workspace as the tensor/NN backend.
- Mirror roughly 10% of the structure/usage of the Python reference at `/home/rustuser/projects/pyth/damped-linoss` (prioritize parity in behavior and examples over 1:1 file mapping).
- For training and inference, follow Candle’s idioms. Where the Python implementation has features Candle can replicate, prefer the Candle way (experimentation encouraged).
- If you need Python, use the virtual environment at `~/.uservenv`.
- Notebooks are encouraged for exploration and documentation.
- Add ample code comments. At the top of each `*.rs` and `*.rs.ipynb` code cell, include two Rust doc comment lines to support future doc scanning.

## References

- Paper (local): `src/dLinOSS paper.pdf` (convertible to text)
- arXiv: https://arxiv.org/abs/2505.12171v1
- Python reference (local): `/home/rustuser/projects/pyth/damped-linoss`
- Former Rust attempt (Burn-based): `/home/rustuser/projects/rust/active/dlinoss-rust`

## Local Candle dependencies

- `candle-core` and `candle-nn` are path dependencies to your local Candle checkout.
- Optional features: `accelerate`, `cuda`, `cudnn`, `mkl`, `metal`.
- Scan/FFT paths are experimental: default to sequential scan; gate FFT and any parallel scan under features.

## Notebooks (evcxr) dependency policy

- Single-dependency rule (strict):
	- Use exactly one `:dep` in each notebook, pointing to the glue crate.
	- Evcxr requires `:dep` to specify a path/git/version. Since we don’t publish this crate, use a path.
	- Examples:
		- Running the notebook from the `notebooks/` folder (preferred):
			- CPU only: `:dep dlinoss-notebooks = { path = "." }`
			- With FFT: `:dep dlinoss-notebooks = { path = ".", features = ["fft"] }`
		- From elsewhere: use an absolute path to the `notebooks/` crate directory:
			- `:dep dlinoss-notebooks = { path = "/home/rustuser/projects/rust/active/dlinossrustcandle/notebooks", features = ["fft"] }`
	- Rationale: The glue crate re-exports all needed items (Candle notebook utilities from upstream `candle_notebooks`, the D‑LinOSS API, `SignalGen`, and `rfft_magnitude` under the `fft` feature). This keeps notebooks portable and avoids dependency drift.
	- Tip: End complex cells with `println!("done");` or return a concrete value to avoid REPL “never type” pitfalls.

### Reference: Candle’s notebooks crate
- Upstream best practices live in Candle’s `candle_notebooks` crate:
	`/home/rustuser/projects/rust/from_github/candle/0aEXPLORATION/research/notebooks/candle_notebooks`
- Our `dlinoss-notebooks` builds on top of that to provide a single, stable dependency for all D‑LinOSS research notebooks.

## High-level requirements

1. Tensor backend: Candle tensors only.
2. D-LinOSS core: continuous-time damped oscillatory SSM and discrete-time counterpart with provably stable parameterization.
3. Discretization: exact 2×2 oscillatory block exponential; Tustin/Bilinear fallback.
4. Execution modes: sequential scan (default); optional parallel scan (feature-gated); optional FFT kernelization (feature-gated).
5. Layer API: transform [B, T, In] → [B, T, Out] with optional initial state.
6. Learnability: all core parameters learnable; stable reparameterization.
7. Stability: ensure spectral radius < 1 in the discrete domain.
8. Tests: stability, impulse/step, linearity, shapes/batching, determinism, plus optional parity with Python.
9. Examples: ultra simple, simple experiment, pulse analysis; optional real-time mini-demo.
10. Docs: README + examples runnable on CPU by default; clearly note GPU/FFT flags and experimental status.
11. Performance: CPU/GPU, dtype control, batching; reasonable runtime up to T≈1e4 on CPU; larger via parallel/FFT (experimental).

## Crate structure (sketch)

- See `src/` for implementation and `examples/` and `tests/` for usage.

## Mathematical specification (concise)

- Continuous-time linear SSM: \(\dot{x} = A x + B u,\ y = C x + D u\)
- Damped oscillatory 2×2 blocks for poles \(p=-\alpha\pm i\omega\), \(\alpha>0\), \(\omega>0\)
- Discretization with step \(\Delta t\): exact for 2×2 blocks; Tustin fallback otherwise
- Stability via positive \(\alpha\) and exact discretization ⇒ \(\rho(A_d)=e^{-\alpha\Delta t}<1\)

## Candle integration and shapes

- Tensors: `candle_core::Tensor` returning `Result<Tensor>`
- Devices/dtypes: `Device` (CPU/GPU), dtype configurable
- Shapes: input `x` is [B, T, In]; optional state `x0` is [B, S]; output `y` is [B, T, Out]

## Testing requirements

- Validate stability, linearity, shapes/batching, determinism, parity (optional), and performance sanity.

## Examples to port

- `ultra_simple_dlinoss.py` → `examples/ultra_simple.rs`
- `simple_dlinoss_experiment.py` → `examples/simple_experiment.rs`
- `continuous_pulse_analysis.py` → `examples/pulse_analysis.rs`
- `realtime_tinydlinoss_experiment.py` → `examples/realtime_tiny.rs` (optional)

## Workspace sub-crates

- `crates/dlinoss-augment`: traits wrapping Candle scan/FFT (no custom impls)
- `crates/dlinoss-display`: display utilities (egui/etui/minifb; FFT when enabled)
- `crates/dlinoss-helpers`: CLI helpers (clap; optional dialog)

## XTask (canonical dev workflow)

- Use `xtask` for all dev actions: build, test, lint, run examples, and verify Candle.
- Default VS Code build task runs: `cargo run -p xtask -- verify-candle --fft && cargo run -p xtask -- ci`.

## Quick build/test etiquette

- Naive path is supported: `cargo build`, `cargo test`, `cargo run --example ...` at the root should always work.
- Prefer the smart path for full checks: `cargo run -p xtask -- ci` (fmt + clippy + tests) and `cargo run -p xtask -- verify-candle --fft` before bigger changes.
- Keep both paths green. If naive `cargo test` breaks, fix it or wire it into `xtask` so it doesn’t drift.

## Windows: building a .exe (CPU by default)

- We ship a tiny binary at `src/bin/dlinoss_demo.rs`. Build it on Windows for a portable `.exe`:
	- `cargo build --release --bin dlinoss_demo`
	- Runs on CPU with no extra drivers. For GUI/TUI, use feature flags and examples (`--features egui` / `--features etui`).
- GPU on Windows (optional, advanced):
	- CUDA: enable `--features cuda[,cudnn]` with NVIDIA CUDA Toolkit + cuDNN installed and on PATH.
	- MKL: may require Intel MKL toolchain and proper env vars; otherwise stick to CPU.
	- Metal/Accelerate are macOS-only; ignore on Windows.
- Recommendation: start with the CPU binary. Add GPU later if needed.

## About “Patch ... was not used in the crate graph” warnings

- You may see Cargo warnings like:
	- `Patch candle-notebooks ... was not used in the crate graph.`
	- `Patch candle-transformers ... was not used in the crate graph.`
- Cause: The upstream Candle workspace declares `[patch.crates-io]` entries for optional crates; our local path deps don’t use them, so Cargo reports them as unused.
- Status: Harmless. If they get noisy, prefer the `xtask` flow which keeps output tighter.

## Versioning

- Keep this file updated as features and findings evolve.