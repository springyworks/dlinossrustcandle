# D-LinOSS Requirements

Crate: dlinossrustcandle  
Workspace: /home/rustuser/projects/rust/active/dlinossrustcandle

## Quick guide for contributors and Copilot

- Don’t reimplement scan/FFT. Use Candle’s built-in `cumsum`/`exclusive_scan` and FFT APIs. The `crates/dlinoss-augment` sub-crate provides traits that call these operations directly.
- Use VS Code search wisely; avoid heavy disk churn. Prefer targeted listings and searches. use gg for fast grep alternative and vscode-commands&vscode-taks
- Add useful Rust doc comments at the top of files for quick overviews.
- When targeting a Windows 11 executable from the Ubuntu 24 environment, place the generated `.exe` in `/media/rustuser/onSSD/FROMUBUNTU24` so it’s accessible from your Windows 11 native boot.
- Don’t suppress warnings. Treat warnings as signals to improve quality.
- Add ample code comments. At the top of each `*.rs` and `*.rs.ipynb` code cell, include two Rust doc comment lines to support future doc scanning.
- Use xtasks where opurtune, and use vscode-tasks, revert to old school rust testing if needed
- Notebooks (evcxr) single-dep rule: each notebook must declare exactly one `:dep`, pointing to the glue 
crate `dlinoss-notebooks = { path = "." }`. Do not add other `:dep`s; the glue crate re-exports Candle notebook utilities, the D‑LinOSS API, and optional  helpers.
- Use gg and rg , instead of grep they are fast
- In the copilot chat output, use continue inserts chat-dialog and check each result. use chat continue/cancel chat-dialog

## Goals

- Implement a D-LinOSS (Damped Linear Oscillatory State-Space) layer/module on Candle tensors using the local Candle workspace as the tensor/NN backend.
- Mirror roughly 10% of the structure/usage of the Python reference at `/home/rustuser/projects/pyth/damped-linoss` (prioritize parity in behavior and examples over 1:1 file mapping).
- For training and inference, follow Candle’s idioms. Where the Python implementation has features Candle can replicate, prefer the Candle way (experimentation encouraged).
- If you need Python, use the virtual environment at `~/.uservenv`.
- Notebooks are encouraged for exploration and documentation.

## References

- Paper (local): `src/dLinOSS paper.pdf` (convertible to text)
- arXiv: https://arxiv.org/abs/2505.12171v1
- Python reference (local): `/home/rustuser/projects/pyth/damped-linoss`
- Former Rust attempt (Burn-based): `/home/rustuser/projects/rust/active/dlinoss-rust`

## Local Candle dependencies

- `candle-core` and `candle-nn` are possible path dependencies to your local Candle (=Candlekos) checkout.
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
			- `:dep dlinoss-notebooks = { path = "/home/rustuser/projects/rust/active/dlinossrustcandle/notebooks", features = ["fft"] }`, todo check-it
	- Rationale: The glue crate re-exports all needed items (Candlekos notebook utilities from upstream `candle_notebooks`, the D‑LinOSS API, `SignalGen`, and `rfft_magnitude` under the `fft` feature). This keeps notebooks portable and avoids dependency drift.
	- Tip: End complex cells with `println!("done");` or return a concrete value to avoid REPL “never type” pitfalls.


## High-level requirements

1. Tensor backend: Candlekos tensors only.
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

## Mathematical specification (concise) .   todo: check it

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

### Test feature gates & scaling

- `large-tests`: Enables long-running stress & scalability tests (e.g. very long sequences / larger batches). Disabled by default to keep CI fast.
	- Run manually: `cargo test --features large-tests -- --nocapture large_sequence_throughput`
	- Additionally guarded by env var: `DLINOSS_LARGE=1` (unset ⇒ test prints skip message and returns Ok)
- `freq-tests`: Reserved for heavier frequency-domain diagnostics (the current lightweight sine sweep test runs unconditionally). Future extended sweeps or spectral estimation benches will live behind this feature.
- `silence-upstream-patch`: As described above, unrelated to logic; only affects Cargo warning noise.

Guidelines:
- Keep default `cargo test` under ~10s on a typical dev CPU.
- Place any test whose runtime exceeds ~1s/T=1e4 elements or allocates >~200MB behind `large-tests`.
- Use `println!` progress breadcrumbs in loops > ~5k iterations so users see forward motion with `--nocapture`.

Planned additions (Grade A path):
- Spectral energy conservation checks vs. discrete-time theoretical transfer function (behind `freq-tests`).
- Cross-validation vs. Python reference exports (golden data fixtures) for a few canonical inputs.
- Randomized linearity & superposition fuzz tests (quick, stay in default set).

## Grade A Test Matrix (Implemented)

Criterion | Test File / Function | Notes / Gate
--------- | -------------------- | ------------
Matrix reconstruction (M,F) correctness | `tests/dlinoss_math.rs` | Analytical helper `build_m_f` vs step dynamics
Impulse/step & linearity | `tests/dlinoss_math.rs` | Superposition, impulse ≡ diff(step)
Shape & batching | `tests/dlinoss_dimensional.rs::output_shapes_multi_dim` | Varied batch sizes
Latent stability (energy heuristic) | `tests/dlinoss_dimensional.rs::stability_energy_decay_zero_input` | Geometric & spike guards
Full state trajectory exposure | Used across tests via `forward_with_state` | Hidden API
Spectral radius bound (power iteration) | `tests/dlinoss_spectral.rs` | ρ(M) < threshold
Multi-frequency gain & phase smoothness | `tests/dlinoss_freq.rs::sine_response_relative_amplitude` | Unwrapped phase spikes bounded
Multi-sine energy boundedness | `tests/dlinoss_freq.rs::mixed_multisine_energy_distribution` | Prevent HF blow-up
nD forward equivalence (flatten vs nd) | `tests/dlinoss_dimensional.rs::nd_forward_roundtrip_equivalence` | 4D spatial case reshape invariant
Zero-length / degenerate dims handling | `tests/dlinoss_dimensional.rs::nd_zero_length_and_degenerate_dims` | Skips forward on T=0
Deterministic constructor + golden checksum | `tests/dlinoss_golden.rs` | Fails on drift; update intentionally
Large 1D throughput stress (gated) | `tests/dlinoss_stress.rs::large_sequence_throughput` | `--features large-tests` + `DLINOSS_LARGE=1`
Large nD spatial throughput stress (gated) | `tests/dlinoss_stress.rs::large_nd_spatial_throughput` | Same gate/env
Error mode: dimension mismatch | `tests/dlinoss_errors.rs` | Clear error message asserts

Pending / Future (not yet implemented):
- Extended frequency-domain analytical comparison vs closed-form transfer (freq-tests gate)
- Python parity fixture ingestion
- Randomized linearity fuzz (multiple random pairs) – optional; current deterministic coverage adequate
- Phase portrait invariants & latent energy spectrum visualization integration tests (post egui quadrant demo)

### Running the suites

Default fast suite:
```
cargo test
```

Include large stress tests:
```
DLINOSS_LARGE=1 cargo test --features large-tests -- --nocapture large
```

Run only frequency tests (verbose):
```
cargo test --test dlinoss_freq -- --nocapture
```

Golden snapshot update procedure (when an intentional numerical change occurs):
1. Run `cargo test --test dlinoss_golden -- --nocapture` and capture the reported `sum` & `mean`.
2. Update `GOLD_SUM` and `GOLD_MEAN` in `tests/dlinoss_golden.rs` **with justification in commit message**.
3. Re-run full suite; ensure no other regressions.

Large nD performance sampling (example command):
```
DLINOSS_LARGE=1 cargo test --features large-tests --test dlinoss_stress -- --nocapture stress_nd
```

Feature summary (test related):
- `large-tests`: Enables high-cost throughput experiments (guarded again by env var so opt-in is explicit)
- `freq-tests` (reserved): Will host heavier spectral analyses; current lightweight sweeps run without it.

### Quality Gate Expectations

Gate | Expectation
---- | -----------
Clippy | No new warnings in core crate (allow test-only allowances)
Fmt | `cargo fmt` produces no diff
Unit tests | All mandatory tests pass under default; gated tests pass when enabled
Determinism | Golden checksum stable unless deliberate update
Stability | No latent energy catastrophic spikes under zero-input decay test
Spectral | Power iteration bound satisfied (< configured threshold)

If any gate fails the CI `xtask` should report and block merge.

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

### Silencing the warning explicitly (optional)

We provide an optional feature flag `silence-upstream-patch` that depends on a dummy optional
local path to `candle-notebooks`. Enabling it convinces Cargo the upstream Candle patch
is "used" so the warning disappears. Normally you should NOT enable it (keep build lean), but
if you ever see the warning again after upstream Candle changes and want a clean log, run:

```
cargo build --features silence-upstream-patch
```

You can also add that feature in CI if you care about completely silent logs. Remove the dummy
dependency + feature once upstream Candle eliminates the unused patch entry.

## Versioning

- Keep this file updated as features and findings evolve.