# dlinossrustcandle

➡ Notebooks quickstart: see [notebooks/README.md](./notebooks/README.md)



> 📋 **Quick Start**: Read [requirements.md](./requirements.md) for complete development setup  

> 🔧 **Architecture**: See [crates/README.md](./crates/README.md#sub-crates-overview) for workspace structure  DLiNOSS (Damped Linear Oscillatory State Space) layer implemented with Candlekos which was build on [Candle](https://github.com/huggingface/candle) tensors.

> 🧪 **Core Implementation**: See [src/README.md](./src/README.md#key-components) for D-LinOSS details  

> 🏗️ **Windows Binary**: [target/release/dlinoss_demo](./target/README.md#size-management) (CPU-only, no drivers needed)Notes:

- We use Candle's native scan-like ops (cumsum/exclusive_scan) and FFT where applicable. No custom scan/fft implementations.

**D-LinOSS** (Damped Linear Oscillatory State Space) layer implemented with [Candle](https://github.com/huggingface/candle) tensors, providing stable damped oscillatory dynamics for neural networks.- A small sub-crate `crates/dlinoss-augment` provides extension traits that simply wrap Candle APIs for ergonomics.



## 🚀 Quick Commands- Uses local path dependencies to your Candle checkout.

- Minimal forward pass using IMEX-like discretization.

```bash

# Test everything (smart discovery)## Usage

cargo run -p xtask -- discover-tests --run --features fft

Run examples (CPU):

# Build Windows .exe (CPU-only)

cargo build --release --bin dlinoss_demo  ```bash

# → Binary at: target/release/dlinoss_democargo run --example ultra_simple

cargo run --example simple_experiment

# Run examplescargo run --example viz_dual_pane

cargo run --example ultra_simplecargo run --example viz_tui

cargo run --features egui --example viz_dual_pane  ```

cargo run --features etui --example viz_tui

Run tests:

# Development workflow  

cargo run -p xtask -- verify-candle --fft && cargo run -p xtask -- ci```bash

```cargo test

```

## 📁 Repository Structure

Features (experimental, off by default):

| Directory | Purpose | Key Files |

|-----------|---------|-----------|- `parallel-scan`: reserved for a future parallel scan implementation. Public scan helpers (inclusive/exclusive prefix sums) are available; the true parallel recurrence is gated and unimplemented.

| **[`src/`](./src/README.md)** | Core D-LinOSS implementation | `dlinoss.rs`, `scan.rs`, `kernelize.rs` |- `fft`: enables Candle’s FFT support (used in examples for spectral visualization).

| **[`crates/`](./crates/README.md)** | Workspace sub-crates | `dlinoss-augment/`, `dlinoss-display/`, `dlinoss-helpers/` |

| **[`examples/`](./src/README.md#usage)** | Demo applications | `ultra_simple.rs`, `viz_dual_pane.rs`, `viz_tui.rs` |Optional example flags:

| **[`tests/`](./src/README.md#testing)** | Integration tests | `stability.rs`, `scan_tests.rs`, `smoke.rs` |

| **[`xtask/`](./crates/README.md#development)** | Development automation | Smart test discovery, candle verification |```bash

# Enable CUDA (if supported by your Candle build)

## 🎯 Key Featurescargo run --features cuda --example viz_dual_pane



### 📊 **Mathematical Foundation**# Enable FFT-backed bottom panes in the GUI example

- **Continuous-time SSM**: `ẋ = Ax + Bu, y = Cx + Du` with damped oscillatory dynamicscargo run --features fft --example viz_dual_pane

- **Exact discretization**: 2×2 block exponential for complex poles `p = -α ± iω`# Or in the TUI example

- **Provable stability**: `α > 0` ensures `ρ(A_d) < 1` in discrete domaincargo run --features fft --example viz_tui

```

### ⚡ **Execution Modes**

- **Sequential scan** (default): Always stable, CPU/GPU compatiblePython baselines (in the Python repo) should be run inside the venv and can use TEMPTEST scripts to avoid BatchNorm collectives.

- **Parallel scan** (feature-gated): Experimental, better performance for long sequences  

- **FFT kernelization** (feature-gated): Fastest for very long sequences, requires `fft` feature## XTask workflow



### 🔧 **Candle Integration**This workspace uses an `xtask` utility as the canonical entry point for development tasks. The default VS Code build task runs a Candle verification probe and then CI:

- **Path dependencies**: Uses local Candle workspace at `../from_github/candle`

- **No custom implementations**: Wraps Candle's `cumsum`/`exclusive_scan` and `rfft`/`irfft`- Verify Candle scan/fft and run CI: cargo run -p xtask -- verify-candle --fft && cargo run -p xtask -- ci

- **Cross-platform**: CPU (always), CUDA/Metal/MKL (feature-gated)- Run the TUI example via xtask: cargo run -p xtask -- run-tui

- Run the GUI example via xtask (with FFT): cargo run -p xtask -- --fft run-gui

## 🧪 Smart Testing System

## What the GUI example shows

Our xtask includes **smart test discovery** to prevent forgotten tests:

`viz_dual_pane` launches an egui window with two main plots (and optionally two bottom plots):

```bash

# Discover all tests in workspace (finds #[test] functions)- Left: D-LinOSS output y[t]

cargo run -p xtask -- discover-tests- Right: cumulative sum of the input over time (prefix sum)

- Bottom-Left (when `--features fft`): a simple FFT visualization of y

# Run all discovered tests (basic + feature-gated)  - Bottom-Right (when `--features fft`): the input sine

cargo run -p xtask -- discover-tests --run --features fft

```

**Test Categories Found**:
- **Integration tests**: [`tests/*.rs`](./src/README.md#testing) (stability, shapes, scan operations)
- **Sub-crate tests**: [`crates/*/src/**/*.rs`](./crates/README.md#testing-sub-crates) (probe tests, utilities)
- **Generated tests**: [`xtask/src/*.rs`](./crates/dlinoss-helpers/src/probe/README.md) (candle verification)

## 🎨 Visualization Examples

### GUI (egui)
```bash
cargo run --features egui --example viz_dual_pane
cargo run --features egui,fft --example viz_dual_pane  # With FFT spectrograms
```

### Terminal UI (ratatui)  
```bash
cargo run --features etui --example viz_tui
cargo run --features etui,fft --example viz_tui  # With frequency domain plots
```

## 🪟 Windows Distribution

Build standalone Windows executable:
```bash
cargo build --release --bin dlinoss_demo
```

**Output location**: [`target/release/dlinoss_demo`](./target/README.md) (on Windows: `dlinoss_demo.exe`)

**Features**:
- ✅ **CPU-only**: No drivers required
- ✅ **Portable**: Single executable  
- ✅ **Optional GPU**: Use `--features cuda` for NVIDIA GPU support (requires CUDA Toolkit)

## 🔗 Related Projects

- **📝 Research Paper**: [`src/dLinOSS paper.pdf`](./src/README.md#documentation)
- **🐍 Python Reference**: `/home/rustuser/projects/pyth/damped-linoss` ([comparison target](./requirements.md#references))
- **🔥 Previous Burn Attempt**: `/home/rustuser/projects/rust/active/dlinoss-rust` ([archived](./requirements.md#references))

## 🛠️ Development

### XTask Workflow
```bash
# Full CI pipeline
cargo run -p xtask -- ci                          # Format + Clippy + Test

# Candle integration verification  
cargo run -p xtask -- verify-candle               # Basic probe
cargo run -p xtask -- verify-candle --fft         # With FFT support

# Quick development commands
cargo run -p xtask -- run-gui                     # Launch GUI demo
cargo run -p xtask -- run-tui                     # Launch TUI demo
cargo run -p xtask -- run-simple                  # Run basic experiment
```

### Feature Matrix
| Feature | Default | Purpose | Requires |
|---------|---------|---------|----------|
| `fft` | ❌ | FFT kernelization & spectrograms | Candle with FFT |
| `cuda` | ❌ | NVIDIA GPU acceleration | CUDA Toolkit |  
| `metal` | ❌ | Apple GPU acceleration | macOS only |
| `mkl` | ❌ | Intel MKL optimization | Intel MKL |
| `egui` | ❌ | GUI visualization | X11/Wayland |
| `etui` | ❌ | Terminal visualization | Terminal |

### Glob Patterns for Search

Use these patterns to find specific file types:

```bash
# All Rust source files
**/*.rs

# Core implementation (excluding tests/examples)  
src/**/*.rs

# Sub-crate sources only
crates/**/src/**/*.rs

# Test files only
{tests,**/tests}/**/*.rs

# Example files
examples/**/*.rs

# Configuration files
**/{Cargo.toml,*.json,*.md}

# Exclude build artifacts
**/*.rs !target/** !**/target/**
```

## 📄 License

MIT OR Apache-2.0
