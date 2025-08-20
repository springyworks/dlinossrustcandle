use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::fs;
use std::path::{Path, PathBuf};
use xshell::{Shell, cmd};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Ui {
    Gui,
    Tui,
    Headless,
}
#[derive(Debug, Subcommand)]
enum Cmd {
    /// Build and test the workspace (fmt + clippy + test)
    Ci,
    /// Build only
    Build,
    /// Check (typecheck without building artifacts)
    Check,
    /// Test the workspace
    Test,
    /// Format code with rustfmt
    Fmt,
    /// Lint with clippy
    Clippy,
    /// Run an example by name, pass-through extra args via --
    RunEx {
        name: String,
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },
    /// Run a binary by name (src/bin/*), pass-through extra args via --
    RunBin {
        name: String,
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },
    /// Run the simple experiment example
    RunSimple,
    /// Run the dual-pane GUI (requires --features egui)
    RunGui,
    /// Run the TUI (requires --features etui)
    RunTui,
    /// Verify local Candle path + scan/fft methods via a TEMPTEST probe crate
    VerifyCandle {
        /// Enable FFT feature for the probe
        #[arg(long)]
        fft: bool,
    },
    /// Discover and validate all tests in the workspace
    DiscoverTests {
        /// Actually run discovered tests (default: just list them)
        #[arg(long)]
        run: bool,
        /// Include feature-gated tests
        #[arg(long)]
        features: Vec<String>,
    },
    /// Build a Windows .exe via cross-compilation (bin or example)
    BuildWindows {
        /// Build in release mode (default: debug)
        #[arg(long)]
        release: bool,
        /// Name of the binary in src/bin (mutually exclusive with --example)
        #[arg(long)]
        bin: Option<String>,
        /// Name of the example in examples/ (mutually exclusive with --bin)
        #[arg(long)]
        example: Option<String>,
        /// Comma-separated features (e.g. fft,egui)
        #[arg(long, value_delimiter = ',')]
        features: Vec<String>,
        /// Copy to shared Windows location (/media/rustuser/onSSD/FROMUBUNTU24)
        #[arg(long)]
        copy_to_windows: bool,
        /// Optional custom output file name (without extension)
        #[arg(long)]
        out_name: Option<String>,
        /// Optional destination directory (defaults to /media/rustuser/onSSD/FROMUBUNTU24)
        #[arg(long)]
        out_dir: Option<String>,
    },
    /// Build+copy the animated GUI example to Windows shared path (one-shot helper)
    ShipWindowsViz,
    /// Scan repository README.md files and inject/update cross links between markers
    DocsIndex {
        /// Write changes to files (default: dry-run prints planned changes)
        #[arg(long)]
        write: bool,
    },
    /// Check-build the notebooks glue crate with selected features
    NotebooksCheck {
        /// Comma-separated features to enable (e.g. gui,fft,audio)
        #[arg(long, value_delimiter = ',')]
        features: Vec<String>,
    },
    /// Run comprehensive workspace testing including all features and combinations
    Comprehensive {
        /// Enable extended feature/test combinations (more thorough but slower)
        #[arg(long)]
        extended: bool,
    },
    /// Assemble PNG frames in images_store/SHOWCASE into an animated GIF
    Gif {
        /// Output gif path (default: images_store/SHOWCASE/anim.gif)
        #[arg(long)]
        out: Option<String>,
        /// Frame delay in hundredths of a second (default 4 -> 25fps)
        #[arg(long, default_value_t = 4)]
        delay_cs: u16,
        /// Max frames (default all)
        #[arg(long)]
        max_frames: Option<usize>,
    },
    /// Run a scripted demonstration against the dlinoss-mcp server (stdin/stdout JSON-RPC)
    McpDemo {
        /// Enable FFT feature when running server
        #[arg(long)]
        fft: bool,
        /// Steps to advance
        #[arg(long, default_value_t = 300)]
        steps: usize,
        /// FFT size
        #[arg(long, default_value_t = 256)]
        fft_size: usize,
    },
    /// Run the MCP server persistently (serve mode) and forward stdio
    McpServe {
        /// Enable FFT feature when running server
        #[arg(long)]
        fft: bool,
        /// Optional readiness file path to write when server starts
        #[arg(long)]
        ready_file: Option<String>,
    },
}

#[derive(Debug, Parser)]
#[command(name = "xtask", version, about = "Workspace tasks for dlinoss")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
    /// Enable FFT feature
    #[arg(long)]
    fft: bool,
    /// UI selection where relevant
    #[arg(long, value_enum, default_value_t = Ui::Gui)]
    ui: Ui,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let sh = Shell::new()?;
    match args.cmd {
        Cmd::Ci => {
            // Limit formatting to this workspace only. Using --all caused rustfmt to attempt
            // traversal of upstream Candle workspace path dependencies, one of which referenced
            // a removed experimental crate (candle_tensor_elementprograms) leading to a manifest error.
            // Plain `cargo fmt` respects the current workspace members and skips external path deps.
            cmd!(sh, "cargo fmt").run()?;
            // Avoid --all-features to prevent pulling macOS-only deps (e.g., metal/objc) on Linux.
            cmd!(sh, "cargo clippy --all-targets -- -D warnings").run()?;
            cmd!(sh, "cargo test").run()?;
        }
        Cmd::Build => {
            cmd!(sh, "cargo build").run()?;
        }
        Cmd::Check => {
            cmd!(sh, "cargo check").run()?;
        }
        Cmd::Test => {
            cmd!(sh, "cargo test").run()?;
        }
        Cmd::Fmt => {
            // See comment in Ci: keep formatting scoped to local workspace members.
            cmd!(sh, "cargo fmt").run()?;
        }
        Cmd::Clippy => {
            // Prefer workspace-wide clippy without --all-features to avoid platform-only deps.
            cmd!(sh, "cargo clippy --workspace --all-targets -- -D warnings").run()?;
        }
        Cmd::RunEx { name, args: extra } => {
            let mut cmdline = format!("cargo run --example {name}");
            if args.fft {
                cmdline.push_str(" --features fft");
            }
            if !extra.is_empty() {
                cmdline.push_str(" -- ");
                cmdline.push_str(&extra.join(" "));
            }
            cmd!(sh, "{cmdline}").run()?;
        }
        Cmd::RunBin { name, args: extra } => {
            let mut cmdline = format!("cargo run --bin {name}");
            if args.fft {
                cmdline.push_str(" --features fft");
            }
            if !extra.is_empty() {
                cmdline.push_str(" -- ");
                cmdline.push_str(&extra.join(" "));
            }
            cmd!(sh, "{cmdline}").run()?;
        }
        Cmd::RunSimple => {
            cmd!(sh, "cargo run --example simple_experiment").run()?;
        }
        Cmd::RunGui => {
            if args.fft {
                cmd!(sh, "cargo run --features egui,fft --example viz_dual_pane").run()?;
            } else {
                cmd!(sh, "cargo run --features egui --example viz_dual_pane").run()?;
            }
        }
        Cmd::RunTui => {
            if args.fft {
                cmd!(sh, "cargo run --features etui,fft --example viz_tui").run()?;
            } else {
                cmd!(sh, "cargo run --features etui --example viz_tui").run()?;
            }
        }
        Cmd::VerifyCandle { fft } => {
            verify_candle_migrated(&sh, fft)?;
        }
        Cmd::DiscoverTests { run, features } => {
            discover_tests(&sh, run, features)?;
        }
        Cmd::BuildWindows {
            release,
            bin,
            example,
            features,
            copy_to_windows,
            out_name,
            out_dir,
        } => {
            build_windows(
                &sh,
                release,
                bin,
                example,
                features,
                copy_to_windows,
                out_name,
                out_dir,
            )?;
        }
        Cmd::ShipWindowsViz => {
            // One-shot: release build of viz_animated_egui with FFT, copy to shared path with a friendly name
            build_windows(
                &sh,
                true,
                None,
                Some("viz_animated_egui".to_string()),
                vec!["fft".to_string()],
                true,
                Some("dlinoss_viz_animated".to_string()),
                None,
            )?;
        }
        Cmd::DocsIndex { write } => {
            docs_index(&sh, write)?;
        }
        Cmd::NotebooksCheck { features } => {
            notebooks_check(&sh, features)?;
        }
        Cmd::Comprehensive { extended } => {
            comprehensive_test(&sh, extended)?;
        }
        Cmd::Gif {
            out,
            delay_cs,
            max_frames,
        } => {
            assemble_gif(out, delay_cs, max_frames)?;
        }
        Cmd::McpDemo {
            fft,
            steps,
            fft_size,
        } => {
            mcp_demo(fft, steps, fft_size)?;
        }
        Cmd::McpServe { fft, ready_file } => {
            use std::io::{BufRead, BufReader};
            use std::process::{Command, Stdio};
            let mut cmd = Command::new("cargo");
            cmd.arg("run").arg("-p").arg("dlinoss-mcp");
            if fft {
                cmd.arg("--features").arg("fft");
            }
            if let Some(path) = ready_file.clone() {
                cmd.env("DLINOSS_MCP_READY_FILE", &path);
            }
            cmd.stdin(Stdio::inherit())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit());
            eprintln!("[xtask] launching dlinoss-mcp (serve mode)...");
            let mut child = cmd.spawn()?;
            let stdout = child.stdout.take().unwrap();
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let line = line?;
                println!("{}", line);
                if line.contains("\"mcp_ready\":true") {
                    eprintln!(
                        "[xtask] MCP server ready (stdio). Attach a client using stdin/stdout JSON-RPC lines."
                    );
                }
            }
            let status = child.wait()?;
            if !status.success() {
                anyhow::bail!("mcp server exited with status {status}");
            }
        }
    }
    Ok(())
}

fn verify_candle_migrated(sh: &Shell, fft: bool) -> Result<()> {
    // Test the migrated probe in dlinoss-helpers
    if fft {
        cmd!(sh, "cargo test -p dlinoss-helpers --features fft").run()?;
    } else {
        cmd!(sh, "cargo test -p dlinoss-helpers").run()?;
    }

    // Also verify dependency tree
    let _ = cmd!(sh, "cargo tree -e features").run();
    Ok(())
}

// Legacy probe kept for reference; current command uses verify_candle_migrated.
#[allow(dead_code)]
fn verify_candle(sh: &Shell, fft: bool) -> Result<()> {
    // Workspace root = CWD when running xtask
    let root = std::env::current_dir()?;
    let candle_core_abs = root
        .join("../../from_github/candle/candle-core")
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("Failed to resolve candle-core path: {e}"))?;

    // 1) Quick dependency presence check via `cargo tree` (advisory). Ignore output.
    let _ = cmd!(sh, "cargo tree -e features").run();

    // 2) Generate TEMPTEST probe crate on the fly
    let probe_dir = root.join("TEMPTEST/candle_probe");
    ensure_dir(&probe_dir)?;
    write_probe_files(&probe_dir, &candle_core_abs)?;

    // 3) Run its tests (without FFT), then optionally with FFT
    let manifest_path = probe_dir.join("Cargo.toml");
    // Remove stale lock to avoid carrying forward unused patch entries
    let lock_path = probe_dir.join("Cargo.lock");
    if lock_path.exists() {
        let _ = fs::remove_file(&lock_path);
    }
    cmd!(sh, "cargo test --manifest-path {manifest_path}").run()?;
    if fft {
        cmd!(
            sh,
            "cargo test --features fft --manifest-path {manifest_path}"
        )
        .run()?;
    }
    Ok(())
}

#[allow(dead_code)]
fn ensure_dir(path: &Path) -> Result<()> {
    if !path.exists() {
        fs::create_dir_all(path)?;
    }
    if !path.join("src").exists() {
        fs::create_dir_all(path.join("src"))?;
    }
    Ok(())
}

#[allow(dead_code)]
fn write_probe_files(probe_dir: &Path, candle_core_abs: &Path) -> Result<()> {
    let cargo_toml = format!(
        r#"[package]
name = "candle_probe"
version = "0.1.0"
edition = "2024"

[workspace]

[dependencies]
anyhow = "1"
candle = {{ package = "candle-core", path = "{}", default-features = false }}

[features]
default = []
fft = ["candle/fft"]
"#,
        candle_core_abs.display()
    );
    fs::write(probe_dir.join("Cargo.toml"), cargo_toml)?;

    let lib_rs = r#"use anyhow::Result;
use candle::{Device, Tensor, DType};

#[test]
fn probe_cumsum_and_exclusive_scan() -> Result<()> {
    let dev = Device::Cpu;
    let x = Tensor::from_slice(&[1f32, 2., 3., 4.], (4,), &dev)?;
    let cs = x.cumsum(0)?; // [1,3,6,10]
    let v = cs.to_vec1::<f32>()?;
    assert_eq!(v, vec![1., 3., 6., 10.]);

    let ex = x.exclusive_scan(0)?; // [0,1,3,6]
    let v = ex.to_vec1::<f32>()?;
    assert_eq!(v, vec![0., 1., 3., 6.]);
    Ok(())
}

#[cfg(feature = "fft")]
#[test]
fn probe_fft_and_ifft_roundtrip() -> Result<()> {
    let dev = Device::Cpu;
    // simple length-8 waveform
    let x = Tensor::from_slice(&[1f32, 0., -1., 0., 1., 0., -1., 0.], (8,), &dev)?;
    // Forward real FFT (rfft), normalized -> complex spectrum (interleaved)
    let f = x.rfft(0usize, true)?;
    // Inverse real FFT (irfft), normalized -> back to real signal of same length
    let xi = f.irfft(0usize, true)?;
    // Real roundtrip within tolerance with matching shapes
    let diff = (xi - &x)?.abs()?;
    let max = diff.max_all()?.to_scalar::<f32>()?;
    assert!(max < 1e-4, "fft/ifft roundtrip max err {} too high", max);
    Ok(())
}
"#;
    fs::write(probe_dir.join("src/lib.rs"), lib_rs)?;
    Ok(())
}

fn discover_tests(sh: &Shell, run: bool, features: Vec<String>) -> Result<()> {
    println!("🔍 Discovering tests in workspace...\n");

    let root = std::env::current_dir()?;
    let mut test_functions = Vec::new();
    let mut test_files = Vec::new();

    // Find all Rust files with test functions
    find_test_functions(&root, &mut test_functions, &mut test_files)?;

    // Report findings
    println!("📊 Test Discovery Summary:");
    println!("├─ Total test functions found: {}", test_functions.len());
    println!("├─ Test files found: {}", test_files.len());
    println!(
        "└─ Feature-gated tests: {}",
        test_functions.iter().filter(|t| t.feature_gated).count()
    );

    println!("\n📁 Test Files by Category:");
    categorize_test_files(&test_files);

    println!("\n🧪 Test Functions by Feature:");
    categorize_test_functions(&test_functions);

    if run {
        println!("\n🚀 Running discovered tests...");
        run_discovered_tests(sh, &features)?;
    } else {
        println!("\n💡 Use --run to execute all discovered tests");
        println!("💡 Use --features fft to include feature-gated tests");
    }

    Ok(())
}

#[derive(Debug)]
struct TestFunction {
    name: String,
    file_path: PathBuf,
    _line_number: usize,
    feature_gated: bool,
    feature_name: Option<String>,
}

#[derive(Debug)]
struct TestFile {
    path: PathBuf,
    test_count: usize,
    category: TestCategory,
}

#[derive(Debug, PartialEq, Eq, Hash)]
enum TestCategory {
    Integration, // tests/ directory
    Unit,        // src/ directory
    SubCrate,    // crates/ directory
    Generated,   // xtask generated
    External,    // outside workspace
}

fn find_test_functions(
    root: &Path,
    functions: &mut Vec<TestFunction>,
    files: &mut Vec<TestFile>,
) -> Result<()> {
    use std::fs;

    fn visit_dir(
        dir: &Path,
        root: &Path,
        functions: &mut Vec<TestFunction>,
        files: &mut Vec<TestFile>,
    ) -> Result<()> {
        if dir.file_name().and_then(|n| n.to_str()) == Some("target") {
            return Ok(()); // Skip target directories
        }

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                visit_dir(&path, root, functions, files)?;
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                analyze_rust_file(&path, root, functions, files)?;
            }
        }
        Ok(())
    }

    visit_dir(root, root, functions, files)
}

fn analyze_rust_file(
    file_path: &Path,
    root: &Path,
    functions: &mut Vec<TestFunction>,
    files: &mut Vec<TestFile>,
) -> Result<()> {
    let content = fs::read_to_string(file_path)?;
    let lines: Vec<&str> = content.lines().collect();
    let mut test_count = 0;
    let mut current_feature_gate: Option<String> = None;

    for (line_num, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        // Check for feature gates
        if trimmed.starts_with("#[cfg(feature = ") {
            if let Some(start) = trimmed.find('"') {
                if let Some(end) = trimmed.rfind('"') {
                    if start < end {
                        current_feature_gate = Some(trimmed[start + 1..end].to_string());
                    }
                }
            }
        }

        // Check for test functions
        if trimmed == "#[test]" {
            // Look for the function name on the next line(s)
            if let Some(next_line) = lines.get(line_num + 1) {
                if let Some(fn_name) = extract_function_name(next_line.trim()) {
                    test_count += 1;
                    functions.push(TestFunction {
                        name: fn_name,
                        file_path: file_path.to_path_buf(),
                        _line_number: line_num + 1,
                        feature_gated: current_feature_gate.is_some(),
                        feature_name: current_feature_gate.clone(),
                    });
                }
            }
            current_feature_gate = None; // Reset after use
        }

        // Reset feature gate on non-attribute lines
        if !trimmed.starts_with('#') && !trimmed.is_empty() {
            current_feature_gate = None;
        }
    }

    if test_count > 0 {
        let category = categorize_file(file_path, root);
        files.push(TestFile {
            path: file_path.to_path_buf(),
            test_count,
            category,
        });
    }

    Ok(())
}

fn extract_function_name(line: &str) -> Option<String> {
    if let Some(fn_pos) = line.find("fn ") {
        let after_fn = &line[fn_pos + 3..];
        if let Some(paren_pos) = after_fn.find('(') {
            return Some(after_fn[..paren_pos].trim().to_string());
        }
    }
    None
}

fn categorize_file(file_path: &Path, root: &Path) -> TestCategory {
    if let Ok(relative) = file_path.strip_prefix(root) {
        let components: Vec<_> = relative.components().collect();
        if components.is_empty() {
            return TestCategory::External;
        }

        match components[0].as_os_str().to_str() {
            Some("tests") => TestCategory::Integration,
            Some("src") => TestCategory::Unit,
            Some("crates") => TestCategory::SubCrate,
            Some("xtask") => TestCategory::Generated,
            _ => TestCategory::External,
        }
    } else {
        TestCategory::External
    }
}

fn categorize_test_files(files: &[TestFile]) {
    let mut by_category = std::collections::HashMap::new();
    for file in files {
        by_category
            .entry(&file.category)
            .or_insert_with(Vec::new)
            .push(file);
    }

    for (category, files) in by_category {
        let total_tests: usize = files.iter().map(|f| f.test_count).sum();
        println!(
            "├─ {:?}: {} files, {} tests",
            category,
            files.len(),
            total_tests
        );
        for file in files {
            println!(
                "│  └─ {} ({} tests)",
                file.path.file_name().unwrap().to_string_lossy(),
                file.test_count
            );
        }
    }
}

fn categorize_test_functions(functions: &[TestFunction]) {
    let mut regular_tests = Vec::new();
    let mut feature_gated = std::collections::HashMap::new();

    for func in functions {
        if func.feature_gated {
            let feature_name = func.feature_name.as_deref().unwrap_or("unknown");
            feature_gated
                .entry(feature_name.to_string())
                .or_insert_with(Vec::new)
                .push(func);
        } else {
            regular_tests.push(func);
        }
    }

    println!("├─ Regular tests: {}", regular_tests.len());
    for func in regular_tests.iter().take(5) {
        println!(
            "│  └─ {}() in {}",
            func.name,
            func.file_path.file_name().unwrap().to_string_lossy()
        );
    }
    if regular_tests.len() > 5 {
        println!("│  └─ ... and {} more", regular_tests.len() - 5);
    }

    for (feature, funcs) in feature_gated {
        println!("├─ Feature '{}': {} tests", feature, funcs.len());
        for func in funcs.iter().take(3) {
            println!(
                "│  └─ {}() in {}",
                func.name,
                func.file_path.file_name().unwrap().to_string_lossy()
            );
        }
        if funcs.len() > 3 {
            println!("│  └─ ... and {} more", funcs.len() - 3);
        }
    }
}

fn run_discovered_tests(sh: &Shell, features: &[String]) -> Result<()> {
    // Run workspace tests
    if features.is_empty() {
        cmd!(sh, "cargo test --workspace").run()?;
    } else {
        let features_str = features.join(",");
        cmd!(sh, "cargo test --workspace --features {features_str}").run()?;
    }

    // Run individual sub-crate tests to ensure nothing is missed
    let crates = ["dlinoss-augment", "dlinoss-display", "dlinoss-helpers"];
    for crate_name in crates {
        println!("Testing {crate_name} individually...");
        if features.is_empty() {
            cmd!(sh, "cargo test -p {crate_name}").run()?;
        } else {
            let features_str = features.join(",");
            cmd!(sh, "cargo test -p {crate_name} --features {features_str}").run()?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_windows(
    sh: &Shell,
    release: bool,
    bin: Option<String>,
    example: Option<String>,
    features: Vec<String>,
    copy_to_windows: bool,
    out_name: Option<String>,
    out_dir: Option<String>,
) -> Result<()> {
    println!("🪟 Building Windows executable via cross-compilation...");

    // Check if Windows target is installed
    let output = cmd!(sh, "rustup target list --installed").read()?;
    if !output.contains("x86_64-pc-windows-gnu") {
        println!("📦 Installing Windows target...");
        cmd!(sh, "rustup target add x86_64-pc-windows-gnu").run()?;
    }

    // Warn if MinGW cross-compiler is missing
    if cmd!(sh, "which x86_64-w64-mingw32-gcc").read().is_err() {
        println!(
            "⚠️  Missing MinGW cross compiler (x86_64-w64-mingw32-gcc). On Ubuntu: sudo apt-get install -y mingw-w64"
        );
    }

    // Resolve target: prefer explicit bin/example, default to example viz_animated_egui
    if bin.is_some() && example.is_some() {
        return Err(anyhow::anyhow!(
            "--bin and --example are mutually exclusive"
        ));
    }
    let is_bin = bin.is_some();
    let target_name = if let Some(b) = bin.clone() {
        b
    } else {
        example
            .clone()
            .unwrap_or_else(|| "viz_animated_egui".to_string())
    };

    // Build for Windows
    let mut cmd_args: Vec<String> = vec![
        "build".into(),
        "--target".into(),
        "x86_64-pc-windows-gnu".into(),
    ];
    if is_bin {
        cmd_args.push("--bin".into());
    } else {
        cmd_args.push("--example".into());
    }
    cmd_args.push(target_name.clone());

    if !features.is_empty() {
        cmd_args.push("--features".into());
        let joined = features.join(",");
        cmd_args.push(joined);
    }
    if release {
        cmd_args.push("--release".into());
    }

    // Build command using owned args
    let cmd_ref: Vec<&str> = cmd_args.iter().map(|s| s.as_str()).collect();
    cmd!(sh, "cargo {cmd_ref...}").run()?;

    // Determine built exe path
    let exe_path = if is_bin {
        if release {
            format!("target/x86_64-pc-windows-gnu/release/{target_name}.exe")
        } else {
            format!("target/x86_64-pc-windows-gnu/debug/{target_name}.exe")
        }
    } else if release {
        format!("target/x86_64-pc-windows-gnu/release/examples/{target_name}.exe")
    } else {
        format!("target/x86_64-pc-windows-gnu/debug/examples/{target_name}.exe")
    };

    if std::path::Path::new(&exe_path).exists() {
        let metadata = std::fs::metadata(&exe_path)?;
        println!("✅ Windows executable built successfully!");
        println!("📍 Location: {}", &exe_path);
        println!("📦 Size: {:.2} MB", metadata.len() as f64 / 1_048_576.0);
        println!("🚀 Ready to copy to your Windows 11 PC!");

        // Verify it's a Windows PE executable
        if let Ok(output) = cmd!(sh, "file {exe_path}").read() {
            if output.contains("PE32+") && output.contains("MS Windows") {
                println!("✅ Verified: Windows PE32+ executable");
            }
        }

        // Copy to Windows shared location if requested
        if copy_to_windows {
            // Prefer .../FROMUBUNTU24/native if present, else .../FROMUBUNTU24
            let default_root = "/media/rustuser/onSSD/FROMUBUNTU24".to_string();
            let default_native = format!("{default_root}/native");
            let auto_default = if std::path::Path::new(&default_native).exists() {
                default_native
            } else {
                default_root
            };
            let dest_dir = out_dir.unwrap_or(auto_default);
            if !std::path::Path::new(&dest_dir).exists() {
                println!("ℹ️  Destination directory does not exist, creating: {dest_dir}");
                std::fs::create_dir_all(&dest_dir)?;
            }
            let base = out_name.unwrap_or(target_name);
            let dest_file = format!("{}/{}.exe", dest_dir.trim_end_matches('/'), base);
            cmd!(sh, "cp {exe_path} {dest_file}").run()?;
            println!("📂 Copied to Windows shared location: {dest_file}");
            println!("🪟 Ready to access from Windows 11!");
        } else {
            println!("ℹ️  Skipped copy. Use --copy-to-windows to copy automatically.");
        }
    } else {
        return Err(anyhow::anyhow!(
            "Build succeeded but executable not found at {}",
            exe_path
        ));
    }

    Ok(())
}

fn docs_index(_sh: &Shell, write: bool) -> Result<()> {
    use std::io::Write;
    let root = std::env::current_dir()?;
    let mut readmes = Vec::new();

    // Collect all README.md files except those under target/ or .git/
    fn visit(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                if name == "target" || name == ".git" {
                    continue;
                }
                visit(&path, out)?;
            } else if path.file_name().and_then(|s| s.to_str()) == Some("README.md") {
                out.push(path);
            }
        }
        Ok(())
    }
    visit(&root, &mut readmes)?;

    // Extract titles and build relative paths map
    #[derive(Clone)]
    struct DocEntry {
        path: PathBuf,
        title: String,
    }
    let mut entries: Vec<DocEntry> = Vec::new();
    for p in &readmes {
        let content = fs::read_to_string(p)?;
        let title = content
            .lines()
            .find_map(|l| {
                let lt = l.trim();
                if lt.starts_with('#') {
                    Some(lt.trim_start_matches('#').trim().to_string())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| {
                p.strip_prefix(&root)
                    .unwrap_or(p)
                    .to_string_lossy()
                    .to_string()
            });
        entries.push(DocEntry {
            path: p.clone(),
            title,
        });
    }

    let start_marker = "<!-- docs-index:start -->";
    let end_marker = "<!-- docs-index:end -->";

    for entry in &entries {
        let content = fs::read_to_string(&entry.path)?;
        let mut new_block = String::new();
        new_block.push_str(start_marker);
        new_block.push('\n');
        new_block.push_str(
            "<!-- Auto-generated by xtask docs-index. Do not edit between markers. -->\n",
        );
        new_block.push_str("### Related READMEs\n\n");

        // Build list of links relative to this file, excluding itself
        for other in &entries {
            if other.path == entry.path {
                continue;
            }
            let rel = pathdiff::diff_paths(&other.path, entry.path.parent().unwrap_or(&root))
                .unwrap_or_else(|| other.path.clone());
            new_block.push_str(&format!("- [{}]({})\n", other.title, rel.display()));
        }
        new_block.push_str(end_marker);

        let updated =
            if let (Some(s), Some(e)) = (content.find(start_marker), content.find(end_marker)) {
                let e_end = e + end_marker.len();
                let mut s1 = String::new();
                s1.push_str(&content[..s]);
                s1.push_str(&new_block);
                s1.push_str(&content[e_end..]);
                s1
            } else {
                // Append a new section at the end
                let mut s1 = content.clone();
                if !s1.ends_with('\n') {
                    s1.push('\n');
                }
                s1.push('\n');
                s1.push_str(&new_block);
                s1.push('\n');
                s1
            };

        if updated != content {
            if write {
                let mut f = fs::File::create(&entry.path)?;
                f.write_all(updated.as_bytes())?;
                println!(
                    "Updated {}",
                    entry
                        .path
                        .strip_prefix(&root)
                        .unwrap_or(&entry.path)
                        .display()
                );
            } else {
                println!(
                    "Would update {} (run with --write to apply)",
                    entry
                        .path
                        .strip_prefix(&root)
                        .unwrap_or(&entry.path)
                        .display()
                );
            }
        }
    }

    Ok(())
}

fn notebooks_check(sh: &Shell, features: Vec<String>) -> Result<()> {
    let manifest = "notebooks/Cargo.toml";
    let mut args = vec![
        "check".to_string(),
        "--manifest-path".to_string(),
        manifest.to_string(),
    ];
    let feats = if features.is_empty() {
        vec!["gui".to_string(), "fft".to_string(), "audio".to_string()]
    } else {
        features
    };
    if !feats.is_empty() {
        args.push("--features".to_string());
        args.push(feats.join(","));
    }
    let args_ref: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    cmd!(sh, "cargo {args_ref...}").run()?;
    Ok(())
}

/// Run comprehensive testing across the workspace including:
/// - Basic compilation check
/// - Feature combination testing  
/// - Test builds with all feature combinations
/// - Workspace clippy (informational)
/// - Documentation build
/// - Format check
/// - Notebooks validation
/// - Discovery and execution of all tests
fn comprehensive_test(sh: &Shell, extended: bool) -> Result<()> {
    println!("🚀 Starting comprehensive D-LinOSS workspace testing");
    println!("===================================================");

    let mut test_count = 0;
    let mut passed_count = 0;

    // Helper function for running tests with status tracking
    let mut run_test = |name: &str, test_fn: &dyn Fn() -> Result<()>| {
        test_count += 1;
        print!("📋 Testing: {name} ... ");
        std::io::Write::flush(&mut std::io::stdout()).ok();
        match test_fn() {
            Ok(_) => {
                println!("✅ PASS");
                passed_count += 1;
            }
            Err(e) => {
                println!("❌ FAIL: {e}");
            }
        }
    };

    // 1. Basic workspace check
    run_test("Workspace compilation", &|| -> Result<()> {
        cmd!(sh, "cargo check --workspace").run()?;
        Ok(())
    });

    // 2. Feature combination testing
    let feature_sets = if extended {
        vec![
            vec![],                       // baseline CPU
            vec!["fft"],                  // FFT only
            vec!["egui"],                 // GUI only
            vec!["etui"],                 // TUI only
            vec!["audio"],                // Audio only
            vec!["fft", "egui"],          // FFT + GUI
            vec!["fft", "etui"],          // FFT + TUI
            vec!["fft", "audio"],         // FFT + Audio
            vec!["egui", "audio"],        // GUI + Audio
            vec!["fft", "egui", "audio"], // All features
        ]
    } else {
        vec![
            vec![],              // baseline CPU
            vec!["fft"],         // FFT only
            vec!["fft", "egui"], // FFT + GUI (most common combo)
        ]
    };

    for features in feature_sets {
        let feature_name = if features.is_empty() {
            "CPU-only".to_string()
        } else {
            features.join("+")
        };

        run_test(
            &format!("Feature combo: {feature_name}"),
            &|| -> Result<()> {
                if features.is_empty() {
                    cmd!(sh, "cargo check --workspace").run()?;
                } else {
                    let features_str = features.join(",");
                    cmd!(sh, "cargo check --workspace --features {features_str}").run()?;
                }
                Ok(())
            },
        );
    }

    // 3. Test builds with feature combinations
    run_test("Test builds (CPU-only)", &|| -> Result<()> {
        cmd!(sh, "cargo test --workspace --no-run").run()?;
        Ok(())
    });

    if extended {
        run_test("Test builds (FFT)", &|| -> Result<()> {
            cmd!(sh, "cargo test --workspace --features fft --no-run").run()?;
            Ok(())
        });

        run_test("Test builds (FFT+GUI)", &|| -> Result<()> {
            cmd!(sh, "cargo test --workspace --features fft,egui --no-run").run()?;
            Ok(())
        });
    }

    // 4. Actual test execution
    run_test("Test execution (CPU-only)", &|| -> Result<()> {
        cmd!(sh, "cargo test --workspace").run()?;
        Ok(())
    });

    if extended {
        run_test("Test execution (FFT)", &|| -> Result<()> {
            cmd!(sh, "cargo test --workspace --features fft").run()?;
            Ok(())
        });
    }

    // 5. Workspace clippy (informational)
    run_test("Workspace clippy (informational)", &|| -> Result<()> {
        let _status = cmd!(sh, "cargo clippy --workspace --all-targets").run();
        // Don't fail on clippy warnings in comprehensive mode
        println!("  (clippy warnings are informational only)");
        Ok(())
    });

    // 6. Documentation build
    run_test("Documentation build", &|| -> Result<()> {
        cmd!(sh, "cargo doc --workspace --no-deps").run()?;
        Ok(())
    });

    // 7. Format check
    run_test("Format check", &|| -> Result<()> {
        cmd!(sh, "cargo fmt --all -- --check").run()?;
        Ok(())
    });

    // 8. Candle verification
    run_test("Candle FFT verification", &|| -> Result<()> {
        cmd!(sh, "cargo test -p dlinoss-helpers --features fft").run()?;
        Ok(())
    });

    // 9. Notebooks validation
    run_test("Notebooks compilation", &|| -> Result<()> {
        notebooks_check(
            sh,
            vec!["gui".to_string(), "fft".to_string(), "audio".to_string()],
        )?;
        Ok(())
    });

    // 10. Test discovery and validation
    run_test("Test discovery", &|| -> Result<()> {
        discover_tests(sh, false, vec![])?; // Just discovery, no execution
        Ok(())
    });

    // 11. Examples compilation check
    run_test("Examples compilation", &|| -> Result<()> {
        cmd!(sh, "cargo check --examples").run()?;
        Ok(())
    });

    if extended {
        run_test("Examples compilation (FFT)", &|| -> Result<()> {
            cmd!(sh, "cargo check --examples --features fft").run()?;
            Ok(())
        });

        run_test("Examples compilation (FFT+GUI)", &|| -> Result<()> {
            cmd!(sh, "cargo check --examples --features fft,egui").run()?;
            Ok(())
        });
    }

    // Summary
    println!();
    println!("📊 COMPREHENSIVE TEST SUMMARY");
    println!("=============================");
    println!("Total tests: {test_count}");
    println!("Passed: {passed_count} ✅");
    println!("Failed: {} ❌", test_count - passed_count);

    let success_rate = if test_count > 0 {
        (passed_count * 100) / test_count
    } else {
        0
    };
    println!("Success rate: {success_rate}%");

    if passed_count == test_count {
        println!();
        println!("🎉 All tests passed! D-LinOSS workspace is healthy.");
        println!();
        println!("Environment variables for even more thorough testing:");
        println!("  Run with --extended for additional feature combinations");
        Ok(())
    } else {
        println!();
        println!("⚠️  Some tests failed. See output above for details.");
        anyhow::bail!(
            "Comprehensive test suite failed with {}/{test_count} tests passing",
            passed_count
        )
    }
}

fn assemble_gif(out: Option<String>, delay_cs: u16, max_frames: Option<usize>) -> Result<()> {
    use gif::{Encoder, Frame, Repeat};
    use image::GenericImageView;
    use std::fs::File;
    let dir = PathBuf::from("images_store/SHOWCASE");
    anyhow::ensure!(dir.exists(), "{} does not exist", dir.display());
    let mut frames: Vec<_> = fs::read_dir(&dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("png"))
        .collect();
    frames.sort();
    if frames.is_empty() {
        anyhow::bail!("No PNG frames found in {}", dir.display());
    }
    if let Some(m) = max_frames {
        if frames.len() > m {
            frames.truncate(m);
        }
    }
    let first = image::open(&frames[0])?;
    let (w, h) = first.dimensions();
    let out_path = out.unwrap_or_else(|| format!("{}/anim.gif", dir.display()));
    let mut file = File::create(&out_path)?;
    let mut encoder = Encoder::new(&mut file, w as u16, h as u16, &[])?;
    encoder.set_repeat(Repeat::Infinite)?;
    for (idx, path) in frames.iter().enumerate() {
        let img = if idx == 0 {
            first.clone()
        } else {
            image::open(path)?
        };
        let mut frame = Frame::default();
        frame.width = w as u16;
        frame.height = h as u16;
        frame.delay = delay_cs; // hundredths of second
        // Convert to RGBA then to indexed? Simpler: use RGBA -> encode as RGBA (gif encoder will quantize)
        let rgba = img.to_rgba8();
        frame.buffer = rgba.into_raw().into();
        encoder.write_frame(&frame)?;
    }
    println!("GIF written to {out_path}");
    Ok(())
}

fn mcp_demo(fft: bool, steps: usize, fft_size: usize) -> Result<()> {
    use serde_json::json;
    use std::io::{BufRead, Write};
    use std::process::{Command, Stdio};
    println!("🚀 Starting dlinoss-mcp demo (fft={fft})");
    // Construct cargo run command for the mcp server
    let mut cmd = Command::new("cargo");
    cmd.arg("run").arg("-p").arg("dlinoss-mcp");
    if fft {
        cmd.arg("--features").arg("fft");
    }
    cmd.stdin(Stdio::piped()).stdout(Stdio::piped());
    let mut child = cmd.spawn()?;
    let mut child_stdin = child
        .stdin
        .take()
        .ok_or_else(|| anyhow::anyhow!("failed to take stdin"))?;
    let child_stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("failed to take stdout"))?;
    let mut lines = std::io::BufReader::new(child_stdout).lines();

    // Helper closure to send JSON and read one line
    let mut send = |val: serde_json::Value| -> Result<serde_json::Value> {
        let line = serde_json::to_string(&val)?;
        writeln!(child_stdin, "{}", line)?;
        child_stdin.flush()?;
        let resp_line = lines
            .next()
            .ok_or_else(|| anyhow::anyhow!("no response"))??;
        let resp: serde_json::Value = serde_json::from_str(&resp_line)?;
        Ok(resp)
    };

    // 1. ping
    let ping = send(json!({"jsonrpc":"2.0","id":1,"method":"dlinoss.ping","params":{}}))?;
    println!("➡ ping -> {}", ping);
    // 2. init
    let init = send(
        json!({"jsonrpc":"2.0","id":2,"method":"dlinoss.init","params":{"state_dim":16,"input_dim":2,"output_dim":3,"t_len":10000}}),
    )?;
    println!("➡ init -> {}", init);
    // 3. step
    let step =
        send(json!({"jsonrpc":"2.0","id":3,"method":"dlinoss.step","params":{"steps":steps}}))?;
    println!("➡ step({steps}) -> {}", step);
    // 4. getState rings
    let rings = send(
        json!({"jsonrpc":"2.0","id":4,"method":"dlinoss.getState","params":{"which":"rings","limit":2048}}),
    )?;
    if let Some(err) = rings.get("error") {
        println!("➡ rings error: {}", err);
    } else {
        let ring_len = rings
            .pointer("/result/data")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        println!("➡ rings -> len={ring_len}");
        // Adjust FFT size if needed
        let mut req_fft = fft_size.min(ring_len.max(0));
        if req_fft == 0 {
            println!("➡ fft skipped (no samples yet)");
        } else {
            // ensure even and at least 16
            if req_fft % 2 == 1 {
                req_fft -= 1;
            }
            if req_fft < 16 {
                req_fft = 16.min(ring_len);
            }
            let fft_req =
                json!({"jsonrpc":"2.0","id":5,"method":"dlinoss.getFft","params":{"size":req_fft}});
            let fft_resp = send(fft_req)?;
            if let Some(err) = fft_resp.get("error") {
                println!("➡ fft error: {}", err);
            } else {
                let spec_len = fft_resp
                    .pointer("/result/spectrum")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0);
                println!("➡ fft(size={req_fft}) -> spectrum_len={spec_len}");
            }
        }
    }

    // Shutdown: drop stdin so server exits
    drop(child_stdin);
    let _ = child.wait();
    println!("✅ MCP demo complete");
    Ok(())
}
