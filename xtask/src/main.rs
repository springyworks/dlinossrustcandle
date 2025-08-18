use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::fs;
use std::path::{Path, PathBuf};
use xshell::{cmd, Shell};

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
            cmd!(sh, "cargo fmt --all").run()?;
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
            cmd!(sh, "cargo fmt --all").run()?;
        }
        Cmd::Clippy => {
            cmd!(sh, "cargo clippy --all-targets --all-features").run()?;
        }
        Cmd::RunEx { name, args: extra } => {
            let mut cmdline = format!("cargo run --example {}", name);
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
            let mut cmdline = format!("cargo run --bin {}", name);
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

fn ensure_dir(path: &Path) -> Result<()> {
    if !path.exists() {
        fs::create_dir_all(path)?;
    }
    if !path.join("src").exists() {
        fs::create_dir_all(path.join("src"))?;
    }
    Ok(())
}

fn write_probe_files(probe_dir: &Path, candle_core_abs: &Path) -> Result<()> {
    let cargo_toml = format!(
        r#"[package]
name = "candle_probe"
version = "0.1.0"
edition = "2021"

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
    line_number: usize,
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
                        line_number: line_num + 1,
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
        println!("Testing {} individually...", crate_name);
        if features.is_empty() {
            cmd!(sh, "cargo test -p {crate_name}").run()?;
        } else {
            let features_str = features.join(",");
            cmd!(sh, "cargo test -p {crate_name} --features {features_str}").run()?;
        }
    }

    Ok(())
}

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
        println!("⚠️  Missing MinGW cross compiler (x86_64-w64-mingw32-gcc). On Ubuntu: sudo apt-get install -y mingw-w64");
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
            format!("target/x86_64-pc-windows-gnu/release/{}.exe", target_name)
        } else {
            format!("target/x86_64-pc-windows-gnu/debug/{}.exe", target_name)
        }
    } else {
        if release {
            format!(
                "target/x86_64-pc-windows-gnu/release/examples/{}.exe",
                target_name
            )
        } else {
            format!(
                "target/x86_64-pc-windows-gnu/debug/examples/{}.exe",
                target_name
            )
        }
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
            let default_native = format!("{}/native", default_root);
            let auto_default = if std::path::Path::new(&default_native).exists() {
                default_native
            } else {
                default_root
            };
            let dest_dir = out_dir.unwrap_or(auto_default);
            if !std::path::Path::new(&dest_dir).exists() {
                println!(
                    "ℹ️  Destination directory does not exist, creating: {}",
                    dest_dir
                );
                std::fs::create_dir_all(&dest_dir)?;
            }
            let base = out_name.unwrap_or(target_name);
            let dest_file = format!("{}/{}.exe", dest_dir.trim_end_matches('/'), base);
            cmd!(sh, "cp {exe_path} {dest_file}").run()?;
            println!("📂 Copied to Windows shared location: {}", dest_file);
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
                if name == "target" || name == ".git" { continue; }
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
    struct DocEntry { path: PathBuf, title: String }
    let mut entries: Vec<DocEntry> = Vec::new();
    for p in &readmes {
        let content = fs::read_to_string(p)?;
        let title = content
            .lines()
            .find_map(|l| {
                let lt = l.trim();
                if lt.starts_with('#') {
                    Some(lt.trim_start_matches('#').trim().to_string())
                } else { None }
            })
            .unwrap_or_else(|| p.strip_prefix(&root).unwrap_or(p).to_string_lossy().to_string());
        entries.push(DocEntry { path: p.clone(), title });
    }

    let start_marker = "<!-- docs-index:start -->";
    let end_marker = "<!-- docs-index:end -->";

    for entry in &entries {
        let content = fs::read_to_string(&entry.path)?;
        let mut new_block = String::new();
        new_block.push_str(start_marker);
        new_block.push('\n');
        new_block.push_str("<!-- Auto-generated by xtask docs-index. Do not edit between markers. -->\n");
        new_block.push_str("### Related READMEs\n\n");

        // Build list of links relative to this file, excluding itself
        for other in &entries {
            if other.path == entry.path { continue; }
            let rel = pathdiff::diff_paths(&other.path, entry.path.parent().unwrap_or(&root))
                .unwrap_or_else(|| other.path.clone());
            new_block.push_str(&format!("- [{}]({})\n", other.title, rel.display()));
        }
        new_block.push_str(end_marker);

        let updated = if let (Some(s), Some(e)) = (content.find(start_marker), content.find(end_marker)) {
            let e_end = e + end_marker.len();
            let mut s1 = String::new();
            s1.push_str(&content[..s]);
            s1.push_str(&new_block);
            s1.push_str(&content[e_end..]);
            s1
        } else {
            // Append a new section at the end
            let mut s1 = content.clone();
            if !s1.ends_with('\n') { s1.push('\n'); }
            s1.push_str("\n");
            s1.push_str(&new_block);
            s1.push('\n');
            s1
        };

        if updated != content {
            if write {
                let mut f = fs::File::create(&entry.path)?;
                f.write_all(updated.as_bytes())?;
                println!("Updated {}", entry.path.strip_prefix(&root).unwrap_or(&entry.path).display());
            } else {
                println!("Would update {} (run with --write to apply)", entry.path.strip_prefix(&root).unwrap_or(&entry.path).display());
            }
        }
    }

    Ok(())
}
