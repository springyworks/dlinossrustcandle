//! dlinoss-helpers: classic CLI args and small utilities.

pub mod probe;

use anyhow::Result;
use clap::{ArgAction, Parser, ValueEnum};

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum UiKind {
    Gui,
    Tui,
    Headless,
}

#[derive(Debug, Parser)]
#[command(name = "dlinoss", version, about = "Run D-LinOSS demos", long_about = None)]
pub struct CommonCli {
    /// UI mode: GUI (egui), TUI (terminal), or Headless
    #[arg(long, value_enum, default_value_t = UiKind::Gui)]
    pub ui: UiKind,
    /// Sequence length
    #[arg(long, default_value_t = 1024)]
    pub t: usize,
    /// Input mode (1..5)
    #[arg(long, default_value_t = 3)]
    pub mode: u8,
    /// Frequency for sine input
    #[arg(long, default_value_t = 0.03)]
    pub freq: f32,
    /// Enable FFT views where available
    #[arg(long, action = ArgAction::SetTrue)]
    pub fft: bool,
    /// Optional interactive CLI dialog (old-school)
    #[arg(long, action = ArgAction::SetTrue)]
    pub dialog: bool,
}

impl CommonCli {
    pub fn parse_or_dialog() -> Result<Self> {
        let mut args = Self::parse();
        if args.dialog {
            // minimal interactive prompts; keep old-school
            use std::io::{stdin, stdout, Write};
            let mut s = String::new();
            print!(
                "UI (gui/tui/headless) [{}]: ",
                match args.ui {
                    UiKind::Gui => "gui",
                    UiKind::Tui => "tui",
                    UiKind::Headless => "headless",
                }
            );
            stdout().flush()?;
            s.clear();
            stdin().read_line(&mut s)?;
            let st = s.trim().to_lowercase();
            args.ui = match st.as_str() {
                "tui" => UiKind::Tui,
                "headless" => UiKind::Headless,
                _ => UiKind::Gui,
            };
            print!("T [{}]: ", args.t);
            stdout().flush()?;
            s.clear();
            stdin().read_line(&mut s)?;
            if let Ok(v) = s.trim().parse() {
                args.t = v;
            }
            print!("Mode (1..5) [{}]: ", args.mode);
            stdout().flush()?;
            s.clear();
            stdin().read_line(&mut s)?;
            if let Ok(v) = s.trim().parse() {
                args.mode = v;
            }
            print!("Freq [{}]: ", args.freq);
            stdout().flush()?;
            s.clear();
            stdin().read_line(&mut s)?;
            if let Ok(v) = s.trim().parse() {
                args.freq = v;
            }
            print!("FFT (y/N) [{}]: ", if args.fft { "y" } else { "n" });
            stdout().flush()?;
            s.clear();
            stdin().read_line(&mut s)?;
            let st = s.trim().to_lowercase();
            args.fft = st == "y" || st == "yes";
        }
        Ok(args)
    }
}
