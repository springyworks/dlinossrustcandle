//! D-LinOSS Notebooks - Single Dependency Re-exports
//!
//! This crate re-exports the main D-LinOSS API with all necessary dependencies
//! so notebooks can use a single `:dep` line for complete functionality.
//!
//! Usage in evcxr:
//! ```
//! :dep dlinoss-notebooks = { path = ".", features = ["fft", "gui", "audio"] }
//! use dlinoss_notebooks::*;
//! ```

// Re-export all Candle core functionality for tensor operations from the root crate
pub use dlinossrustcandle::{
    // Core D-LinOSS types
    DLinOssLayer,
    DLinOssLayerConfig,
    DType,
    // Re-export candle types that the root crate already imports from your local candlekos
    Device,
    Tensor,
};

// Re-export scan operations from the root crate's augment module
pub use dlinossrustcandle::TensorScanExt;

// FFT operations (re-exported from root crate if available)
#[cfg(feature = "fft")]
pub use dlinossrustcandle::TensorFftExt;

// Re-export essential utilities
pub use anyhow::{bail, ensure, Context, Error as AnyhowError, Result};
// Also expose an `anyhow` module for convenience in notebooks (e.g., `use dlinoss_notebooks::anyhow`).
pub mod anyhow {
    pub use anyhow::*;
}

// Re-export display and notebook utilities
pub use base64;
pub use html_escape;

// Optional GUI re-exports (feature-gated)
#[cfg(feature = "gui")]
pub use eframe;

#[cfg(feature = "gui")]
pub use egui_plot;

// Re-export display helpers from the root crate (egui dual-pane)
#[cfg(feature = "gui")]
pub use dlinossrustcandle::display_egui;

// Optional audio re-exports (feature-gated)
#[cfg(feature = "audio")]
pub use cpal;

/// Very small CPAL helper to play a mono f32 buffer at a given sample rate.
#[cfg(feature = "audio")]
pub mod audio_utils {
    use super::Result;
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use std::collections::VecDeque;
    use std::sync::mpsc;

    pub fn play_mono_f32(buffer: Vec<f32>, sample_rate: u32) -> Result<()> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow::anyhow!("No default output device"))?;
        let mut supported = device.supported_output_configs()?;
        let cfg = supported
            .find(|c| c.channels() == 1 && c.sample_format() == cpal::SampleFormat::F32)
            .map(|c| c.with_max_sample_rate())
            .unwrap_or_else(|| {
                // fallback to first config
                supported
                    .next()
                    .expect("no output configs available")
                    .with_max_sample_rate()
            })
            .config();
        // Capture total length before moving the buffer into the closure
        let total_len = buffer.len();
        let mut idx = 0usize;
        let stream = device.build_output_stream(
            &cpal::StreamConfig {
                channels: 1,
                sample_rate: cpal::SampleRate(sample_rate),
                buffer_size: cfg.buffer_size.clone(),
            },
            move |data: &mut [f32], _| {
                for sample in data.iter_mut() {
                    *sample = if idx < total_len { buffer[idx] } else { 0.0 };
                    idx = idx.saturating_add(1);
                }
            },
            move |err| eprintln!("audio error: {err}"),
            None,
        )?;
        stream.play()?;
        // Let the buffer play; in a real app, block by UI/event loop.
        std::thread::sleep(std::time::Duration::from_millis(
            (total_len as u64) * 1000 / (sample_rate as u64),
        ));
        Ok(())
    }

    /// Start a non-blocking mono f32 output stream and return a Sender to push audio frames.
    /// Each sent Vec<f32> will be enqueued and consumed by the audio callback.
    /// The stream is started (playing) before returning.
    pub fn start_mono_stream_queue(
        sample_rate: u32,
    ) -> Result<(cpal::Stream, mpsc::Sender<Vec<f32>>)> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow::anyhow!("No default output device"))?;
        let mut supported = device.supported_output_configs()?;
        let cfg = supported
            .find(|c| c.channels() == 1 && c.sample_format() == cpal::SampleFormat::F32)
            .map(|c| c.with_max_sample_rate())
            .unwrap_or_else(|| {
                supported
                    .next()
                    .expect("no output configs available")
                    .with_max_sample_rate()
            })
            .config();

        let (tx, rx) = mpsc::channel::<Vec<f32>>();
        let mut queue: VecDeque<f32> = VecDeque::new();
        let stream = device.build_output_stream(
            &cpal::StreamConfig {
                channels: 1,
                sample_rate: cpal::SampleRate(sample_rate),
                buffer_size: cfg.buffer_size.clone(),
            },
            move |data: &mut [f32], _| {
                // Drain any pending buffers without blocking
                while let Ok(mut chunk) = rx.try_recv() {
                    queue.extend(chunk.drain(..));
                }
                for sample in data.iter_mut() {
                    *sample = queue.pop_front().unwrap_or(0.0);
                }
            },
            move |err| eprintln!("audio error: {err}"),
            None,
        )?;
        stream.play()?;
        Ok((stream, tx))
    }
}

// Optional windowing re-exports (feature-gated)
#[cfg(feature = "display")]
pub use minifb;

/// Convenience function to create a CPU device
pub fn cpu_device() -> Device {
    Device::Cpu
}

/// Convenience function to create a CUDA device (if available)
#[cfg(feature = "cuda")]
pub fn cuda_device(device_id: usize) -> Result<Device> {
    Device::new_cuda(device_id).map_err(|e| anyhow::anyhow!("Failed to create CUDA device: {}", e))
}

/// Set working directory for notebooks to ensure reproducible paths.
/// Automatically detects the correct notebook directory based on the manifest.
/// Call this in the first cell of each notebook for consistent behavior.
pub fn set_notebook_cwd() -> Result<()> {
    use std::env;

    // Strategy 1: Try to find notebooks directory by looking for Cargo.toml with dlinoss-notebooks
    let current = env::current_dir()?;

    // Check if we're already in a notebooks directory with the right Cargo.toml
    let cargo_toml = current.join("Cargo.toml");
    if cargo_toml.exists() {
        let content = std::fs::read_to_string(&cargo_toml)?;
        if content.contains("dlinoss-notebooks") && content.contains("dlinossrustcandle") {
            // We're already in the notebooks directory
            return Ok(());
        }
    }

    // Strategy 2: Look for notebooks directory in workspace ancestors
    let mut search_path = current.clone();
    for _ in 0..5 {
        // Limit search depth
        let notebooks_path = search_path.join("notebooks");
        let notebooks_cargo = notebooks_path.join("Cargo.toml");

        if notebooks_cargo.exists() {
            let content = std::fs::read_to_string(&notebooks_cargo)?;
            if content.contains("dlinoss-notebooks") {
                env::set_current_dir(&notebooks_path)?;
                return Ok(());
            }
        }

        match search_path.parent() {
            Some(parent) => search_path = parent.to_path_buf(),
            None => break,
        }
    }

    // Strategy 3: Look for workspace with notebooks subdirectory
    let mut search_path = current;
    for _ in 0..5 {
        let workspace_cargo = search_path.join("Cargo.toml");
        if workspace_cargo.exists() {
            let content = std::fs::read_to_string(&workspace_cargo)?;
            if content.contains("dlinossrustcandle") && content.contains("[workspace]") {
                let notebooks_path = search_path.join("notebooks");
                if notebooks_path.exists() {
                    env::set_current_dir(&notebooks_path)?;
                    return Ok(());
                }
            }
        }

        match search_path.parent() {
            Some(parent) => search_path = parent.to_path_buf(),
            None => break,
        }
    }

    // If all strategies fail, give helpful error
    anyhow::bail!(
        "Could not find dlinoss-notebooks directory. \
         Please run the notebook from within the notebooks/ folder or \
         from the workspace root containing notebooks/"
    )
}

/// Set relative image store directory from the current working directory.
/// Call after set_notebook_cwd() to ensure images are saved in the right place.
pub fn set_image_store_rel_dir(dir_name: &str) -> Result<()> {
    std::fs::create_dir_all(dir_name)?;
    Ok(())
}

// (single re-export above)

/// Quick real-FFT magnitude spectrum helper for notebook visualization.
/// Returns magnitude spectrum bins for a 1D real signal.
#[cfg(feature = "fft")]
pub fn rfft_magnitude(signal: &[f32]) -> Result<Vec<f32>> {
    let device = Device::Cpu;
    let t = Tensor::from_slice(signal, signal.len(), &device)?;
    let fft_result = t.rfft(0, true)?; // real FFT, normalized
    let complex_data = fft_result.to_vec1::<f32>()?;
    // Complex interleaved: [re0, im0, re1, im1, ...]
    let mut magnitudes = Vec::with_capacity(complex_data.len() / 2);
    for chunk in complex_data.chunks_exact(2) {
        let re = chunk[0];
        let im = chunk[1];
        magnitudes.push((re * re + im * im).sqrt());
    }
    Ok(magnitudes)
}

/// Placeholder for when FFT is not enabled.
#[cfg(not(feature = "fft"))]
pub fn rfft_magnitude(_signal: &[f32]) -> Result<Vec<f32>> {
    Ok(vec![])
}

/// Common signal generators for quick experiments in notebooks.
/// Shapes: returns tensors shaped [B=1, T, In=1] on CPU, dtype f32.
pub struct SignalGen;

impl SignalGen {
    /// Unit impulse at t=0.
    pub fn impulse(t: usize) -> Result<Tensor> {
        let dev = Device::Cpu;
        let mut v = vec![0.0f32; t];
        if t > 0 {
            v[0] = 1.0;
        }
        Ok(Tensor::from_slice(&v, (1, t, 1), &dev)?)
    }

    /// Constant step of amplitude `amp` for all t.
    pub fn step(t: usize, amp: f32) -> Result<Tensor> {
        let dev = Device::Cpu;
        let v = vec![amp; t];
        Ok(Tensor::from_slice(&v, (1, t, 1), &dev)?)
    }

    /// Sine wave at `freq_hz` with sample period `dt` seconds.
    pub fn sine(t: usize, freq_hz: f32, dt: f32) -> Result<Tensor> {
        let dev = Device::Cpu;
        let v: Vec<f32> = (0..t)
            .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 * dt).sin())
            .collect();
        Ok(Tensor::from_slice(&v, (1, t, 1), &dev)?)
    }

    /// Linear chirp from `f0` to `f1` over the window, sampled at `dt` seconds.
    pub fn chirp(t: usize, f0: f32, f1: f32, dt: f32) -> Result<Tensor> {
        let dev = Device::Cpu;
        let duration = (t as f32) * dt;
        let v: Vec<f32> = (0..t)
            .map(|i| {
                let time = i as f32 * dt;
                let freq = if duration > 0.0 {
                    f0 + (f1 - f0) * time / duration
                } else {
                    f0
                };
                (2.0 * std::f32::consts::PI * freq * time).sin()
            })
            .collect();
        Ok(Tensor::from_slice(&v, (1, t, 1), &dev)?)
    }
}

/// Run a simple D-LinOSS experiment for testing
pub fn simple_dlinoss_experiment() -> Result<()> {
    println!("🧪 D-LinOSS Simple Experiment");

    // Create device and basic configuration
    let device = cpu_device();
    let config = DLinOssLayerConfig::default();

    // Create D-LinOSS layer
    let layer = DLinOssLayer::new(config, &device)?;

    // Generate simple test signal
    let test_signal = SignalGen::sine(100, 10.0, 0.01)?;

    // Process through layer
    let output = layer.forward(&test_signal, None)?;

    println!("✅ Successfully created and ran D-LinOSS layer!");
    println!(
        "📊 Input shape: {:?}, Output shape: {:?}",
        test_signal.shape(),
        output.shape()
    );

    Ok(())
}

// Minimal realtime GUI+audio glue so notebooks can import `dlinoss_notebooks::realtime_gui`
// and run a tiny processing loop. This is a lightweight placeholder; a richer egui/cpal
// implementation can be added later in the display/helpers crates and re-exported here.
#[cfg(any(feature = "gui", feature = "audio"))]
pub mod realtime_gui {
    use super::{Result, Tensor};

    /// Trait representing a processor that takes an input tensor [B,T,In] and returns [B,T,Out].
    pub trait TensorProcessor {
        fn process(&mut self, input: &Tensor) -> Result<Tensor>;
    }

    /// Run a minimal realtime demo. For now, this performs a few processing steps without spawning
    /// a real GUI/audio stream. It validates that the processing pipeline is wired correctly.
    pub fn run_realtime_demo_with_audio<P: TensorProcessor>(mut proc: P) -> Result<()> {
        // Generate a tiny input window and call the user processor a couple of times.
        // Shapes: [B=1, T=128, In=1]
        let x = super::SignalGen::sine(128, 2.0, 1e-2)?;
        let _y1 = proc.process(&x)?;
        let _y2 = proc.process(&x)?;
        Ok(())
    }
}
