//! dlinoss-notebooks: glue crate for research notebooks
//! Re-exports candle-notebooks utilities and the D-LinOSS API for single-dep usage.

pub use anyhow::Result;
pub use candle_notebooks::*;

// Re-export the core D-LinOSS API so notebooks can `use dlinoss_notebooks::*`.
pub use dlinossrustcandle::{DLinOssLayer, DLinOssLayerConfig};

// Re-export helpful traits when present (no feature gating needed for scan ext)
pub use dlinossrustcandle::TensorScanExt;

#[cfg(feature = "fft")]
pub use dlinossrustcandle::TensorFftExt;

/// Quick real-FFT magnitude spectrum helper for notebook visualization.
/// Returns magnitude spectrum bins for a 1D real signal.
#[cfg(feature = "fft")]
pub fn rfft_magnitude(signal: &[f32]) -> Result<Vec<f32>> {
    use candle_core::{Device, Tensor};
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
    pub fn impulse(t: usize) -> Result<candle_core::Tensor> {
        use candle_core::{Device, Tensor};
        let dev = Device::Cpu;
        let mut v = vec![0.0f32; t];
        if t > 0 {
            v[0] = 1.0;
        }
        Ok(Tensor::from_slice(&v, (1, t, 1), &dev)?)
    }

    /// Constant step of amplitude `amp` for all t.
    pub fn step(t: usize, amp: f32) -> Result<candle_core::Tensor> {
        use candle_core::{Device, Tensor};
        let dev = Device::Cpu;
        let v = vec![amp; t];
        Ok(Tensor::from_slice(&v, (1, t, 1), &dev)?)
    }

    /// Sine wave at `freq_hz` with sample period `dt` seconds.
    pub fn sine(t: usize, freq_hz: f32, dt: f32) -> Result<candle_core::Tensor> {
        use candle_core::{Device, Tensor};
        let dev = Device::Cpu;
        let v: Vec<f32> = (0..t)
            .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 * dt).sin())
            .collect();
        Ok(Tensor::from_slice(&v, (1, t, 1), &dev)?)
    }

    /// Linear chirp from `f0` to `f1` over the window, sampled at `dt` seconds.
    pub fn chirp(t: usize, f0: f32, f1: f32, dt: f32) -> Result<candle_core::Tensor> {
        use candle_core::{Device, Tensor};
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

#[cfg(feature = "gui")]
pub mod realtime_gui {
    use super::*;
    use eframe::{egui, App, Frame, NativeOptions};
    use egui_plot::{Line, Plot, PlotPoints};
    use std::time::Instant;

    struct TinyApp {
        device: candle_core::Device,
        layer: DLinOssLayer,
        t: f32,
        input: Vec<f32>,
        output: Vec<f32>,
        last: Instant,
    }

    impl TinyApp {
        fn new() -> Result<Self> {
            let device = candle_core::Device::Cpu;
            let cfg = DLinOssLayerConfig { state_dim: 16, input_dim: 1, output_dim: 1, delta_t: 0.01, dtype: candle_core::DType::F32 };
            let layer = DLinOssLayer::new(cfg, &device)?;
            Ok(Self { device, layer, t: 0.0, input: vec![0.0; 256], output: vec![0.0; 256], last: Instant::now() })
        }
        fn step(&mut self) -> Result<()> {
            let now = Instant::now();
            let dt = (now - self.last).as_secs_f32().clamp(0.0, 0.05);
            self.last = now;
            self.t += dt;
            // generate 24 new samples of a swept sine for fun
            let len = 24usize;
            let dt_samp = 0.01f32;
            let values: Vec<f32> = (0..len).map(|i| {
                let tt = self.t + i as f32 * dt_samp;
                let freq = 0.5 + 0.5 * (0.5 * tt).sin();
                (2.0 * std::f32::consts::PI * freq * tt).sin()
            }).collect();
            let x = candle_core::Tensor::from_slice(&values, (1, len, 1), &self.device)?;
            let y = self.layer.forward(&x, None)?;
            let xin = x.squeeze(0)?.squeeze(1)?.to_vec1::<f32>()?;
            let yout = y.squeeze(0)?.squeeze(1)?.to_vec1::<f32>()?;
            for i in 0..len {
                self.input.remove(0); self.input.push(xin[i]);
                self.output.remove(0); self.output.push(yout[i]);
            }
            Ok(())
        }
    }

    impl App for TinyApp {
        fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
            ctx.request_repaint();
            if let Err(e) = self.step() { eprintln!("step error: {e}"); }
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.heading("D-LinOSS Realtime Demo");
                let in_pts: PlotPoints = self.input.iter().enumerate().map(|(i,&y)| [i as f64, y as f64]).collect::<Vec<_>>().into();
                let out_pts: PlotPoints = self.output.iter().enumerate().map(|(i,&y)| [i as f64, y as f64]).collect::<Vec<_>>().into();
                Plot::new("io").height(260.0).show(ui, |p| {
                    p.line(Line::new(in_pts).name("input").color(egui::Color32::LIGHT_GREEN));
                    p.line(Line::new(out_pts).name("output").color(egui::Color32::LIGHT_BLUE));
                });
            });
        }
    }

    /// Launch a minimal realtime egui window animating DLinOSS input/output in place.
    pub fn run_realtime_demo() -> Result<()> {
        let app = TinyApp::new()?;
        let opts = NativeOptions::default();
        eframe::run_native("DLinOSS Realtime", opts, Box::new(|_| Ok::<Box<dyn App>, anyhow::Error>(Box::new(app))))
            .map_err(|e| anyhow::anyhow!("GUI error: {e}"))
    }
}
