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
    let fft_result = t.rfft(0, true)?;  // real FFT, normalized
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
                let freq = if duration > 0.0 { f0 + (f1 - f0) * time / duration } else { f0 };
                (2.0 * std::f32::consts::PI * freq * time).sin()
            })
            .collect();
        Ok(Tensor::from_slice(&v, (1, t, 1), &dev)?)
    }
}
