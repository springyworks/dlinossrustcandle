//! Experimental kernelization (FFT) stubs. Not implemented yet.

use anyhow::{Result, bail};
use candlekos::Tensor;

/// Generate an impulse response kernel (stub).
#[cfg(feature = "fft")]
pub fn impulse_kernel(_len: usize) -> Result<Tensor> {
    bail!("fft feature is experimental and not implemented yet")
}

/// Convolve input sequence with a kernel via FFT (stub).
#[cfg(feature = "fft")]
pub fn convolve_fft(_x: &Tensor, _k: &Tensor) -> Result<Tensor> {
    bail!("fft feature is experimental and not implemented yet")
}
