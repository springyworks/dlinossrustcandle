//! dlinoss-augment: extension traits that wrap Candle's scan/fft ops.

use anyhow::Result;
use candlekos::Tensor;

pub trait TensorScanExt {
    /// Inclusive prefix-sum along axis: cumsum
    fn prefix_sum_along(&self, axis: usize) -> Result<Tensor>;
    /// Exclusive prefix-sum along axis
    fn exclusive_prefix_sum_along(&self, axis: usize) -> Result<Tensor>;
    /// Inclusive prefix-sum across time for [B,T,D]
    fn prefix_sum_time_btd(&self) -> Result<Tensor>;
    /// Exclusive prefix-sum across time for [B,T,D]
    fn exclusive_prefix_sum_time_btd(&self) -> Result<Tensor>;
}

impl TensorScanExt for Tensor {
    fn prefix_sum_along(&self, axis: usize) -> Result<Tensor> {
        Ok(self.cumsum(axis)?)
    }
    fn exclusive_prefix_sum_along(&self, axis: usize) -> Result<Tensor> {
        Ok(self.exclusive_scan(axis)?)
    }
    fn prefix_sum_time_btd(&self) -> Result<Tensor> {
        let dims = self.dims();
        anyhow::ensure!(dims.len() == 3, "expected [B,T,D]");
        Ok(self.cumsum(1)?)
    }
    fn exclusive_prefix_sum_time_btd(&self) -> Result<Tensor> {
        let dims = self.dims();
        anyhow::ensure!(dims.len() == 3, "expected [B,T,D]");
        Ok(self.exclusive_scan(1)?)
    }
}

#[cfg(feature = "fft")]
pub trait TensorFftExt {
    /// 1D FFT convenience: real input along axis 0, normalized, return interleaved (re,im)
    fn fft_real_norm(&self) -> Result<Tensor>;
}

#[cfg(feature = "fft")]
impl TensorFftExt for Tensor {
    fn fft_real_norm(&self) -> Result<Tensor> {
        // Axis 0 by convention for 1D vectors
        Ok(self.rfft(0usize, true)?)
    }
}

// Optional extras from Candle's exploration augment crate
#[cfg(feature = "augment")]
pub use candle_tensor_augment::TensorAugment;

#[cfg(feature = "augment-expr")]
pub use candle_tensor_augment::TensorMathFill;
