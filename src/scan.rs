//! Experimental scan APIs. Parallel path is not implemented yet.
//!
//! This module also exposes convenience wrappers over Candle's scan-like
//! tensor ops along a chosen axis (notably time when using [B, T, D]).
//! These are useful for associative operations like prefix-sums.

use anyhow::Result;
use candlekos::Tensor;

/// Placeholder for a parallel-scan compatible representation of the recurrence.
#[derive(Debug, Clone)]
pub struct AssocElem {
    // In a full impl, this would hold block factors for associative combination.
    _phantom: (),
}

/// Run the existing sequential forward as a scan wrapper.
pub fn scan_sequential<F>(mut step: F, x0: &Tensor, inputs: &Tensor) -> Result<(Tensor, Tensor)>
where
    F: FnMut(&Tensor, &Tensor) -> Result<(Tensor, Tensor)>,
{
    let dims = inputs.dims();
    anyhow::ensure!(dims.len() == 3, "inputs must be [B,T,D]");
    let b = dims[0];
    let t = dims[1];
    let _d = dims[2];

    // quick shape check for x0
    anyhow::ensure!(x0.dims().get(0) == Some(&b), "x0 batch mismatch");

    let mut state = x0.clone();
    let mut ys = Vec::with_capacity(t);
    for k in 0..t {
        let u_k = inputs.narrow(1, k, 1)?.squeeze(1)?; // [B,D]
        let (y_k, new_state) = step(&state, &u_k)?;
        ys.push(y_k.unsqueeze(1)?);
        state = new_state;
    }
    Ok((Tensor::cat(&ys.iter().collect::<Vec<_>>(), 1)?, state))
}

/// Inclusive prefix-sum along the provided axis.
///
/// For a tensor `x` this returns `y` with `y[..., i, ...] = sum_{k<=i} x[..., k, ...]`.
pub fn prefix_sum_along(inputs: &Tensor, axis: usize) -> Result<Tensor> {
    Ok(inputs.cumsum(axis)?)
}

/// Exclusive prefix-sum along the provided axis.
///
/// For a tensor `x` this returns `y` with `y[..., 0, ...] = 0` and
/// `y[..., i, ...] = sum_{k<i} x[..., k, ...]` for i>0.
pub fn exclusive_prefix_sum_along(inputs: &Tensor, axis: usize) -> Result<Tensor> {
    Ok(inputs.exclusive_scan(axis)?)
}

/// Convenience: inclusive prefix-sum across time for [B, T, D] tensors.
pub fn prefix_sum_time_btd(inputs: &Tensor) -> Result<Tensor> {
    let dims = inputs.dims();
    anyhow::ensure!(dims.len() == 3, "expected [B,T,D]");
    Ok(inputs.cumsum(1)?)
}

/// Convenience: exclusive prefix-sum across time for [B, T, D] tensors.
pub fn exclusive_prefix_sum_time_btd(inputs: &Tensor) -> Result<Tensor> {
    let dims = inputs.dims();
    anyhow::ensure!(dims.len() == 3, "expected [B,T,D]");
    Ok(inputs.exclusive_scan(1)?)
}

/// Parallel scan stub (feature-gated): not implemented yet.
#[cfg(feature = "parallel-scan")]
pub fn scan_parallel(_elems: &[AssocElem]) -> Result<()> {
    anyhow::bail!("parallel-scan feature is experimental and not implemented yet")
}
