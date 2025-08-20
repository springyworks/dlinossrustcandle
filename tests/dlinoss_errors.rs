//! Error mode tests: ensure clear failures on dimension mismatch and invalid parameters.
use anyhow::Result;
use candlekos::{DType, Device, Tensor};
use dlinossrustcandle::{DLinOssLayer, DLinOssLayerConfig};

#[test]
fn forward_dimension_mismatch() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = DLinOssLayerConfig {
        state_dim: 4,
        input_dim: 3,
        output_dim: 2,
        delta_t: 1e-2,
        dtype: DType::F32,
    };
    let layer = DLinOssLayer::new(cfg.clone(), &dev)?;
    // Provide tensor with wrong last dim (2 instead of 3)
    let bad = Tensor::zeros((1, 5, 2), cfg.dtype, &dev)?;
    let err = layer.forward(&bad, None).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("input_dim mismatch"));
    Ok(())
}

#[test]
fn deterministic_invalid_dims() {
    let dev = Device::Cpu;
    let cfg = DLinOssLayerConfig {
        state_dim: 0,
        input_dim: 1,
        output_dim: 1,
        delta_t: 1e-2,
        dtype: DType::F32,
    };
    let err = DLinOssLayer::deterministic(cfg, &dev).unwrap_err();
    assert!(format!("{err}").contains("dimensions must be > 0"));
}
