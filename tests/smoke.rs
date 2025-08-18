use anyhow::Result;
use candle::{DType, Device, Tensor};
use dlinossrustcandle::{DLinOssLayer, DLinOssLayerConfig};

#[test]
fn forward_shapes() -> Result<()> {
    let device = Device::Cpu;
    let cfg = DLinOssLayerConfig {
        state_dim: 8,
        input_dim: 5,
        output_dim: 3,
        delta_t: 1e-2,
        dtype: DType::F32,
    };
    let layer = DLinOssLayer::new(cfg, &device)?;

    let batch = 2usize;
    let seq = 4usize;
    let input = Tensor::rand(0.0f64, 1.0, (batch, seq, 5), &device)?.to_dtype(DType::F32)?;
    let out = layer.forward(&input, None)?;
    assert_eq!(out.dims(), &[batch, seq, 3]);
    Ok(())
}
