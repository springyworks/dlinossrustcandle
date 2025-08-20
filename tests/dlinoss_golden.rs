//! Golden snapshot reproducibility test using deterministic constructor.
use anyhow::Result;
use candlekos::{DType, Device, Tensor};
use dlinossrustcandle::{DLinOssLayer, DLinOssLayerConfig};

fn checksum(t: &Tensor) -> Result<(f64, f64)> {
    let sum = t.sum_all()?.to_scalar::<f32>()? as f64;
    let mean = t.mean_all()?.to_scalar::<f32>()? as f64;
    Ok((sum, mean))
}

#[test]
fn golden_forward_checksum() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = DLinOssLayerConfig {
        state_dim: 7,
        input_dim: 2,
        output_dim: 3,
        delta_t: 1e-2,
        dtype: DType::F32,
    };
    let layer = DLinOssLayer::deterministic(cfg.clone(), &dev)?;
    // Mixed input: impulses + sine tail.
    let t_len = 40usize;
    let mut u = vec![0f32; t_len * cfg.input_dim];
    // Impulses on channel 0 at k=0,5,10; channel1 sinusoid thereafter.
    for k in [0usize, 5, 10] {
        u[k * cfg.input_dim] = 1.0;
    }
    for k in 0..t_len {
        u[k * cfg.input_dim + 1] = (0.3 * k as f32).sin();
    }
    let u_tensor = Tensor::from_slice(&u, (1, t_len, cfg.input_dim), &dev)?;
    let y = layer.forward(&u_tensor, None)?; // [1,T,q]
    let (sum, mean) = checksum(&y)?;
    // Initial golden values; update only intentionally.
    // Locked baseline produced by deterministic params + input recipe (commit baseline)
    const GOLD_SUM: f64 = 0.04721886292099953;
    const GOLD_MEAN: f64 = 0.00039349053986370564;
    let ds = (sum - GOLD_SUM).abs();
    let dm = (mean - GOLD_MEAN).abs();
    assert!(
        ds < 1e-5 && dm < 1e-6,
        "Golden drift sum diff={ds} mean diff={dm} (got sum={sum} mean={mean})"
    );
    Ok(())
}
