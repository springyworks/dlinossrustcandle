use anyhow::Result;
use candlekos::{DType, Device, Tensor};
use dlinossrustcandle::{DLinOssLayer, DLinOssLayerConfig};

#[test]
fn impulse_response_decays() -> Result<()> {
    let device = Device::Cpu;
    let cfg = DLinOssLayerConfig {
        state_dim: 8,
        input_dim: 1,
        output_dim: 1,
        delta_t: 1e-2,
        dtype: DType::F32,
    };
    let mut layer = DLinOssLayer::new(cfg, &device)?;

    // Deterministic, stable parameters to avoid random non-monotonic transients
    let m = 8usize;
    let dt = 1e-2f64;
    // G = 0.5 for all states (positive damping)
    let g = Tensor::ones(m, DType::F32, &device)?.affine(0.0, 0.5)?;
    // Compute IMEX band and pick mid-band A
    let s = g.affine(dt, 1.0)?; // 1 + dt*G
    let sqrt_s = s.sqrt()?;
    let two_plus_dtg = s.affine(1.0, 1.0)?;
    let two_sqrt = sqrt_s.affine(2.0, 0.0)?;
    let inv_dt2 = 1.0 / (dt * dt);
    let a_low = two_plus_dtg.sub(&two_sqrt)?.affine(inv_dt2, 0.0)?;
    let a_high = two_plus_dtg.add(&two_sqrt)?.affine(inv_dt2, 0.0)?;
    let a_mid = a_low.add(&a_high)?.affine(0.5, 0.0)?;
    // B = small ones, C selects first state
    let b = Tensor::ones((m, 1), DType::F32, &device)?.affine(0.1, 0.0)?;
    let c_head = Tensor::new(1f32, &device)?.reshape((1, 1))?;
    let c_tail = Tensor::zeros((1, m - 1), DType::F32, &device)?;
    let c_row = Tensor::cat(&[&c_head, &c_tail], 1)?; // [1,m]

    layer.g = g;
    layer.a = a_mid;
    layer.b = b;
    layer.c = c_row;

    let t = 64usize;
    // Build impulse input [1, T, 1]
    let head = Tensor::new(1f32, &device)?.reshape((1, 1, 1))?;
    let tail = Tensor::zeros((1, t - 1, 1), DType::F32, &device)?;
    let x = Tensor::cat(&[&head, &tail], 1)?;

    let y = layer.forward(&x, None)?; // [1,T,1]
    let y = y.squeeze(0)?.squeeze(1)?; // [T]
    let first_half = y.narrow(0, 0, t / 2)?;
    let last_half = y.narrow(0, t / 2, t - t / 2)?;
    // Compare energy (L2) to allow oscillations while ensuring net decay.
    let e_first: f32 = (first_half.sqr()?.sum_all()?).to_scalar::<f32>()?;
    let e_last: f32 = (last_half.sqr()?.sum_all()?).to_scalar::<f32>()?;
    assert!(
        e_last <= e_first,
        "impulse response energy should decay: last {e_last} > first {e_first}"
    );
    Ok(())
}
