use anyhow::Result;
use candlekos::{DType, Device, Tensor};
use dlinossrustcandle::{DLinOssLayer, DLinOssLayerConfig};

// Power iteration to approximate spectral radius of a square matrix.
fn power_iteration(mat: &Tensor, iters: usize) -> Result<f32> {
    let (n, n2) = mat.dims2()?;
    assert_eq!(n, n2, "matrix square");
    let device = mat.device();
    let mut v = Tensor::rand(0.0f64, 1.0, n, device)?.to_dtype(mat.dtype())?; // [n]
    for _ in 0..iters {
        v = mat.matmul(&v.unsqueeze(1)?)?.squeeze(1)?; // [n]
        let norm = (v.sqr()?.sum_all()?).to_scalar::<f32>()?.sqrt();
        if norm > 0.0 {
            v = v.affine(1.0 / norm as f64, 0.0)?;
        }
    }
    // Rayleigh quotient estimate
    let mv = mat.matmul(&v.unsqueeze(1)?)?.squeeze(1)?; // [n]
    let num = (&v * &mv)?.sum_all()?.to_scalar::<f32>()?;
    let denom = v.sqr()?.sum_all()?.to_scalar::<f32>()?;
    Ok((num / denom).abs())
}

#[test]
fn spectral_radius_reasonable_bound() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = DLinOssLayerConfig {
        state_dim: 12,
        input_dim: 4,
        output_dim: 4,
        delta_t: 1e-2,
        dtype: DType::F32,
    };
    let layer = DLinOssLayer::new(cfg, &dev)?;
    let (m_mat, _f) = layer.build_m_f()?; // [2m,2m]
    let rho = power_iteration(&m_mat, 40)?;
    // Generous bound; may refine after empirical characterization.
    assert!(rho < 3.0, "spectral radius too large: {rho}");
    Ok(())
}
