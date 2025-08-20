//! Analytical and foundational tests for DLinOssLayer.
//! These tests validate discrete transition construction, impulse/step responses,
//! and provide progress output for long loops to avoid the appearance of hanging.

use anyhow::Result;
use candlekos::{DType, Device, Tensor};
use dlinossrustcandle::{DLinOssLayer, DLinOssLayerConfig};

// Helper to build deterministic tensors from slices.
fn t1(data: &[f32], dev: &Device) -> Result<Tensor> {
    Ok(Tensor::from_slice(data, data.len(), dev)?)
}
fn t2(data: &[f32], shape: (usize, usize), dev: &Device) -> Result<Tensor> {
    Ok(Tensor::from_slice(data, shape, dev)?)
}

#[test]
fn build_m_f_matches_manual_step() -> Result<()> {
    let dev = Device::Cpu;
    let m = 3usize;
    let p = 2usize;
    let q = 2usize;
    let dt = 1e-2;
    // Choose a,g inside an expected stable band (positive g, moderate a) so clamping is no-op.
    let a = t1(&[4.0, 3.5, 2.0], &dev)?; // arbitrary within band
    let g = t1(&[0.2, 0.1, 0.05], &dev)?; // damping
    let b = t2(&[0.5, 0.1, 0.2, 0.3, 0.05, 0.4], (m, p), &dev)?;
    let c = t2(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0], (q, m), &dev)?; // observe first & last state components

    let layer = DLinOssLayer::from_components(a, g, b, c, dt)?;
    let (m_mat, f_mat) = layer.build_m_f()?; // [2m,2m], [2m,p]

    // Manual single step from zero state with simple unit input
    let w0 = Tensor::zeros((1, 2 * m), DType::F32, &dev)?; // [batch=1,2m]
    let u = Tensor::from_slice(&[1.0f32, 0.0], (1, p), &dev)?; // [1,p]
    // w1 = w0 * M^T + u * F^T
    let w1_a = w0.matmul(&m_mat.transpose(0, 1)?)?;
    let w1_b = u.matmul(&f_mat.transpose(0, 1)?)?;
    let w1_manual = w1_a.add(&w1_b)?; // [1,2m]

    // Run forward with length=1 sequence containing u
    let seq = u.unsqueeze(1)?; // [1,1,p]
    let y = layer.forward(&seq, None)?; // [1,1,q]
    // Extract internal w1 by re-simulating using build_m_f (already done) -> compare some invariants.

    // Reconstruct x_k = second half of w1
    let x1_manual = w1_manual.narrow(1, m, m)?; // [1,m]
    // Output y should be x1_manual * C^T (done internally) – verify consistency.
    let c_t = layer.c.transpose(0, 1)?; // [m,q]
    let y_expected = x1_manual.matmul(&c_t)?; // [1,q]

    let diff = (y.squeeze(1)? - &y_expected)?.abs()?;
    let max_err = diff.max_all()?.to_scalar::<f32>()?;
    assert!(
        max_err < 1e-5,
        "M,F reconstruction mismatch max_err={max_err}"
    );

    Ok(())
}

#[test]
fn impulse_and_step_response_consistency() -> Result<()> {
    let dev = Device::Cpu;
    let m = 2usize;
    let p = 1usize;
    let q = 1usize;
    let dt = 5e-2;
    // Pick modest parameters
    let a = t1(&[2.0, 1.5], &dev)?;
    let g = t1(&[0.3, 0.4], &dev)?;
    let b = t2(&[0.8, 0.2], (m, p), &dev)?; // column vector
    let c = t2(&[1.0, 0.0], (q, m), &dev)?; // observe first component
    let layer = DLinOssLayer::from_components(a, g, b, c, dt)?;
    let (m_mat, f_mat) = layer.build_m_f()?;

    let t_len = 64usize;
    // Impulse input: u0=1, rest 0
    let mut u_data = vec![0f32; t_len];
    u_data[0] = 1.0;
    let u_tensor = Tensor::from_slice(&u_data, (1, t_len, p), &dev)?; // [1,T,1]
    let y_imp = layer.forward(&u_tensor, None)?; // [1,T,1]

    // Step input: all 1
    let step_data = vec![1f32; t_len];
    let step_tensor = Tensor::from_slice(&step_data, (1, t_len, p), &dev)?;
    let y_step = layer.forward(&step_tensor, None)?; // [1,T,1]

    // Recreate impulse by unrolling recurrence for cross-check
    let mut w = Tensor::zeros((1, 2 * m), DType::F32, &dev)?;
    let mut ref_imp: Vec<f32> = Vec::with_capacity(t_len);
    for k in 0..t_len {
        if k == 0 {
            let uk = Tensor::from_slice(&[1f32], (1, p), &dev)?;
            w = w
                .matmul(&m_mat.transpose(0, 1)?)?
                .add(&uk.matmul(&f_mat.transpose(0, 1)?)?)?;
        } else {
            // zero input
            w = w.matmul(&m_mat.transpose(0, 1)?)?;
        }
        let xk = w.narrow(1, m, m)?;
        let yk = xk.matmul(&layer.c.transpose(0, 1)?)?; // [1,1]
        ref_imp.push(yk.squeeze(0)?.squeeze(0)?.to_scalar::<f32>()?);
    }

    // Compare impulse sequence (tolerance)
    let imp_vec = y_imp.squeeze(0)?.squeeze(1)?.to_vec1::<f32>()?; // length T
    let mut max_rel = 0f32;
    let eps = 1e-6;
    for (a, b) in imp_vec.iter().zip(ref_imp.iter()) {
        let rel = (a - b).abs() / (b.abs() + eps);
        if rel > max_rel {
            max_rel = rel;
        }
    }
    assert!(max_rel < 1e-4, "impulse mismatch max_rel={max_rel}");

    // Step response should be cumulative sum of impulse response
    let step_vec = y_step.squeeze(0)?.squeeze(1)?.to_vec1::<f32>()?;
    let mut cumsum = 0f32;
    let mut max_step_rel = 0f32;
    for k in 0..t_len {
        cumsum += imp_vec[k];
        let rel = (step_vec[k] - cumsum).abs() / (cumsum.abs() + eps);
        if rel > max_step_rel {
            max_step_rel = rel;
        }
    }
    assert!(max_step_rel < 1e-4, "step mismatch max_rel={max_step_rel}");

    Ok(())
}

#[test]
fn linearity_properties() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = DLinOssLayerConfig {
        state_dim: 4,
        input_dim: 3,
        output_dim: 2,
        delta_t: 1e-2,
        dtype: DType::F32,
    };
    let layer = DLinOssLayer::new(cfg, &dev)?;
    let b = 2usize;
    let t = 16usize;
    let p = 3usize;
    let x1 = Tensor::rand(0.0f64, 1.0, (b, t, p), &dev)?.to_dtype(DType::F32)?;
    let x2 = Tensor::rand(0.0f64, 1.0, (b, t, p), &dev)?.to_dtype(DType::F32)?;
    let y1 = layer.forward(&x1, None)?;
    let y2 = layer.forward(&x2, None)?;
    let x_sum = x1.add(&x2)?;
    let y_sum_direct = layer.forward(&x_sum, None)?;
    let y_sum_linear = y1.add(&y2)?;
    let diff = (y_sum_direct - &y_sum_linear)?.abs()?;
    let max = diff.max_all()?.to_scalar::<f32>()?;
    assert!(max < 5e-5, "linearity add violated max_err={max}");

    // Scalar homogeneity
    let alpha = 0.37f32;
    let x_scaled = x1.affine(alpha as f64, 0.0)?; // alpha * x1
    let y_scaled = layer.forward(&x_scaled, None)?;
    let y1_scaled = y1.affine(alpha as f64, 0.0)?;
    let diff2 = (y_scaled - &y1_scaled)?.abs()?;
    let max2 = diff2.max_all()?.to_scalar::<f32>()?;
    assert!(max2 < 5e-5, "linearity scale violated max_err={max2}");
    Ok(())
}
