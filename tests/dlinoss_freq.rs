//! Frequency response style test: drive with sine waves at different frequencies and
//! compare steady-state amplitude ratios. Provides progress prints for long loops.
use anyhow::Result;
use candlekos::{DType, Device, Tensor};
use dlinossrustcandle::{DLinOssLayer, DLinOssLayerConfig};
use std::f32::consts::PI;

fn rms(t: &Tensor) -> Result<f32> {
    Ok(((t.sqr()?).mean_all()?).to_scalar::<f32>()?.sqrt())
}

#[test]
fn sine_response_relative_amplitude() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = DLinOssLayerConfig {
        state_dim: 8,
        input_dim: 1,
        output_dim: 1,
        delta_t: 1.5e-2,
        dtype: DType::F32,
    };
    let layer = DLinOssLayer::new(cfg.clone(), &dev)?;
    let dt = cfg.delta_t as f32;
    let t_len = 600usize; // longer for better steady-state discrimination
    // Define a sweep of angular frequencies (rad/s) within Nyquist (< PI/dt).
    let max_omega = 0.6 * PI / dt; // safety margin
    let freqs: [f32; 6] = [
        max_omega * 0.02,
        max_omega * 0.05,
        max_omega * 0.10,
        max_omega * 0.18,
        max_omega * 0.30,
        max_omega * 0.45,
    ];

    let mut ratios = Vec::new();
    let mut phase_lags = Vec::new();

    for (fi, &omega) in freqs.iter().enumerate() {
        let mut u = Vec::with_capacity(t_len);
        for k in 0..t_len {
            u.push((omega * dt * (k as f32)).sin());
        }
        let u_tensor = Tensor::from_slice(&u, (1, t_len, 1), &dev)?;
        let y = layer.forward(&u_tensor, None)?; // [1,T,1]
        // Transient removal: first 30%
        let steady_start = (t_len as f32 * 0.3) as usize;
        let steady_len = t_len - steady_start;
        let u_steady = u_tensor.narrow(1, steady_start, steady_len)?;
        let y_steady = y.narrow(1, steady_start, steady_len)?;
        let u_flat = u_steady.squeeze(0)?.squeeze(1)?;
        let y_flat = y_steady.squeeze(0)?.squeeze(1)?;
        let u_rms = rms(&u_flat)?;
        let y_rms = rms(&y_flat)?;
        let ratio = if u_rms > 1e-6 { y_rms / u_rms } else { 0.0 };

        // Phase lag estimation via cross-correlation peak (circular approx)
        let u_vec = u_flat.to_vec1::<f32>()?;
        let y_vec = y_flat.to_vec1::<f32>()?;
        let n = u_vec.len();
        let mut best_shift = 0isize;
        let mut best_corr = f32::MIN;
        // Limit search window to a fraction of a period (improves speed)
        let approx_period = (2.0 * PI / omega / dt).round() as isize;
        let search_radius = (approx_period.min((n as isize) / 4)).max(4);
        for shift in -search_radius..=search_radius {
            let mut acc = 0f32;
            for i in 0..n {
                let j = i as isize + shift;
                if j >= 0 && (j as usize) < n {
                    acc += u_vec[i] * y_vec[j as usize];
                }
            }
            if acc > best_corr {
                best_corr = acc;
                best_shift = shift;
            }
        }
        let phase_lag = (best_shift as f32) * dt * omega; // radians (approx)
        ratios.push(ratio);
        phase_lags.push(phase_lag);
        println!("[freq-test] fi={fi} omega={omega:.3} ratio={ratio:.4} phase_lag={phase_lag:.3}");
        assert!(ratio.is_finite() && ratio >= 0.0);
    }

    // Basic monotonic damping expectation: higher freq should not have *larger* gain than very low freq beyond tolerance.
    let low = ratios.first().cloned().unwrap_or(0.0);
    let high = ratios.last().cloned().unwrap_or(0.0);
    // Allow some looseness (system may be relatively flat); just ensure not explosive high-frequency amplification.
    assert!(
        high <= low * 3.0 + 1e-3,
        "High-frequency gain unexpectedly large: low={low} high={high}"
    );

    // Phase lag: unwrap to mitigate 2π wrapping then apply a very loose smoothness constraint.
    let mut unwrapped = Vec::with_capacity(phase_lags.len());
    let mut accum = 0f32;
    let mut prev = phase_lags[0];
    unwrapped.push(prev);
    for &ph in &phase_lags[1..] {
        let mut d = ph - prev;
        // wrap into [-PI, PI]
        let two_pi = 2.0 * PI;
        while d > PI {
            d -= two_pi;
        }
        while d < -PI {
            d += two_pi;
        }
        accum += d;
        unwrapped.push(unwrapped[0] + accum);
        prev = ph;
    }
    // Count large reversals (second derivative spikes)
    let mut spikes = 0;
    for w in unwrapped.windows(3) {
        let d1 = w[1] - w[0];
        let d2 = w[2] - w[1];
        if (d2 - d1).abs() > 4.0 {
            // allow big tolerance, just block wild swings
            spikes += 1;
        }
    }
    assert!(
        spikes <= 2,
        "Phase lag highly irregular: raw={:?} unwrapped={:?}",
        phase_lags,
        unwrapped
    );

    Ok(())
}

#[test]
fn mixed_multisine_energy_distribution() -> Result<()> {
    // Drive the layer with a sum of several sines and confirm output energy remains bounded and roughly proportional.
    let dev = Device::Cpu;
    let cfg = DLinOssLayerConfig {
        state_dim: 10,
        input_dim: 1,
        output_dim: 1,
        delta_t: 2.0e-2,
        dtype: DType::F32,
    };
    let layer = DLinOssLayer::new(cfg.clone(), &dev)?;
    let dt = cfg.delta_t as f32;
    let t_len = 800usize;
    let base = 4.0f32;
    let comps = [1.0, 2.3, 3.7, 5.1];
    let mut u = Vec::with_capacity(t_len);
    for k in 0..t_len {
        let t = k as f32 * dt;
        let mut val = 0.0f32;
        for (ci, c) in comps.iter().enumerate() {
            val += (base * c * t).sin() * (1.0 / (ci as f32 + 1.0));
        }
        u.push(val);
    }
    let u_tensor = Tensor::from_slice(&u, (1, t_len, 1), &dev)?;
    let y = layer.forward(&u_tensor, None)?;
    let u_rms = rms(&u_tensor.squeeze(0)?.squeeze(1)?)?;
    let y_rms = rms(&y.squeeze(0)?.squeeze(1)?)?;
    println!("[freq-test-mixed] u_rms={u_rms:.4} y_rms={y_rms:.4}");
    assert!(u_rms > 1e-3 && y_rms.is_finite() && y_rms < u_rms * 5.0 + 1e-3);
    Ok(())
}
