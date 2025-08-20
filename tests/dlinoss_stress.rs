//! Large / stress test gated by feature `large-tests` and env var DLINOSS_LARGE=1.
#![cfg(feature = "large-tests")]
use anyhow::{Result, bail};
use candlekos::{DType, Device, Tensor};
use dlinossrustcandle::{DLinOssLayer, DLinOssLayerConfig};
use std::time::Instant;

#[test]
fn large_sequence_throughput() -> Result<()> {
    if std::env::var("DLINOSS_LARGE").ok().as_deref() != Some("1") {
        eprintln!("[stress] skipped (set DLINOSS_LARGE=1 to run)");
        return Ok(());
    }
    let dev = Device::Cpu;
    let cfg = DLinOssLayerConfig {
        state_dim: 64,
        input_dim: 8,
        output_dim: 8,
        delta_t: 1e-2,
        dtype: DType::F32,
    };
    let layer = DLinOssLayer::new(cfg.clone(), &dev)?;
    let batch = 8usize;
    let t_len = 20_000usize; // moderate; adjust if slow
    let x = Tensor::rand(0.0f64, 1.0, (batch, t_len, cfg.input_dim), &dev)?.to_dtype(cfg.dtype)?;
    let start = Instant::now();
    let y = layer.forward(&x, None)?; // [B,T,Out]
    let dur = start.elapsed();
    // Sanity checks
    assert_eq!(y.dims(), &[batch, t_len, cfg.output_dim]);
    let max_abs = y.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(max_abs.is_finite());
    println!(
        "[stress] completed batch={batch} T={t_len} in {:?} (max_abs={:.3})",
        dur, max_abs
    );
    Ok(())
}

#[test]
fn large_nd_spatial_throughput() -> Result<()> {
    if std::env::var("DLINOSS_LARGE").ok().as_deref() != Some("1") {
        eprintln!("[stress-nd] skipped (set DLINOSS_LARGE=1 to run)");
        return Ok(());
    }
    let dev = Device::Cpu;
    let cfg = DLinOssLayerConfig {
        state_dim: 48,
        input_dim: 2,
        output_dim: 2,
        delta_t: 5e-3,
        dtype: DType::F32,
    };
    let layer = DLinOssLayer::new(cfg.clone(), &dev)?;
    // 4D spatial volume e.g. (H,W,D) collapsed via forward_nd
    let (b, t, h, w, d, p) = (2usize, 1200usize, 16usize, 12usize, 8usize, cfg.input_dim);
    let total_spatial = h * w * d;
    let x = Tensor::rand(0.0f64, 1.0, (b, t, h, w, d, p), &dev)?.to_dtype(cfg.dtype)?;
    let start = Instant::now();
    let y = layer.forward_nd(&x, None)?; // [b,t,h,w,d,q]
    let elapsed = start.elapsed();
    assert_eq!(y.dims(), &[b, t, h, w, d, cfg.output_dim]);
    let samples = (b * total_spatial * t) as f64;
    let secs = elapsed.as_secs_f64().max(1e-9);
    let rate = samples / secs;
    println!(
        "[stress-nd] processed {} samples in {:.3}s => {:.1} samples/s",
        samples, secs, rate
    );
    // Basic sanity bound: ensure rate is positive and output finite.
    let max_abs = y.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(max_abs.is_finite());
    Ok(())
}
