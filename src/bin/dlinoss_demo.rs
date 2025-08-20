//! Minimal D-LinOSS demo binary
//!
//! Purpose: provide an easy-to-build executable (CPU by default), including on Windows.
//! Build: cargo build --release --bin dlinoss_demo

use anyhow::Result;
use candlekos::{DType, Device, Tensor};
use dlinossrustcandle::{DLinOssLayer, DLinOssLayerConfig};

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = DLinOssLayerConfig {
        state_dim: 8,
        input_dim: 1,
        output_dim: 1,
        delta_t: 1e-2,
        dtype: DType::F32,
    };
    let layer = DLinOssLayer::new(cfg, &dev)?;

    // Tiny impulse signal [B=1, T=16, In=1]
    let t = 16usize;
    let head = Tensor::new(1f32, &dev)?.reshape((1, 1, 1))?;
    let tail = Tensor::zeros((1, t - 1, 1), DType::F32, &dev)?;
    let x = Tensor::cat(&[&head, &tail], 1)?;

    let y = layer.forward(&x, None)?; // [1,T,1]
    let y = y.squeeze(0)?.squeeze(1)?; // [T]
    let v = y.to_vec1::<f32>()?;
    println!("D-LinOSS demo output (first 8 of {t}):");
    for (i, val) in v.iter().take(8).enumerate() {
        println!("t={i:02}: {val:.4}");
    }
    Ok(())
}
