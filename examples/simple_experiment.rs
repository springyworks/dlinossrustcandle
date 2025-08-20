use anyhow::Result;
use candlekos::{DType, Device, Tensor};
use dlinossrustcandle::{DLinOssLayer, DLinOssLayerConfig};

fn make_signals(t: usize) -> Result<Vec<(&'static str, Tensor)>> {
    let dev = Device::Cpu;
    let ones = Tensor::ones((t, 1), DType::F32, &dev)?;
    let time = Tensor::arange(0f32, t as f32, &dev)?;
    let sine = time.affine(0.2, 0.0)?.sin()?.reshape((t, 1))?;
    let square = sine.sign()?;
    let decay = time.affine(-0.05, 0.0)?.exp()?.reshape((t, 1))?;
    let head = Tensor::new(1f32, &dev)?.reshape((1, 1))?;
    let tail = Tensor::zeros((t - 1, 1), DType::F32, &dev)?;
    let impulse = Tensor::cat(&[&head, &tail], 0)?;
    let ramp = time.affine(1.0 / (t as f64 - 1.0), 0.0)?.reshape((t, 1))?;
    Ok(vec![
        ("step", ones),
        ("sine", sine),
        ("square", square),
        ("exp_decay", decay),
        ("impulse", impulse),
        ("ramp", ramp),
    ])
}

fn main() -> Result<()> {
    println!("\n=== Simple D-LinOSS Experiment (Rust) ===");
    let dev = Device::Cpu;
    let cfg = DLinOssLayerConfig {
        state_dim: 16,
        input_dim: 1,
        output_dim: 1,
        delta_t: 1e-2,
        dtype: DType::F32,
    };
    let layer = DLinOssLayer::new(cfg, &dev)?;

    let t = 100usize;
    let signals = make_signals(t)?;
    for (name, s) in signals {
        let x = s.reshape((1usize, t, 1usize))?;
        let y = layer.forward(&x, None)?; // [1,T,1]
        let y = y.squeeze(0)?.squeeze(1)?;
        let ymin: f32 = y.min_all()?.to_scalar::<f32>()?;
        let ymax: f32 = y.max_all()?.to_scalar::<f32>()?;
        println!("{name:>10}: out range [{ymin:.3},{ymax:.3}]");
    }
    Ok(())
}
