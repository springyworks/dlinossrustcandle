use anyhow::Result;
use candlekos::{DType, Device, Tensor};
use dlinossrustcandle::{DLinOssLayer, DLinOssLayerConfig};

fn make_signals(seq: usize) -> Result<Vec<(&'static str, Tensor)>> {
    let dev = Device::Cpu;
    let ones = Tensor::ones((seq, 1), DType::F32, &dev)?; // step
    let t = Tensor::arange(0f32, seq as f32, &dev)?; // time
    let t = t.affine(0.2, 0.0)?; // scale
    let sine = t.sin()?.reshape((seq, 1))?;
    // impulse at t=0 via concatenation
    let head = Tensor::new(1f32, &dev)?.reshape((1, 1))?;
    let tail = Tensor::zeros((seq - 1, 1), DType::F32, &dev)?;
    let impulse = Tensor::cat(&[&head, &tail], 0)?;
    Ok(vec![("step", ones), ("sine", sine), ("impulse", impulse)])
}

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

    let seq_len = 64usize;
    let signals = make_signals(seq_len)?;
    for (name, s) in signals {
        let x = s.reshape((1usize, seq_len, 1usize))?; // [B=1,T,In=1]
        let y = layer.forward(&x, None)?; // [1,T,1]
        let y2 = y.squeeze(0)?.squeeze(1)?; // [T]
        let y_min: f32 = y2.min_all()?.to_scalar::<f32>()?;
        let y_max: f32 = y2.max_all()?.to_scalar::<f32>()?;
        println!("{name} -> out range [{y_min:.3},{y_max:.3}]");
    }
    Ok(())
}
