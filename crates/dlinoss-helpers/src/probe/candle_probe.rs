#[cfg(test)]
use candle::{Device, Tensor};
#[cfg(test)]
use anyhow::Result;

#[test]
fn probe_cumsum_and_exclusive_scan() -> Result<()> {
    let dev = Device::Cpu;
    let x = Tensor::from_slice(&[1f32, 2., 3., 4.], (4,), &dev)?;
    let cs = x.cumsum(0)?; // [1,3,6,10]
    let v = cs.to_vec1::<f32>()?;
    assert_eq!(v, vec![1., 3., 6., 10.]);

    let ex = x.exclusive_scan(0)?; // [0,1,3,6]
    let v = ex.to_vec1::<f32>()?;
    assert_eq!(v, vec![0., 1., 3., 6.]);
    Ok(())
}

#[cfg(feature = "fft")]
#[test]
fn probe_fft_and_ifft_roundtrip() -> Result<()> {
    let dev = Device::Cpu;
    // simple length-8 waveform
    let x = Tensor::from_slice(&[1f32, 0., -1., 0., 1., 0., -1., 0.], (8,), &dev)?;
    // Forward real FFT (rfft), normalized -> complex spectrum (interleaved)
    let f = x.rfft(0usize, true)?;
    // Inverse real FFT (irfft), normalized -> back to real signal of same length
    let xi = f.irfft(0usize, true)?;
    // Real roundtrip within tolerance with matching shapes
    let diff = (xi - &x)?.abs()?;
    let max = diff.max_all()?.to_scalar::<f32>()?;
    assert!(max < 1e-4, "fft/ifft roundtrip max err {} too high", max);
    Ok(())
}
