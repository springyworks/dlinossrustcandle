use anyhow::Result;
use candlekos::Tensor;
use dlinossrustcandle::TensorScanExt;

#[test]
fn test_prefix_sum_along_1d() -> Result<()> {
    let dev = candlekos::Device::Cpu;
    let x = Tensor::new(&[1f32, 2., 3., 4.], &dev)?;
    let y = x.prefix_sum_along(0)?;
    let v = y.to_vec1::<f32>()?;
    assert_eq!(v, vec![1., 3., 6., 10.]);
    Ok(())
}

#[test]
fn test_exclusive_prefix_sum_along_1d() -> Result<()> {
    let dev = candlekos::Device::Cpu;
    let x = Tensor::new(&[1f32, 2., 3., 4.], &dev)?;
    let y = x.exclusive_prefix_sum_along(0)?;
    let v = y.to_vec1::<f32>()?;
    assert_eq!(v, vec![0., 1., 3., 6.]);
    Ok(())
}

#[test]
fn test_prefix_sum_time_btd() -> Result<()> {
    let dev = candlekos::Device::Cpu;
    // [B=1, T=4, D=2]
    let data: [f32; 8] = [1., 2., 3., 4., 5., 6., 7., 8.];
    let x = Tensor::from_slice(&data, (1, 4, 2), &dev)?;
    let y = x.prefix_sum_time_btd()?;
    let v = y.to_vec3::<f32>()?;
    assert_eq!(
        v,
        vec![vec![
            vec![1., 2.],
            vec![4., 6.],
            vec![9., 12.],
            vec![16., 20.]
        ],]
    );
    Ok(())
}

#[test]
fn test_exclusive_prefix_sum_time_btd() -> Result<()> {
    let dev = candlekos::Device::Cpu;
    // [B=1, T=4, D=2]
    let data: [f32; 8] = [1., 2., 3., 4., 5., 6., 7., 8.];
    let x = Tensor::from_slice(&data, (1, 4, 2), &dev)?;
    let y = x.exclusive_prefix_sum_time_btd()?;
    let v = y.to_vec3::<f32>()?;
    assert_eq!(
        v,
        vec![vec![
            vec![0., 0.],
            vec![1., 2.],
            vec![4., 6.],
            vec![9., 12.]
        ],]
    );
    Ok(())
}
