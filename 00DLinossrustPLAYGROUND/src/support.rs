//! Support utilities for the D-LinOSS playground (ring buffers, simple plotting helpers, FFT naive fallback).
//! Exposed for integration/unit tests.

/// Push a value into a ring-like Vec with a soft drop (drops 1/4 of capacity when full).
pub fn push_ring(buf: &mut Vec<f32>, v: f32, cap: usize) {
    if cap == 0 {
        return;
    }
    if buf.len() >= cap {
        let drop = (cap / 4).max(1);
        buf.drain(0..drop);
    }
    buf.push(v);
}

/// Push a 2D point into a ring-like Vec with soft drop.
pub fn push_ring_phase(buf: &mut Vec<[f32; 2]>, v: [f32; 2], cap: usize) {
    if cap == 0 {
        return;
    }
    if buf.len() >= cap {
        let drop = (cap / 4).max(1);
        buf.drain(0..drop);
    }
    buf.push(v);
}

/// Compute latent energy (||x||) from a full state row w = [v, x] where both halves have equal length.
pub fn latent_energy_from_state_row(row: &[f32]) -> f32 {
    if row.is_empty() {
        return 0.0;
    }
    let half = row.len() / 2;
    if half == 0 {
        return 0.0;
    }
    let x = &row[half..];
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

/// Naive DFT magnitude (real input) returning (frequency_bin, magnitude).
pub fn compute_fft_naive(window: &[f32]) -> Vec<[f32; 2]> {
    let n = window.len();
    if n < 2 {
        return Vec::new();
    }
    let half = n / 2;
    let mut out = Vec::with_capacity(half);
    for k in 0..half {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        let kk = k as f32;
        for (t, &v) in window.iter().enumerate() {
            let th = 2.0 * std::f32::consts::PI * kk * (t as f32) / (n as f32);
            re += v * th.cos();
            im -= v * th.sin();
        }
        let mag = (re * re + im * im).sqrt() / (n as f32);
        out.push([k as f32, mag]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_soft_drop() {
        let mut v = Vec::new();
        for i in 0..300 {
            push_ring(&mut v, i as f32, 64);
        }
        assert!(v.len() <= 64);
        assert_eq!(v.last().cloned().unwrap() as i32, 299);
    }

    #[test]
    fn energy_half_norm() {
        let row: Vec<f32> = (0..10).map(|x| x as f32).collect(); // v=[0..4], x=[5..9]
        let expected =
            (5f32.powi(2) + 6f32.powi(2) + 7f32.powi(2) + 8f32.powi(2) + 9f32.powi(2)).sqrt();
        let got = latent_energy_from_state_row(&row);
        assert!(
            (got - expected).abs() < 1e-6,
            "energy mismatch got {got} expected {expected}"
        );
    }

    #[test]
    fn fft_sine_peak() {
        // sine of freq bin 3 in length 64 window
        let n = 64usize;
        let freq = 3usize;
        let mut sig = Vec::with_capacity(n);
        for t in 0..n {
            let th = 2.0 * std::f32::consts::PI * (freq as f32) * (t as f32) / (n as f32);
            sig.push(th.sin());
        }
        let spec = compute_fft_naive(&sig);
        // find max bin (skip DC maybe)
        let (max_bin, _max_mag) = spec
            .iter()
            .enumerate()
            .skip(1)
            .max_by(|a, b| a.1[1].partial_cmp(&b.1[1]).unwrap())
            .map(|(i, p)| (i, p[1]))
            .unwrap();
        assert_eq!(max_bin, freq, "expected dominant bin {freq} got {max_bin}");
    }
}
