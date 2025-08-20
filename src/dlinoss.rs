use anyhow::Result;
use candle::{DType, Device, Tensor};

/// Configuration for the D-LinOSS layer
#[derive(Debug, Clone)]
pub struct DLinOssLayerConfig {
    /// State dimension m (per oscillator)
    pub state_dim: usize,
    /// Input dimension p
    pub input_dim: usize,
    /// Output dimension q
    pub output_dim: usize,
    /// Discretization step
    pub delta_t: f64,
    /// Data type for parameters and tensors
    pub dtype: DType,
}

impl Default for DLinOssLayerConfig {
    fn default() -> Self {
        Self {
            state_dim: 64,
            input_dim: 64,
            output_dim: 64,
            delta_t: 1e-2,
            dtype: DType::F32,
        }
    }
}

/// Damped Linear Oscillatory State-Space Layer implemented with Candle tensors.
///
/// Continuous-time (paper):
///   x''(t) = -A x(t) - G x'(t) + B u(t)
///   y(t) = C x(t)
/// Using an IMEX discretization as in the burn implementation.
#[derive(Debug, Clone)]
pub struct DLinOssLayer {
    pub a: Tensor, // [m]
    pub g: Tensor, // [m]
    pub b: Tensor, // [m, p]
    pub c: Tensor, // [q, m]
    pub delta_t: f64,
    state_dim: usize,
    input_dim: usize,
    output_dim: usize,
    device: Device,
    dtype: DType,
}

impl DLinOssLayer {
    pub fn new(cfg: DLinOssLayerConfig, device: &Device) -> Result<Self> {
        let m = cfg.state_dim;
        let p = cfg.input_dim;
        let q = cfg.output_dim;

        // Simple random inits (TODO: better init as per paper Section 3.1)
        let a = Tensor::rand(0.0f64, 1.0, m, device)?.to_dtype(cfg.dtype)?;
        let g = Tensor::rand(0.0f64, 1.0, m, device)?.to_dtype(cfg.dtype)?;
        let b = Tensor::rand(0.0f64, 1.0, (m, p), device)?.to_dtype(cfg.dtype)?;
        let c = Tensor::rand(0.0f64, 1.0, (q, m), device)?.to_dtype(cfg.dtype)?;

        Ok(Self {
            a,
            g,
            b,
            c,
            delta_t: cfg.delta_t,
            state_dim: m,
            input_dim: p,
            output_dim: q,
            device: device.clone(),
            dtype: cfg.dtype,
        })
    }

    /// Forward pass.
    ///
    /// input: [batch, seq_len, p]
    /// init_state: optional [batch, 2m]
    /// returns: [batch, seq_len, q]
    pub fn forward(&self, input: &Tensor, init_state: Option<&Tensor>) -> Result<Tensor> {
        let dims = input.dims();
        anyhow::ensure!(dims.len() == 3, "input must be [batch, seq, p]");
        let batch = dims[0];
        let seq_len = dims[1];
        let p = dims[2];
        anyhow::ensure!(
            p == self.input_dim,
            "input_dim mismatch: {} != {}",
            p,
            self.input_dim
        );
        // Touch output_dim and add a debug-only shape check to keep clippy happy and validate C.
        #[cfg(debug_assertions)]
        {
            let (qc, mc) = self.c.dims2()?;
            debug_assert_eq!(qc, self.output_dim, "C has wrong q dimension");
            debug_assert_eq!(mc, self.state_dim, "C has wrong m dimension");
        }

        let m = self.state_dim;
        let w_dim = 2 * m;

        let w0 = match init_state {
            Some(w) => w.clone(),
            None => Tensor::zeros((batch, w_dim), self.dtype, &self.device)?,
        };

        // parameters processed with stability constraints (IMEX damped band)
        let dt = self.delta_t; // f64
        // Ensure G >= eps
        let g_diag = self.g.abs()?.affine(1.0, 1e-4)?; // g = |g| + eps (f64)
        // s = 1 + dt * G
        let s_diag = g_diag.affine(dt, 1.0)?; // [m]
        let sqrt_s = s_diag.sqrt()?; // [m]
        let two_plus_dtg = s_diag.affine(1.0, 1.0)?; // (2 + dt*G)
        let two_sqrt = sqrt_s.affine(2.0, 0.0)?; // 2 * sqrt(1 + dt*G)
        let inv_dt2 = 1.0 / (dt * dt);
        let a_low = two_plus_dtg.sub(&two_sqrt)?.affine(inv_dt2, 0.0)?; // [m]
        let a_high = two_plus_dtg.add(&two_sqrt)?.affine(inv_dt2, 0.0)?; // [m]
        // Clamp a into [a_low, a_high] using relu-trick
        let a_minus_low = self.a.sub(&a_low)?;
        let a_minus_high = self.a.sub(&a_high)?;
        let a_diag = a_low
            .add(&a_minus_low.relu()?)?
            .sub(&a_minus_high.relu()?)?; // [m]
        let s_inv = s_diag.recip()?; // diag
        let s_inv_a = (&s_inv * &a_diag)?;

        // Build blocks using diagonal broadcast.
        // No Tensor::eye in candle; construct diagonal via elementwise with identity-like trick.
        // We'll create diagonal matrices by multiplying a column vector by a one-hot row mask.
        // Simpler: create zeros (m,m) then scatter_set diagonal. Use from_vec helper for indexes.
        // But candle lacks direct diag_set; emulate using broadcasting.
        // Use: diag(v) = zeros(m,m).add( v.reshape([m,1]) * eye ), but we still need eye.
        // Workaround: use batch element-wise by expanding along dims and masking with eq of arange.
        let ar = Tensor::arange(0u32, m as u32, &self.device)?; // [m]
        let row = ar.unsqueeze(1)?; // [m,1]
        let col = ar.unsqueeze(0)?; // [1,m]
        let eye = row.broadcast_eq(&col)?; // [m,m] bool (u8)
        let eye = eye.to_dtype(self.dtype)?;

        let m11 = eye.broadcast_mul(&s_inv.unsqueeze(0)?)?; // [m,m]
        let m12 = eye.broadcast_mul(&s_inv_a.affine(-dt, 0.0)?.unsqueeze(0)?)?; // [m,m]
        let m21 = eye.broadcast_mul(&s_inv.affine(dt, 0.0)?.unsqueeze(0)?)?; // [m,m]
        let s_inv_a_dt2 = s_inv_a.affine(dt * dt, 0.0)?;
        let diag_dt2 = eye.broadcast_mul(&s_inv_a_dt2.unsqueeze(0)?)?;
        let m22 = eye.sub(&diag_dt2)?; // [m,m]

        let m_top = Tensor::cat(&[&m11, &m12], 1)?; // [m,2m]
        let m_bottom = Tensor::cat(&[&m21, &m22], 1)?; // [m,2m]
        let m_mat = Tensor::cat(&[&m_top, &m_bottom], 0)?; // [2m,2m]

        // F = [[dt * S_inv * B], [dt^2 * S_inv * B]]
        let s_inv_col = s_inv.unsqueeze(1)?; // [m,1]
        let s_inv_b = s_inv_col.broadcast_mul(&self.b)?; // [m,p]
        let f_top = s_inv_b.affine(dt, 0.0)?; // [m,p]
        let f_bottom = s_inv_b.affine(dt * dt, 0.0)?; // [m,p]
        let f_mat = Tensor::cat(&[&f_top, &f_bottom], 0)?; // [2m,p]

        // Iterate sequence
        let mut w_k = w0;
        let mut outs: Vec<Tensor> = Vec::with_capacity(seq_len);
        for k in 0..seq_len {
            let u_k = input.narrow(1, k, 1)?.squeeze(1)?; // [batch,p]
            let m_w = w_k.matmul(&m_mat.transpose(0, 1)?)?; // [batch,2m]
            let f_u = u_k.matmul(&f_mat.transpose(0, 1)?)?; // [batch,2m]
            w_k = m_w.add(&f_u)?;

            let x_k = w_k.narrow(1, m, m)?; // [batch,m]
            let y = x_k.matmul(&self.c.transpose(0, 1)?)?; // [batch,q]
            outs.push(y.unsqueeze(1)?);
        }
        Ok(Tensor::cat(&outs.iter().collect::<Vec<_>>(), 1)?)
    }
}
