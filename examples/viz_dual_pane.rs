use anyhow::Result;
use candlekos::{DType, Device, Tensor};
#[cfg(feature = "fft")]
use dlinossrustcandle::TensorFftExt;
use dlinossrustcandle::{DLinOssLayer, DLinOssLayerConfig, TensorScanExt};
use eframe::{App, Frame, NativeOptions, egui};
use egui::{CentralPanel, Color32, Context};
use egui_plot::{Line, Plot};

// Small helper to convert a [T, D] tensor to (x,y) pairs for D=1.
fn tensor_to_xy(points: &Tensor) -> Result<Vec<[f64; 2]>> {
    let (t, d) = points.dims2()?;
    anyhow::ensure!(d == 1, "Expected last dim 1 for plotting");
    let v = points.reshape((t,))?.to_vec1::<f32>()?;
    let pts: Vec<[f64; 2]> = v
        .iter()
        .enumerate()
        .map(|(i, &y)| [i as f64, y as f64])
        .collect();
    Ok(pts)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputMode {
    Sine,
    Step,
    SinePlusStep,
    Square,
    Noise,
}

struct DualPaneApp {
    // Candle + model
    device: Device,
    layer: DLinOssLayer,
    t: usize,
    freq: f32,
    mode: InputMode,
    show_bottom: bool,
    show_fft: bool,
    obs_index: usize,

    // Plots
    left: Vec<[f64; 2]>,
    right: Vec<[f64; 2]>,
    bottom_left: Option<Vec<[f64; 2]>>,
    bottom_right: Option<Vec<[f64; 2]>>,
}

impl App for DualPaneApp {
    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
        // Keyboard controls
        let mut need_recompute = false;
        ctx.input(|i| {
            for ev in &i.events {
                if let egui::Event::Key {
                    key,
                    pressed,
                    modifiers,
                    ..
                } = ev
                {
                    if !pressed {
                        continue;
                    }
                    match key {
                        egui::Key::Num1 => {
                            self.mode = InputMode::Sine;
                            need_recompute = true;
                        }
                        egui::Key::Num2 => {
                            self.mode = InputMode::Step;
                            need_recompute = true;
                        }
                        egui::Key::Num3 => {
                            self.mode = InputMode::SinePlusStep;
                            need_recompute = true;
                        }
                        egui::Key::Num4 => {
                            self.mode = InputMode::Square;
                            need_recompute = true;
                        }
                        egui::Key::Num5 => {
                            self.mode = InputMode::Noise;
                            need_recompute = true;
                        }
                        egui::Key::Plus | egui::Key::Equals => {
                            self.freq = (self.freq * 1.1).min(0.45);
                            need_recompute = true;
                        }
                        egui::Key::Minus => {
                            self.freq = (self.freq * 0.9).max(0.001);
                            need_recompute = true;
                        }
                        egui::Key::F => {
                            self.show_fft = !self.show_fft;
                            need_recompute = true;
                        }
                        egui::Key::V => {
                            self.show_bottom = !self.show_bottom;
                        }
                        egui::Key::R => {
                            if let Err(e) = stable_init_layer(&mut self.layer, self.obs_index) {
                                eprintln!("reset error: {e}");
                            }
                            need_recompute = true;
                        }
                        egui::Key::C => {
                            // cycle observation index
                            let m = self.layer.a.dims1().unwrap_or(1);
                            self.obs_index = (self.obs_index + 1) % m;
                            if let Err(e) = set_observe_index(&mut self.layer, self.obs_index) {
                                eprintln!("obs set error: {e}");
                            }
                            need_recompute = true;
                        }
                        egui::Key::Q | egui::Key::Escape => {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                        _ => {}
                    }
                    // Allow Shift +/- to step more aggressively
                    if modifiers.shift && (*key == egui::Key::Plus || *key == egui::Key::Equals) {
                        self.freq = (self.freq * 1.25).min(0.49);
                        need_recompute = true;
                    }
                    if modifiers.shift && *key == egui::Key::Minus {
                        self.freq = (self.freq * 0.75).max(0.0005);
                        need_recompute = true;
                    }
                }
            }
        });

        if need_recompute && let Err(e) = self.recompute() {
            eprintln!("recompute error: {e}");
        }

        CentralPanel::default().show(ctx, |ui| {
            ui.columns(2, |cols| {
                // Left column
                cols[0].heading("Left");
                let line_left = Line::new(self.left.clone());
                Plot::new("left-plot")
                    .view_aspect(2.0)
                    .show(&mut cols[0], |p| {
                        p.line(line_left);
                    });
                if self.show_bottom
                    && let Some(bl) = &self.bottom_left
                {
                    cols[0].heading("Bottom Left");
                    let line_bl = Line::new(bl.clone());
                    Plot::new("bottom_left")
                        .height(200.0)
                        .show(&mut cols[0], |plot_ui| {
                            plot_ui.line(line_bl.name("BL").color(Color32::LIGHT_GREEN));
                        });
                }

                // Right column
                cols[1].heading("Right");
                let line_right = Line::new(self.right.clone());
                Plot::new("right-plot")
                    .view_aspect(2.0)
                    .show(&mut cols[1], |p| {
                        p.line(line_right);
                    });
                if self.show_bottom
                    && let Some(br) = &self.bottom_right
                {
                    cols[1].heading("Bottom Right");
                    let line_br = Line::new(br.clone());
                    Plot::new("bottom_right")
                        .height(200.0)
                        .show(&mut cols[1], |plot_ui| {
                            plot_ui.line(line_br.name("BR").color(Color32::LIGHT_BLUE));
                        });
                }
            });
        });
    }
}

fn set_observe_index(layer: &mut DLinOssLayer, idx: usize) -> Result<()> {
    let m = layer.a.dims1()?;
    let idx = idx.min(m.saturating_sub(1));
    let device = layer.a.device().clone();
    let c_mask = Tensor::arange(0u32, m as u32, &device)?
        .eq(idx as u32)?
        .to_dtype(DType::F32)?; // [m] one-hot at idx
    layer.c = c_mask.reshape((1, m))?; // [1,m]
    Ok(())
}

fn stable_init_layer(layer: &mut DLinOssLayer, obs_index: usize) -> Result<()> {
    // Initialize to a stable, mild-damped configuration.
    let device = layer.a.device().clone();
    let m = layer.a.dims1()?;
    let g_const = 0.3f32;
    layer.g = Tensor::full(g_const, (m,), &device)?.to_dtype(DType::F32)?;
    let dt = layer.delta_t;
    let s = layer.g.affine(dt, 1.0)?; // 1 + dt*G
    let sqrt_s = s.sqrt()?;
    let two_plus_dtg = s.affine(1.0, 1.0)?; // (2 + dt*G)
    let two_sqrt = sqrt_s.affine(2.0, 0.0)?;
    let inv_dt2 = 1.0 / (dt * dt);
    let a_low = two_plus_dtg.sub(&two_sqrt)?.affine(inv_dt2, 0.0)?; // [m]
    let a_high = two_plus_dtg.add(&two_sqrt)?.affine(inv_dt2, 0.0)?; // [m]
    layer.a = (&a_low + &a_high)?.affine(0.5, 0.0)?; // midpoint in the band
    layer.b = layer.b.affine(0.1, 0.0)?; // soften coupling
    set_observe_index(layer, obs_index)?;
    Ok(())
}

impl DualPaneApp {
    fn recompute(&mut self) -> Result<()> {
        // Build input [B=1, T, D=1] based on mode
        let t = self.t;
        let device = &self.device;
        let time = Tensor::arange(0f32, t as f32, device)?.reshape((t, 1))?; // [T,1]
        let scale = 2.0f64 * std::f64::consts::PI * (self.freq as f64);
        let phase = time.affine(scale, 0.0)?; // [T,1]
        let sine = phase.sin()?; // [T,1]
        let step = Tensor::from_iter((0..t).map(|i| if i > t / 3 { 1f32 } else { 0f32 }), device)?
            .reshape((t, 1))?;
        let square = sine.sign()?; // -1/0/1; good enough visualization
        let noise = Tensor::randn(0.0f32, 1.0f32, (t, 1), device)?;

        let x = match self.mode {
            InputMode::Sine => sine.clone(),
            InputMode::Step => step.clone(),
            InputMode::SinePlusStep => (&sine + &step)?, // elementwise sum into one column
            InputMode::Square => square.clone(),
            InputMode::Noise => noise.clone(),
        };
        let input = x.reshape((1, t, 1))?; // [B=1, T, 1]

        let y = self.layer.forward(&input, None)?; // [1, T, 1]
        self.left = tensor_to_xy(&y.squeeze(0)?)?; // y[t]

        // Use augment trait (wraps Candle cumsum)
        let cumsum = input.squeeze(0)?.prefix_sum_along(0)?; // [T,1]
        self.right = tensor_to_xy(&cumsum)?;

        // Bottom panes
        self.bottom_right = Some(tensor_to_xy(&sine)?);

        #[cfg(feature = "fft")]
        {
            if self.show_fft {
                let y1d = y.squeeze(0)?; // [T,1]
                let y1d = y1d.reshape((t,))?; // [T]
                // Use augment trait (wraps Candle fft)
                let spec = y1d.fft_real_norm()?; // real->complex, normalized
                let spec_len = spec.dims1()?;
                let pairs = spec_len / 2;
                let spec2 = spec.reshape((pairs, 2))?; // [pairs,2]
                let re = spec2.narrow(1, 0, 1)?;
                let im = spec2.narrow(1, 1, 1)?;
                let mag = (re.sqr()? + im.sqr()?)?.sqrt()?; // [pairs,1]
                self.bottom_left = Some(tensor_to_xy(&mag)?);
            } else {
                self.bottom_left = None;
            }
        }
        #[cfg(not(feature = "fft"))]
        {
            self.bottom_left = None;
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    // Config: simple 1D in/out to make plotting easy
    let device = Device::Cpu;
    let cfg = DLinOssLayerConfig {
        state_dim: 16,
        input_dim: 1,
        output_dim: 1,
        delta_t: 1e-2,
        dtype: DType::F32,
    };
    let mut layer = DLinOssLayer::new(cfg, &device)?;
    let obs_index = 0usize;
    stable_init_layer(&mut layer, obs_index)?;

    let mut app = DualPaneApp {
        device: device.clone(),
        layer,
        t: 1024,
        freq: 0.03,
        mode: InputMode::SinePlusStep,
        show_bottom: true,
        show_fft: false,
        obs_index,

        left: vec![],
        right: vec![],
        bottom_left: None,
        bottom_right: None,
    };
    app.recompute()?;

    let options = NativeOptions::default();
    if let Err(e) = eframe::run_native(
        "DLinOSS Dual Pane",
        options,
        Box::new(|_cc| Ok::<Box<dyn App>, _>(Box::new(app))),
    ) {
        eprintln!("GUI error: {e}");
    }
    Ok(())
}
