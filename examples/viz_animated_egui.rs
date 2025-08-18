use anyhow::Result;
use candle::{DType, Device, Tensor};
use std::time::Instant;

#[cfg(feature = "fft")]
use dlinossrustcandle::TensorFftExt;
use dlinossrustcandle::{DLinOssLayer, DLinOssLayerConfig, TensorScanExt};
use eframe::{egui, App, Frame, NativeOptions};
use egui::{CentralPanel, Context, RichText};
use egui_plot::{Line, Plot, PlotPoints};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputMode {
    Sine,
    Step,
    SinePlusStep,
    Square,
    Noise,
    Chirp,
    Pulse,
    Sweep,
}

impl InputMode {
    fn name(&self) -> &'static str {
        match self {
            InputMode::Sine => "Sine Wave",
            InputMode::Step => "Step Function", 
            InputMode::SinePlusStep => "Sine + Step",
            InputMode::Square => "Square Wave",
            InputMode::Noise => "Gaussian Noise",
            InputMode::Chirp => "Chirp (Freq Sweep)",
            InputMode::Pulse => "Pulse Train",
            InputMode::Sweep => "Amplitude Sweep",
        }
    }

    fn color(&self) -> egui::Color32 {
        match self {
            InputMode::Sine => egui::Color32::LIGHT_BLUE,
            InputMode::Step => egui::Color32::YELLOW,
            InputMode::SinePlusStep => egui::Color32::from_rgb(255, 100, 255),
            InputMode::Square => egui::Color32::RED,
            InputMode::Noise => egui::Color32::WHITE,
            InputMode::Chirp => egui::Color32::GREEN,
            InputMode::Pulse => egui::Color32::BLUE,
            InputMode::Sweep => egui::Color32::LIGHT_GREEN,
        }
    }
}

struct AnimatedDLinOSSApp {
    device: Device,
    layer: DLinOssLayer,
    
    // Animation state
    time_step: f32,
    last_update: Instant,
    animation_speed: f32,
    window_size: usize,
    
    // Settings
    freq: f32,
    mode: InputMode,
    obs_index: usize,
    show_fft: bool,
    auto_cycle: bool,
    paused: bool,
    
    // Ring buffers for streaming visualization
    input_history: Vec<f32>,
    output_history: Vec<f32>,
    cumsum_history: Vec<f32>,
    fft_history: Vec<f32>,
    
    // Statistics
    input_rms: f32,
    output_rms: f32,
    processing_time_ms: f32,
    frames_per_second: f32,
    last_frame_time: Instant,
}

impl AnimatedDLinOSSApp {
    fn new() -> Result<Self> {
        let device = Device::Cpu;
        let cfg = DLinOssLayerConfig {
            state_dim: 32,
            input_dim: 1,
            output_dim: 1,
            delta_t: 5e-3,
            dtype: DType::F32,
        };
        let mut layer = DLinOssLayer::new(cfg, &device)?;
        
        // Initialize with stable parameters
        stable_init_layer(&mut layer, 0)?;
        
        let window_size = 512;
        let now = Instant::now();
        
        Ok(Self {
            device,
            layer,
            time_step: 0.0,
            last_update: now,
            animation_speed: 1.0,
            window_size,
            freq: 0.05,
            mode: InputMode::SinePlusStep,
            obs_index: 0,
            show_fft: false,
            auto_cycle: false,
            paused: false,
            input_history: vec![0.0; window_size],
            output_history: vec![0.0; window_size],
            cumsum_history: vec![0.0; window_size],
            fft_history: vec![0.0; window_size / 4],
            input_rms: 0.0,
            output_rms: 0.0,
            processing_time_ms: 0.0,
            frames_per_second: 60.0,
            last_frame_time: now,
        })
    }

    fn generate_input_signal(&self, length: usize) -> Result<Tensor> {
        let device = &self.device;
        let t_start = self.time_step;
        let dt = 0.015;
        
        let signal = match self.mode {
            InputMode::Sine => {
                let values: Vec<f32> = (0..length)
                    .map(|i| {
                        let t = t_start + (i as f32) * dt;
                        (2.0 * std::f32::consts::PI * self.freq * t).sin()
                    })
                    .collect();
                Tensor::from_slice(&values, (length, 1), device)?
            }
            InputMode::Step => {
                let values: Vec<f32> = (0..length)
                    .map(|i| {
                        let t = t_start + (i as f32) * dt;
                        if (t % (1.0 / self.freq)) < (0.5 / self.freq) { 1.2 } else { -0.6 }
                    })
                    .collect();
                Tensor::from_slice(&values, (length, 1), device)?
            }
            InputMode::SinePlusStep => {
                let values: Vec<f32> = (0..length)
                    .map(|i| {
                        let t = t_start + (i as f32) * dt;
                        let sine = (2.0 * std::f32::consts::PI * self.freq * t).sin();
                        let step = if (t % (2.0 / self.freq)) < (1.0 / self.freq) { 0.8 } else { -0.4 };
                        sine + step
                    })
                    .collect();
                Tensor::from_slice(&values, (length, 1), device)?
            }
            InputMode::Square => {
                let values: Vec<f32> = (0..length)
                    .map(|i| {
                        let t = t_start + (i as f32) * dt;
                        if (2.0 * std::f32::consts::PI * self.freq * t).sin() >= 0.0 { 1.5 } else { -1.5 }
                    })
                    .collect();
                Tensor::from_slice(&values, (length, 1), device)?
            }
            InputMode::Noise => {
                Tensor::randn(0.0f32, 0.9f32, (length, 1), device)?
            }
            InputMode::Chirp => {
                let values: Vec<f32> = (0..length)
                    .map(|i| {
                        let t = t_start + (i as f32) * dt;
                        let freq_t = self.freq * (1.0 + 0.8 * t.sin());
                        (2.0 * std::f32::consts::PI * freq_t * t).sin()
                    })
                    .collect();
                Tensor::from_slice(&values, (length, 1), device)?
            }
            InputMode::Pulse => {
                let values: Vec<f32> = (0..length)
                    .map(|i| {
                        let t = t_start + (i as f32) * dt;
                        let period = 1.0 / (self.freq * 2.0);
                        let pulse_width = period * 0.15;
                        if (t % period) < pulse_width { 3.0 } else { 0.0 }
                    })
                    .collect();
                Tensor::from_slice(&values, (length, 1), device)?
            }
            InputMode::Sweep => {
                let values: Vec<f32> = (0..length)
                    .map(|i| {
                        let t = t_start + (i as f32) * dt;
                        let amp = 0.3 + 1.2 * (0.1 * t).sin().abs();
                        amp * (2.0 * std::f32::consts::PI * self.freq * t).sin()
                    })
                    .collect();
                Tensor::from_slice(&values, (length, 1), device)?
            }
        };
        
        Ok(signal)
    }

    fn update_animation(&mut self) -> Result<()> {
        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f32();
        self.last_update = now;
        
        // Calculate FPS
        let frame_dt = now.duration_since(self.last_frame_time).as_secs_f32();
        self.frames_per_second = 0.9 * self.frames_per_second + 0.1 * (1.0 / frame_dt.max(0.001));
        self.last_frame_time = now;
        
        if self.auto_cycle {
            // Auto-cycle through modes every 4 seconds
            if self.time_step % 4.0 < dt {
                let modes = [
                    InputMode::Sine, InputMode::Square, InputMode::SinePlusStep,
                    InputMode::Chirp, InputMode::Pulse, InputMode::Sweep, InputMode::Noise
                ];
                let current_idx = modes.iter().position(|&m| m == self.mode).unwrap_or(0);
                self.mode = modes[(current_idx + 1) % modes.len()];
            }
        }
        
        self.time_step += dt * self.animation_speed;
        
        // Generate and process new data chunk
        let chunk_size = 24;
        let start_time = Instant::now();
        
        let input_chunk = self.generate_input_signal(chunk_size)?;
        let input_batch = input_chunk.reshape((1, chunk_size, 1))?;
        let output_batch = self.layer.forward(&input_batch, None)?;
        
        self.processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
        
        // Extract values
        let input_vals = input_chunk.squeeze(1)?.to_vec1::<f32>()?;
        let output_vals = output_batch.squeeze(0)?.squeeze(1)?.to_vec1::<f32>()?;
        
        // Calculate cumsum for this chunk
        let cumsum_chunk = input_chunk.prefix_sum_along(0)?;
        let cumsum_vals = cumsum_chunk.squeeze(1)?.to_vec1::<f32>()?;
        
        // Update ring buffers (streaming fashion)
        for i in 0..chunk_size.min(input_vals.len()) {
            self.input_history.remove(0);
            self.input_history.push(input_vals[i]);
            
            self.output_history.remove(0);
            self.output_history.push(output_vals[i]);
            
            self.cumsum_history.remove(0);
            self.cumsum_history.push(cumsum_vals[i]);
        }
        
        // Update statistics
        self.input_rms = (self.input_history.iter().map(|x| x * x).sum::<f32>() / self.input_history.len() as f32).sqrt();
        self.output_rms = (self.output_history.iter().map(|x| x * x).sum::<f32>() / self.output_history.len() as f32).sqrt();
        
        // FFT processing
        #[cfg(feature = "fft")]
        if self.show_fft && self.output_history.len() >= 128 {
            let fft_size = 128;
            let fft_input = Tensor::from_slice(
                &self.output_history[self.output_history.len() - fft_size..], 
                (fft_size,), 
                &self.device
            )?;
            
            if let Ok(spec) = fft_input.fft_real_norm() {
                if let Ok(spec_vals) = spec.to_vec1::<f32>() {
                    let pairs = spec_vals.len() / 2;
                    self.fft_history.clear();
                    for i in 0..pairs.min(self.window_size / 4) {
                        let re = spec_vals[i * 2];
                        let im = spec_vals[i * 2 + 1];
                        let mag = (re * re + im * im).sqrt();
                        self.fft_history.push(mag);
                    }
                }
            }
        }
        
        Ok(())
    }
}

impl App for AnimatedDLinOSSApp {
    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
        // Request continuous repaints for animation
        ctx.request_repaint();
        
        // Update animation if not paused
        if !self.paused {
            if let Err(e) = self.update_animation() {
                eprintln!("Animation error: {e}");
            }
        }
        
        // Handle keyboard input
        ctx.input(|i| {
            for event in &i.events {
                if let egui::Event::Key {
                    key,
                    pressed: true,
                    modifiers,
                    ..
                } = event
                {
                    match key {
                        egui::Key::Num1 => self.mode = InputMode::Sine,
                        egui::Key::Num2 => self.mode = InputMode::Step,
                        egui::Key::Num3 => self.mode = InputMode::SinePlusStep,
                        egui::Key::Num4 => self.mode = InputMode::Square,
                        egui::Key::Num5 => self.mode = InputMode::Noise,
                        egui::Key::Num6 => self.mode = InputMode::Chirp,
                        egui::Key::Num7 => self.mode = InputMode::Pulse,
                        egui::Key::Num8 => self.mode = InputMode::Sweep,
                        egui::Key::Plus | egui::Key::Equals => {
                            self.freq = (self.freq * if modifiers.shift { 1.5 } else { 1.1 }).min(0.5);
                        }
                        egui::Key::Minus => {
                            self.freq = (self.freq * if modifiers.shift { 0.67 } else { 0.9 }).max(0.001);
                        }
                        egui::Key::Space => self.paused = !self.paused,
                        egui::Key::A => self.auto_cycle = !self.auto_cycle,
                        egui::Key::F => self.show_fft = !self.show_fft,
                        egui::Key::C => {
                            let m = self.layer.a.dims1().unwrap_or(1);
                            self.obs_index = (self.obs_index + 1) % m;
                            if let Err(e) = set_observe_index(&mut self.layer, self.obs_index) {
                                eprintln!("Observer index error: {e}");
                            }
                        }
                        egui::Key::R => {
                            if let Err(e) = stable_init_layer(&mut self.layer, self.obs_index) {
                                eprintln!("Reset error: {e}");
                            }
                        }
                        egui::Key::ArrowUp => self.animation_speed = (self.animation_speed * 1.2).min(5.0),
                        egui::Key::ArrowDown => self.animation_speed = (self.animation_speed * 0.8).max(0.1),
                        egui::Key::Q | egui::Key::Escape => {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                        _ => {}
                    }
                }
            }
        });

        // Main UI layout
        CentralPanel::default().show(ctx, |ui| {
            // Title and controls
            ui.horizontal(|ui| {
                ui.heading(RichText::new("🚀 D-LinOSS Real-Time Animation").strong());
                ui.separator();
                
                ui.label(format!("Mode: {}", self.mode.name()));
                ui.separator();
                
                ui.label(format!("Freq: {:.3} Hz", self.freq));
                ui.separator();
                
                if self.paused {
                    ui.label(RichText::new("⏸ PAUSED").color(egui::Color32::RED));
                } else {
                    ui.label(RichText::new("▶ RUNNING").color(egui::Color32::GREEN));
                }
                
                if self.auto_cycle {
                    ui.label(RichText::new("🔄 AUTO-CYCLE").color(egui::Color32::BLUE));
                }
            });
            
            ui.separator();
            
            // Statistics row
            ui.horizontal(|ui| {
                ui.label(format!("Input RMS: {:.3}", self.input_rms));
                ui.separator();
                ui.label(format!("Output RMS: {:.3}", self.output_rms));
                ui.separator();
                ui.label(format!("Processing: {:.2} ms", self.processing_time_ms));
                ui.separator();
                ui.label(format!("FPS: {:.1}", self.frames_per_second));
                ui.separator();
                ui.label(format!("Window: {} samples", self.window_size));
                ui.separator();
                ui.label(format!("Observer: {}/{}", self.obs_index, self.layer.a.dims1().unwrap_or(1)));
            });
            
            ui.separator();

            // Control instructions
            ui.horizontal(|ui| {
                ui.label(RichText::new("Controls:").strong());
                ui.label("1-8: Input modes");
                ui.label("+/-: Frequency");
                ui.label("Space: Pause");
                ui.label("A: Auto-cycle");
                ui.label("F: FFT");
                ui.label("C: Observer");
                ui.label("R: Reset");
                ui.label("↑↓: Speed");
                ui.label("Q: Quit");
            });
            
            ui.separator();

            // Main plot area - split into columns
            ui.columns(2, |columns| {
                // Left column: Input/Output overlay
                columns[0].vertical(|ui| {
                    ui.heading("Input & D-LinOSS Output");
                    
                    let input_points: PlotPoints = self.input_history.iter()
                        .enumerate()
                        .map(|(i, &y)| [i as f64, y as f64])
                        .collect::<Vec<_>>()
                        .into();
                    
                    let output_points: PlotPoints = self.output_history.iter()
                        .enumerate()
                        .map(|(i, &y)| [i as f64, y as f64])
                        .collect::<Vec<_>>()
                        .into();
                    
                    Plot::new("input_output_plot")
                        .height(250.0)
                        .allow_zoom(true)
                        .allow_drag(true)
                        .show(ui, |plot_ui| {
                            plot_ui.line(
                                Line::new(input_points)
                                    .color(self.mode.color())
                                    .name(format!("Input: {}", self.mode.name()))
                                    .width(2.0)
                            );
                            plot_ui.line(
                                Line::new(output_points)
                                    .color(egui::Color32::LIGHT_BLUE)
                                    .name("D-LinOSS Output")
                                    .width(2.5)
                            );
                        });
                });

                // Right column: Cumsum or FFT
                columns[1].vertical(|ui| {
                    if self.show_fft && !self.fft_history.is_empty() {
                        ui.heading("FFT Spectrum");
                        
                        let fft_points: PlotPoints = self.fft_history.iter()
                            .enumerate()
                            .map(|(i, &y)| [i as f64, y as f64])
                            .collect::<Vec<_>>()
                            .into();
                        
                        Plot::new("fft_plot")
                            .height(250.0)
                            .allow_zoom(true)
                            .show(ui, |plot_ui| {
                                plot_ui.line(
                                    Line::new(fft_points)
                                        .color(egui::Color32::YELLOW)
                                        .name("Magnitude Spectrum")
                                        .width(2.0)
                                );
                            });
                    } else {
                        ui.heading("Cumulative Sum");
                        
                        let cumsum_points: PlotPoints = self.cumsum_history.iter()
                            .enumerate()
                            .map(|(i, &y)| [i as f64, y as f64])
                            .collect::<Vec<_>>()
                            .into();
                        
                        Plot::new("cumsum_plot")
                            .height(250.0)
                            .allow_zoom(true)
                            .show(ui, |plot_ui| {
                                plot_ui.line(
                                    Line::new(cumsum_points)
                                        .color(egui::Color32::GREEN)
                                        .name("Cumulative Sum")
                                        .width(2.0)
                                );
                            });
                    }
                });
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
        .to_dtype(DType::F32)?;
    layer.c = c_mask.reshape((1, m))?;
    Ok(())
}

fn stable_init_layer(layer: &mut DLinOssLayer, obs_index: usize) -> Result<()> {
    let device = layer.a.device().clone();
    let m = layer.a.dims1()?;
    let g_const = 0.2f32;
    layer.g = Tensor::full(g_const, (m,), &device)?.to_dtype(DType::F32)?;
    let dt = layer.delta_t;
    let s = layer.g.affine(dt, 1.0)?;
    let sqrt_s = s.sqrt()?;
    let two_plus_dtg = s.affine(1.0, 1.0)?;
    let two_sqrt = sqrt_s.affine(2.0, 0.0)?;
    let inv_dt2 = 1.0 / (dt * dt);
    let a_low = two_plus_dtg.sub(&two_sqrt)?.affine(inv_dt2, 0.0)?;
    let a_high = two_plus_dtg.add(&two_sqrt)?.affine(inv_dt2, 0.0)?;
    layer.a = (&a_low + &a_high)?.affine(0.5, 0.0)?;
    layer.b = layer.b.affine(0.12, 0.0)?;
    set_observe_index(layer, obs_index)?;
    Ok(())
}

fn main() -> Result<()> {
    let app = AnimatedDLinOSSApp::new()?;
    
    let options = NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("🚀 D-LinOSS Real-Time Animation"),
        ..Default::default()
    };
    
    if let Err(e) = eframe::run_native(
        "D-LinOSS Real-Time Animation",
        options,
        Box::new(|_cc| Ok::<Box<dyn App>, _>(Box::new(app))),
    ) {
        eprintln!("GUI error: {}", e);
    }
    
    Ok(())
}

// Touch dev-only patched crates so Cargo considers them part of the graph when building examples.
// This follows the repo guidance: don't suppress warnings; if a patch is declared, deliberately use it.
#[allow(dead_code)]
mod _dev_patch_touch {
    #[allow(unused_imports)]
    use candle_notebooks as _nb;
    #[allow(unused_imports)]
    use candle_transformers as _ct;
}