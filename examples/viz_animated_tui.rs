use std::io;
use std::time::{Duration, Instant};

use anyhow::Result;
use candle::{DType, Device, Tensor};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line as TxtLine, Span};
use ratatui::widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph, Sparkline};
use ratatui::Terminal;

#[cfg(feature = "fft")]
use dlinossrustcandle::TensorFftExt;
use dlinossrustcandle::{DLinOssLayer, DLinOssLayerConfig, TensorScanExt};

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

    fn color(&self) -> Color {
        match self {
            InputMode::Sine => Color::Cyan,
            InputMode::Step => Color::Yellow,
            InputMode::SinePlusStep => Color::Magenta,
            InputMode::Square => Color::Red,
            InputMode::Noise => Color::White,
            InputMode::Chirp => Color::Green,
            InputMode::Pulse => Color::Blue,
            InputMode::Sweep => Color::LightBlue,
        }
    }
}

struct AnimatedState {
    device: Device,
    layer: DLinOssLayer,
    window_size: usize,
    freq: f32,
    mode: InputMode,
    obs_index: usize,
    show_fft: bool,
    auto_cycle: bool,

    // Animation state
    time_step: f32,
    last_update: Instant,
    animation_speed: f32,

    // Ring buffers for streaming data
    input_history: Vec<f32>,
    output_history: Vec<f32>,
    cumsum_history: Vec<f32>,

    // Statistics
    input_rms: f32,
    output_rms: f32,
    processing_time_ms: f32,

    // Display data
    input_plot: Vec<(f64, f64)>,
    output_plot: Vec<(f64, f64)>,
    cumsum_plot: Vec<(f64, f64)>,
    fft_plot: Vec<(f64, f64)>,
    sparkline_data: Vec<u64>,
}

//

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
    let g_const = 0.25f32;
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
    layer.b = layer.b.affine(0.15, 0.0)?;
    set_observe_index(layer, obs_index)?;
    Ok(())
}

impl AnimatedState {
    fn new() -> Result<Self> {
        let device = Device::Cpu;
        let cfg = DLinOssLayerConfig {
            state_dim: 24,
            input_dim: 1,
            output_dim: 1,
            delta_t: 8e-3,
            dtype: DType::F32,
        };
        let mut layer = DLinOssLayer::new(cfg, &device)?;
        let obs_index = 0;
        stable_init_layer(&mut layer, obs_index)?;

        let window_size = 512;
        Ok(Self {
            device,
            layer,
            window_size,
            freq: 0.04,
            mode: InputMode::SinePlusStep,
            obs_index,
            show_fft: false,
            auto_cycle: false,

            time_step: 0.0,
            last_update: Instant::now(),
            animation_speed: 1.0,

            input_history: vec![0.0; window_size],
            output_history: vec![0.0; window_size],
            cumsum_history: vec![0.0; window_size],
            // fft_plot will be computed on demand; keep vector empty initially
            input_rms: 0.0,
            output_rms: 0.0,
            processing_time_ms: 0.0,

            input_plot: vec![],
            output_plot: vec![],
            cumsum_plot: vec![],
            fft_plot: vec![],
            sparkline_data: vec![50; 20],
        })
    }

    fn generate_input_signal(&self, length: usize) -> Result<Tensor> {
        let device = &self.device;
        let t_start = self.time_step;
        let dt = 0.02;

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
                        if (t % (1.0 / self.freq)) < (0.5 / self.freq) {
                            1.0
                        } else {
                            -0.5
                        }
                    })
                    .collect();
                Tensor::from_slice(&values, (length, 1), device)?
            }
            InputMode::SinePlusStep => {
                let values: Vec<f32> = (0..length)
                    .map(|i| {
                        let t = t_start + (i as f32) * dt;
                        let sine = (2.0 * std::f32::consts::PI * self.freq * t).sin();
                        let step = if (t % (1.0 / self.freq)) < (0.5 / self.freq) {
                            0.8
                        } else {
                            -0.3
                        };
                        sine + step
                    })
                    .collect();
                Tensor::from_slice(&values, (length, 1), device)?
            }
            InputMode::Square => {
                let values: Vec<f32> = (0..length)
                    .map(|i| {
                        let t = t_start + (i as f32) * dt;
                        if (2.0 * std::f32::consts::PI * self.freq * t).sin() >= 0.0 {
                            1.0
                        } else {
                            -1.0
                        }
                    })
                    .collect();
                Tensor::from_slice(&values, (length, 1), device)?
            }
            InputMode::Noise => Tensor::randn(0.0f32, 0.8f32, (length, 1), device)?,
            InputMode::Chirp => {
                let values: Vec<f32> = (0..length)
                    .map(|i| {
                        let t = t_start + (i as f32) * dt;
                        let freq_t = self.freq * (1.0 + 0.5 * t);
                        (2.0 * std::f32::consts::PI * freq_t * t).sin()
                    })
                    .collect();
                Tensor::from_slice(&values, (length, 1), device)?
            }
            InputMode::Pulse => {
                let values: Vec<f32> = (0..length)
                    .map(|i| {
                        let t = t_start + (i as f32) * dt;
                        let period = 1.0 / (self.freq * 3.0);
                        let pulse_width = period * 0.1;
                        if (t % period) < pulse_width {
                            2.0
                        } else {
                            0.0
                        }
                    })
                    .collect();
                Tensor::from_slice(&values, (length, 1), device)?
            }
            InputMode::Sweep => {
                let values: Vec<f32> = (0..length)
                    .map(|i| {
                        let t = t_start + (i as f32) * dt;
                        let amp = 0.5 + 0.5 * (0.1 * t).sin();
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

        if self.auto_cycle {
            // Auto-cycle through modes every 5 seconds
            if self.time_step % 5.0 < dt {
                let modes = [
                    InputMode::Sine,
                    InputMode::Square,
                    InputMode::SinePlusStep,
                    InputMode::Chirp,
                    InputMode::Pulse,
                    InputMode::Noise,
                ];
                let current_idx = modes.iter().position(|&m| m == self.mode).unwrap_or(0);
                self.mode = modes[(current_idx + 1) % modes.len()];
            }
        }

        self.time_step += dt * self.animation_speed;

        // Generate new chunk of data
        let chunk_size = 32;
        let start_time = Instant::now();

        let input_chunk = self.generate_input_signal(chunk_size)?;
        let input_batch = input_chunk.reshape((1, chunk_size, 1))?;
        let output_batch = self.layer.forward(&input_batch, None)?;

        self.processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

        // Extract data and update ring buffers
        let input_vals = input_chunk.squeeze(1)?.to_vec1::<f32>()?;
        let output_vals = output_batch.squeeze(0)?.squeeze(1)?.to_vec1::<f32>()?;

        // Calculate cumulative sum for current chunk
        let cumsum_chunk = input_chunk.prefix_sum_along(0)?;
        let cumsum_vals = cumsum_chunk.squeeze(1)?.to_vec1::<f32>()?;

        // Update ring buffers
        for (i, &val) in input_vals.iter().enumerate() {
            if i < chunk_size {
                self.input_history.remove(0);
                self.input_history.push(val);

                self.output_history.remove(0);
                self.output_history.push(output_vals[i]);

                self.cumsum_history.remove(0);
                self.cumsum_history.push(cumsum_vals[i]);
            }
        }

        // Calculate RMS values
        self.input_rms = (self.input_history.iter().map(|x| x * x).sum::<f32>()
            / self.input_history.len() as f32)
            .sqrt();
        self.output_rms = (self.output_history.iter().map(|x| x * x).sum::<f32>()
            / self.output_history.len() as f32)
            .sqrt();

        // Update sparkline data with output amplitude
        let recent_amp = self
            .output_history
            .iter()
            .rev()
            .take(10)
            .map(|x| x.abs())
            .fold(0.0, f32::max);
        self.sparkline_data.remove(0);
        self.sparkline_data.push((recent_amp * 100.0) as u64);

        // Convert to plot data
        self.input_plot = self
            .input_history
            .iter()
            .enumerate()
            .map(|(i, &y)| (i as f64, y as f64))
            .collect();
        self.output_plot = self
            .output_history
            .iter()
            .enumerate()
            .map(|(i, &y)| (i as f64, y as f64))
            .collect();
        self.cumsum_plot = self
            .cumsum_history
            .iter()
            .enumerate()
            .map(|(i, &y)| (i as f64, y as f64))
            .collect();

        // FFT if enabled
        #[cfg(feature = "fft")]
        if self.show_fft && self.output_history.len() >= 64 {
            let fft_input = Tensor::from_slice(
                &self.output_history[self.output_history.len() - 64..],
                (64,),
                &self.device,
            )?;
            if let Ok(spec) = fft_input.fft_real_norm() {
                if let Ok(spec_vals) = spec.to_vec1::<f32>() {
                    let pairs = spec_vals.len() / 2;
                    self.fft_plot = (0..pairs.min(self.window_size / 4))
                        .map(|i| {
                            let re = spec_vals[i * 2];
                            let im = spec_vals[i * 2 + 1];
                            let mag = (re * re + im * im).sqrt();
                            (i as f64, mag as f64)
                        })
                        .collect();
                }
            }
        }

        Ok(())
    }
}

fn create_help_text() -> Vec<TxtLine<'static>> {
    vec![TxtLine::from(vec![
        Span::styled(
            "D-LinOSS Real-Time Demo ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("— ", Style::default().fg(Color::Gray)),
        Span::styled("1-8", Style::default().fg(Color::Green)),
        Span::styled(" inputs, ", Style::default().fg(Color::Gray)),
        Span::styled("+/-", Style::default().fg(Color::Green)),
        Span::styled(" freq, ", Style::default().fg(Color::Gray)),
        Span::styled("Space", Style::default().fg(Color::Green)),
        Span::styled(" pause, ", Style::default().fg(Color::Gray)),
        Span::styled("A", Style::default().fg(Color::Green)),
        Span::styled(" auto-cycle, ", Style::default().fg(Color::Gray)),
        Span::styled("F", Style::default().fg(Color::Green)),
        Span::styled(" FFT, ", Style::default().fg(Color::Gray)),
        Span::styled("Q", Style::default().fg(Color::Green)),
        Span::styled(" quit", Style::default().fg(Color::Gray)),
    ])]
}

fn main() -> Result<()> {
    // Terminal setup
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    crossterm::execute!(
        stdout,
        crossterm::terminal::EnterAlternateScreen,
        crossterm::event::EnableMouseCapture
    )?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let mut app = AnimatedState::new()?;
    let mut paused = false;
    let mut running = true;

    while running {
        // Update animation if not paused
        if !paused {
            if let Err(e) = app.update_animation() {
                eprintln!("Animation error: {e}");
            }
        }

        // Draw
        terminal.draw(|f| {
            let size = f.size();

            // Main layout: header + body
            let main_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(3), Constraint::Min(0)].as_ref())
                .split(size);

            // Header with help
            let help_para = Paragraph::new(create_help_text())
                .block(Block::default().borders(Borders::ALL).title("Controls"))
                .alignment(Alignment::Left);
            f.render_widget(help_para, main_chunks[0]);

            // Body layout: top plots + bottom status
            let body_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(75), Constraint::Percentage(25)].as_ref())
                .split(main_chunks[1]);

            // Top plots: left (input/output) + right (cumsum/fft)
            let plot_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
                .split(body_chunks[0]);

            // Left plot: Input and Output overlaid
            let input_dataset = Dataset::default()
                .name("Input")
                .marker(ratatui::symbols::Marker::Dot)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(app.mode.color()))
                .data(&app.input_plot);

            let output_dataset = Dataset::default()
                .name("D-LinOSS Output")
                .marker(ratatui::symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Cyan))
                .data(&app.output_plot);

            let left_chart = Chart::new(vec![input_dataset, output_dataset])
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(format!("Input: {} | Output", app.mode.name())),
                )
                .x_axis(
                    Axis::default()
                        .title("Time")
                        .bounds([0.0, app.window_size as f64]),
                )
                .y_axis(Axis::default().title("Amplitude").bounds([-3.0, 3.0]));
            f.render_widget(left_chart, plot_chunks[0]);

            // Right plot: Cumsum or FFT
            let right_title = if app.show_fft && !app.fft_plot.is_empty() {
                "FFT Spectrum"
            } else {
                "Cumulative Sum"
            };

            let right_data = if app.show_fft && !app.fft_plot.is_empty() {
                &app.fft_plot
            } else {
                &app.cumsum_plot
            };

            let right_dataset = Dataset::default()
                .name(right_title)
                .marker(ratatui::symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Green))
                .data(right_data);

            let right_bounds = if app.show_fft {
                [0.0, 2.0]
            } else {
                [-5.0, 5.0]
            };
            let right_chart = Chart::new(vec![right_dataset])
                .block(Block::default().borders(Borders::ALL).title(right_title))
                .x_axis(
                    Axis::default()
                        .title("Index")
                        .bounds([0.0, right_data.len() as f64]),
                )
                .y_axis(Axis::default().title("Value").bounds(right_bounds));
            f.render_widget(right_chart, plot_chunks[1]);

            // Bottom status area: split into stats and sparkline
            let status_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(70), Constraint::Percentage(30)].as_ref())
                .split(body_chunks[1]);

            // Stats panel
            let stats_text = vec![
                TxtLine::from(vec![
                    Span::styled("Mode: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        app.mode.name(),
                        Style::default()
                            .fg(app.mode.color())
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!(" | Freq: {:.3} Hz", app.freq),
                        Style::default().fg(Color::White),
                    ),
                ]),
                TxtLine::from(vec![
                    Span::styled("Input RMS: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{:.3}", app.input_rms),
                        Style::default().fg(Color::Yellow),
                    ),
                    Span::styled(" | Output RMS: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{:.3}", app.output_rms),
                        Style::default().fg(Color::Cyan),
                    ),
                ]),
                TxtLine::from(vec![
                    Span::styled("Processing: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{:.2} ms", app.processing_time_ms),
                        Style::default().fg(Color::Green),
                    ),
                    Span::styled(
                        format!(
                            " | Observer: {}/{}",
                            app.obs_index,
                            app.layer.a.dims1().unwrap_or(1)
                        ),
                        Style::default().fg(Color::Magenta),
                    ),
                ]),
                TxtLine::from(vec![
                    Span::styled("Status: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        if paused { "PAUSED" } else { "RUNNING" },
                        Style::default()
                            .fg(if paused { Color::Red } else { Color::Green })
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        if app.auto_cycle { " | AUTO-CYCLE" } else { "" },
                        Style::default().fg(Color::Blue),
                    ),
                ]),
            ];

            let stats_para = Paragraph::new(stats_text)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Real-Time Statistics"),
                )
                .alignment(Alignment::Left);
            f.render_widget(stats_para, status_chunks[0]);

            // Sparkline for recent output activity
            let sparkline = Sparkline::default()
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Output Activity"),
                )
                .data(&app.sparkline_data)
                .style(Style::default().fg(Color::Yellow));
            f.render_widget(sparkline, status_chunks[1]);
        })?;

        // Input handling with real-time polling
        if event::poll(Duration::from_millis(16))? {
            // ~60 FPS
            if let Event::Key(KeyEvent {
                code, modifiers, ..
            }) = event::read()?
            {
                match code {
                    KeyCode::Char('1') => app.mode = InputMode::Sine,
                    KeyCode::Char('2') => app.mode = InputMode::Step,
                    KeyCode::Char('3') => app.mode = InputMode::SinePlusStep,
                    KeyCode::Char('4') => app.mode = InputMode::Square,
                    KeyCode::Char('5') => app.mode = InputMode::Noise,
                    KeyCode::Char('6') => app.mode = InputMode::Chirp,
                    KeyCode::Char('7') => app.mode = InputMode::Pulse,
                    KeyCode::Char('8') => app.mode = InputMode::Sweep,
                    KeyCode::Char('+') | KeyCode::Char('=') => {
                        app.freq = (app.freq
                            * if modifiers.contains(KeyModifiers::SHIFT) {
                                1.5
                            } else {
                                1.1
                            })
                        .min(0.5);
                    }
                    KeyCode::Char('-') => {
                        app.freq = (app.freq
                            * if modifiers.contains(KeyModifiers::SHIFT) {
                                0.67
                            } else {
                                0.9
                            })
                        .max(0.001);
                    }
                    KeyCode::Char(' ') => paused = !paused,
                    KeyCode::Char('a') | KeyCode::Char('A') => app.auto_cycle = !app.auto_cycle,
                    KeyCode::Char('f') | KeyCode::Char('F') => app.show_fft = !app.show_fft,
                    KeyCode::Char('c') | KeyCode::Char('C') => {
                        let m = app.layer.a.dims1().unwrap_or(1);
                        app.obs_index = (app.obs_index + 1) % m;
                        if let Err(e) = set_observe_index(&mut app.layer, app.obs_index) {
                            eprintln!("obs set error: {e}");
                        }
                    }
                    KeyCode::Char('r') | KeyCode::Char('R') => {
                        if let Err(e) = stable_init_layer(&mut app.layer, app.obs_index) {
                            eprintln!("reset error: {e}");
                        }
                    }
                    KeyCode::Up => app.animation_speed = (app.animation_speed * 1.2).min(5.0),
                    KeyCode::Down => app.animation_speed = (app.animation_speed * 0.8).max(0.1),
                    KeyCode::Char('q') | KeyCode::Esc => running = false,
                    _ => {}
                }
            }
        }
    }

    // Cleanup
    disable_raw_mode()?;
    crossterm::execute!(
        io::stdout(),
        crossterm::event::DisableMouseCapture,
        crossterm::terminal::LeaveAlternateScreen
    )?;

    Ok(())
}
