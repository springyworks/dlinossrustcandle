use std::io;
use std::time::Duration;

use anyhow::Result;
use candlekos::{DType, Device, Tensor};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::{Line as TxtLine, Span};
use ratatui::widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph};

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
}

struct AppState {
    device: Device,
    layer: DLinOssLayer,
    t: usize,
    freq: f32,
    mode: InputMode,
    obs_index: usize,
    show_fft: bool,
    // prepared plot data
    left: Vec<(f64, f64)>,
    right: Vec<(f64, f64)>,
    bottom_left: Vec<(f64, f64)>,
    bottom_right: Vec<(f64, f64)>,
}

fn tensor_to_xy(points: &Tensor) -> Result<Vec<(f64, f64)>> {
    let (t, d) = points.dims2()?;
    anyhow::ensure!(d == 1, "Expected last dim 1 for plotting");
    let v = points.reshape((t,))?.to_vec1::<f32>()?;
    Ok(v.into_iter()
        .enumerate()
        .map(|(i, y)| (i as f64, y as f64))
        .collect())
}

fn set_observe_index(layer: &mut DLinOssLayer, idx: usize) -> Result<()> {
    let m = layer.a.dims1()?;
    let idx = idx.min(m.saturating_sub(1));
    let device = layer.a.device().clone();
    let c_mask = Tensor::arange(0u32, m as u32, &device)?
        .eq(idx as u32)?
        .to_dtype(DType::F32)?; // [m]
    layer.c = c_mask.reshape((1, m))?;
    Ok(())
}

fn stable_init_layer(layer: &mut DLinOssLayer, obs_index: usize) -> Result<()> {
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
    let a_low = two_plus_dtg.sub(&two_sqrt)?.affine(inv_dt2, 0.0)?;
    let a_high = two_plus_dtg.add(&two_sqrt)?.affine(inv_dt2, 0.0)?;
    layer.a = (&a_low + &a_high)?.affine(0.5, 0.0)?;
    layer.b = layer.b.affine(0.1, 0.0)?;
    set_observe_index(layer, obs_index)?;
    Ok(())
}

impl AppState {
    fn recompute(&mut self) -> Result<()> {
        let t = self.t;
        let device = &self.device;
        let time = Tensor::arange(0f32, t as f32, device)?.reshape((t, 1))?; // [T,1]
        let scale = 2.0f64 * std::f64::consts::PI * (self.freq as f64);
        let phase = time.affine(scale, 0.0)?;
        let sine = phase.sin()?;
        let step = Tensor::from_iter((0..t).map(|i| if i > t / 3 { 1f32 } else { 0f32 }), device)?
            .reshape((t, 1))?;
        let square = sine.sign()?;
        let noise = Tensor::randn(0.0f32, 1.0f32, (t, 1), device)?;
        let x = match self.mode {
            InputMode::Sine => sine.clone(),
            InputMode::Step => step.clone(),
            InputMode::SinePlusStep => (&sine + &step)?,
            InputMode::Square => square.clone(),
            InputMode::Noise => noise.clone(),
        };
        let input = x.reshape((1, t, 1))?;
        let y = self.layer.forward(&input, None)?; // [1,T,1]
        self.left = tensor_to_xy(&y.squeeze(0)?)?;
        let cumsum = input.squeeze(0)?.prefix_sum_along(0)?;
        self.right = tensor_to_xy(&cumsum)?;
        self.bottom_right = tensor_to_xy(&sine)?;
        // bottom_left: optionally FFT magnitude if feature enabled
        #[cfg(feature = "fft")]
        {
            let y1d = y.squeeze(0)?.reshape((t,))?;
            let spec = y1d.fft_real_norm()?;
            let spec_len = spec.dims1()?;
            let pairs = spec_len / 2;
            let spec2 = spec.reshape((pairs, 2))?;
            let re = spec2.narrow(1, 0, 1)?;
            let im = spec2.narrow(1, 1, 1)?;
            let mag = (re.sqr()? + im.sqr()?)?.sqrt()?; // [pairs,1]
            self.bottom_left = tensor_to_xy(&mag)?;
        }
        #[cfg(not(feature = "fft"))]
        {
            self.bottom_left.clear();
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    // Setup model
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

    // Terminal init
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

    let mut app = AppState {
        device: device.clone(),
        layer,
        t: 1024,
        freq: 0.03,
        mode: InputMode::SinePlusStep,
        obs_index,
        show_fft: false,
        left: vec![],
        right: vec![],
        bottom_left: vec![],
        bottom_right: vec![],
    };
    app.recompute()?;

    let mut running = true;
    while running {
        // Draw
        terminal.draw(|f| {
            let size = f.size();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(3), Constraint::Min(10)].as_ref())
                .split(size);

            let help = Paragraph::new(vec![TxtLine::from(vec![Span::styled(
                "DLinOSS TUI — 1-5 inputs, +/- freq, C cycle, F FFT, R reset, Q/Esc quit",
                Style::default().fg(Color::Yellow),
            )])])
            .block(Block::default().borders(Borders::ALL).title("Help"));
            f.render_widget(help, chunks[0]);

            let main_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
                .split(chunks[1]);

            // Left (model output)
            let data_left: Vec<(f64, f64)> = app.left.clone();
            let datasets_left = vec![
                Dataset::default()
                    .name("y[t]")
                    .marker(ratatui::symbols::Marker::Dot)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(Color::Cyan))
                    .data(&data_left),
            ];
            let chart_left = Chart::new(datasets_left)
                .block(Block::default().borders(Borders::ALL).title("Output y[t]"))
                .x_axis(Axis::default().title("t").bounds([0.0, app.t as f64]))
                .y_axis(Axis::default().title("amp").bounds([-2.0, 2.0]));
            f.render_widget(chart_left, main_chunks[0]);

            // Right (cumsum or input)
            let data_right: Vec<(f64, f64)> = app.right.clone();
            let datasets_right = vec![
                Dataset::default()
                    .name("cumsum(x)")
                    .marker(ratatui::symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(Color::Green))
                    .data(&data_right),
            ];
            let chart_right = Chart::new(datasets_right)
                .block(Block::default().borders(Borders::ALL).title("Cumsum input"))
                .x_axis(Axis::default().title("t").bounds([0.0, app.t as f64]))
                .y_axis(Axis::default().title("amp").bounds([-2.0, 2.0]));
            f.render_widget(chart_right, main_chunks[1]);
        })?;

        // Input handling with small poll timeout
        if event::poll(Duration::from_millis(20))?
            && let Event::Key(KeyEvent {
                code, modifiers, ..
            }) = event::read()?
        {
            match code {
                KeyCode::Char('1') => {
                    app.mode = InputMode::Sine;
                    app.recompute()?;
                }
                KeyCode::Char('2') => {
                    app.mode = InputMode::Step;
                    app.recompute()?;
                }
                KeyCode::Char('3') => {
                    app.mode = InputMode::SinePlusStep;
                    app.recompute()?;
                }
                KeyCode::Char('4') => {
                    app.mode = InputMode::Square;
                    app.recompute()?;
                }
                KeyCode::Char('5') => {
                    app.mode = InputMode::Noise;
                    app.recompute()?;
                }
                KeyCode::Char('+') | KeyCode::Char('=') => {
                    app.freq = (app.freq
                        * if modifiers.contains(KeyModifiers::SHIFT) {
                            1.25
                        } else {
                            1.10
                        })
                    .min(0.49);
                    app.recompute()?;
                }
                KeyCode::Char('-') => {
                    app.freq = (app.freq
                        * if modifiers.contains(KeyModifiers::SHIFT) {
                            0.75
                        } else {
                            0.90
                        })
                    .max(0.0005);
                    app.recompute()?;
                }
                KeyCode::Char('f') | KeyCode::Char('F') => {
                    app.show_fft = !app.show_fft;
                    app.recompute()?;
                }
                KeyCode::Char('c') | KeyCode::Char('C') => {
                    let m = app.layer.a.dims1().unwrap_or(1);
                    app.obs_index = (app.obs_index + 1) % m;
                    set_observe_index(&mut app.layer, app.obs_index)?;
                    app.recompute()?;
                }
                KeyCode::Char('r') | KeyCode::Char('R') => {
                    stable_init_layer(&mut app.layer, app.obs_index)?;
                    app.recompute()?;
                }
                KeyCode::Char('q') | KeyCode::Esc => {
                    running = false;
                }
                _ => {}
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
