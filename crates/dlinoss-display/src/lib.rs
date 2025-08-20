//! dlinoss-display: building blocks for GUI/TUI display.
//! Features:
//! - `egui`: egui/egui_plot dual-pane plotting helpers.
//! - `etui`: ratatui-based terminal plotting helpers.
//! - `minifb`: minimal windowing stub.

use anyhow::Result;
use candlekos::Tensor;
#[allow(unused_imports)]
use dlinoss_augment::TensorScanExt;

/// Convert [T,1] to points for plotting
pub fn tensor_to_xy(points: &Tensor) -> Result<Vec<[f64; 2]>> {
    let (t, d) = points.dims2()?;
    anyhow::ensure!(d == 1, "Expected last dim 1 for plotting");
    let v = points.reshape((t,))?.to_vec1::<f32>()?;
    Ok(v.into_iter()
        .enumerate()
        .map(|(i, y)| [i as f64, y as f64])
        .collect())
}

#[cfg(feature = "egui")]
pub mod egui_dual {
    use super::*;
    use eframe::{App, Frame, NativeOptions, egui};
    use egui_plot::{Line, Plot};

    pub struct DualData {
        pub left: Vec<[f64; 2]>,
        pub right: Vec<[f64; 2]>,
        pub bottom_left: Option<Vec<[f64; 2]>>,
        pub bottom_right: Option<Vec<[f64; 2]>>,
        pub title: String,
    }

    pub struct DualPaneApp {
        pub data: DualData,
    }
    impl App for DualPaneApp {
        fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.columns(2, |cols| {
                    cols[0].heading("Left");
                    Plot::new("left").view_aspect(2.0).show(&mut cols[0], |p| {
                        p.line(Line::new(self.data.left.clone()));
                    });
                    if let Some(bl) = &self.data.bottom_left {
                        cols[0].heading("Bottom Left");
                        Plot::new("bottom-left")
                            .view_aspect(2.0)
                            .show(&mut cols[0], |p| {
                                p.line(Line::new(bl.clone()));
                            });
                    }
                    cols[1].heading("Right");
                    Plot::new("right").view_aspect(2.0).show(&mut cols[1], |p| {
                        p.line(Line::new(self.data.right.clone()));
                    });
                    if let Some(br) = &self.data.bottom_right {
                        cols[1].heading("Bottom Right");
                        Plot::new("bottom-right")
                            .view_aspect(2.0)
                            .show(&mut cols[1], |p| {
                                p.line(Line::new(br.clone()));
                            });
                    }
                });
            });
        }
    }

    pub fn run_dual(data: DualData) -> Result<()> {
        let options = NativeOptions::default();
        let title = data.title.clone();
        match eframe::run_native(
            &title,
            options,
            Box::new(|_cc| Ok::<Box<dyn App>, _>(Box::new(DualPaneApp { data }))),
        ) {
            Ok(_) => Ok(()),
            Err(e) => Err(anyhow::anyhow!("GUI error: {}", e)),
        }
    }
}

#[cfg(feature = "etui")]
pub mod tui {
    use super::*;
    use crossterm::event::{self, Event, KeyCode, KeyEvent};
    use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
    use ratatui::Terminal;
    use ratatui::backend::CrosstermBackend;
    use ratatui::layout::{Constraint, Direction, Layout};
    use ratatui::style::{Color, Style};
    use ratatui::widgets::{Axis, Block, Borders, Chart, Dataset, GraphType};
    use std::io;
    use std::time::Duration;

    pub fn draw_dual(left: &[(f64, f64)], right: &[(f64, f64)], title: &str) -> Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        crossterm::execute!(
            stdout,
            crossterm::terminal::EnterAlternateScreen,
            crossterm::event::EnableMouseCapture
        )?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let mut running = true;
        while running {
            terminal.draw(|f| {
                let size = f.size();
                let chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(size);
                let ds_left = vec![
                    Dataset::default()
                        .name("left")
                        .graph_type(GraphType::Line)
                        .style(Style::default().fg(Color::Cyan))
                        .data(left),
                ];
                let chart_left = Chart::new(ds_left)
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title(format!("{} - Left", title)),
                    )
                    .x_axis(Axis::default())
                    .y_axis(Axis::default());
                f.render_widget(chart_left, chunks[0]);
                let ds_right = vec![
                    Dataset::default()
                        .name("right")
                        .graph_type(GraphType::Line)
                        .style(Style::default().fg(Color::Green))
                        .data(right),
                ];
                let chart_right = Chart::new(ds_right)
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title(format!("{} - Right", title)),
                    )
                    .x_axis(Axis::default())
                    .y_axis(Axis::default());
                f.render_widget(chart_right, chunks[1]);
            })?;
            if event::poll(Duration::from_millis(50))? {
                if let Event::Key(KeyEvent { code, .. }) = event::read()? {
                    if matches!(code, KeyCode::Char('q') | KeyCode::Esc) {
                        running = false;
                    }
                }
            }
        }
        disable_raw_mode()?;
        crossterm::execute!(
            io::stdout(),
            crossterm::event::DisableMouseCapture,
            crossterm::terminal::LeaveAlternateScreen
        )?;
        Ok(())
    }
}

#[cfg(feature = "minifb")]
pub mod mini {
    pub fn hello_window_title() -> &'static str {
        "DLinOSS - minifb window"
    }
}
