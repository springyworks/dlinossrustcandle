//! D-LinOSS Playground 3D + egui integration (phase 1 scaffold).
//! Features:
//! - Bevy 3D scene with instanced spheres representing a 3D slice of a 4D tensor.
//! - egui 4-pane dashboard (time series, latent energy, FFT placeholder, 3D controls).
//! - Deterministic D-LinOSS forward simulation updating a synthetic volume.
//! - Frame capture toggle (writes PNG frames under images_store/showcase/).
//!
//! Next phases (not yet implemented):
//! - True FFT live spectrum (reuse candle FFT when feature enabled)
//! - Phase portrait & latent component selection
//! - Headless export batch mode
//! - Colormap editor & auto scaling

use anyhow::Result;
use bevy::prelude::*;
use bevy::render::render_resource::PrimitiveTopology;
use bevy_egui::{EguiContexts, EguiPlugin, egui};
use candlekos::{DType, Device, Tensor};
use dlinossrustcandle::{DLinOssLayer, DLinOssLayerConfig};
use image::{ImageBuffer, Rgba};
use std::env;
use std::fs::{self, File};
use std::io::BufWriter;
mod support;
#[cfg(feature = "fft")]
use candlekos::Tensor as _TensorFftExtImportGuard;
use support::{
    compute_fft_naive as support_fft_naive, latent_energy_from_state_row, push_ring,
    push_ring_phase,
}; // ensure feature pulls in rfft symbol

// ---- Simulation Resources ----

#[derive(Resource)]
struct SimConfig {
    h: usize,
    w: usize,
    d: usize,
    paused: bool,
    capture: bool,
    frame_counter: u64,
}

#[derive(Resource)]
struct DLinOssSim {
    layer: DLinOssLayer,
    last_volume: Vec<f32>, // h * w * d values (current slice at given time)
    device: Device,
    input_dim: usize,
    t_len: usize,
    current_t: usize,
    input_cache: Vec<f32>,
    output_ring: Vec<f32>,
    input_ring: Vec<f32>,
    ring_capacity: usize,
    latent_energy_ring: Vec<f32>,
    phase_points: Vec<[f32; 2]>,
    latent_pair: (usize, usize),
    state_snapshot: Option<Tensor>, // last w_seq (small window)
    spectrum_cache: Vec<[f32; 2]>,
}

#[derive(Component)]
struct VoxelSphere {
    idx: usize,
}

// Camera
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    sim_cfg: Res<SimConfig>,
) {
    // Camera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(-4.0, 6.0, 14.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 10_000.0,
            range: 100.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(8.0, 12.0, 8.0),
        ..default()
    });

    // Base sphere mesh reused (scaled per instance)
    let sphere = Mesh::from(Sphere { radius: 0.5 });
    let sphere_handle = meshes.add(sphere);
    let base_mat = materials.add(StandardMaterial {
        base_color: Color::rgba_linear(0.8, 0.2, 0.2, 0.6),
        unlit: false,
        alpha_mode: bevy::prelude::AlphaMode::Blend,
        ..default()
    });

    let (h, w, d) = (sim_cfg.h, sim_cfg.w, sim_cfg.d);
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let idx = z * h * w + y * w + x;
                let xf = x as f32 - (w as f32) * 0.5;
                let yf = y as f32 - (h as f32) * 0.5;
                let zf = z as f32 - (d as f32) * 0.5;
                commands.spawn((
                    PbrBundle {
                        mesh: sphere_handle.clone(),
                        material: base_mat.clone(),
                        transform: Transform::from_xyz(xf, yf, zf).with_scale(Vec3::splat(0.05)),
                        ..default()
                    },
                    VoxelSphere { idx },
                ));
            }
        }
    }

    // Floor grid (optional simple plane)
    let mut plane = Mesh::new(PrimitiveTopology::TriangleList, Default::default());
    plane.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![
            [-20.0, -8.0, -20.0],
            [20.0, -8.0, -20.0],
            [20.0, -8.0, 20.0],
            [-20.0, -8.0, 20.0],
        ],
    );
    plane.insert_indices(bevy::render::mesh::Indices::U32(vec![0, 1, 2, 0, 2, 3]));
    commands.spawn(PbrBundle {
        mesh: meshes.add(plane),
        material: materials.add(StandardMaterial {
            base_color: Color::rgba(0.1, 0.1, 0.15, 1.0),
            perceptual_roughness: 0.9,
            metallic: 0.0,
            ..default()
        }),
        ..default()
    });
}

// Update simulation and sphere scales
fn sim_step(
    mut sim: ResMut<DLinOssSim>,
    mut spheres: Query<(&VoxelSphere, &mut Transform)>,
    sim_cfg: Res<SimConfig>,
) {
    if sim_cfg.paused {
        return;
    }
    // Construct input batch [1,T,input_dim] incremental small window
    let chunk = 8usize;
    let remaining = (sim.t_len - sim.current_t).min(chunk);
    if remaining == 0 {
        return;
    }
    let start = sim.current_t;
    let end = start + remaining;
    let input_dim = sim.input_dim;
    for t in start..end {
        let phase = t as f32 * 0.07;
        let base_idx = t * input_dim;
        if let Some(slot) = sim.input_cache.get_mut(base_idx) {
            *slot = phase.sin();
        }
        if input_dim > 1 {
            if let Some(slot) = sim.input_cache.get_mut(base_idx + 1) {
                *slot = (0.5 * phase).cos();
            }
        }
    }
    let u_slice = &sim.input_cache[start * sim.input_dim..end * sim.input_dim];
    let u_tensor = Tensor::from_slice(u_slice, (1, remaining, sim.input_dim), &sim.device).ok();
    if let Some(u) = u_tensor {
        if let Ok((w_seq, y)) = sim.layer.forward_with_state(&u, None) {
            // Store last state only (w_k at end)
            if let Ok(last_state) = w_seq.narrow(1, w_seq.dims()[1] - 1, 1) {
                sim.state_snapshot = last_state.squeeze(1).ok();
            }
            // Collapse output last dimension if >1 via mean, map to volume
            if let Ok(mut v) = y.squeeze(0) {
                // [remaining, q]
                if v.dims().len() == 2 {
                    let q = v.dims()[1];
                    if q > 1 {
                        v = v.mean(1).unwrap_or(v);
                    }
                }
                if let Ok(vals) = v.to_vec1::<f32>() {
                    // Map latest sample to the volume pattern (simple temporal diffusion)
                    if let Some(&latest) = vals.last() {
                        let len = sim.last_volume.len().max(1);
                        for (i, slot) in sim.last_volume.iter_mut().enumerate() {
                            *slot =
                                0.92 * *slot + 0.08 * (latest * (((i % len) as f32 * 0.001).sin()));
                        }
                    }
                    {
                        let cap = sim.ring_capacity;
                        for val in vals.iter() {
                            push_ring(&mut sim.output_ring, *val, cap);
                        }
                    }
                }
            }
            // Keep matching inputs (reuse slice used) approximate per-step first channel
            {
                let cap = sim.ring_capacity;
                for tloc in 0..remaining {
                    let v = sim.input_cache[(start + tloc) * sim.input_dim];
                    push_ring(&mut sim.input_ring, v, cap);
                }
            }
            // Latent energy: ||x||^2 using last state part (x portion of w = [v,x])
            // Extract values needed first (avoid double borrow later)
            let (norm_val, phase_pair_opt) = if let Some(ref w_last) = sim.state_snapshot {
                if let Ok(wv) = w_last.to_vec2::<f32>() {
                    if let Some(row) = wv.first() {
                        let half = row.len() / 2;
                        let x = &row[half..];
                        let norm = latent_energy_from_state_row(row);
                        let (i, j) = sim.latent_pair;
                        let phase_pair = if i < half && j < half {
                            Some([x[i], x[j]])
                        } else {
                            None
                        };
                        (Some(norm), phase_pair)
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };
            let cap = sim.ring_capacity; // copy to avoid immutable borrow later
            if let Some(nv) = norm_val {
                push_ring(&mut sim.latent_energy_ring, nv, cap);
            }
            if let Some(pp) = phase_pair_opt {
                push_ring_phase(&mut sim.phase_points, pp, cap / 4);
            }
        }
    }
    sim.current_t += remaining;
    // Update sphere transforms based on new volume values
    for (sphere, mut tf) in spheres.iter_mut() {
        if let Some(val) = sim.last_volume.get(sphere.idx) {
            let mag = val.abs();
            let scale = (0.02 + mag * 0.18).clamp(0.01, 0.5);
            tf.scale = Vec3::splat(scale);
        }
    }
}

// Egui UI (4 panes skeleton)
fn ui_system(
    mut contexts: EguiContexts,
    mut sim_cfg: ResMut<SimConfig>,
    mut sim: ResMut<DLinOssSim>,
) {
    let ctx = contexts.ctx_mut();
    egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.heading("D-LinOSS 3D Playground");
            if ui
                .button(if sim_cfg.paused { "Resume" } else { "Pause" })
                .clicked()
            {
                sim_cfg.paused = !sim_cfg.paused;
            }
            if ui
                .button(if sim_cfg.capture {
                    "Stop Capture"
                } else {
                    "Start Capture"
                })
                .clicked()
            {
                sim_cfg.capture = !sim_cfg.capture;
            }
            ui.label(format!("Frame: {}", sim_cfg.frame_counter));
            ui.separator();
            ui.label(format!("Grid: {}x{}x{}", sim_cfg.h, sim_cfg.w, sim_cfg.d));
        });
    });
    egui::SidePanel::left("left_panel").show(ctx, |ui| {
        ui.heading("Time Series");
        simple_plot(ui, "Input/Output", &sim.input_ring, &sim.output_ring, 160.0);
        ui.separator();
        ui.heading("Latent Energy (Approx)");
        simple_plot_single(
            ui,
            "Energy",
            &sim.latent_energy_ring,
            120.0,
            egui::Color32::RED,
        );
    });
    egui::SidePanel::right("right_panel").show(ctx, |ui| {
        ui.heading("3D Controls");
        ui.label("Camera orbit & colormap: TODO");
        ui.separator();
        ui.heading("FFT / Spectrum");
        if ui.button("FFT 256").clicked() {
            compute_fft_window(&mut sim, 256);
        }
        if ui.button("FFT 512").clicked() {
            compute_fft_window(&mut sim, 512);
        }
        spectrum_plot(ui, &sim.spectrum_cache, 140.0);
        ui.separator();
        ui.heading("Phase Portrait");
        ui.horizontal(|ui| {
            ui.label("i");
            ui.add(egui::DragValue::new(&mut sim.latent_pair.0).clamp_range(0..=63));
            ui.label("j");
            ui.add(egui::DragValue::new(&mut sim.latent_pair.1).clamp_range(0..=63));
        });
        phase_plot(ui, &sim.phase_points, 180.0);
    });
    egui::TopBottomPanel::bottom("bottom_panel").show(ctx, |ui| {
        ui.label("Phase Portrait & Diagnostics (future pane)");
    });
}

// Frame capture placeholder
fn capture_system(sim_cfg: Res<SimConfig>, sim: Res<DLinOssSim>) {
    if !sim_cfg.capture {
        return;
    }
    let dir = "images_store/SHOWCASE";
    let _ = fs::create_dir_all(dir);
    // Simple 2D projection of current volume onto an image (XZ slice stacking Y rows)
    let (h, w, d) = (sim_cfg.h, sim_cfg.w, sim_cfg.d);
    let slice_h = d as u32; // one pixel per depth
    let img_h = h as u32 * slice_h;
    let img_w = w as u32;
    let mut imgbuf: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(img_w, img_h);
    for y in 0..h {
        for x in 0..w {
            for z in 0..d {
                let idx = z * h * w + y * w + x;
                if let Some(val) = sim.last_volume.get(idx) {
                    let norm = (val.abs() * 4.0).min(1.0);
                    let r = (norm * 255.0) as u8;
                    let g = ((1.0 - norm) * 64.0) as u8;
                    let b = ((norm * norm) * 255.0) as u8;
                    let py = y as u32 * slice_h + z as u32;
                    imgbuf.put_pixel(x as u32, py, Rgba([r, g, b, 255]));
                }
            }
        }
    }
    let path = format!("{}/frame_{:06}.png", dir, sim_cfg.frame_counter);
    if let Ok(f) = File::create(&path) {
        let mut w = BufWriter::new(f);
        #[allow(deprecated)]
        {
            let encoder = image::codecs::png::PngEncoder::new(&mut w);
            let _ = encoder.encode(&imgbuf, img_w, img_h, image::ColorType::Rgba8);
        }
    }
}

fn frame_counter(mut sim_cfg: ResMut<SimConfig>) {
    sim_cfg.frame_counter += 1;
}
fn compute_fft_window(sim: &mut DLinOssSim, size: usize) {
    if sim.output_ring.len() < size {
        return;
    }
    let window = &sim.output_ring[sim.output_ring.len() - size..];
    if window.len() != size {
        return;
    }
    let tmp: Vec<f32> = window.to_vec();
    #[cfg(feature = "fft")]
    {
        if !fill_spectrum_candle(sim, &tmp) {
            fill_spectrum_from_naive(sim, &tmp);
        }
    }
    #[cfg(not(feature = "fft"))]
    {
        fill_spectrum_from_naive(sim, &tmp);
    }
}
fn fill_spectrum_from_naive(sim: &mut DLinOssSim, window: &[f32]) {
    let spec = support_fft_naive(window);
    sim.spectrum_cache.clear();
    for p in spec.into_iter().take(512) {
        sim.spectrum_cache.push(p);
    }
}
#[cfg(feature = "fft")]
fn fill_spectrum_candle(sim: &mut DLinOssSim, window: &[f32]) -> bool {
    use candlekos::DType;
    // Create tensor and run rfft (normalized) then compute magnitude of complex interleaved output
    let dev = &sim.device;
    let t = match Tensor::from_slice(window, (window.len(),), dev) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let len = window.len();
    let fft = match t.rfft(0usize, true) {
        Ok(f) => f,
        Err(_) => return false,
    };
    // rfft output shape should be [len/2 + 1, 2] (real, imag)
    let dims = fft.dims();
    if dims.len() != 2 || dims[1] != 2 {
        return false;
    }
    let rows = dims[0];
    let data = match fft.to_vec2::<f32>() {
        Ok(v) => v,
        Err(_) => return false,
    };
    sim.spectrum_cache.clear();
    // Skip last Nyquist bin for consistency with naive half-size; cap to 512
    for k in 0..rows.min(512) {
        let re = data[k][0];
        let im = data[k][1];
        let mag = (re * re + im * im).sqrt() / (len as f32);
        sim.spectrum_cache.push([k as f32, mag]);
    }
    true
}

fn spectrum_plot(ui: &mut egui::Ui, pts: &[[f32; 2]], height: f32) {
    ui.label("Spectrum");
    let desired = egui::Vec2::new(ui.available_width(), height);
    let (rect, _) = ui.allocate_exact_size(desired, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_stroke(rect, 0.0, egui::Stroke::new(1.0, egui::Color32::DARK_GRAY));
    if pts.is_empty() {
        return;
    }
    let max_x = pts.last().map(|p| p[0]).unwrap_or(1.0).max(1.0);
    let mut max_y = 1e-6f32;
    for p in pts {
        if p[1] > max_y {
            max_y = p[1];
        }
    }
    for p in pts.iter().take(512) {
        let x = p[0] / max_x;
        let y = (p[1] / max_y).min(1.0);
        let top = egui::pos2(
            rect.left() + x * rect.width(),
            rect.bottom() - y * rect.height(),
        );
        let bottom = egui::pos2(rect.left() + x * rect.width(), rect.bottom());
        painter.line_segment(
            [bottom, top],
            egui::Stroke::new(1.0, egui::Color32::LIGHT_GREEN),
        );
    }
}

fn phase_plot(ui: &mut egui::Ui, pts: &[[f32; 2]], height: f32) {
    ui.label("Phase / Scatter");
    let desired = egui::Vec2::new(ui.available_width(), height);
    let (rect, _resp) = ui.allocate_exact_size(desired, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_stroke(rect, 0.0, egui::Stroke::new(1.0, egui::Color32::DARK_GRAY));
    if pts.is_empty() {
        return;
    }
    // Determine bounds
    let (mut min_x, mut max_x) = (pts[0][0], pts[0][0]);
    let (mut min_y, mut max_y) = (pts[0][1], pts[0][1]);
    for p in pts.iter().skip(1) {
        if p[0] < min_x {
            min_x = p[0];
        }
        if p[0] > max_x {
            max_x = p[0];
        }
        if p[1] < min_y {
            min_y = p[1];
        }
        if p[1] > max_y {
            max_y = p[1];
        }
    }
    let sx = (max_x - min_x).max(1e-6);
    let sy = (max_y - min_y).max(1e-6);
    for p in pts.iter().rev().take(2000) {
        // draw recent points
        let x = (p[0] - min_x) / sx;
        let y = (p[1] - min_y) / sy;
        let pos = egui::pos2(
            rect.left() + x * rect.width(),
            rect.bottom() - y * rect.height(),
        );
        painter.circle_filled(
            pos,
            1.0,
            egui::Color32::from_rgba_unmultiplied(200, 220, 90, 180),
        );
    }
}

// push_ring_phase now imported from support

// Very lightweight plot helpers (no external deps) using egui painter.
fn simple_plot(ui: &mut egui::Ui, title: &str, a: &[f32], b: &[f32], height: f32) {
    ui.label(title);
    let desired = egui::Vec2::new(ui.available_width(), height);
    let (rect, _resp) = ui.allocate_exact_size(desired, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_stroke(rect, 0.0, egui::Stroke::new(1.0, egui::Color32::DARK_GRAY));
    let max_len = a.len().max(b.len()).max(1);
    let (min_a, max_a) = min_max(a);
    let (min_b, max_b) = min_max(b);
    let min_v = min_a.min(min_b);
    let max_v = max_a.max(max_b).max(min_v + 1e-6);
    let to_point = |i: usize, v: f32| {
        let x = i as f32 / (max_len - 1).max(1) as f32;
        let y = if max_v > min_v {
            (v - min_v) / (max_v - min_v)
        } else {
            0.5
        };
        egui::pos2(
            rect.left() + x * rect.width(),
            rect.bottom() - y * rect.height(),
        )
    };
    polyline(painter.clone(), a, egui::Color32::LIGHT_BLUE, to_point);
    polyline(painter, b, egui::Color32::GOLD, to_point);
}

fn simple_plot_single(
    ui: &mut egui::Ui,
    title: &str,
    a: &[f32],
    height: f32,
    color: egui::Color32,
) {
    ui.label(title);
    let desired = egui::Vec2::new(ui.available_width(), height);
    let (rect, _resp) = ui.allocate_exact_size(desired, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_stroke(rect, 0.0, egui::Stroke::new(1.0, egui::Color32::DARK_GRAY));
    let max_len = a.len().max(1);
    let (min_a, max_a) = min_max(a);
    let max_v = max_a.max(min_a + 1e-6);
    let to_point = |i: usize, v: f32| {
        let x = i as f32 / (max_len - 1).max(1) as f32;
        let y = if max_v > min_a {
            (v - min_a) / (max_v - min_a)
        } else {
            0.5
        };
        egui::pos2(
            rect.left() + x * rect.width(),
            rect.bottom() - y * rect.height(),
        )
    };
    polyline(painter, a, color, to_point);
}

fn min_max(slice: &[f32]) -> (f32, f32) {
    if slice.is_empty() {
        return (0.0, 0.0);
    }
    let mut min_v = slice[0];
    let mut max_v = slice[0];
    for &v in slice.iter().skip(1) {
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
    }
    (min_v, max_v)
}

fn polyline<F: Fn(usize, f32) -> egui::Pos2>(
    painter: egui::Painter,
    data: &[f32],
    color: egui::Color32,
    map: F,
) {
    if data.len() < 2 {
        return;
    }
    let mut points = Vec::with_capacity(data.len());
    for (i, &v) in data.iter().enumerate() {
        points.push(map(i, v));
    }
    painter.add(egui::Shape::line(points, egui::Stroke::new(1.0, color)));
}

pub fn main() -> Result<()> {
    // Initialize deterministic layer
    let device = Device::Cpu;
    let cfg = DLinOssLayerConfig {
        state_dim: 16,
        input_dim: 2,
        output_dim: 3,
        delta_t: 1e-2,
        dtype: DType::F32,
    };
    let layer = DLinOssLayer::deterministic(cfg.clone(), &device)?;
    let h = 12usize;
    let w = 10usize;
    let d = 8usize; // moderate grid
    let volume = vec![0f32; h * w * d];
    let sim_res = DLinOssSim {
        layer,
        last_volume: volume,
        device: device.clone(),
        input_dim: cfg.input_dim,
        t_len: 10_000,
        current_t: 0,
        input_cache: vec![0f32; 10_000 * cfg.input_dim],
        output_ring: Vec::with_capacity(4096),
        input_ring: Vec::with_capacity(4096),
        ring_capacity: 4096,
        latent_energy_ring: Vec::with_capacity(4096),
        phase_points: Vec::with_capacity(4096),
        latent_pair: (0, 1),
        state_snapshot: None,
        spectrum_cache: Vec::with_capacity(1024),
    };
    let sim_cfg = SimConfig {
        h,
        w,
        d,
        paused: false,
        capture: false,
        frame_counter: 0,
    };

    let headless = env::var("DLINOSS_HEADLESS")
        .ok()
        .map(|v| v == "1")
        .unwrap_or(false);
    let mut app = App::new();
    app.insert_resource(sim_cfg)
        .insert_resource(sim_res)
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: if headless {
                None
            } else {
                Some(Window {
                    title: "D-LinOSS 3D Playground".into(),
                    resolution: (1380.0, 900.0).into(),
                    ..default()
                })
            },
            ..default()
        }))
        .add_plugins(EguiPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, (sim_step, ui_system, capture_system, frame_counter));
    if headless {
        let max_frames: u64 = env::var("DLINOSS_FRAMES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300);
        for _ in 0..max_frames {
            app.update();
        }
    } else {
        app.run();
    }
    Ok(())
}
