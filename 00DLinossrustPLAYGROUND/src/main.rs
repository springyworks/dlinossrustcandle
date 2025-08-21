use anyhow::Result;
use std::env;
use bevy::core_pipeline::tonemapping::Tonemapping; // added for tonemapper enum usage
use bevy::input::mouse::MouseButton;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;
use bevy_egui::{EguiContexts, egui, EguiPlugin};
use dlinossrustcandle::{Device, DType, Tensor, DLinOssLayer, DLinOssLayerConfig};
mod support;
use support::{latent_energy_from_state_row, push_ring_phase};

macro_rules! info { ($($t:tt)*) => { println!($($t)*); } }

// Core visualization configuration
#[derive(Resource)]
struct VisualizationConfig {
    depth_scale_attenuation: bool,
    reverse_time_scroll: bool,
    time_window: usize,
    show_center_axes: bool,
    show_corner_axes: bool,
    base_scale: f32,
    scale_factor: f32,
    max_scale: f32,
    spike_value: f32,
    energy_scale: f32,
    show_internal_grid: bool,
    internal_grid_opacity: f32,
    internal_grid_stride: usize,
    wave_mode: WaveMode,
    show_volume_stats: bool,
    show_energy_derivative: bool,
    show_advanced_scales: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WaveMode { Normal, ComplexWave }

#[derive(Resource)]
struct SimConfig {
    h: usize,
    w: usize,
    d: usize,
    paused: bool,
    capture: bool,
    frame_counter: u64,
    capture_start_frame: Option<u64>,
    max_capture_frames: u64,
    capture_prefix: String,
}

impl Default for SimConfig { fn default() -> Self { Self { h:8,w:8,d:8,paused:false,capture:false,frame_counter:0,capture_start_frame:None,max_capture_frames:300,capture_prefix:"sess".into() } } }

#[derive(Clone, Copy, PartialEq, Eq)]
enum CameraAnimMode { Idle, FlyThrough, HalluFly, AutoRotate }

#[derive(Resource)]
struct CameraAnimState {
    mode: CameraAnimMode,
    t: f32,
    loop_index: u64,
    approach: f32,
    pause: f32,
    retreat: f32,
    total: f32,
    base_transform: Transform,
}

impl Default for CameraAnimState { fn default() -> Self { Self { mode: CameraAnimMode::Idle, t:0.0, loop_index:0, approach:4.0, pause:3.0, retreat:4.0, total:11.0, base_transform: Transform::IDENTITY } } }

#[derive(Component)]
struct VoxelSphere { idx: usize }

#[derive(Clone, Copy)]
struct TimedSample { idx: u64, value: f32 }

struct TimedRing { data: Vec<TimedSample>, capacity: usize, next_idx: u64 }
impl TimedRing {
    fn new(capacity: usize) -> Self { Self { data: Vec::with_capacity(capacity.min(2048)), capacity, next_idx:0 } }
    fn push(&mut self, v: f32) { if self.capacity>0 && self.data.len()>=self.capacity { let drop = (self.capacity/4).max(1); self.data.drain(0..drop); } self.data.push(TimedSample{ idx:self.next_idx, value:v }); self.next_idx+=1; }
    fn clear(&mut self){ self.data.clear(); }
    fn is_empty(&self)->bool{ self.data.is_empty() }
    fn latest_idx(&self)->u64{ self.data.last().map(|s|s.idx).unwrap_or(0) }
    fn iter(&self)->std::slice::Iter<'_,TimedSample>{ self.data.iter() }
}

#[derive(Resource)]
struct DLinOssSim {
    layer: DLinOssLayer,
    last_volume: Vec<f32>,
    output_ring: TimedRing,
    input_ring: TimedRing,
    latent_energy_ring: TimedRing,
    phase_points: Vec<[f32;2]>,
    latent_energy_peak: f32,
    camera_depth_ring: TimedRing,
    input_dim: usize,
    device: Device,
    current_t: usize,
    t_len: usize,
    input_cache: Vec<f32>,
    ring_capacity: usize,
    latent_pair: (usize,usize),
    state_snapshot: Option<Tensor>,
    spectrum_cache: Vec<[f32;2]>,
    volume_mean_ring: TimedRing,
    volume_var_ring: TimedRing,
    latent_energy_deriv_ring: TimedRing,
    last_latent_energy: f32,
}

// (Other existing use statements for your layer, rings, etc. should remain below or above as appropriate)
// Camera orbit/zoom controller (turntable pattern)
#[derive(Resource)]
struct CameraOrbitController {
    pub yaw: f32,
    pub pitch: f32,
    pub radius: f32,
    pub target: Vec3,
    #[allow(dead_code)]
    pub dragging: bool,
    #[allow(dead_code)]
    pub last_mouse_pos: Vec2,
}

impl Default for CameraOrbitController {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            radius: 14.0,
            target: Vec3::ZERO,
            dragging: false,
            last_mouse_pos: Vec2::ZERO,
        }
    }
}

fn camera_orbit_system(
    mut controller: ResMut<CameraOrbitController>,
    mut query: Query<&mut Transform, With<Camera3d>>,
    buttons: Res<bevy::input::ButtonInput<bevy::input::mouse::MouseButton>>,
    mut mouse_motion: EventReader<bevy::input::mouse::MouseMotion>,
    mut scroll: EventReader<bevy::input::mouse::MouseWheel>,
    anim: Option<Res<CameraAnimState>>,
    mut egui_ctxs: EguiContexts,
) {
    if let Some(anim) = anim {
        if anim.mode != CameraAnimMode::Idle {
            return;
        }
    }
    let egui_ctx = egui_ctxs.ctx_mut();
    if egui_ctx.is_pointer_over_area() || egui_ctx.wants_pointer_input() {
        return;
    }
    for ev in scroll.read() {
        controller.radius -= ev.y * 1.2;
        controller.radius = controller.radius.clamp(3.0, 80.0);
    }
    // Dynamic minimum pitch so camera never drops below ground plane (y = GROUND_Y)
    const GROUND_Y: f32 = -8.0; // must match plane spawn
    let ground_clearance = 0.3; // keep camera this much above plane
    let min_pitch_from_ground = ((GROUND_Y + ground_clearance) / controller.radius)
        .clamp(-0.99, 0.99)
        .asin();
    // Hard safety clamp (do not let camera flip upside-down)
    let min_pitch = min_pitch_from_ground.max(-0.65); // ~ -37 deg
    let max_pitch = 1.35; // slightly less than previous to avoid grazing top singularity
    let mut delta = Vec2::ZERO;
    if buttons.pressed(MouseButton::Left) {
        for ev in mouse_motion.read() {
            delta += ev.delta;
        }
        if delta.length_squared() > 0.0 {
            controller.yaw -= delta.x * 0.012;
            controller.pitch -= delta.y * 0.012;
            controller.pitch = controller.pitch.clamp(min_pitch, max_pitch);
        }
    }
    let yaw = controller.yaw;
    let pitch = controller.pitch;
    let r = controller.radius;
    let target = controller.target;
    let x = r * yaw.cos() * pitch.cos();
    let y = r * pitch.sin();
    let z = r * yaw.sin() * pitch.cos();
    let pos = target + Vec3::new(x, y, z);
    for mut tf in query.iter_mut() {
        tf.translation = pos;
        tf.look_at(target, Vec3::Y);
    }
}
// D-LinOSS Playground 3D + egui integration (phase 1 scaffold).
// Features:
// - Bevy 3D scene with instanced spheres representing a 3D slice of a 4D tensor.
// - egui 4-pane dashboard (time series, latent energy, FFT placeholder, 3D controls).
// - Deterministic D-LinOSS forward simulation updating a synthetic volume.
// - Frame capture toggle (writes PNG frames under images_store/showcase/).
// (ui_system function defined later – stray previous inline UI removed)


// Fly Through & HalluFly docs:
// - Fly Through: cinematic path that starts far (z ~ 42), dives toward/through the voxel cube, tightens
//   orbit radius inside, then retreats, looping seamlessly. Subtle spin near closest approach for depth cue.
// - HalluFly: same spatial path but with aggressive multi-axis spin while "inside" plus an altered voxel field:
//   spheres receive an added swirl * breathe modulation (coordinated radial + angular wave) producing a
//   hallucinatory pulsation synchronized with camera motion. Toggle via top-bar buttons.
// Implementation details:
//   camera_anim_system drives transform using time-normalized phase with sinusoidal easing to avoid jerk.
//   sim_step checks for HalluFly to blend extra volumetric patterning (swirl+breathe) into last_volume.
//   camera_orbit_system is disabled while an animation mode is active so manual input doesn't fight it.
// Future tuning ideas:
//   * Parameter sliders for duration, near/far depth, spin intensity.
//   * Path variants (figure-eight, helical ascent, spline-defined waypoints).
//   * Per-voxel color modulation keyed off swirl phase for psychedelic palettes.
// 2025-08 additions:
//   * Random per-loop jitter (SmallRng) blended stronger near interior for organic flight feel.
//   * Three extended pause windows (centers 0.33 / 0.50 / 0.67 of loop) each 3x previous length for showcase.
//     A smooth pause factor (0..1) derived from proximity to centers modulates spin & highlight intensity.
//   * Highlight cluster: sparse index selection scaled up while paused to draw attention.
//   * Opal glass material: translucent whitish spheres with soft emissive tint for better light play.
//   * Light toggle: UI button zeroes / restores intensities for point & spot lights.

fn camera_anim_system(
    time: Res<Time>,
    mut q_cam: Query<&mut Transform, (With<Camera3d>, Without<VoxelSphere>)>,
    mut state: ResMut<CameraAnimState>,
) {
    if state.mode == CameraAnimMode::Idle {
        return;
    }
    let dt = time.delta_seconds();
    state.t += dt;
    let total = state.total;
    if state.t >= total {
        state.t -= total; // loop smoothly
        state.loop_index += 1;
    }
    let t = state.t;
    let (approach, pause, retreat) = (state.approach, state.pause, state.retreat);
    let mut phase_kind = "approach";
    let near_pos = Vec3::new(8.0, 4.0, 8.0);
    let far_pos = Vec3::new(-6.0, 7.0, 44.0);
    // Helper smoothstep easing
    let smooth = |x: f32| (x * x * (3.0 - 2.0 * x)).clamp(0.0, 1.0);
    let pos;
    if state.mode == CameraAnimMode::AutoRotate {
        // Slow orbit around origin at fixed radius
        let radius = 18.0;
        let speed = 0.18; // radians/sec (slow)
        let angle = state.t * speed;
        let x = radius * angle.cos();
        let z = radius * angle.sin();
        let y = 7.0 + 2.0 * (angle * 0.3).sin();
        pos = Vec3::new(x, y, z);
        if let Ok(mut tf) = q_cam.get_single_mut() {
            *tf = Transform::from_translation(pos).looking_at(Vec3::ZERO, Vec3::Y);
        }
        return;
    }
    if t < approach {
        let p = smooth(t / approach);
        pos = far_pos.lerp(near_pos, p);
    } else if t < approach + pause {
        phase_kind = "pause";
        pos = near_pos;
    } else {
        phase_kind = "retreat";
        let tr = t - approach - pause;
        let q = smooth(tr / retreat);
        pos = near_pos.lerp(far_pos, q);
    }
    if let Ok(mut tf) = q_cam.get_single_mut() {
        *tf = Transform::from_translation(pos).looking_at(Vec3::ZERO, Vec3::Y);
        // Gentle, subtle sway only in HalluFly (still toned down)
        if matches!(state.mode, CameraAnimMode::HalluFly) {
            let sway = (t * 0.4).sin() * 0.15;
            tf.rotate_local_y(sway);
            if phase_kind == "pause" {
                tf.rotate_local_x((t * 0.25).sin() * 0.05);
            }
        }
    }
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            depth_scale_attenuation: true,
            reverse_time_scroll: true,
            time_window: 600,
            show_center_axes: false,
            show_corner_axes: true,
            base_scale: 0.08,
            scale_factor: 0.55,
            max_scale: 1.2,
            spike_value: 12.0,
            energy_scale: 1.0,
            show_internal_grid: true,
            internal_grid_opacity: 0.12,
            internal_grid_stride: 1,
            wave_mode: WaveMode::Normal,
            show_volume_stats: true,
            show_energy_derivative: true,
            show_advanced_scales: false,
        }
    }
}

// Axis marker kinds for visibility toggling
#[derive(Component, Clone, Copy, PartialEq, Eq)]
enum AxisKind {
    Center,
    Corner,
}

#[derive(Component)]
struct AxisMarker(pub AxisKind);

#[derive(Component)]
struct SceneLight;

// Marker for internal grid line entities
#[derive(Component)]
struct InternalGridLine {
    axis: u8, // 0=X,1=Y,2=Z
    i: usize,
    j: usize,
}

#[derive(Resource, Default)]
struct LightControl {
    enabled: bool,
    // store original intensities so toggling restores them
    original_point: f32,
    original_spot: f32,
    original_directional: f32,
    original_ambient: f32,
    intensity_scale: f32,
}

fn axis_visibility_system(
    vis: Res<VisualizationConfig>,
    mut q: Query<(&AxisMarker, &mut Visibility)>,
) {
    if !vis.is_changed() {
        return;
    }
    for (marker, mut v) in q.iter_mut() {
        let show = match marker.0 {
            AxisKind::Center => vis.show_center_axes,
            AxisKind::Corner => vis.show_corner_axes,
        };
        *v = if show {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}

fn exposure_apply_system(
    exposure: Res<ExposureSettings>,
    light_control: Res<LightControl>,
    mut q_point: Query<&mut PointLight>,
    mut q_spot: Query<&mut SpotLight>,
    mut q_dir: Query<&mut DirectionalLight>,
    mut ambient: ResMut<AmbientLight>,
) {
    if !exposure.is_changed() && !light_control.is_changed() {
        return;
    }
    if !light_control.enabled {
        return;
    }
    let mult = 2f32.powf(exposure.ev) * light_control.intensity_scale;
    for mut pl in q_point.iter_mut() {
        pl.intensity = light_control.original_point * mult;
    }
    for mut sl in q_spot.iter_mut() {
        sl.intensity = light_control.original_spot * mult;
    }
    for mut dl in q_dir.iter_mut() {
        dl.illuminance = light_control.original_directional * mult;
    }
    ambient.brightness = light_control.original_ambient * mult;
}

// Apply config-driven visibility for internal grid lines and update opacity
fn internal_grid_system(
    vis: Res<VisualizationConfig>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut q: Query<(&InternalGridLine, &Handle<StandardMaterial>, &mut Visibility)>,
) {
    if !vis.is_changed() { return; }
    for (line, mat_h, mut visib) in q.iter_mut() {
        let show = vis.show_internal_grid
            && (line.i % vis.internal_grid_stride == 0)
            && (line.j % vis.internal_grid_stride == 0);
        *visib = if show { Visibility::Visible } else { Visibility::Hidden };
        if let Some(mat) = materials.get_mut(mat_h) {
            let mut c = mat.base_color;
            c.set_a(vis.internal_grid_opacity);
            mat.base_color = c;
        }
    }
}

// Camera
#[derive(Resource, Debug)]
struct ExposureSettings {
    ev: f32,             // exposure value in EV (stops)
    pending_apply: bool, // flag to reapply lights after change
}

impl Default for ExposureSettings {
    fn default() -> Self {
        Self {
            ev: 0.0,
            pending_apply: true,
        }
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    sim_cfg: Res<SimConfig>,
) {
    // Camera
    let cam_start = Transform::from_xyz(-4.0, 6.0, 14.0).looking_at(Vec3::ZERO, Vec3::Y);
    // Camera with initial tonemapping (ACES as neutral cinematic baseline) - avoid duplicate component in bundle
    let cam_id = commands.spawn(Camera3dBundle { transform: cam_start, ..default() }).id();
    commands.entity(cam_id).insert(Tonemapping::AcesFitted);
    // Insert camera animation state resource (after camera spawn)
    commands.insert_resource(CameraAnimState {
        base_transform: cam_start,
        ..default()
    });
    commands.spawn((
        PointLightBundle {
            point_light: PointLight {
                intensity: 7000.0,
                range: 80.0,
                shadows_enabled: true,
                ..default()
            },
            transform: Transform::from_xyz(8.0, 12.0, 8.0),
            ..default()
        },
        SceneLight,
    ));

    // Add a soft spotlight sweeping from above-front to emphasize form
    commands.spawn((
        SpotLightBundle {
            spot_light: SpotLight {
                intensity: 18000.0,
                color: Color::rgba(1.0, 0.95, 0.9, 1.0),
                outer_angle: 50_f32.to_radians(),
                inner_angle: 20_f32.to_radians(),
                range: 120.0,
                shadows_enabled: true,
                ..default()
            },
            transform: Transform::from_xyz(-14.0, 18.0, 6.0)
                .looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
            ..default()
        },
        SceneLight,
    ));

    // Initialize light control resource with original intensities
    commands.insert_resource(LightControl {
        enabled: true,
        original_point: 7000.0,
        original_spot: 18000.0,
        original_directional: 25000.0,
        original_ambient: 0.35,
        intensity_scale: 1.0,
    });
    commands.insert_resource(ExposureSettings::default());

    // Base sphere mesh reused (scaled per instance)
    let sphere = Mesh::from(Sphere { radius: 0.5 });
    let sphere_handle = meshes.add(sphere);
    // Opaque metallic spheres for clearer lighting response
    let base_mat = materials.add(StandardMaterial {
        base_color: Color::rgb_linear(0.70, 0.72, 0.75),
        metallic: 0.95,
        perceptual_roughness: 0.18,
        reflectance: 0.5,
        emissive: Color::BLACK,
        unlit: false,
        ..default()
    });

    // Add a directional light to create stronger specular highlights across spheres
    commands.spawn((
        DirectionalLightBundle {
            directional_light: DirectionalLight {
                illuminance: 25_000.0,
                shadows_enabled: true,
                ..default()
            },
            transform: Transform::from_xyz(-12.0, 24.0, 14.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        SceneLight,
    ));

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

    // Axes (X=Red, Y=Green, Z=Blue) and bounding box wireframe to reduce spatial disorientation
    let axis_radius = 0.02;
    let axis_len = (h.max(w).max(d) as f32) * 0.6 + 2.0;
    let axis_mesh = Mesh::from(Capsule3d {
        radius: axis_radius,
        half_length: axis_len * 0.5,
        ..default()
    });
    // X axis
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(axis_mesh.clone()),
            material: materials.add(StandardMaterial {
                base_color: Color::rgba(1.0, 0.1, 0.1, 1.0),
                unlit: true,
                ..default()
            }),
            transform: Transform::from_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2))
                .with_translation(Vec3::X * axis_len * 0.5),
            ..default()
        },
        AxisMarker(AxisKind::Center),
    ));
    // Y axis
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(axis_mesh.clone()),
            material: materials.add(StandardMaterial {
                base_color: Color::rgba(0.1, 1.0, 0.1, 1.0),
                unlit: true,
                ..default()
            }),
            transform: Transform::from_translation(Vec3::Y * axis_len * 0.5),
            ..default()
        },
        AxisMarker(AxisKind::Center),
    ));
    // Z axis
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(axis_mesh.clone()),
            material: materials.add(StandardMaterial {
                base_color: Color::rgba(0.1, 0.4, 1.0, 1.0),
                unlit: true,
                ..default()
            }),
            transform: Transform::from_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2))
                .with_translation(Vec3::Z * axis_len * 0.5),
            ..default()
        },
        AxisMarker(AxisKind::Center),
    ));

    // Bounding box wireframe: spawn thin boxes along edges
    let bx = w as f32 * 0.5;
    let by = h as f32 * 0.5;
    let bz = d as f32 * 0.5;
    // Corner axes (from minimum corner) for alternative reference
    let corner_origin = Vec3::new(-bx, -by, -bz);
    let corner_len = axis_len;
    let axis_mesh_corner = Mesh::from(Capsule3d {
        radius: axis_radius,
        half_length: corner_len * 0.5,
        ..default()
    });
    // X from corner
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(axis_mesh_corner.clone()),
            material: materials.add(StandardMaterial {
                base_color: Color::rgba(1.0, 0.4, 0.4, 0.8),
                unlit: true,
                ..default()
            }),
            transform: Transform::from_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2))
                .with_translation(corner_origin + Vec3::X * corner_len * 0.5),
            ..default()
        },
        AxisMarker(AxisKind::Corner),
    ));
    // Y
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(axis_mesh_corner.clone()),
            material: materials.add(StandardMaterial {
                base_color: Color::rgba(0.4, 1.0, 0.4, 0.8),
                unlit: true,
                ..default()
            }),
            transform: Transform::from_translation(corner_origin + Vec3::Y * corner_len * 0.5),
            ..default()
        },
        AxisMarker(AxisKind::Corner),
    ));
    // Z
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(axis_mesh_corner.clone()),
            material: materials.add(StandardMaterial {
                base_color: Color::rgba(0.4, 0.6, 1.0, 0.8),
                unlit: true,
                ..default()
            }),
            transform: Transform::from_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2))
                .with_translation(corner_origin + Vec3::Z * corner_len * 0.5),
            ..default()
        },
        AxisMarker(AxisKind::Corner),
    ));
    let edge_radius = 0.01;
    let edge_mesh = Mesh::from(Capsule3d {
        radius: edge_radius,
        half_length: 0.5,
        ..default()
    });
    let edge_mat = materials.add(StandardMaterial {
        base_color: Color::rgba(0.8, 0.8, 0.9, 0.4),
        unlit: true,
        alpha_mode: bevy::prelude::AlphaMode::Blend,
        ..default()
    });
    let mut spawn_edge = |a: Vec3, b: Vec3| {
        let dir = b - a;
        let len = dir.length();
        if len > 0.0 {
            let center = a + dir * 0.5;
            // Align capsule along dir using look_at then rotate to match capsule local up
            let rot = Quat::from_rotation_arc(Vec3::Y, dir.normalize());
            commands.spawn(PbrBundle {
                mesh: meshes.add(edge_mesh.clone()),
                material: edge_mat.clone(),
                transform: Transform::from_translation(center)
                    .with_rotation(rot)
                    .with_scale(Vec3::new(1.0, len, 1.0)),
                ..default()
            });
        }
    };
    // 12 edges of the box
    let corners = [
        Vec3::new(-bx, -by, -bz),
        Vec3::new(bx, -by, -bz),
        Vec3::new(-bx, by, -bz),
        Vec3::new(bx, by, -bz),
        Vec3::new(-bx, -by, bz),
        Vec3::new(bx, -by, bz),
        Vec3::new(-bx, by, bz),
        Vec3::new(bx, by, bz),
    ];
    // bottom rectangle
    spawn_edge(corners[0], corners[1]);
    spawn_edge(corners[1], corners[3]);
    spawn_edge(corners[3], corners[2]);
    spawn_edge(corners[2], corners[0]);
    // top rectangle
    spawn_edge(corners[4], corners[5]);
    spawn_edge(corners[5], corners[7]);
    spawn_edge(corners[7], corners[6]);
    spawn_edge(corners[6], corners[4]);
    // vertical edges
    spawn_edge(corners[0], corners[4]);
    spawn_edge(corners[1], corners[5]);
    spawn_edge(corners[2], corners[6]);
    spawn_edge(corners[3], corners[7]);

    // Internal lattice grid lines (skip outer shell). Provide visual depth cues.
    let grid_mesh = meshes.add(Mesh::from(Capsule3d { radius: 0.004, half_length: 0.5, ..default() }));
    let (h, w, d) = (sim_cfg.h, sim_cfg.w, sim_cfg.d);
    let len_x = w as f32;
    let len_y = h as f32;
    let len_z = d as f32;
    let base_mat = materials.add(StandardMaterial {
        base_color: Color::rgba(0.9, 0.9, 1.0, 0.12),
        unlit: true,
        alpha_mode: bevy::prelude::AlphaMode::Blend,
        ..default()
    });
    // X-axis lines
    if w > 1 { for z in 1..d.saturating_sub(1) { for y in 1..h.saturating_sub(1) {
        let yf = y as f32 - h as f32 * 0.5;
        let zf = z as f32 - d as f32 * 0.5;
        commands.spawn((PbrBundle {
            mesh: grid_mesh.clone(),
            material: base_mat.clone(),
            transform: Transform::from_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2))
                .with_translation(Vec3::new(0.0, yf, zf))
                .with_scale(Vec3::new(1.0, len_x, 1.0)),
            ..default()
        }, InternalGridLine { axis: 0, i: y, j: z }));
    }}}
    // Y-axis lines
    if h > 1 { for z in 1..d.saturating_sub(1) { for x in 1..w.saturating_sub(1) {
        let xf = x as f32 - w as f32 * 0.5;
        let zf = z as f32 - d as f32 * 0.5;
        commands.spawn((PbrBundle {
            mesh: grid_mesh.clone(),
            material: base_mat.clone(),
            transform: Transform::from_translation(Vec3::new(xf, 0.0, zf))
                .with_scale(Vec3::new(1.0, len_y, 1.0)),
            ..default()
        }, InternalGridLine { axis: 1, i: x, j: z }));
    }}}
    // Z-axis lines
    if d > 1 { for y in 1..h.saturating_sub(1) { for x in 1..w.saturating_sub(1) {
        let xf = x as f32 - w as f32 * 0.5;
        let yf = y as f32 - h as f32 * 0.5;
        commands.spawn((PbrBundle {
            mesh: grid_mesh.clone(),
            material: base_mat.clone(),
            transform: Transform::from_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2))
                .with_translation(Vec3::new(xf, yf, 0.0))
                .with_scale(Vec3::new(1.0, len_z, 1.0)),
            ..default()
        }, InternalGridLine { axis: 2, i: x, j: y }));
    }}}

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
            base_color: Color::rgba(0.06, 0.08, 0.10, 0.55), // semi-transparent, subtle tint
            perceptual_roughness: 1.0,
            metallic: 0.0,
            alpha_mode: bevy::prelude::AlphaMode::Blend,
            unlit: true, // keep it visually quiet
            ..default()
        }),
        ..default()
    });

    // ---- Lighting Debug: Textured rectangle (checkerboard) ----
    // Smaller vivid checkerboard so texture visibility is obvious (red vs blue)
    let tex_size = 64u32;
    let tile = 8; // tile size in pixels
    let mut data = Vec::with_capacity((tex_size * tex_size * 4) as usize);
    for y in 0..tex_size {
        for x in 0..tex_size {
            let a = (x / tile + y / tile) % 2 == 0;
            let (r, g, b) = if a {
                (210u8, 40u8, 40u8)
            } else {
                (40u8, 40u8, 210u8)
            };
            data.extend_from_slice(&[r, g, b, 255]);
        }
    }
    let image = Image::new(
        Extent3d {
            width: tex_size,
            height: tex_size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    let checker_handle = images.add(image);
    let quad_mesh = Mesh::from(bevy::prelude::Plane3d::default());
    let quad = meshes.add(quad_mesh);
    let quad_mat = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(checker_handle.clone()),
        unlit: true, // make sure texture pattern shows regardless of lighting for debugging
        ..default()
    });
    // Place it behind the voxel cube, much smaller so it does not dominate view
    commands.spawn(PbrBundle {
        mesh: quad,
        material: quad_mat,
        transform: Transform::from_xyz(0.0, 0.0, -10.0).with_scale(Vec3::new(6.0, 6.0, 6.0)),
        ..default()
    });
}

// Update simulation and sphere scales
fn sim_step(
    mut sim: ResMut<DLinOssSim>,
    mut spheres: Query<(&VoxelSphere, &mut Transform)>,
    sim_cfg: Res<SimConfig>,
    vis: Option<Res<VisualizationConfig>>,
    camera_q: Query<&Transform, (With<Camera3d>, Without<VoxelSphere>)>,
    anim: Option<Res<CameraAnimState>>,
    // local flag to detect if latent energy updated this frame
    mut latent_pushed: Local<bool>,
) {
    if sim_cfg.paused {
        return;
    }
    // Construct input batch [1,T,input_dim] incremental small window
    let chunk = 8usize;
    if sim.current_t >= sim.t_len {
        sim.current_t = 0;
    }
    let remaining = chunk; // always process a fixed chunk for steady rhythm
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
                    for val in vals.iter() { sim.output_ring.push(*val); }
                }
            }
            // Keep matching inputs (reuse slice used) approximate per-step first channel
            for tloc in 0..remaining {
                let v = sim.input_cache[(start + tloc) * sim.input_dim];
                sim.input_ring.push(v);
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
            if let Some(nv) = norm_val {
                // Push two samples per step for double-rate graph
                sim.latent_energy_ring.push(nv);
                sim.latent_energy_ring.push(nv);
                if nv > sim.latent_energy_peak {
                    sim.latent_energy_peak = nv;
                } else {
                    // slow decay of peak to allow scale to relax
                    sim.latent_energy_peak =
                        0.995 * sim.latent_energy_peak + 0.005 * nv.max(1e-6);
                }
                *latent_pushed = true;
            }
            if let Some(pp) = phase_pair_opt {
                let cap = sim.ring_capacity; // capture to appease borrow checker (already mutable used above)
                push_ring_phase(&mut sim.phase_points, pp, cap / 4);
            }
        }
    }
    // Ensure latent energy plot scrolls every frame (duplicate last if no new value)
    if !*latent_pushed {
        if let Some(last) = sim.latent_energy_ring.data.last().map(|s| s.value) {
            sim.latent_energy_ring.push(last);
            sim.latent_energy_ring.push(last);
        }
    }
    *latent_pushed = false;
    sim.current_t += remaining;
    // Update sphere transforms based on new volume values
    // Optional depth-based attenuation for better perspective size perception
    let cam_tf = camera_q.iter().next();
    let use_depth = vis
        .as_ref()
        .map(|v| v.depth_scale_attenuation)
        .unwrap_or(false);
    // Procedural continuous animation: update last_volume with a 3D multi-frequency wave
    // Use current_t as a proxy for time and advance a fractional phase each iteration.
    let tphase = sim_cfg.frame_counter as f32 * 0.09; // increased speed: 3x faster
    let (h, w, d) = (sim_cfg.h, sim_cfg.w, sim_cfg.d);
    let hallu = anim
        .as_ref()
        .map(|a| a.mode == CameraAnimMode::HalluFly)
        .unwrap_or(false);
    // New animation path uses explicit pause segment; derive simple pause factor when within pause window
    let pause_factor = if let Some(a) = anim.as_ref() {
        if matches!(a.mode, CameraAnimMode::FlyThrough | CameraAnimMode::HalluFly) {
            let cycle_t = a.t % a.total;
            if cycle_t >= a.approach && cycle_t < a.approach + a.pause {
                // normalized 0..1 across pause segment with cosine ease peak in middle
                let local = (cycle_t - a.approach) / a.pause;
                (std::f32::consts::PI * (local - 0.5)).cos().powi(2)
            } else {
                0.0
            }
        } else { 0.0 }
    } else { 0.0 };
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let idx = z * h * w + y * w + x;
                let xf = x as f32 / w as f32;
                let yf = y as f32 / h as f32;
                let zf = z as f32 / d as f32;
                let mut v = 0.0;
                let wave_mode = vis.as_ref().map(|v| v.wave_mode).unwrap_or(WaveMode::Normal);
                match wave_mode {
                    WaveMode::Normal => {
                        v = (tphase + xf * 6.28).sin() * (tphase * 2.1 + yf * 10.0).cos() * 0.5
                            + 0.5 * (tphase * 3.9 + zf * 12.57).sin();
                        if hallu {
                            let radial =
                                ((xf - 0.5).powi(2) + (yf - 0.5).powi(2) + (zf - 0.5).powi(2)).sqrt();
                            let swirl = (tphase * 2.4 + (xf + yf * 1.3 - zf) * 18.0).sin();
                            let breathe = (tphase * 1.7 + radial * 10.0).cos();
                            v += 0.6 * swirl * breathe;
                        }
                    }
                    WaveMode::ComplexWave => {
                        // Complex synthetic wave: sum of multiple sin/cos with phase offsets
                        let freq1 = 3.0 + 2.0 * (tphase * 0.6).sin();
                        let freq2 = 5.0 + 1.5 * (tphase * 0.9).cos();
                        let freq3 = 7.0 + 1.0 * (tphase * 1.2).sin();
                        let phase1 = tphase * 3.6 + xf * freq1 + yf * freq2;
                        let phase2 = tphase * 2.4 + zf * freq3 + xf * freq2;
                        let phase3 = tphase * 4.5 + yf * freq1 + zf * freq2;
                        v = 0.4 * (phase1).sin() + 0.3 * (phase2).cos() + 0.2 * (phase3).sin();
                        v += 0.15 * ((xf + yf + zf + tphase * 1.5) * 12.0).cos();
                    }
                }
                if let Some(slot) = sim.last_volume.get_mut(idx) {
                    *slot = 0.85 * *slot + 0.15 * v;
                }
            }
        }
    }
    let (base_scale, scale_factor, max_scale) = vis
        .as_ref()
        .map(|v| (v.base_scale, v.scale_factor, v.max_scale))
        .unwrap_or((0.05, 0.4, 0.8));
    for (sphere, mut tf) in spheres.iter_mut() {
        if let Some(val) = sim.last_volume.get(sphere.idx) {
            let mag = val.abs();
            let mut scale = (base_scale + mag * scale_factor).clamp(0.01, max_scale);
            if use_depth {
                if let Some(cam) = cam_tf {
                    let dist = cam.translation.distance(tf.translation).max(0.001);
                    let atten = (8.0 / dist).clamp(0.25, 1.2);
                    scale *= atten;
                }
            }
            if pause_factor > 0.0 {
                if sphere.idx % 97 == 0 {
                    scale *= 1.0 + pause_factor * 1.2; // up to 2.2x at peak pause
                }
            }
            tf.scale = Vec3::splat(scale);
        }
    }
    // Push camera distance to origin for third time-series pane
    if let Some(cam) = cam_tf {
    let dist = cam.translation.length();
    sim.camera_depth_ring.push(dist);
    }
    // --- Extra metrics ---
    // Volume statistics (mean & variance) using simple pass (avoid realloc) then push.
    if !sim.last_volume.is_empty() {
        let mut sum = 0.0f32; let mut sum2 = 0.0f32; let n = sim.last_volume.len() as f32;
        for &v in &sim.last_volume { sum += v; sum2 += v * v; }
        let mean = sum / n;
        let var = (sum2 / n - mean * mean).max(0.0);
        sim.volume_mean_ring.push(mean);
        sim.volume_var_ring.push(var);
    }
    // Latent energy derivative (approx) from latest peak-corrected nv (reusing last pushed value if exists)
    let prev_latent = sim.last_latent_energy;
    let latest_latent = sim.latent_energy_ring.data.last().map(|s| s.value);
    if let Some(last) = latest_latent {
        if prev_latent != 0.0 { sim.latent_energy_deriv_ring.push(last - prev_latent); }
        sim.last_latent_energy = last;
    }
}

// Egui UI (4 panes skeleton)
// OLD ui_system implementation removed.

// Clean ui_system implementation
fn ui_system(
    mut contexts: EguiContexts,
    mut sim_cfg: ResMut<SimConfig>,
    mut sim: ResMut<DLinOssSim>,
    mut vis: ResMut<VisualizationConfig>,
    mut anim: ResMut<CameraAnimState>,
    mut light_control: ResMut<LightControl>,
    mut q_point: Query<&mut PointLight>,
    mut q_spot: Query<&mut SpotLight>,
    mut q_dir: Query<&mut DirectionalLight>,
    mut ambient: ResMut<AmbientLight>,
    mut exposure: ResMut<ExposureSettings>,
    mut q_cam_tone: Query<&mut Tonemapping, With<Camera3d>>,
) {
    let ctx = contexts.ctx_mut();
    egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.heading("D-LinOSS 3D Playground");
                ui.separator();
                ui.label(format!("Frame: {}", sim_cfg.frame_counter));
                ui.label(format!("Grid: {}x{}x{}", sim_cfg.h, sim_cfg.w, sim_cfg.d));
            });
            ui.horizontal_wrapped(|ui| {
                if ui.button(if sim_cfg.paused { "Run Simulation" } else { "Pause Simulation" }).clicked() { sim_cfg.paused = !sim_cfg.paused; }
                if ui.button("Restart").clicked() {
                    sim.current_t = 0; sim.last_volume.fill(0.0); sim.output_ring.clear(); sim.input_ring.clear();
                    sim.latent_energy_ring.clear(); sim.phase_points.clear(); sim.state_snapshot = None; sim.spectrum_cache.clear();
                    sim_cfg.frame_counter = 0; sim_cfg.paused = false; sim.latent_energy_peak = 1e-6;
                }
                if ui.button("Spike Cell").clicked() { if let Some(first) = sim.last_volume.get_mut(0) { *first = vis.spike_value; }}
                if ui.button(if sim_cfg.capture { "Stop Capture" } else { "Start Capture" }).clicked() {
                    sim_cfg.capture = !sim_cfg.capture;
                    if sim_cfg.capture { sim_cfg.capture_start_frame = Some(sim_cfg.frame_counter); info!("Capture started (prefix={}, max {} frames)", sim_cfg.capture_prefix, sim_cfg.max_capture_frames); }
                    else { info!("Capture stopped"); }
                }
                if ui.button("AutoRotate").clicked() { anim.t = 0.0; anim.mode = CameraAnimMode::AutoRotate; }
                if ui.button(match anim.mode { CameraAnimMode::Idle => "Fly Through", _ => "Stop Fly" }).clicked() { anim.t = 0.0; anim.mode = match anim.mode { CameraAnimMode::Idle => CameraAnimMode::FlyThrough, _ => CameraAnimMode::Idle }; }
                if ui.button(match anim.mode { CameraAnimMode::HalluFly => "Stop HalluFly", _ => "HalluFly" }).clicked() { anim.t = 0.0; anim.mode = match anim.mode { CameraAnimMode::HalluFly => CameraAnimMode::Idle, _ => CameraAnimMode::HalluFly }; }
            });
            ui.horizontal_wrapped(|ui| {
                if ui.button(if light_control.enabled { "Lighting ON (toggle)" } else { "Lighting OFF (toggle)" }).clicked() {
                    light_control.enabled = !light_control.enabled;
                    if light_control.enabled {
                        for mut pl in q_point.iter_mut() { pl.intensity = light_control.original_point * light_control.intensity_scale; }
                        for mut sl in q_spot.iter_mut() { sl.intensity = light_control.original_spot * light_control.intensity_scale; }
                        for mut dl in q_dir.iter_mut() { dl.illuminance = light_control.original_directional * light_control.intensity_scale; }
                        ambient.brightness = light_control.original_ambient * light_control.intensity_scale; info!("Lighting toggled ON");
                    } else {
                        for mut pl in q_point.iter_mut() { pl.intensity = 0.0; }
                        for mut sl in q_spot.iter_mut() { sl.intensity = 0.0; }
                        for mut dl in q_dir.iter_mut() { dl.illuminance = 0.0; }
                        ambient.brightness = 0.0; info!("Lighting toggled OFF");
                    }
                }
                ui.add(egui::Slider::new(&mut light_control.intensity_scale, 0.05..=3.0).text("Light scale"));
                if ui.button("Apply Scale").clicked() && light_control.enabled {
                    for mut pl in q_point.iter_mut() { pl.intensity = light_control.original_point * light_control.intensity_scale; }
                    for mut sl in q_spot.iter_mut() { sl.intensity = light_control.original_spot * light_control.intensity_scale; }
                    for mut dl in q_dir.iter_mut() { dl.illuminance = light_control.original_directional * light_control.intensity_scale; }
                    ambient.brightness = light_control.original_ambient * light_control.intensity_scale;
                }
                ui.add(egui::Slider::new(&mut exposure.ev, -4.0..=4.0).text("Exposure EV"));
                if ui.button("Tone Reinhard").clicked() { for mut t in q_cam_tone.iter_mut() { *t = Tonemapping::Reinhard; }}
                let pt_sum: f32 = q_point.iter().map(|p| p.intensity).sum();
                let sp_sum: f32 = q_spot.iter().map(|s| s.intensity).sum();
                let dir_sum: f32 = q_dir.iter().map(|d| d.illuminance).sum();
                ui.label(format!("Lights {} | Pt {:.0} Sp {:.0} Dir {:.0} Amb {:.2}", if light_control.enabled { "ON" } else { "OFF" }, pt_sum, sp_sum, dir_sum, ambient.brightness));
            });
        });
    });

    egui::SidePanel::left("left_panel").show(ctx, |ui| {
        ui.heading("Time Series");
        timed_plot_dual(ui, "Input/Output", &sim.input_ring, &sim.output_ring, 160.0, vis.reverse_time_scroll, vis.time_window);
        ui.separator();
        ui.heading("Latent Energy (Approx)");
        timed_plot_single(ui, "Energy (||x||)", &sim.latent_energy_ring, 120.0, egui::Color32::RED, vis.reverse_time_scroll, vis.time_window, Some(sim.latent_energy_peak), vis.energy_scale);
        ui.separator();
        ui.heading("Camera Depth");
        timed_plot_single(ui, "Depth (cam dist)", &sim.camera_depth_ring, 100.0, egui::Color32::LIGHT_GREEN, vis.reverse_time_scroll, vis.time_window, None, 1.0);
        if vis.show_volume_stats {
            ui.separator();
            ui.heading("Volume Mean / Var");
            timed_plot_dual(ui, "Mean/Var", &sim.volume_mean_ring, &sim.volume_var_ring, 120.0, vis.reverse_time_scroll, vis.time_window);
        }
        if vis.show_energy_derivative {
            ui.separator();
            ui.heading("Latent Energy Derivative");
            timed_plot_single(ui, "dEnergy", &sim.latent_energy_deriv_ring, 100.0, egui::Color32::from_rgb(180,140,255), vis.reverse_time_scroll, vis.time_window, None, 1.0);
        }
    });

    egui::SidePanel::right("right_panel").show(ctx, |ui| {
        ui.heading("3D Controls");
        ui.label("Visualization Toggles:");
        ui.checkbox(&mut vis.depth_scale_attenuation, "Depth size attenuation");
        ui.checkbox(&mut vis.reverse_time_scroll, "Time scroll RTL (legacy flag)");
        ui.add(egui::Slider::new(&mut vis.time_window, 100..=4096).text("Time window"));
        ui.checkbox(&mut vis.show_center_axes, "Center axes");
        ui.checkbox(&mut vis.show_corner_axes, "Corner axes");
        ui.checkbox(&mut vis.show_internal_grid, "Internal voxel grid");
        ui.add(egui::Slider::new(&mut vis.internal_grid_opacity, 0.0..=1.0).text("Grid opacity"));
        ui.add(egui::Slider::new(&mut vis.internal_grid_stride, 1..=8).text("Grid stride"));
        ui.add(egui::Slider::new(&mut vis.energy_scale, 0.1..=10.0).logarithmic(true).text("Energy scale"));
        ui.separator();
        ui.label("Sphere Size Scaling:");
        ui.add(egui::Slider::new(&mut vis.base_scale, 0.005..=0.5).text("Base"));
        ui.add(egui::Slider::new(&mut vis.scale_factor, 0.05..=2.0).text("Gain"));
        ui.add(egui::Slider::new(&mut vis.max_scale, 0.1..=4.0).text("Max"));
        ui.add(egui::Slider::new(&mut vis.spike_value, 1.0..=100.0).logarithmic(true).text("Spike val"));
        ui.horizontal(|ui| {
            ui.label("Wave Mode:");
            if ui.button(match vis.wave_mode { WaveMode::Normal => "Normal", WaveMode::ComplexWave => "Complex Wave" }).clicked() {
                vis.wave_mode = match vis.wave_mode { WaveMode::Normal => WaveMode::ComplexWave, WaveMode::ComplexWave => WaveMode::Normal };
            }
        });
        ui.separator();
        ui.collapsing("Advanced Scale & Metrics", |ui| {
            ui.checkbox(&mut vis.show_volume_stats, "Show volume mean/var");
            ui.checkbox(&mut vis.show_energy_derivative, "Show energy derivative");
            ui.label("(Size sliders duplicated above for quick access)");
        });
    });
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
// Timed plotting (handles monotonic indices, gaps, reverse scroll)
fn timed_plot_dual(
    ui: &mut egui::Ui,
    title: &str,
    a: &TimedRing,
    b: &TimedRing,
    height: f32,
    reverse: bool,
    max_visible: usize,
) {
    ui.label(title);
    let desired = egui::Vec2::new(ui.available_width(), height);
    let (rect, _resp) = ui.allocate_exact_size(desired, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_stroke(rect, 0.0, egui::Stroke::new(1.0, egui::Color32::DARK_GRAY));
    if a.is_empty() && b.is_empty() { return; }
    let latest = a.latest_idx().max(b.latest_idx());
    let need_scroll = (latest as usize) >= max_visible;
    let visible_start = if need_scroll { latest.saturating_sub(max_visible as u64) } else { 0 };
    // Collect visible samples with normalized x [0,1]
    let series = [(a, egui::Color32::LIGHT_BLUE), (b, egui::Color32::GOLD)];
    // Determine min/max values
    let mut min_v = f32::MAX;
    let mut max_v = f32::MIN;
    for (ring, _) in &series {
        for s in ring.iter().filter(|s| s.idx >= visible_start) {
            if s.value < min_v { min_v = s.value; }
            if s.value > max_v { max_v = s.value; }
        }
    }
    if max_v <= min_v { max_v = min_v + 1e-6; }
    for (ring, color) in &series {
        let mut pts: Vec<egui::Pos2> = Vec::new();
        let mut last_idx_opt: Option<u64> = None;
        for samp in ring.iter().filter(|s| s.idx >= visible_start) {
            // Break on gaps >1 to avoid vertical lines
            if let Some(last) = last_idx_opt { if samp.idx > last + 1 { if pts.len()>1 { painter.add(egui::Shape::line(pts.clone(), egui::Stroke::new(1.0,*color))); } pts.clear(); } }
            last_idx_opt = Some(samp.idx);
            let domain_span = if need_scroll { max_visible as u64 } else { max_visible as u64 }; // constant span for clarity
            let rel = if need_scroll {
                (samp.idx - visible_start) as f32 / (domain_span.max(1) as f32)
            } else {
                // pad left (or right) with blank until filled
                (samp.idx as f32) / (domain_span.max(1) as f32)
            };
            // Always scroll left-to-right after initial blank phase
            let x = rel;
            let norm = (samp.value - min_v) / (max_v - min_v);
            let pos = egui::pos2(rect.left() + x * rect.width(), rect.bottom() - norm * rect.height());
            pts.push(pos);
        }
        if pts.len() > 1 { painter.add(egui::Shape::line(pts, egui::Stroke::new(1.0, *color))); }
    }
    // Draw an outline baseline tick markers for 0..max_visible samples (optional lightweight)
}

fn timed_plot_single(
    ui: &mut egui::Ui,
    title: &str,
    ring: &TimedRing,
    height: f32,
    color: egui::Color32,
    reverse: bool,
    max_visible: usize,
    fixed_peak: Option<f32>,
    scale: f32,
) {
    ui.label(title);
    let desired = egui::Vec2::new(ui.available_width(), height);
    let (rect, _resp) = ui.allocate_exact_size(desired, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_stroke(rect, 0.0, egui::Stroke::new(1.0, egui::Color32::DARK_GRAY));
    if ring.is_empty() { return; }
    let latest = ring.latest_idx();
    let need_scroll = (latest as usize) >= max_visible;
    let visible_start = if need_scroll { latest.saturating_sub(max_visible as u64) } else { 0 };
    let mut min_v = f32::MAX;
    let mut max_v = f32::MIN;
    let mut sum = 0.0f32; let mut count = 0usize;
    for s in ring.iter().filter(|s| s.idx >= visible_start) {
        let v = s.value * scale;
        if v < min_v { min_v = v; }
        if v > max_v { max_v = v; }
        sum += v; count += 1;
    }
    if let Some(pk) = fixed_peak { let pk = pk * scale; if pk > max_v { max_v = pk; } }
    if max_v <= min_v { max_v = min_v + 1e-6; }
    let mut pts: Vec<egui::Pos2> = Vec::new();
    let mut last_idx_opt: Option<u64> = None;
    for samp in ring.iter().filter(|s| s.idx >= visible_start) {
        if let Some(last) = last_idx_opt { if samp.idx > last + 1 { if pts.len()>1 { painter.add(egui::Shape::line(pts.clone(), egui::Stroke::new(1.0,color))); } pts.clear(); } }
        last_idx_opt = Some(samp.idx);
        let domain_span = max_visible as u64; // fixed domain length
        let rel = if need_scroll {
            (samp.idx - visible_start) as f32 / (domain_span.max(1) as f32)
        } else {
            (samp.idx as f32) / (domain_span.max(1) as f32)
        };
    // Always scroll left-to-right after initial blank phase
    let x = rel;
        let v = samp.value * scale;
        let norm = (v - min_v) / (max_v - min_v);
        let pos = egui::pos2(rect.left() + x * rect.width(), rect.bottom() - norm * rect.height());
        pts.push(pos);
    }
    if pts.len() > 1 { painter.add(egui::Shape::line(pts, egui::Stroke::new(1.0, color))); }
    if count > 1 {
        let mean = sum / count as f32;
        let norm_mean = (mean - min_v)/(max_v-min_v);
        let y = rect.bottom() - norm_mean * rect.height();
        painter.line_segment([egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)], egui::Stroke::new(1.0, egui::Color32::from_gray(110)));
    }
}


fn frame_counter(mut sim_cfg: ResMut<SimConfig>) { sim_cfg.frame_counter = sim_cfg.frame_counter.wrapping_add(1); }

fn capture_system(mut _sim_cfg: ResMut<SimConfig>) { /* stub: implement frame capture later */ }

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
    let ring_capacity = 4096;
    let sim_res = DLinOssSim {
        layer,
        last_volume: volume,
        device: device.clone(),
        input_dim: cfg.input_dim,
        t_len: 10_000,
        current_t: 0,
        input_cache: vec![0f32; 10_000 * cfg.input_dim],
        output_ring: TimedRing::new(ring_capacity),
        input_ring: TimedRing::new(ring_capacity),
        ring_capacity: ring_capacity,
        latent_energy_ring: TimedRing::new(ring_capacity),
        phase_points: Vec::with_capacity(4096),
        latent_pair: (0, 1),
        state_snapshot: None,
        spectrum_cache: Vec::with_capacity(1024),
        latent_energy_peak: 1e-6,
        camera_depth_ring: TimedRing::new(ring_capacity),
    volume_mean_ring: TimedRing::new(ring_capacity),
    volume_var_ring: TimedRing::new(ring_capacity),
    latent_energy_deriv_ring: TimedRing::new(ring_capacity),
    last_latent_energy: 0.0,
    };
    // Generate session prefix (UTC seconds) for capture naming
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let sim_cfg = SimConfig {
        h,
        w,
        d,
        paused: false,
        capture: false,
        frame_counter: 0,
        capture_start_frame: None,
        max_capture_frames: 300, // ~5s at 60fps
        capture_prefix: format!("sess{}", ts),
    };

    let headless = env::var("DLINOSS_HEADLESS")
        .ok()
        .map(|v| v == "1")
        .unwrap_or(false);
    let mut app = App::new();
    app.insert_resource(sim_cfg)
        .insert_resource(sim_res)
        .insert_resource(CameraOrbitController::default());
    app.insert_resource(VisualizationConfig::default());

    if headless {
        // In headless mode use MinimalPlugins to avoid initializing render resources
        // (PipelineCache, shaders, etc.) which require a window. This allows running
        // update systems for a fixed number of frames in CI or headless environments.
        app.add_plugins(MinimalPlugins);
    } else {
        app.add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "D-LinOSS 3D Playground".into(),
                resolution: (1380.0, 900.0).into(),
                ..default()
            }),
            ..default()
        }));
    }

    // Only add Egui and UI systems when not running headless. In headless mode
    // there's no primary window / Egui contexts and calling ctx_mut() panics.
    if !headless {
        app.add_plugins(EguiPlugin)
            .add_systems(Startup, setup)
            .add_systems(
                Update,
                (
                    sim_step,
                    ui_system,
                    exposure_apply_system,
                    capture_system,
                    frame_counter,
                    camera_orbit_system,
                    axis_visibility_system,
                    internal_grid_system,
                    camera_anim_system,
                ),
            );
        // Ambient light for subtle base illumination (helps metallic translucent spheres)
        app.insert_resource(AmbientLight {
            color: Color::rgb_linear(0.25, 0.27, 0.30),
            brightness: 0.35,
        });
    } else {
        // Headless: register only non-UI systems. Do not run `setup` because it
        // depends on render assets (Mesh/Material) which are unavailable with
        // MinimalPlugins in headless mode.
        app.add_systems(Update, (sim_step, capture_system, frame_counter));
    }
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
