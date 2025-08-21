# Bevy + Egui Lighting & Texturing Integration Report

This report distills the lessons and working patterns derived from constructing the standalone lighting lab used to diagnose “flat” rendering appearances when mixing Bevy 3D (PBR) with an egui overlay.

## 1. Root Causes of "No Lighting" / Flat Look

| Symptom | Underlying Cause | Resolution |
|---------|------------------|-----------|
| Meshes look uniformly shaded (plasticky grey) | Single weak light, no specular range, default tonemapping compressing highlights | Add calibrated light rig (directional + point/spot fill) & raise intensities to realistic lumen / lux ranges |
| Colors appear dull / washed | sRGB vs linear misunderstanding or tonemapping curve clipping | Ensure base_color textures are in `Rgba8UnormSrgb`; keep `StandardMaterial` values in linear; experiment with ACES / AgX |
| Texture detail not visible | Using fully metallic + low roughness without enough light variation | Provide contrasty checkerboard + unlit reference patch |
| Lighting disappeared while interacting with UI | Camera or orbit system blocked by egui pointer gating incorrectly | Gate camera input on `egui_ctx.is_pointer_over_area()` only, not entire update logic |
| Exposure too dark / too bright after edits | Directly scaling light intensities destructively | Adopt non-destructive exposure model (store original intensities + EV factor) |

## 2. Recommended Light Rig (Baseline)

Use at least:
- Directional Light: key illumination (e.g., illuminance 20_000 – 35_000 lux) with shadows.
- Point Light: warm fill (intensity 5_000 – 9_000) at an offset above/side.
- Spot Light: angled highlight (intensity 12_000 – 18_000) with inner/outer cone to rake surfaces.
- Ambient Light: low level (0.25–0.40 brightness linear) for softened dark areas (avoid crushing blacks).

Keep a global intensity scaling slider (0.05–3.0) to adapt to different monitors / HDR conditions and a master ON/OFF toggle. Store original intensities so toggling is reversible.

## 3. Tonemapping & Exposure

Bevy 0.13 supports multiple tonemappers (`Tonemapping` component). Practical set:
`Reinhard`, `BlenderFilmic`, `AcesFitted`, `AgX`, `TonyMcMapface`, `SomewhatBoringDisplayTransform`.

UI Pattern:
- Combo box to pick `Tonemapping` enum variant (applied to camera entity).
- Exposure EV slider (e.g., -8.0 .. +8.0). Keep exposure separate: do NOT permanently scale lights—apply a derived multiplier each frame or store base intensities and reapply after EV change.

Interim (destructive) method acceptable for prototyping but switch to non-destructive before content capture.

## 4. Procedural Texture & Unlit Reference

Provide a small, vivid checkerboard quad using `Rgba8UnormSrgb`. Two variants:
- Lit material (PBR) to check shadow & specular.
- Unlit material (set `unlit: true`) for absolute color reference (verifies tonemapper impact versus raw sRGB expectation).

## 5. Material Diagnostic Grid

Spawn grid of spheres varying `(metallic, perceptual_roughness)` to visualize highlight shape & microfacet response. Example loops:
```
for (i, metallic) in [0.0, 0.25, 0.5, 0.75, 1.0].iter().enumerate() {
  for (j, rough) in [0.05, 0.15, 0.35, 0.55, 0.75].iter().enumerate() { /* spawn */ }
}
```
Use neutral base color (0.70–0.75 linear) and high reflectance (0.5) initially.

## 6. Orbit + Fly-Through Cameras with Egui

Key points:
- Orbit (turntable) system ignores input if egui wants pointer.
- Fly-through path disables manual orbit to avoid conflicting transforms.
- Smooth parametric path uses eased phase variable; jitter & pauses add cinematic variation.
- Always call `look_at(target, Vec3::Y)` after updating translation when not adding local rotations.

## 7. Gentle Animation Patterns

Avoid violent motion that obscures lighting evaluation. Use low-frequency sinusoidal bobbing or slow rotations. If multiple systems need mutable `Transform` on same entities, use a `ParamSet` to prevent query conflicts rather than merging unrelated logic.

## 8. Non-Destructive Exposure Strategy (Target Refactor)

Current (prototype): directly scales light component intensities—compounds error after multiple EV tweaks.

Preferred:
1. Store `base_*` intensities in a resource.
2. Maintain `exposure_ev: f32`.
3. Derived multiplier = `2f32.powf(exposure_ev)`. (One EV = doubling.)
4. Each frame (or on change) set: `point_light.intensity = base_point * multiplier`, etc.

## 9. Migration Checklist (Playground Adoption)

1. Update `Cargo.toml`: add `tonemapping_luts` feature to Bevy (enables extended tonemappers).
2. Add `Tonemapping` + (later) custom `Exposure` component to camera at spawn.
3. Introduce `LightControl` resource with original intensities + scale + enabled flag.
4. Ensure ambient light resource inserted early.
5. Add UI controls: lighting toggle, scale slider, tonemapping combo, exposure EV slider.
6. Add checkerboard quad + (optional) material grid for reference.
7. Verify orbit controller gating with egui pointer tests.
8. Validate that textures use sRGB formats and materials keep `unlit=false` except for reference patch.
9. Capture baseline screenshots across tonemappers for visual regression.
10. Implement non-destructive exposure before final capture/export.

## 10. Common Pitfalls & Resolutions

| Pitfall | Fix |
|--------|-----|
| Duplicate component (e.g., `Tonemapping`) causing panic | Ensure only one instance per camera; update existing component instead of inserting anew |
| Mutably borrowing `Transform` from multiple queries | Use `ParamSet` or consolidate logic; never attempt two mutable queries on same component set in one system |
| Washed highlights after brightening | Try `AgX` or `ACES` tonemapping; reduce global scale, adjust exposure EV negative |
| No specular sparkle | Increase directional light illuminance; ensure roughness not too high; confirm material not `unlit` |
| Exposure flicker | Avoid per-frame random scaling; only change on user input |

## 11. Future Enhancements
- Light presets (Studio, Rim, Night, HDR test).
- Per-light color pickers with Kelvin-to-RGB approximation slider.
- Histogram or luminance heatmap overlay.
- Auto-exposure prototype (average log luminance target).
- Panning (shift + middle-drag) support for orbit camera.
- Spline-defined camera keyframe editor stored in ron/json.

## 12. Reference Snippet: Camera Spawn with Tonemapping
```rust
commands.spawn((
    Camera3dBundle { 
        transform: start_tf,
        ..default()
    },
    Tonemapping::AcesFitted,
));
```
(Attach exposure resource/system separately.)

---
Prepared for D-LinOSS Playground integration. Use this as living documentation; update as migration proceeds.
