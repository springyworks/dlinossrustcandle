//! D-LinOSS on Candle: Damped Linear Oscillatory State-Space layer

pub mod dlinoss;

pub use dlinoss::{DLinOssLayer, DLinOssLayerConfig};
// Re-export core Candle types for convenience in notebooks and downstream crates
pub use candlekos::{DType, Device, Tensor};

// Experimental helpers are provided via the dlinoss-augment sub-crate which wraps Candle ops.
#[cfg(feature = "fft")]
pub use dlinoss_augment::TensorFftExt;
pub use dlinoss_augment::TensorScanExt;

#[cfg(feature = "fft")]
pub mod kernelize;

// Optional display and helpers crates are re-exported for convenience
#[cfg(feature = "egui")]
pub use dlinoss_display::egui_dual as display_egui;
#[cfg(feature = "minifb")]
pub use dlinoss_display::mini as display_minifb;
#[cfg(feature = "etui")]
pub use dlinoss_display::tui as display_tui;

#[cfg(feature = "cli")]
pub use dlinoss_helpers as helpers;
