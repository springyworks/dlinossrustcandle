//! Minimal references to dev-only patched crates so Cargo considers them used.
//! Intentionally trivial to avoid build weight while keeping patches exercised.

#[test]
fn touch_candle_notebooks_and_transformers() {
    // Touch a couple of public modules/types to mark usage.
    // Avoiding heavy runtime work; this is compile-time linkage.
    use candle_notebooks as _nb; // alias import proves the crate resolves
    use candle_transformers as _ct; // likewise for transformers

    // Also touch a submodule path name to ensure the crate graph walks it.
    #[allow(unused)]
    fn _noop<T>(_t: T) {}
    _noop(_nb::ah::anyhow!("ok"));
    // Use a path symbol from transformers to ensure linkage
    #[allow(unused)]
    use _ct::utils as _ct_utils;
}
