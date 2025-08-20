# Contributing

Thanks for your interest in improving this project! We welcome issues, ideas, and PRs.

## Development workflow

- Use the xtask entrypoints:
  - `cargo run -p xtask -- ci` (fmt + clippy + tests)
  - `cargo run -p xtask -- verify-candle --fft` (Candle probe)
  - `cargo run -p xtask -- discover-tests --run --features fft`
- Keep clippy clean (`-D warnings`). Prefer small, focused PRs.
- Add doc comments to new modules and public items.

## Commit style

- Conventional but lightweight: feat:, fix:, docs:, refactor:, chore:, test:
- Mention relevant paths/files.

## Testing

- Add unit/integration tests for new features; ensure CPU-only tests pass by default.
- Feature-gated tests (e.g. `fft`) are welcome; ensure they’re optional.

## Notebooks

- Follow the single-:dep policy using the glue crate (`dlinoss-notebooks`).
- Place notebooks under `notebooks/` and keep them runnable with CPU defaults.

## Code of Conduct

- See CODE_OF_CONDUCT.md.

## License

- By contributing, you agree your contributions are licensed under MIT OR Apache-2.0.
