# Jupytext Experiment Sub-crate

A minimal Cargo crate to experiment with Jupytext-paired Rust notebooks that still work great with rust-analyzer.

Why?
- Keep editing and linting in a normal Rust file (`.rs`) so rust-analyzer shines.
- Pair it with an `.ipynb` using Jupytext for rich output (plots, tables) when desired.

## How pairing works
- This folder has `.jupytext.toml` with:
  
  ```toml
  formats = "ipynb,rs:percent"
  ```
  
- The `percent` format uses `// %%` cell markers in `.rs`. Jupytext can convert back and forth:
  - `hello_notebook.rs` <-> `hello_notebook.ipynb`

## Try it
- Ensure your VS Code Jupytext Sync is configured to use a Python with `jupytext` installed.
- Open `hello_notebook.rs` and run: 

```bash
cargo run -p jupytext-experiment-subcrate --bin hello_notebook
```

- To create the paired `.ipynb`, in VS Code use the Jupytext Sync command (or CLI):

```bash
# CLI (optional if you have jupytext installed in ~/.uservenv)
~/.uservenv/bin/python -m jupytext --to ipynb hello_notebook.rs
```

Open `hello_notebook.ipynb` to see the cells and run them with a notebook kernel (e.g., Rust evcxr if desired).

## Notes
- Keep code standard Rust, not evcxr directives, for rust-analyzer friendliness.
- If you want to use the evcxr kernel in the notebook, place evcxr-specific directives only in the `.ipynb` side or gated blocks.
