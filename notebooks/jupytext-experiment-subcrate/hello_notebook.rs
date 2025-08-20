// %% [markdown]
// # Hello Notebook (Jupytext + Rust)
//
// This file is paired with an `.ipynb` via Jupytext (`.jupytext.toml`: `formats = "ipynb,rs:percent"`).
// You can open the paired notebook for rich outputs while keeping this plain Rust file for rust-analyzer.
//
// Run as a binary:
// ```bash
// cargo run -p jupytext-experiment-subcrate --bin hello_notebook
// ```
//
// Notes:
// - Keep the code in standard Rust form so rust-analyzer works.
// - Use simple `// %%` cell markers for sections.

// %%
fn main() -> anyhow::Result<()> {
    println!("Hello from a Jupytext-paired Rust notebook! ✨");

    // %% [markdown]
    // ## A tiny computation
    // Below we do a simple calculation and print the result. In a notebook, this could be visualized.

    // %%
    let xs: Vec<i32> = (0..10).collect();
    let sum: i32 = xs.iter().sum();
    println!("Sum 0..9 = {}", sum);

    // %% [markdown]
    // ### Next steps
    // - Add plotting by emitting CSV or using a minimal terminal-based plot.
    // - Split code into more cells using `// %%` markers.

    Ok(())
}

// %% [markdown]
// # Hello Notebook (Jupytext + Rust)
//
