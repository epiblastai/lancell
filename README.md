# lancell

Multimodal cell database built on LanceDB and Zarr.

Lancell stores single-cell data across heterogeneous assays (gene expression, protein abundance, chromatin accessibility, imaging) in a unified system optimized for both interactive queries and large-scale training workloads.

## Setup

Requires Python 3.13 and a Rust toolchain.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install Python deps
uv sync

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build the Rust extension
maturin develop --release
```
