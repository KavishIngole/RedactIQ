#!/usr/bin/env bash
# RedactIQ - Setup and run script for Intel Xeon environments
# This script provisions the environment and starts the server.

set -euo pipefail

echo "=== RedactIQ Setup ==="

# 1. Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# 2. Install dependencies
echo "Installing dependencies..."
pip install -e ".[dev]"

# 3. (Optional) Install Intel extensions
if [ "${INSTALL_INTEL_EXTENSIONS:-0}" = "1" ]; then
    echo "Installing Intel extensions..."
    pip install intel-extension-for-pytorch oneccl-bind-pt
fi

# 4. Generate synthetic data if not present
if [ ! -f "data/train.jsonl" ]; then
    echo "Generating synthetic training data..."
    python -m redactiq.data.generate
fi

# 5. Set Intel CPU optimizations
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(nproc)}
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1

echo ""
echo "=== Setup complete ==="
echo ""
echo "Available commands:"
echo "  python -m redactiq.data.generate     # Generate synthetic data"
echo "  python scripts/fine_tune.py               # Fine-tune LLM with LoRA"
echo "  python scripts/train_anomaly.py           # Train anomaly detector"
echo "  python -m redactiq.serving.api        # Start API server"
echo "  python -m redactiq.ui.app                 # Start Gradio UI"
echo "  python scripts/evaluate.py                # Run evaluation"
echo "  pytest tests/                             # Run tests"
