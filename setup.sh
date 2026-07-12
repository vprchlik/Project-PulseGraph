#!/usr/bin/env bash
set -euo pipefail

echo "=== PulseGraph Setup ==="

# 1. Git setup
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
    git remote add origin https://github.com/vprchlik/Project-PulseGraph.git
    echo "Git initialized with remote: https://github.com/vprchlik/Project-PulseGraph.git"
else
    echo "Git already initialized"
fi

# 2. Python environment
if [ ! -d .venv ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing Python dependencies..."
pip install -e ".[dev]"

# 3. Create data directories
mkdir -p data/raw data/processed data/cache reports

# 4. Environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example -- set GITHUB_TOKEN if collecting new data"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Run tests:"
echo "     source .venv/bin/activate"
echo "     pytest tests/ -q"
echo "  2. (Optional) Set GITHUB_TOKEN in .env, then ingest data:"
echo "     python scripts/ingest_stars.py --per-regime 80"
echo "  3. Run the primary experiment:"
echo "     python scripts/run_experiment.py --include-chronos --tag real_full"
echo "  4. See README.md and RESULTS.md for full reproduction and audit scripts."
