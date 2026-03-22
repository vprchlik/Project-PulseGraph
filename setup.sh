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
    echo "Created .env from .env.example -- edit it with your API keys"
fi

# 5. Frontend
if [ -d frontend ]; then
    echo "Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys (GITHUB_TOKEN, etc.)"
echo "  2. Run the feasibility study:"
echo "     source .venv/bin/activate"
echo "     python scripts/phase0_feasibility.py --data-source synthetic --baselines-only"
echo "  3. Run tests:"
echo "     pytest tests/ -v"
echo "  4. Start the API:"
echo "     uvicorn pulsegraph.api.app:app --reload --port 8000"
echo "  5. Start the frontend:"
echo "     cd frontend && npm run dev"
