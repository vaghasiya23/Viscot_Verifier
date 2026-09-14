#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=== Verifier Pipeline 2.0 Environment Setup ==="

# Check for uv
if ! command -v uv &>/dev/null; then
    if [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "Installing uv package manager..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

echo "Creating Python 3.11 virtual environment..."
uv venv --python 3.11 .venv

echo "Installing requirements..."
uv pip install --python .venv/bin/python -r requirements.txt

echo ""
echo "=== Setup Completed Successfully! ==="
echo "To activate this environment in your terminal, run:"
echo "    source $DIR/.venv/bin/activate"
