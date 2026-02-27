#!/usr/bin/env bash
#
# RCMAS setup script
#
# Creates a virtual environment, installs dependencies, and runs a
# quick sanity check.  Safe to run multiple times.
#
# Usage:
#   ./setup.sh
#
set -e

MIN_PYTHON="3.10"

# ── Locate Python ──────────────────────────────────────────────────

PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python not found. Please install Python >= $MIN_PYTHON."
    echo "       https://www.python.org/downloads/"
    exit 1
fi

# ── Check version ─────────────────────────────────────────────────

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "ERROR: Python >= $MIN_PYTHON is required (found $PY_VERSION)."
    exit 1
fi

echo "Using $PYTHON ($PY_VERSION)"

# ── Create virtual environment ────────────────────────────────────

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv/ ..."
    "$PYTHON" -m venv .venv
else
    echo "Virtual environment .venv/ already exists."
fi

# ── Activate and install ──────────────────────────────────────────

source .venv/bin/activate

echo "Installing RCMAS and dependencies ..."
pip install --upgrade pip --quiet
pip install -e ".[dev]" --quiet

# ── Sanity check ──────────────────────────────────────────────────

echo ""
echo "Running tests ..."
pytest -q

# ── Done ──────────────────────────────────────────────────────────

echo ""
echo "========================================="
echo "  Setup complete!"
echo "========================================="
echo ""
echo "Activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Try an example:"
echo "  rcmas co --grid grids/symmetric/4x4.txt --agents 2 --horizon 8"
echo ""
