#!/usr/bin/env bash
#
# RCMAS setup script
#
# Creates the virtual environment .venv/ with the pinned versions of every
# dependency (z3-solver 5.3.3.0 and the test tools), installs the tool in
# editable mode, checks the solver version, and runs the test suite.
#
#   - With uv (https://docs.astral.sh/uv/) installed, the environment is
#     built from uv.lock (`uv sync --frozen`).
#   - Without uv, it is built with venv and pip from requirements-lock.txt,
#     which uv exported from the same lock, with hash checking.
#
# Safe to run more than once.
#
# Usage:
#   ./setup.sh               # set up and run the tests
#   ./setup.sh --no-tests    # set up only
#   RCMAS_NO_UV=1 ./setup.sh # use pip even if uv is installed
#
set -euo pipefail

cd "$(dirname "$0")"

RUN_TESTS=1
for arg in "$@"; do
    case "$arg" in
        --no-tests) RUN_TESTS=0 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

MIN_PYTHON="3.10"
Z3_PIN=$(sed -n 's/.*"z3-solver==\([^"]*\)".*/\1/p' pyproject.toml)

if command -v uv >/dev/null 2>&1 && [ "${RCMAS_NO_UV:-0}" != "1" ]; then
    echo "Creating .venv/ from uv.lock with $(uv --version) ..."
    uv sync --frozen --extra dev
else
    PYTHON=""
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PYTHON="$candidate"
            break
        fi
    done
    if [ -z "$PYTHON" ]; then
        echo "ERROR: Python not found. Please install Python >= $MIN_PYTHON." >&2
        exit 1
    fi
    if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
        echo "ERROR: Python >= $MIN_PYTHON is required (found $("$PYTHON" -V 2>&1))." >&2
        exit 1
    fi
    echo "Using $("$PYTHON" -V 2>&1) and pip with requirements-lock.txt"
    if [ ! -d ".venv" ]; then
        "$PYTHON" -m venv .venv
    fi
    .venv/bin/python -m pip install --upgrade pip --quiet
    .venv/bin/python -m pip install --require-hashes -r requirements-lock.txt --quiet
    .venv/bin/python -m pip install --no-deps -e . --quiet
fi

FOUND=$(.venv/bin/python -c 'import importlib.metadata as m; print(m.version("z3-solver"))')
if [ "$FOUND" != "$Z3_PIN" ]; then
    echo "ERROR: .venv has z3-solver $FOUND, but pyproject.toml pins $Z3_PIN." >&2
    echo "       Remove .venv/ and run ./setup.sh again." >&2
    exit 1
fi
echo "Python $(.venv/bin/python -c 'import platform; print(platform.python_version())'), z3-solver $FOUND"

if [ "$RUN_TESTS" = "1" ]; then
    echo ""
    echo "Running tests ..."
    .venv/bin/python -m pytest -q
fi

echo ""
echo "Setup complete. Activate the environment with"
echo "  source .venv/bin/activate"
echo "and try"
echo "  rcmas co --grid grids/symmetric/4x4.txt --agents 2 --horizon 8"
echo "Reproduce the thesis experiments with ./reproduce.sh (see README.md)."
