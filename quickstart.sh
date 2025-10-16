#!/bin/bash

# Quick start script for RCMAS project

set -e

echo "=============================================="
echo "RCMAS Project Quick Start"
echo "=============================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install project
echo ""
echo "Installing RCMAS in development mode..."
pip install -e ".[dev]"

# Run tests
echo ""
echo "Running tests..."
source venv/bin/activate && pytest tests/ -v

# Run example
echo ""
echo "Running basic example..."
source venv/bin/activate && python examples/basic_simulation.py

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment in the future:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest"
echo ""
echo "To run examples:"
echo "  python examples/basic_simulation.py"
echo "  python examples/complex_territory.py"
echo ""
echo "See DEVELOPMENT.md for more information."
