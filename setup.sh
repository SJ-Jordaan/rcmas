#!/bin/bash

# Setup script for RCMAS project
# This script creates the conda environment and installs the package

echo "🚀 Setting up RCMAS project..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^rcmas "; then
    echo "⚠️  Environment 'rcmas' already exists."
    read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n rcmas -y
    else
        echo "✅ Using existing environment."
        conda activate rcmas
        pip install -e ".[dev]"
        exit 0
    fi
fi

# Create conda environment
echo "📦 Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Activate environment
echo "🔄 Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate rcmas

# Install package in development mode
echo "📥 Installing package in development mode..."
pip install -e ".[dev]"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate rcmas"
echo ""
echo "To run tests:"
echo "  pytest"
echo ""
echo "To see all available commands:"
echo "  make help"
