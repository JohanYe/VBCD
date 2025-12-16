#!/usr/bin/env bash

# VBCD Installation Script
# This script installs all Python dependencies in a reproducible way using uv
# Usage: bash install.sh

set -e  # Exit on any error

echo "================================"
echo "VBCD Dependency Installation"
echo "================================"

# Ensure uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first."
    exit 1
fi

echo "Using uv package manager for reproducibility..."
echo ""

# Create virtual environment with Python 3.10
echo "Creating virtual environment (.venv) with Python 3.10..."
uv venv --python=3.10 .venv
source .venv/bin/activate
echo "Virtual environment activated!"
echo ""

# Install most dependencies with uv (skip project constraints)
echo "Installing core dependencies with uv..."
uv pip install \
    accelerate \
    addict \
    h5py \
    imageio \
    numpy \
    open3d \
    pillow \
    plyfile \
    pyvista \
    PyYAML \
    scikit-image \
    scipy \
    timm \
    trimesh \
    rtree \
    tensorboard \
    torch==2.4.1+cu118 \
    torchaudio==2.4.1+cu118 \
    torchvision==0.19.1+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

echo "Installing torch-geometric and dependencies..."
uv pip install \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    pyg_lib \
    torch-geometric==2.6.1 \
    -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
    
# Install remaining packages
echo ""
echo "Installing remaining dependencies..."
uv pip install \
    tqdm==4.67.1 \
    trimesh==4.8.1 \
    omegaconf==1.4.1 \
    einops==0.8.1 \
    iopath==0.1.10 \
    ninja==1.13.0 \
    regex

# Install pytorch3d from prebuilt wheels
echo ""
echo "Installing pytorch3d..."
uv pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.4.1cu118

echo ""
echo "================================"
echo "Installation Complete!"
echo "================================"
echo ""
echo "Virtual environment is at: .venv"
echo "To activate it in the future, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To verify the installation, run:"
echo "  python -c \"import torch; import torch_geometric; import pytorch3d; print('All imports successful!')\""
