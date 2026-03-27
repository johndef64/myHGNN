conda env create -f environment.yml
conda activate graph-ml-cu128


#!/usr/bin/env bash
set -e

ENV_NAME=graph-ml-cu128

echo "Installing PyTorch 2.5.1 cu124..."
conda run -n $ENV_NAME pip install \
  torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

echo "Installing PyG matching torch 2.5.1 cu124..."
conda run -n $ENV_NAME pip install --no-cache-dir --only-binary=:all: \
  pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

echo "Installing torch-geometric + extras..."
conda run -n $ENV_NAME pip install \
  torch-geometric termcolor torcheval

echo "Fix numpy/scipy/sklearn..."
conda run -n $ENV_NAME pip install --force-reinstall \
  "numpy<2.0" \
  "scipy<1.14" \
  "scikit-learn<1.5"

echo "Done."
