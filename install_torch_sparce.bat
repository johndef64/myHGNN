REM Install specific torch sparse version compatible with torch 2.4.0 and CUDA 12.4

REM Replace existing torch installation
pip uninstall torch torchvision torchaudio -y
pip install torch==2.4.0  --index-url https://download.pytorch.org/whl/cu124


REM Install/Replace torch-sparse and torch-scatter from PyG wheels
pip uninstall torch_geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -y
REM pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install torch-geometric

REM pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

REM Alternative conda installation commands
REM ---
REM conda install pytorch-sparse -c pyg
REM ---
REM conda install pyg -c pyg