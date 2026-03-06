#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

export CUDA_WHL_LIB_DIR=$(python -c 'import nvidia.cuda_runtime; print(nvidia.cuda_runtime.__path__[0])')/lib
export CUBLAS_WHL_LIB_DIR=$(python -c 'import nvidia.cublas; print(nvidia.cublas.__path__[0])')/lib
export CUDNN_WHL_LIB_DIR=$(python -c 'import nvidia.cudnn; print(nvidia.cudnn.__path__[0])')/lib
export LD_LIBRARY_PATH="$CUDA_WHL_LIB_DIR:$CUBLAS_WHL_LIB_DIR:$CUDNN_WHL_LIB_DIR:${LD_LIBRARY_PATH:-}"

exec python app.py