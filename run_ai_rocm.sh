#!/usr/bin/env bash
# run_ai_rocm.sh

docker pull rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_2.5.0
docker run -it --gpus all --network=host -e HSA_OVERRIDE_GFX_VERSION=11.0.0 -v $HOME/Stocks:/workspace rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_2.5.0
