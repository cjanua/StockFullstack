#!/usr/bin/env bash
# run_ai_rocm.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker pull rocm/pytorch:rocm6.4.2_ubuntu24.04_py3.12_pytorch_release_2.6.0
docker run -it --rm --device=/dev/dxg --mount type=bind,source=/usr/lib/wsl,target=/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host --shm-size 8G -v /opt/rocm/lib:/host_rocm_lib:ro -v $SCRIPT_DIR:/workspace rocm/pytorch:rocm6.4.2_ubuntu24.04_py3.12_pytorch_release_2.6.0