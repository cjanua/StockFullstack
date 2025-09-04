# ai/Dockerfile

FROM rocm/pytorch:rocm6.4.2_ubuntu24.04_py3.12_pytorch_release_2.6.0

# Set working directory
WORKDIR /workspace

# Add uv to PATH for subsequent layers
ENV PATH="/root/.local/bin:$PATH"

# Copy pyproject.toml and install dependencies
COPY pyproject.toml /workspace/
RUN pip install -e .

# Default command to keep container running
CMD ["sleep", "infinity"]