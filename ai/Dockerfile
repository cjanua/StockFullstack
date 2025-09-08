FROM rocm/pytorch:rocm6.4.2_ubuntu24.04_py3.12_pytorch_release_2.6.0

WORKDIR /workspace

# Copy the entire project structure (since we're building from project root)
COPY . /workspace

RUN pwd
RUN ls

# Install the project package from the root (includes stock_fullstack)
RUN pip install . --resume-retries 5

# Set working directory to ai for running main.py
WORKDIR /workspace/ai

# Default command to keep container running
CMD ["python", "main.py"]