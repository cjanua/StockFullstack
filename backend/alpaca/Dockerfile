# backend/alpaca/Dockerfile

# Use Python 3.13 slim image
FROM python:3.13-slim

# Create proper directory structure to match imports
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libgmp-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create directory structure to match imports
RUN mkdir -p /app/backend/alpaca

# Copy application files into the proper structure
COPY . /app/backend/alpaca

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/backend/alpaca/req.txt

# Make the startup script executable
RUN chmod +x /app/backend/alpaca/start_portfolio_service.sh

# Expose the application port
EXPOSE 8001

# Set working directory for startup
WORKDIR /app/backend/alpaca

# Set the default command to run the service
CMD ["./start_portfolio_service.sh"]
