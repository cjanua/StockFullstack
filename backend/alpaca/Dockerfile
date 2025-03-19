# alpaca/Docekrfile
# Use Python 3.13 slim image
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libgmp-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy application files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r req.txt

# Make the startup script executable
RUN chmod +x ./start_portfolio_service.sh

# Expose the application port
EXPOSE 8001

# Set the default command to run the service
CMD ["./start_portfolio_service.sh"]
