# Dockerized Stock Trading System

This project has been dockerized to minimize host system dependencies. The only requirements are:

- Docker
- Docker Compose

## Project Structure

The project consists of the following services:

- **AI Service**: Machine learning and trading algorithms
- **Portfolio Service**: Integration with Alpaca API for portfolio management
- **Dashboard**: Web-based UI for monitoring and managing trades
- **Redis**: Cache and message broker

## Getting Started

### Prerequisites

1. Install Docker: [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)
2. Install Docker Compose: [https://docs.docker.com/compose/install/](https://docs.docker.com/compose/install/)

### Environment Setup

Create a `.env` file in the project root with your Alpaca API credentials:

```
ALPACA_KEY=your_alpaca_key
ALPACA_SECRET=your_alpaca_secret
```

### Running the System

Use the included `run_docker.sh` script to manage the system:

```bash
# Start all services
./run_docker.sh up

# Stop all services
./run_docker.sh down

# Rebuild containers
./run_docker.sh build

# View logs
./run_docker.sh logs

# View logs for a specific service
./run_docker.sh logs ai-service

# Run only specific services
./run_docker.sh ai        # Run AI service
./run_docker.sh portfolio # Run portfolio service
./run_docker.sh dashboard # Run dashboard

# Open a shell in a container
./run_docker.sh shell ai-service
```

## Development

For development purposes, you can mount your local code into the containers:

1. Edit the `docker-compose.yaml` file
2. Add volume mounts for the services you want to develop

Example:
```yaml
ai-service:
  volumes:
    - ./ai:/app/ai
```

This will allow you to make changes to the code without rebuilding the Docker image.

## Hardware Acceleration

The AI service is configured to use ROCm for AMD GPU acceleration. If you have an NVIDIA GPU, modify the Dockerfile to use CUDA instead.
