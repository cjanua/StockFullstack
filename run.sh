#!/usr/bin/env bash
# run.sh

# Detect if .env file exists, if not create it from template
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "Creating .env file from .env.example"
        cp .env.example .env
        echo "Please update your .env file with your credentials"
    else
        echo "Creating empty .env file"
        touch .env
        echo "Please add your credentials to the .env file"
    fi
fi

# Function to print usage instructions
print_usage() {
    echo "Usage: $0 [command]"
    echo "Commands:"
    echo "  up [--dev]          Start all services (use --dev for hot reload)"
    echo "  down                Stop all services"
    echo "  restart             Restart all services"
    echo "  build               Build all Docker images"
    echo "  logs [service]      View logs of all or a specific service"
    echo "  [service] [--dev]   Run select service only (portfolio | dashboard)"
    echo "  shell [service]     Open a shell in a container"
    echo
    echo "Available services: portfolio-service, dashboard, redis"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! docker compose version &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

DEV_MODE="false"
RUN_DIR=".run"
WATCHER_PID_FILE="$RUN_DIR/watcher.pid"
WATCHER_LOG_FILE="$RUN_DIR/watcher.log"

mkdir -p "$RUN_DIR"

is_watcher_running() {
  if [ -f "$WATCHER_PID_FILE" ]; then
    local pid=$(cat "$WATCHER_PID_FILE")
    # Check if a process with this PID exists. The >/dev/null hides output.
    if ps -p "$pid" > /dev/null; then
      return 0 # 0 means true (is running)
    fi
  fi
  return 1 # 1 means false (is not running)
}

build() {
    
}

# Basic arg parsing to support optional --dev flag
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --dev)
            DEV_MODE="true"
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
done
set -- "${ARGS[@]}"

# Parse command line arguments
if [ $# -eq 0 ]; then
        print_usage
        exit 0
fi

case "$1" in
    dev)
        echo "🚀 Starting full development environment..."
        ./run.sh up --dev
        ./run.sh watcher start
        ;;

    watcher)
        case "$2" in
        start)
            if is_watcher_running; then
            echo "✅ Watcher is already running with PID $(cat $WATCHER_PID_FILE)."
            exit 0
            fi
            echo "Starting Dockerfile watcher in the background..."
            # nohup ensures the process keeps running even if you close the terminal
            nohup ./watch-and-rebuild.sh > "$WATCHER_LOG_FILE" 2>&1 &
            # Save the PID of the last backgrounded process to our file
            echo $! > "$WATCHER_PID_FILE"
            sleep 1 # Give it a second to start up
            if is_watcher_running; then
                echo "✅ Watcher started with PID $(cat $WATCHER_PID_FILE)."
                echo "--> View logs with: ./run.sh watcher logs"
            else
                echo "❌ Watcher failed to start. Check logs for errors:"
                cat "$WATCHER_LOG_FILE"
            fi
            ;;
        stop)
            if ! is_watcher_running; then
            echo "Watcher is not running."
            # Clean up a potentially stale PID file
            rm -f "$WATCHER_PID_FILE"
            exit 0
            fi
            echo "Stopping watcher..."
            kill $(cat "$WATCHER_PID_FILE")
            rm "$WATCHER_PID_FILE"
            echo "✅ Watcher stopped."
            ;;
        status)
            if is_watcher_running; then
            echo "✅ Watcher is running with PID $(cat $WATCHER_PID_FILE)."
            else
            echo "Watcher is stopped."
            fi
            ;;
        logs)
            if [ -f "$WATCHER_LOG_FILE" ]; then
            # tail -f will follow the log file in real-time
            tail -f "$WATCHER_LOG_FILE"
            else
            echo "Log file not found. Is the watcher running?"
            fi
            ;;
        *)
            echo "Usage: $0 watcher [start|stop|status|logs]"
            exit 1
            ;;
        esac
        ;;
    up)
        echo "Starting all services..."
        if [ "$DEV_MODE" = "true" ]; then
            echo "Dev mode enabled (hot reload)"
            docker compose -f docker-compose.yaml -f docker-compose.dev.yaml up --build -d
        else
            docker compose up -d
        fi
        ;;
    down)
        echo "Stopping all services..."
        docker compose down
        ;;
    restart)
        echo "Restarting all services..."
        docker compose down
        docker compose up -d
        ;;
    build)
        echo "Building all Docker images..."
        docker compose build
        ;;
    logs)
        if [ -z "$2" ]; then
            echo "Showing logs for all services..."
            docker compose logs -f
        else
            echo "Showing logs for $2..."
            docker compose logs -f "$2"
        fi
        ;;
    # ai)
    #     echo "Starting AI service..."
    #     docker compose up -d ai-service
    #     docker compose logs -f ai-service
    #     ;;
    portfolio)
        echo "Starting portfolio service..."
        if [ "$DEV_MODE" = "true" ]; then
            echo "Dev mode enabled (hot reload)"
            docker compose -f docker-compose.yaml -f docker-compose.dev.yaml up -d redis portfolio-service
        else
            docker compose up -d redis portfolio-service
        fi
        docker compose logs -f portfolio-service
        ;;
    dashboard)
        echo "Starting dashboard service..."
        if [ "$DEV_MODE" = "true" ]; then
            echo "Dev mode enabled (hot reload)"
            docker compose -f docker-compose.yaml -f docker-compose.dev.yaml up -d redis portfolio-service dashboard
            docker compose logs -f dashboard
        else
            docker compose up -d redis portfolio-service dashboard
            docker compose logs -f dashboard
        fi
        ;;
    shell)
        if [ -z "$2" ]; then
            echo "Please specify a service name"
            echo "Available services: portfolio-service, dashboard, redis"
            exit 1
        else
            echo "Opening shell in $2..."
            docker compose exec "$2" /bin/bash || docker compose exec "$2" /bin/sh
        fi
        ;;
    *)
        print_usage
        exit 1
        ;;
esac
