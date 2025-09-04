#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_DIR="$PROJECT_DIR/.run"
WATCHER_PID_FILE="$RUN_DIR/watcher.pid"
WATCHER_LOG_FILE="$RUN_DIR/watcher.log"

mkdir -p "$RUN_DIR"

is_watcher_running() {
  if [ -f "$WATCHER_PID_FILE" ]; then
    local pid=$(cat "$WATCHER_PID_FILE")
    if ps -p "$pid" > /dev/null; then
      return 0
    fi
  fi
  return 1
}

start_watcher() {
  if is_watcher_running; then
    return 0
  fi
  nohup $PROJECT_DIR/scripts/watch-and-rebuild.sh > "$WATCHER_LOG_FILE" 2>&1 &
  echo $! > "$WATCHER_PID_FILE"
  sleep 1
  if is_watcher_running; then
      echo $(cat $WATCHER_PID_FILE)
      return 0
  else
      cat "$WATCHER_LOG_FILE"
      return 1
  fi
}

stop_watcher() {
  if ! is_watcher_running; then
      rm -f "$WATCHER_PID_FILE"
      return 0
  fi
  kill $(cat "$WATCHER_PID_FILE")
  rm "$WATCHER_PID_FILE"
  if ! is_watcher_running; then
      rm -f "$WATCHER_PID_FILE"
      return 0
  fi
  return 1
}

restart_watcher() {
  stop_watcher
  start_watcher
  return 0
}

log_watcher() {
  if [ -f "$WATCHER_LOG_FILE" ]; then
      tail -f "$WATCHER_LOG_FILE"
  fi
}

watcher() {
    local cmd=$1
    case "$cmd" in
        start)
            start_watcher
            ;;
        stop)
            stop_watcher
            ;;
        restart)
            echo "Stopping $(cat $WATCHER_PID_FILE)"
            restart_watcher
            echo "Started $(cat $WATCHER_PID_FILE)"
            ;;
        status)
            if is_watcher_running; then
                echo "✅ Watcher is running with PID $(cat $WATCHER_PID_FILE)."
            else
                echo "Watcher is stopped."
            fi
            ;;
        logs)
            log_watcher
            ;;
        *)
            echo "Usage: $0 [start|stop|status|logs]"
            exit 1
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  if [[ "$1" == "watcher" ]]; then
    watcher "$2"
  else
    echo "Usage: $0 watcher [start|stop|restart|status|logs]"
    exit 1
  fi
fi

exit 0