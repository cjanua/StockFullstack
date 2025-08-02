#!/usr/bin/env bash

set -euo pipefail

main() {
  if [[ $# -eq 0 ]]; then
    cmd_help
    exit 1
  fi

  local command="$1"
  shift
  case "$command" in
    init)
      cmd_init "$@"
      ;;
    help)
      cmd_help
      ;;
    *)
      echo "Error: Unknown command: $command" >&2
      cmd_help
      exit 1
      ;;
  esac
}

cmd_init() {
  echo "Initializing with args: $*"
  # Your init logic here
}

cmd_help() {
  cat <<EOF
Usage: $(basename "$0") <command>

Available commands:
  init    Initialize something
  help    Display this help message
EOF
}

main "$@"