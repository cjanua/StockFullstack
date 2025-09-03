#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo $SCRIPT_DIR
read -p "Enter your username: " USERNAME
if echo "$(getent passwd "$USERNAME")" | grep -q "$USERNAME"; then
  touch "$SCRIPT_DIR/.user"
  if echo "$(cat "$SCRIPT_DIR/.user")" | grep -q "$USERNAME"; then
    echo "User $USERNAME already exists."
  else
    echo "$USERNAME" >> "$SCRIPT_DIR/.user"
  fi
  exit 0
fi

if [ $(whoami) != "root" ]; then
  echo "This script must be run as root."
  exit 1
fi

useradd -m "$USERNAME"
echo "Please set the password for $USERNAME:"
passwd "$USERNAME"
usermod -aG sudo "$USERNAME"

touch "$SCRIPT_DIR/.user"
echo "$USERNAME" > "$SCRIPT_DIR/.user"

exit 0