#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

sudo rm /etc/wsl.conf

if [ ! -f "$SCRIPT_DIR/../unix/.user" ]; then
  echo "No user found. Please create a user first."
  exit 1
fi

cp $SCRIPT_DIR/wsl.conf $SCRIPT_DIR/../../.tmp/wsl.conf
echo "
[user]
default = $(cat $SCRIPT_DIR/../unix/.user)
" >> $SCRIPT_DIR/../../.tmp/wsl.conf

sudo cp $SCRIPT_DIR/../../.tmp/wsl.conf /etc/wsl.conf

sudo reboot now