#!/usr/bin/env bash

CONFIG_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

sudo rm /etc/wsl.conf
sudo cp $CONFIG_DIR/wsl.conf /etc/

sudo reboot now