#!/usr/bin/env bash

sudo apt autoremove --purge -y

sudo apt autoclean
sudo apt clean

sudo journalctl --vacuum-time=1weeks

rm -rf /tmp/* ~/.cache/*

docker image prune -f
docker builder prune -f
