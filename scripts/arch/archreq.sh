#!/bin/bash

sudo pacman -Syyu
sudo pacman -S --needed base-devel git

if ! which python >/dev/null 2>&1; then
    sudo pacman -S python python-pip
fi

if ! which yay >/dev/null 2>&1; then
    git clone https://aur.archlinux.org/yay.git
    cd yay
    makepkg -si
    cd ..
    rm -rf ./yay
fi

if ! which bun >/dev/null 2>&1; then
    curl -fsSL https://bun.sh/install | bash
    source ~/.bashrc 
fi

if ! which gh >/dev/null 2>&1; then
    yay -Sy github-cli
    source ~/.bashrc 
fi