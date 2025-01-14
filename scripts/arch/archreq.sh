#!/bin/bash

sudo pacman -Syyu
yay -Syyu
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

if ! which git >/dev/null 2>&1; then
    sudo pacman -S git
fi

if ! which gh >/dev/null 2>&1; then
    yay -Sy github-cli
    source ~/.bashrc 
fi

# if ! which snyk >/dev/null 2>&1; then
#     curl https://static.snyk.io/cli/latest/snyk-linux -o snyk
#     chmod +x ./snyk
#     mv ./snyk /usr/local/bin/ 
# fi

if ! which direnv >/dev/null 2>&1; then
    yay -Sy direnv
    echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
    source ~/.bashrc
fi

direnv allow

pip install --upgrade pip
pip install -r req.txt

cd frontend
bun install
cd ..

direnv reload