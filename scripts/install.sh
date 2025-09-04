#!/usr/bin/env bash

if [ -f /etc/os-release ]; then
  . /etc/os-release
  if [ "$ID" = "ubuntu" ]; then
    scripts/ubuntu/install.sh
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$(cd "$SCRIPT_DIR/../" && pwd)"
chmod +x "$INSTALL_DIR/dtf"

# Add to .bashrc if not already there
if ! grep -q "export PATH=\"$INSTALL_DIR:\$PATH\"" ~/.bashrc; then
  echo -e "\n# Add dtf to path\nexport PATH=\"$INSTALL_DIR:\$PATH\"" >> ~/.bashrc
  echo "Please run 'source ~/.bashrc' or open a new terminal."
fi

# Add to .zshrc if not already there
if [ -f ~/.zshrc ] && ! grep -q "export PATH=\"$INSTALL_DIR:\$PATH\"" ~/.zshrc; then
  echo -e "\n# Add dtf to path\nexport PATH=\"$INSTALL_DIR:\$PATH\"" >> ~/.zshrc
  echo "Please run 'source ~/.zshrc' or open a new terminal."
fi