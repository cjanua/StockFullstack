#!/bin/bash

# Check if running as root
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root" 
    exit 1
fi

# Get username
read -p "Enter username: " USERNAME

# Create user and set password
useradd -m -G wheel -s /bin/bash "$USERNAME"

# Set password securely (avoiding echo to terminal)
echo "$USERNAME:qwert" | chpasswd

# Configure sudo access
sed -i 's/# %wheel ALL=(ALL:ALL) ALL/%wheel ALL=(ALL:ALL) ALL/' /etc/sudoers

# Set up basic user environment
cp /etc/skel/.bash_profile "/home/$USERNAME/"
cp /etc/skel/.bashrc "/home/$USERNAME/"
chown -R "$USERNAME:$USERNAME" "/home/$USERNAME"

# Force password change on first login
passwd -e "$USERNAME"

# Configure WSL to use the new user as default
echo "[user]
default=$USERNAME" > /etc/wsl.conf

echo "User $USERNAME has been created successfully."
echo "Please change your password immediately upon first login."
echo "WSL has been configured to automatically log in as $USERNAME"
echo "Please restart WSL for changes to take effect."