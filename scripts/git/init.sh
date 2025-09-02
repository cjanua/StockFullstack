#!/usr/bin/env bash

# Get current Git user name and email, suppressing errors if they don't exist
current_name=$(git config --global user.name 2>/dev/null)
current_email=$(git config --global user.email 2>/dev/null)

# Check if either the name or email is empty
if [[ -z "$current_name" || -z "$current_email" ]]; then
  echo "Git user name or email is not configured."
  read -p "Enter your Git user name: " name
  read -p "Enter your Git user email: " email

  git config --global user.name "$name"
  git config --global user.email "$email"

  echo "Git user name and email have been set globally."
else
  echo "Git user name and email are already set:"
  echo "  Name: $current_name"
  echo "  Email: $current_email"
fi