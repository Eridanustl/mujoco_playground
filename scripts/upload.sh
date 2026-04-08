#!/bin/bash

# Check if the username argument is provided
if [ -z "$1" ]; then
  echo "Error: No username provided."
  echo "Usage: $0 <username>"
  exit 1
fi

USER_NAME=$1
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_NAME=$(basename "$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || dirname "$SCRIPT_DIR")")

echo "Uploading project [${PROJECT_NAME}] to user [${USER_NAME}]..."

rsync -avz \
    -e 'ssh -p16000' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude 'logs/' \
    --exclude 'outputs/' \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '.vscode/' \
    --exclude 'mujoco_playground/external_deps' \
    ./ ${USER_NAME}@10.41.206.215:~/code/${PROJECT_NAME}