#!/bin/bash

# Check if the username argument is provided
if [ -z "$1" ]; then
  echo "Error: No username provided."
  echo "Usage: $0 <username>"
  exit 1
fi

# Assign the first argument to the USER_NAME variable
USER_NAME=$1

# Execute rsync using the provided username
echo "Syncing logs for user [${USER_NAME}]..."
REMOTE_DIR="/home/${USER_NAME}/code/mujoco_playground"
rsync -avz -e 'ssh -p16000' "${USER_NAME}@10.41.206.215:${REMOTE_DIR}/logs/" ./logs/
rsync -avz -e 'ssh -p16000' "${USER_NAME}@10.41.206.215:${REMOTE_DIR}/rollout0.mp4" ./