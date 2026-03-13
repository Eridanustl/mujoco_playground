#!/bin/bash

docker_image=mujoco-playground:latest

build_image() {
    echo "Building image $docker_image..."
    docker build -t "$docker_image" .
}

if git_root=$(git rev-parse --show-toplevel 2>/dev/null); then
    project_name=$(basename "$git_root")
    git_branch=$(git rev-parse --abbrev-ref HEAD)
    safe_branch_name=${git_branch//\//_}
    id="${project_name}_${safe_branch_name}"
else
    id="mujoco_playground"
fi

# Build image if it doesn't exist
if ! docker image inspect "$docker_image" &>/dev/null; then
    build_image
fi

if [ "$(docker ps -q --filter "name=^${id}$")" ]; then
    echo "Container $id is already running, attaching..."
    docker exec -it "$id" bash
else
    container_dir="/root/code/mujoco_playground"
    echo "Creating new container $id..."
    docker run \
        --rm \
        --name="$id" \
        --interactive \
        --tty \
        --gpus all \
        --network host \
        --env NVIDIA_DRIVER_CAPABILITIES=all \
        --volume "$(pwd):${container_dir}" \
        --volume "mujoco_venv:${container_dir}/.venv" \
        --volume "$(pwd)/scripts/.bashrc:/root/.bashrc" \
        --volume "$HOME/.ssh:/root/.ssh:ro" \
        --volume "$HOME/.cache:/root/.cache:rw" \
        --workdir "${container_dir}" \
        "$docker_image" \
        bash --rcfile /root/.bashrc
fi
