FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
  python3 python3-venv python3-dev \
  curl git build-essential \
  && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /root/code/mujoco_playground
COPY pyproject.toml uv.lock README.md ./
COPY mujoco_playground ./mujoco_playground

RUN uv venv --python 3.12
ENV VIRTUAL_ENV=/root/code/mujoco_playground/.venv
ENV PATH="/root/code/mujoco_playground/.venv/bin:$PATH"

RUN uv pip install -U "jax[cuda12]" --index-url https://pypi.org/simple
RUN uv --no-config sync --all-extras --active
RUN uv pip install tensorboard
RUN uv pip install nvitop

# Remove copied source — the real project will be bind-mounted at runtime
RUN find /root/code/mujoco_playground -mindepth 1 -maxdepth 1 ! -name '.venv' -exec rm -rf {} +

ENV JAX_DEFAULT_MATMUL_PRECISION=highest

CMD ["bash"]
