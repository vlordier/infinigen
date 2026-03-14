ARG APP_IMAGE=python:3.11-slim
FROM ${APP_IMAGE}
ARG APP_IMAGE

RUN if [ "$APP_IMAGE" = "nvidia/cuda:12.0.0-devel-ubuntu22.04" ]; then \
    echo "Using CUDA image" && \
    apt-get update && \
    apt-get install -y unzip sudo git g++ libglm-dev libglew-dev libglfw3-dev libgles2-mesa-dev zlib1g-dev wget cmake vim libxi6 libgconf-2-4 python3.11 python3.11-venv python3-pip libxkbcommon-x11-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*; \
else \
    echo "Using Python image" && \
    apt-get update -yq && \
    apt-get install -yq cmake g++ libgconf-2-4 libgles2-mesa-dev libglew-dev libglfw3-dev libglm-dev libxi6 sudo unzip vim zlib1g-dev libxkbcommon-x11-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*; \
fi

RUN python3 -m pip install --no-cache-dir uv

WORKDIR /opt/infinigen
COPY . .

ENV VIRTUAL_ENV=/opt/infinigen/.venv
ENV PATH="/opt/infinigen/.venv/bin:${PATH}"
RUN uv venv $VIRTUAL_ENV --python 3.11 && \
    uv pip install -e ".[dev]"
