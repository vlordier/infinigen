ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8.22
ARG APP_IMAGE=python:3.11-bookworm
FROM ${UV_IMAGE} AS uv

FROM ${APP_IMAGE}
ARG APP_IMAGE
COPY --from=uv /uv /uvx /bin/
ENV UV_PROJECT_ENVIRONMENT="/opt/infinigen/.venv"
ENV PATH="/opt/infinigen/.venv/bin:${PATH}"
RUN if [ "$APP_IMAGE" = "nvidia/cuda:12.0.0-devel-ubuntu22.04" ]; then \
    echo "Using CUDA image" && \
    apt-get update && \
    apt-get install -y unzip sudo git g++ libglm-dev libglew-dev libglfw3-dev libgles2-mesa-dev zlib1g-dev wget cmake vim libxi6 libgconf-2-4 && \
    apt-get install -y libxkbcommon-x11-0; \
else \
    echo "Using Python image" && \
    apt-get update -yq && \
    apt-get install -yq cmake g++ libgconf-2-4 libgles2-mesa-dev libglew-dev libglfw3-dev libglm-dev libxi6 sudo unzip vim zlib1g-dev && \
    apt-get install -y libxkbcommon-x11-0; \
fi

RUN mkdir /opt/infinigen
WORKDIR /opt/infinigen
COPY . .
ENV INFINIGEN_MINIMAL_INSTALL=True
RUN uv python install 3.11 && \
    uv sync --frozen --extra dev --python 3.11
