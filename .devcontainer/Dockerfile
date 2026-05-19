FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip3 install --no-cache-dir \
    ultralytics \
    opencv-python-headless \
    numpy \
    pillow \
    matplotlib

WORKDIR /workspace
