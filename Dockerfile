
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ca-certificates \
    cuda-nvrtc-11-8 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install \
    torch==2.0.1 \
    torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

RUN python -c "import cv2, einops, kornia, matplotlib, motmetrics, networkx, numpy, PIL, scipy, torch, torchvision; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)"

COPY . .

CMD ["/bin/bash"]
