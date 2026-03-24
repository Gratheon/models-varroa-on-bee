FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-server.txt ./

RUN set -eux; \
    pip install --upgrade pip; \
    (pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision || pip install torch torchvision); \
    pip install -r requirements-server.txt

COPY detect.py server.py ./
COPY yolo11n.pt ./yolo11n.pt

EXPOSE 8752

CMD ["python", "/app/server.py"]
