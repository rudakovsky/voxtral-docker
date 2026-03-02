FROM python:3.11-slim

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    ca-certificates \
    libsndfile1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio \
  && pip install --no-cache-dir \
    transformers \
    datasets \
    soundfile \
    "mistral-common[audio]"

COPY transcribe.py /app/transcribe.py

ENTRYPOINT ["python", "transcribe.py"]
