FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ffmpeg: used by helpers/audio_helper.py to extract/transcode the audio
# track from the shared source file (which may be a video container).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Model weights are baked into the AMI at /model-cache/{SERVICE_NAME}/ and bind-mounted
# into the container at runtime (not part of the image). /tmp/shared_jobs is likewise a
# host bind mount shared with the video_ai_service and scene_detection containers.
RUN mkdir -p /model-cache /tmp/shared_jobs

CMD ["python", "worker.py"]
