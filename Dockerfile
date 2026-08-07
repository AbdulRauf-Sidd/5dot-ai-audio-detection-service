FROM python:3.13.3-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN useradd -m -u 1000 appuser

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p /model-cache /tmp/shared_jobs && \
    chown -R appuser:appuser /app /model-cache /tmp/shared_jobs

USER appuser

CMD ["python", "worker.py"]
