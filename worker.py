"""ai_audio worker.

Standalone script (no FastAPI/Celery): long-polls its own SQS queue in an
infinite loop, does raw-SQL Postgres reads/writes, and reports completion to
the core service via webhook. See config/project_config.py for env vars.
"""

import json
import logging
import os
import tempfile
import time

import boto3
import torch

import db
import shared_storage
import webhook
from config.project_config import (
    AWS_REGION,
    DEVICE,
    IDLE_TIMEOUT_SECONDS,
    SERVICE_NAME,
    SQS_QUEUE_URL,
)
from helpers.audio_helper import extract_audio, infer_chunk, split_audio_into_chunks
from ml_models.audio import load_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(SERVICE_NAME)

CHUNK_LENGTH_SECONDS = 5  # matches video_ai_service's chunking convention, since
                          # detection_chunks rows are pre-created with shared
                          # segment_start/segment_end boundaries for a given job
INFERENCE_MAX_ATTEMPTS = 3  # 1 initial attempt + 2 retries, per transient errors like CUDA OOM

_model = None
_head = None


def extract_job_id(message: dict) -> str:
    body = message["Body"]
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict) and "job_id" in parsed:
            return str(parsed["job_id"])
    except (json.JSONDecodeError, TypeError):
        pass
    return body.strip()


def _process_chunks(job_id: str, source_path: str) -> list[tuple[float, float, dict]]:
    work_dir = tempfile.mkdtemp(prefix=f"{job_id}_")
    try:
        wav_path = extract_audio(source_path, os.path.join(work_dir, "audio.wav"))
        chunks = split_audio_into_chunks(wav_path, CHUNK_LENGTH_SECONDS)
        if not chunks:
            raise RuntimeError("No audio chunks could be extracted.")
        device = torch.device(DEVICE)
        return [
            (start, end, infer_chunk(audio, _model, _head, device))
            for start, end, audio in chunks
        ]
    finally:
        for name in os.listdir(work_dir):
            try:
                os.remove(os.path.join(work_dir, name))
            except OSError:
                pass
        try:
            os.rmdir(work_dir)
        except OSError:
            pass


def _run_inference_with_retry(job_id: str, source_path: str) -> list[tuple[float, float, dict]]:
    last_exc = None
    for attempt in range(1, INFERENCE_MAX_ATTEMPTS + 1):
        try:
            return _process_chunks(job_id, source_path)
        except Exception as exc:
            last_exc = exc
            logger.warning("Inference attempt %s/%s failed for job %s: %s",
                            attempt, INFERENCE_MAX_ATTEMPTS, job_id, exc)
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
    raise last_exc


def process_job(conn, job_id: str) -> None:
    job = db.fetch_job(conn, job_id)
    if not job:
        logger.error("Job %s not found in detection_requests", job_id)
        return
    if not job.get("detect_ai_audio"):
        logger.info("Job %s did not request audio detection, skipping", job_id)
        return

    db.mark_processing(conn, job_id)

    try:
        source_path = shared_storage.get_source_file(job)
        chunk_results = _run_inference_with_retry(job_id, source_path)

        for i, (start, end, result) in enumerate(chunk_results):
            db.update_chunk(conn, job_id, i, result["p_fake"], start, end)

        overall_score = sum(r["p_fake"] for _, _, r in chunk_results) / len(chunk_results)
        db.save_result(conn, job_id, overall_score)

        webhook.notify(job_id, "complete", {"score": overall_score})

    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        db.mark_failed(conn, job_id, str(exc))
        webhook.notify(job_id, "failed", {"error": str(exc)})

    finally:
        shared_storage.cleanup_if_last(conn, job_id)


def main():
    global _model, _head
    logger.info("Loading %s model from /model-cache/%s ...", SERVICE_NAME, SERVICE_NAME)
    _model, _head = load_models()
    logger.info("Model loaded, connecting to Postgres...")
    conn = db.connect()

    sqs = boto3.client("sqs", region_name=AWS_REGION)
    last_activity_at = time.time()

    logger.info("Polling %s", SQS_QUEUE_URL)
    while True:
        resp = sqs.receive_message(
            QueueUrl=SQS_QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
        )
        messages = resp.get("Messages", [])

        if not messages:
            if time.time() - last_activity_at >= IDLE_TIMEOUT_SECONDS:
                logger.info("Idle: no messages in the last %ss", IDLE_TIMEOUT_SECONDS)
                last_activity_at = time.time()
            continue

        last_activity_at = time.time()
        message = messages[0]
        job_id = extract_job_id(message)

        logger.info("Received job %s", job_id)
        try:
            process_job(conn, job_id)
        except Exception:
            logger.exception("Unhandled error processing job %s", job_id)
        finally:
            sqs.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=message["ReceiptHandle"])
            logger.info("Deleted SQS message for job %s", job_id)


if __name__ == "__main__":
    main()
