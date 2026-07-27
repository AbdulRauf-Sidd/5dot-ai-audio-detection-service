# ai_audio

Standalone SQS worker (no FastAPI/Celery/SQLAlchemy) for AI-generated-audio
detection. Long-polls its own SQS queue, extracts the audio track from the
shared job source file via ffmpeg, scores it in 5-second chunks with a
WavLM-based deepfense detector, and writes results to Postgres
(`detection_requests.overall_ai_audio_score` /
`detection_chunks.ai_audio_score`) before notifying the core service via
webhook. See `worker.py` for the main loop and `config/project_config.py`
for the required environment variables.

Same architecture as the sibling `video_ai_service` and `scene_detection`
workers: raw-SQL Postgres access (`db.py`), shared `/tmp/shared_jobs`
download coordination (`shared_storage.py`), and model weights baked into
the AMI at `/model-cache/ai_audio/` rather than downloaded at runtime.

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the worker:

```bash
python3 worker.py
```

Run the smoke test (checks module wiring; also verifies DB/model-weights/ffmpeg if available):

```bash
python3 test.py
```
