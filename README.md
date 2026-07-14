# Audio Authenticity API

Minimal FastAPI service around the model in `predictor.py`. Each prediction is
saved to a local SQLite database (`predictions.db`).

Install the project dependencies in the same virtual environment as the model:

```bash
python3 -m pip install -r requirements.txt
```

Start the API:

```bash
python3 -m uvicorn main:app --reload
```

Send an audio file for prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@path/to/audio.wav"
```

`GET /health` reports whether the model has loaded. Interactive API docs are
available at `http://127.0.0.1:8000/docs`.
