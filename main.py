from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

import models
from database import Base, engine, get_db
from predictor import load_model, predict


model = None
head = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the database and load the ML model once at application startup."""
    global model, head
    Base.metadata.create_all(bind=engine)
    model, head = load_model()
    yield


app = FastAPI(title="Audio Authenticity API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def create_prediction(
    file: Annotated[UploadFile, File(description="Audio file supported by libsndfile")],
    db: Session = Depends(get_db),
):
    if model is None or head is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    try:
        audio_bytes = await file.read()
        audio, _ = sf.read(BytesIO(audio_bytes), dtype="float32", always_2d=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Upload a valid audio file") from exc

    if audio.size == 0:
        raise HTTPException(status_code=400, detail="Audio file is empty")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # convert stereo/multichannel to mono

    result = predict(audio, model, head)
    record = models.Prediction(filename=file.filename or "audio", **result)
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "id": record.id,
        "filename": record.filename,
        "label": record.label,
        "p_real": record.p_real,
        "p_fake": record.p_fake,
        "created_at": record.created_at,
    }
