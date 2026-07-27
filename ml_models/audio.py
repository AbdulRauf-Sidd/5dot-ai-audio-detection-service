import threading
from pathlib import Path

import torch

import predictor
from config.project_config import DEVICE, MODEL_CACHE_DIR

model = None
head = None

_load_lock = threading.Lock()


def load_models():
    global model, head

    with _load_lock:
        if model is None or head is None:
            print(f"[*] Loading deepfense audio model from {MODEL_CACHE_DIR} ...")
            # Baked into the AMI at MODEL_CACHE_DIR -- loaded offline, no
            # Hugging Face Hub / network calls (the WavLM frontend is built from
            # the local architecture stub, all weights come from best_model.pth).
            model, head = predictor.load_model(Path(MODEL_CACHE_DIR), torch.device(DEVICE))
            print("[+] deepfense audio model loaded")

    return model, head
