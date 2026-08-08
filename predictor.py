from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from deepfense.models import *
from deepfense.utils.registry import build_detector


class _AMSoftmaxHead(nn.Module):
    def __init__(self, in_features, num_classes=2):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))

    def forward(self, x):
        return F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1))


def load_model(model_dir: Path, device: torch.device):
    cfg       = OmegaConf.load(model_dir / "config.yaml")
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    stub = model_dir / "_wavlm_large_stub.pt"
    model_cfg["frontend"]["args"]["ckpt_path"] = str(stub)
    model_cfg["frontend"]["args"]["source"]    = "fairseq"

    model = build_detector(cfg.model.type, model_cfg)
    state = torch.load(model_dir / "audio_model.pth", map_location=device)
    model.load_state_dict(state["model_state"], strict=False)
    model.to(device).eval()

    w                    = state["loss_fn"]["weight"]
    head                 = _AMSoftmaxHead(*reversed(w.shape)).to(device)
    head.load_state_dict(state["loss_fn"])
    head.eval()

    return model, head


def predict(audio: np.ndarray, model, head, device: torch.device) -> dict:
    x = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(head(model(x)["embeddings"]).float(), dim=-1).cpu().numpy()[0]
    p_real, p_fake = float(probs[0]), float(probs[1])
    return {"label": "real" if p_real >= 0.5 else "fake", "p_real": p_real, "p_fake": p_fake}
