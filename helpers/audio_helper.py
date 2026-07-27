import subprocess

import numpy as np
import soundfile as sf

import predictor

SAMPLE_RATE = 16000  # deepfense/WavLM was trained on 16kHz mono audio


def extract_audio(input_path: str, output_path: str, sample_rate: int = SAMPLE_RATE) -> str:
    """Extract a mono PCM WAV audio track from `input_path` via ffmpeg.

    The shared source file may be a plain audio upload or a video container
    (when the same job also requested video/lipsync/changes detection), so
    this always demuxes+decodes through ffmpeg rather than assuming a
    directly soundfile-readable format.
    """
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-acodec", "pcm_s16le",
        "-loglevel", "error",
        output_path,
    ]
    subprocess.run(cmd, check=True)
    return output_path


def split_audio_into_chunks(wav_path: str, chunk_length: int = 5) -> list[tuple[float, float, np.ndarray]]:
    """Load the extracted WAV and slice it in memory into `chunk_length`-second
    windows (start, end, samples). The last chunk may be shorter."""
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    duration = len(audio) / sr
    chunks = []
    start = 0.0
    while start < duration:
        end = min(start + chunk_length, duration)
        chunks.append((start, end, audio[int(start * sr):int(end * sr)]))
        start += chunk_length
    return chunks


def infer_chunk(audio: np.ndarray, model, head, device) -> dict:
    try:
        return predictor.predict(audio, model, head, device)
    except Exception as e:
        print(f"[ERROR] Chunk inference failed: {e}")
        return {"label": "real", "p_real": 1.0, "p_fake": 0.0, "error": str(e)}
