"""
backend/agent.py - pneumonia model (ONNX) behind the website.

Put these files from the Colab export zip into backend/model/:
    pneumonia_densenet121.onnx
    model_meta.json
The website calls analyze(image_bytes) and expects the dictionary returned below.
"""

import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parent / "model"
ONNX_PATH = MODEL_DIR / "pneumonia_densenet121.onnx"
META_PATH = MODEL_DIR / "model_meta.json"

IMG = 224  # the model's input size
URGENT_P = (
    0.80  # probability above which a case is "Urgent". Ask a clinician to confirm.
)

SCOPE_NOTE = (
    "This model only looks for pneumonia. It was trained on children's chest X-rays "
    "(ages 1 to 5) from one hospital, so results on adults or other scanners are unreliable."
)

_session = None
_threshold = 0.5


def _load():
    """Load the model once, on the first request."""
    global _session, _threshold
    if _session is None:
        if not ONNX_PATH.exists():
            raise FileNotFoundError(
                "Model file not found. Put pneumonia_densenet121.onnx in the backend/model folder."
            )
        import onnxruntime as ort

        _session = ort.InferenceSession(
            str(ONNX_PATH), providers=["CPUExecutionProvider"]
        )
        if META_PATH.exists():
            _threshold = float(json.loads(META_PATH.read_text())["threshold"])
    return _session


def _sanity(img):
    """Cheap checks that the upload looks like a chest X-ray photo. Returns a list of problems."""
    problems = []
    w, h = img.size
    if min(w, h) < 128:
        problems.append(f"The image is too small ({w}x{h}).")
    if not (0.6 <= w / h <= 1.6):
        problems.append("The shape of the image is unusual for a chest X-ray.")
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    colour = (
        np.abs(rgb[..., 0] - rgb[..., 1]).mean()
        + np.abs(rgb[..., 1] - rgb[..., 2]).mean()
    )
    if colour > 12:
        problems.append("The image is in colour. Chest X-rays are grayscale.")
    if rgb.mean(axis=2).std() < 15:
        problems.append("The image has almost no contrast.")
    return problems


def _heatmap(cam, size):
    """7x7 attention map -> PNG (black = nothing, red/yellow = strongest)."""
    c = np.maximum(cam, 0)
    c = c / (c.max() + 1e-8)
    w, h = size
    scale = min(1.0, 512 / max(w, h))
    w, h = max(1, int(w * scale)), max(1, int(h * scale))
    v = (
        np.asarray(
            Image.fromarray((c * 255).astype("uint8")).resize((w, h), Image.BILINEAR),
            dtype=np.float32,
        )
        / 255
    )
    rgb = np.stack(
        [np.clip(v * 2, 0, 1), np.clip(v * 2 - 0.8, 0, 1), np.zeros_like(v)], axis=-1
    )
    buf = io.BytesIO()
    Image.fromarray((rgb * 255).astype("uint8")).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def analyze(image_bytes: bytes) -> dict:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception:
        return {
            "recognized": False,
            "priority": None,
            "score": 0,
            "findings": {},
            "reasons": ["This file could not be read as an image."],
        }

    problems = _sanity(img)
    if problems:
        return {
            "recognized": False,
            "priority": None,
            "score": 0,
            "findings": {},
            "reasons": problems,
        }

    sess = _load()
    x = np.asarray(
        img.convert("RGB").resize((IMG, IMG), Image.BILINEAR), dtype=np.float32
    )
    x = x.transpose(2, 0, 1)[None] / 255.0
    prob, cam = sess.run(None, {"image": x})
    p = float(np.asarray(prob).ravel()[0])

    if p >= max(URGENT_P, _threshold):
        priority = "Urgent"
    elif p >= _threshold:
        priority = "Review soon"
    else:
        priority = "Routine"

    return {
        "recognized": True,
        "priority": priority,
        "score": round(p, 3),
        "findings": {"Pneumonia": round(p, 3)},
        "heatmap": _heatmap(np.asarray(cam)[0], img.size),
        "summary": SCOPE_NOTE,
    }
