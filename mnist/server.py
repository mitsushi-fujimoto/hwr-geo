from __future__ import annotations
from pydantic import BaseModel

import os
import time
from typing import List, Optional, Dict, Any
from PIL import Image

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn

# Checkpoint file path
#   Override: export MODEL_PATH=mnist_net2.pth
MODEL_PATH: str = os.getenv("MODEL_PATH", "mnist_net2.pth")

# Toggle debug image saving
#   Enable: export DEBUG_SAVE=1
#   Change directory: export DEBUG_DIR=debug_images
DEBUG_SAVE: bool = os.getenv("DEBUG_SAVE", "0") == "1"
DEBUG_DIR: str = os.getenv("DEBUG_DIR", "debug_images")

# CORS (Cross-Origin Resource Sharing) settings
#   Example: export ALLOW_ORIGINS="http://localhost:5173,http://127.0.0.1:5173"
_allow_origins_env = os.getenv(
    "ALLOW_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000,http://localhost:5173,http://127.0.0.1:5173",
)
ALLOW_ORIGINS: List[str] = [o.strip() for o in _allow_origins_env.split(",") if o.strip()]

# Load model (called at startup via lifespan)
def load_model() -> Net2:
    # Verify checkpoint file exists
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model checkpoint not found: {MODEL_PATH}")

    # Load checkpoint file
    ckpt = torch.load(MODEL_PATH, map_location="cpu")

    # Verify required keys exist
    for key in ("n_input", "n_output", "n_hidden", "state_dict"):
        if key not in ckpt:
            raise RuntimeError(f"Checkpoint missing key '{key}': {MODEL_PATH}")

    m = Net2(int(ckpt["n_input"]), int(ckpt["n_output"]), int(ckpt["n_hidden"]))
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m

@asynccontextmanager
async def lifespan(app: FastAPI):
    global net
    net = load_model()
    yield

# Create FastAPI instance
app = FastAPI(lifespan=lifespan)

# Allow CORS for React (running on a different port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Net2 model definition (fully connected, 2 hidden layers)
class Net2(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_hidden: int):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_hidden)
        self.l3 = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        return x

#  Requests / Responses
class PredictRequest(BaseModel):
    pixels: List[int]  # length 784, values 0..255 (MNIST scale)

class PredictResponse(BaseModel):
    pred: int
    topk: List[Dict[str, Any]]  # [{"i": int, "p": float}, ...]  p in [0,1]
    saved_path: Optional[str] = None

# Global variable for lifespan-loaded model
net: Optional[Net2] = None

# Save 784 pixels (0..255) as 28x28 grayscale PNG.
def save_pixels_as_png(pixels_0_255: List[int], pred_label: Optional[str] = None) -> str:
    os.makedirs(DEBUG_DIR, exist_ok=True)

    # 784 -> 28x28
    img = Image.new("L", (28, 28))  # L=grayscale
    img.putdata([int(max(0, min(255, v))) for v in pixels_0_255])

    ts = time.strftime("%Y%m%d_%H%M%S")
    label = f"{pred_label}_" if pred_label else ""
    path = f"{DEBUG_DIR}/{label}{ts}.png"
    img.save(path)
    return path

# 0..255 -> 0..1 -> Normalize(0.5,0.5) => [-1,1]
def preprocess(pixels_0_255: List[int]) -> torch.Tensor:
    x = torch.tensor(pixels_0_255, dtype=torch.float32)

    # Clamp to guard against out-of-range values
    x = torch.clamp(x, 0.0, 255.0)

    x = x / 255.0
    x = (x - 0.5) / 0.5
    return x.view(1, -1)

# Extract top-K from probabilities
# returns: [{"i": class_index, "p": probability}, ...] sorted desc
def topk_from_probs(probs: torch.Tensor, k: int = 2) -> List[Dict[str, Any]]:
    k = max(1, min(k, probs.numel()))
    vals, idxs = torch.topk(probs, k=k)
    return [{"i": int(i.item()), "p": float(v.item())} for v, i in zip(vals, idxs)]

# Health check endpoint
#   Open http://localhost:8001/health to verify:
#   {"ok":true,"model_loaded":true,"debug_save":false}
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "model_loaded": net is not None, "debug_save": DEBUG_SAVE}

# API
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if len(req.pixels) != 784:
        raise HTTPException(status_code=400, detail=f"pixels length must be 784, got {len(req.pixels)}")

    if net is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    x = preprocess(req.pixels)

    with torch.no_grad():
        logits = net(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs).item())

    # Return only top-2
    top2 = topk_from_probs(probs, k=2)

    # Save image (for debugging)
    saved_path: Optional[str] = None
    if DEBUG_SAVE:
        saved_path = save_pixels_as_png(req.pixels, pred_label=f"pred{pred}")

    return PredictResponse(pred=pred, topk=top2, saved_path=saved_path)
