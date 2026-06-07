import torch
import numpy as np

def select_device(device: str = "") -> str:
    """
    Select the best available device (cuda or cpu).
    If 'cuda' is requested but not available, fall back to 'cpu'.
    """
    device = str(device).lower().strip()
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    if not device:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

def check_requirements():
    """Check for necessary environment requirements."""
    pass
