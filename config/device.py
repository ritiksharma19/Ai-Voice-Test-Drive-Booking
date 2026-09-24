"""
config/device.py
GPU-first device selection for local models (NVIDIA CUDA → CPU fallback).

Does not import torch: faster-whisper runs on CTranslate2, which reports CUDA
availability itself. On Windows, the cuBLAS/cuDNN DLLs shipped by the
`nvidia-cublas-cu12` / `nvidia-cudnn-cu12` pip wheels are registered on the
DLL search path so no system-wide CUDA toolkit install is required.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from config.logging_config import get_logger

logger = get_logger("config.device")


@lru_cache(maxsize=1)
def register_cuda_dlls() -> list[str]:
    """Add pip-installed NVIDIA runtime DLL folders to the Windows search path."""
    if sys.platform != "win32":
        return []
    added: list[str] = []
    for base in map(Path, sys.path):
        nvidia_root = base / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for bin_dir in nvidia_root.glob("*/bin"):
            path = str(bin_dir)
            try:
                os.add_dll_directory(path)
            except OSError:
                continue
            # CTranslate2 loads cuBLAS/cuDNN with LoadLibrary, which consults PATH.
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
            added.append(path)
    if added:
        logger.info("Registered %d NVIDIA DLL folder(s) from pip wheels", len(added))
    return added


@lru_cache(maxsize=1)
def cuda_device_count() -> int:
    register_cuda_dlls()
    try:
        import ctranslate2  # type: ignore
        return ctranslate2.get_cuda_device_count()
    except Exception as exc:  # pragma: no cover - depends on host
        logger.debug("CUDA probe failed: %s", exc)
        return 0


def get_optimal_device(preferred: str = "auto") -> str:
    """'cuda' when an NVIDIA GPU is usable, else 'cpu'. `preferred` can force either."""
    if preferred in ("cuda", "cpu"):
        return preferred
    return "cuda" if cuda_device_count() > 0 else "cpu"


def get_whisper_device(preferred: str = "auto", compute_type: str = "auto") -> tuple[str, str]:
    """(device, compute_type) for faster-whisper / CTranslate2."""
    device = get_optimal_device(preferred)
    if compute_type != "auto":
        return device, compute_type
    # float16 is the fastest accurate path on any CUDA GPU from Pascal onward;
    # int8 keeps CPU inference usable.
    return device, ("float16" if device == "cuda" else "int8")
