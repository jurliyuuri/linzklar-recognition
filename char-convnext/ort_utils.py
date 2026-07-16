"""ONNX Runtime helpers (CUDA lib preload + session factory)."""

from __future__ import annotations

import ctypes
import os
import site
from pathlib import Path

_LIBS_READY = False


def _nvidia_lib_dirs() -> list[Path]:
    candidates: list[Path] = []
    try:
        import nvidia  # type: ignore

        for root in getattr(nvidia, "__path__", []):
            root_path = Path(root)
            for libdir in root_path.glob("*/lib"):
                if libdir.is_dir():
                    candidates.append(libdir)
            for libdir in root_path.glob("**/lib"):
                if libdir.is_dir() and libdir not in candidates:
                    candidates.append(libdir)
    except ImportError:
        pass

    search_roots = list(site.getsitepackages())
    try:
        search_roots.append(site.getusersitepackages())
    except Exception:
        pass

    for sp in search_roots:
        nvidia_root = Path(sp) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for libdir in nvidia_root.glob("*/lib"):
            if libdir.is_dir() and libdir not in candidates:
                candidates.append(libdir)

    # Prefer CUDA 13 paths first (onnxruntime-gpu 1.27+).
    return sorted(candidates, key=lambda p: (0 if "cu13" in p.parts else 1, str(p)))


def ensure_nvidia_lib_path() -> None:
    """Make pip-shipped NVIDIA shared libs visible to onnxruntime-gpu.

    Setting ``LD_LIBRARY_PATH`` after process start is unreliable for ``dlopen``,
    so we also ``ctypes.CDLL(..., RTLD_GLOBAL)`` preload key CUDA libraries.
    """
    global _LIBS_READY
    if _LIBS_READY:
        return

    lib_dirs = _nvidia_lib_dirs()
    if lib_dirs:
        prefix = os.pathsep.join(str(p) for p in lib_dirs)
        current = os.environ.get("LD_LIBRARY_PATH", "")
        if prefix not in current:
            os.environ["LD_LIBRARY_PATH"] = (
                prefix if not current else f"{prefix}{os.pathsep}{current}"
            )

    # Preload shared libs from nvidia wheels (CUDA EP needs many of these).
    # Load CUDA 13 / cuDNN first, then remaining dirs. Skip static archives.
    ordered_dirs = sorted(
        lib_dirs,
        key=lambda p: (
            0 if "cu13" in p.parts else 1 if "cudnn" in p.parts else 2,
            str(p),
        ),
    )
    loaded: set[str] = set()
    for libdir in ordered_dirs:
        for path in sorted(libdir.glob("lib*.so*")):
            if not path.is_file() or path.name in loaded:
                continue
            # Skip pure symlinks that already resolve to a loaded file name.
            try:
                ctypes.CDLL(str(path.resolve()), mode=ctypes.RTLD_GLOBAL)
                loaded.add(path.name)
            except OSError:
                continue

    _LIBS_READY = True


def make_session(onnx_path: str | Path, providers: str = "auto"):
    """Create an optimized InferenceSession with CUDA when available."""
    ensure_nvidia_lib_path()
    import onnxruntime as ort
    from onnxruntime import InferenceSession, SessionOptions
    from onnxruntime import GraphOptimizationLevel

    opts = SessionOptions()
    opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    available = ort.get_available_providers()
    if providers == "auto":
        chosen = []
        if "CUDAExecutionProvider" in available:
            chosen.append("CUDAExecutionProvider")
        chosen.append("CPUExecutionProvider")
    else:
        chosen = [p.strip() for p in providers.split(",") if p.strip()]

    return InferenceSession(str(onnx_path), sess_options=opts, providers=chosen)
