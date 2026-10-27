"""Facts about the machine and software behind a run, and GPU peak-memory probes.

`collect_environment` feeds `RunDir.log_environment` (`environment.json`); the peak-memory helpers
bracket each seed's summary-method run. `torch` is imported lazily and every probe degrades to
`None`, so CPU-only machines and machines without git or nvidia-smi still work.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

_PACKAGES = ("transformers", "sentence-transformers", "scikit-learn", "numpy")


def _cuda_torch():
    """The `torch` module if CUDA is usable, else None."""
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


def reset_peak_gpu_memory() -> None:
    """Reset CUDA's peak-memory counters (no-op without CUDA)."""
    torch = _cuda_torch()
    if torch is not None:
        torch.cuda.reset_peak_memory_stats()


def peak_gpu_memory_gb() -> tuple[float | None, float | None]:
    """`(max allocated, max reserved)` in GB since the last reset; `(None, None)` without CUDA."""
    torch = _cuda_torch()
    if torch is None:
        return None, None
    return torch.cuda.max_memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9


def _run(cmd: list[str]) -> str | None:
    """Stripped stdout of `cmd`, or None on any failure."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out or None


def _git_state() -> dict:
    repo = Path(__file__).resolve().parent
    commit = _run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    if commit is None:
        return {"git_commit": None, "git_dirty": None}
    # Not `_run`: it maps empty output (clean) and a failed command (unknown) both to None.
    try:
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, timeout=10, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": commit, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(status.strip())}


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _torch_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"torch": None}
    cuda = torch.cuda.is_available()
    info = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": cuda,
        "gpu_name": None,
        "gpu_total_memory_gb": None,
    }
    if cuda:
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_total_memory_gb"] = props.total_memory / 1e9
    return info


def collect_environment(summarizer=None, embedder: str | None = None, dataset=None) -> dict:
    """A JSON-serializable description of the run environment.

    `summarizer` contributes its `describe()`, `dataset` (a `data.Dataset`) its name and size.
    """
    uname = platform.uname()
    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        **_git_state(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "is_wsl": "microsoft" in uname.release.lower(),
        **_torch_info(),
        "nvidia_driver": _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "packages": {name: _version(name) for name in _PACKAGES},
        "summarizer": None if summarizer is None else summarizer.describe(),
        "embedder": embedder,
        "dataset": None
        if dataset is None
        else {"name": dataset.name, "n_texts": len(dataset.texts), "n_clusters": dataset.n_clusters},
    }
