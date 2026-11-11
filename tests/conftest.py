"""Keep the suite hermetic: runs behave as on a machine without `nvidia-smi` unless a test injects a runner."""

import pytest

from kslimeans import gpu_monitor


@pytest.fixture(autouse=True)
def _no_nvidia_smi(monkeypatch):
    def missing(cmd):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(gpu_monitor, "_run_cmd", missing)
    monkeypatch.setattr(gpu_monitor, "_find_windows_smi", lambda: None)
