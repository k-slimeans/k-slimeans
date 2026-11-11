"""Was the GPU idle during a run? Snapshots, a pre-run idle check and a background sampler.

On a shared machine (here: WSL2 with a Windows desktop on the same card) another process can slow a run
without any trace in the run's own logs; the CLINC150 timings were inflated ~1.7x that way. This module
records what the card was doing:

- `gpu_snapshot` reads card-level memory, utilization and power from `nvidia-smi` (plus, best effort, the
  Windows-side process list from `nvidia-smi.exe`, which WSL's `nvidia-smi` cannot see),
- `check_idle` samples the card for a few seconds before a run and compares the median against thresholds,
- `GpuMonitor` samples it in a background thread during the run into `gpu_samples.csv`,
- `GpuSession` ties the three to a config key (`gpu_check`) and writes `gpu.json` into the run directory.

Everything degrades to an explicit "unavailable" result on machines without `nvidia-smi`. The subprocess
runner is injectable (`run`: argv list -> stdout, raising `OSError`/`SubprocessError` on failure) so tests
need no GPU.
"""

from __future__ import annotations

import csv
import json
import shutil
import statistics
import subprocess
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable

from .config import check_keys

Runner = Callable[[list[str]], str]

_GPU_FIELDS = ("index", "timestamp", "name", "memory.used", "memory.total", "utilization.gpu", "power.draw")
_GPU_QUERY = ["--query-gpu=" + ",".join(_GPU_FIELDS), "--format=csv,noheader,nounits"]
_APPS_QUERY = ["--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"]
_WINDOWS_SMI = "/mnt/c/Windows/System32/nvidia-smi.exe"

DEFAULT_THRESHOLDS = {"max_memory_used_mib": 1400, "max_utilization_pct": 15, "max_power_w": 65}
"""Idle limits on the median sample. Calibrated on this machine: the Windows desktop alone is ~1.0-1.1 GB,
2-3 % and ~43 W (a browser tab can add 200-500 MiB); the load seen before the CLINC150 run was ~1.5 GB, 43 %
and 85 W."""

SAMPLE_COLUMNS = ("timestamp", "memory_used_mib", "utilization_pct", "power_w", "torch_reserved_mib", "foreign_mib_est")


class GpuNotIdleError(RuntimeError):
    """Raised by a `require_idle: fail` policy when the pre-run check finds the GPU busy."""


_TIMEOUT_S = 5
_JOIN_TIMEOUT_S = 2 * _TIMEOUT_S


def _run_cmd(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT_S, check=True).stdout


def _find_windows_smi() -> str | None:
    """Path of the Windows `nvidia-smi.exe` reachable through WSL interop, if any."""
    if Path(_WINDOWS_SMI).exists():
        return _WINDOWS_SMI
    return shutil.which("nvidia-smi.exe")


def _number(field: str) -> float | None:
    """A numeric csv field, or None for `[N/A]`, `N/A`, `[Not Supported]` and the like."""
    try:
        return float(field)
    except ValueError:
        return None


def parse_gpu_query(text: str) -> list[dict]:
    """Rows of `nvidia-smi --query-gpu=<_GPU_FIELDS> --format=csv,noheader,nounits` (LF or CRLF)."""
    gpus = []
    for fields in csv.reader(text.splitlines(), skipinitialspace=True):
        if not fields:
            continue
        if len(fields) != len(_GPU_FIELDS):
            raise ValueError(f"Unexpected nvidia-smi row {fields!r}, expected {len(_GPU_FIELDS)} fields")
        index, timestamp, name, used, total, util, power = (f.strip() for f in fields)
        gpus.append(
            {
                "index": int(index),
                "timestamp": timestamp,
                "name": name,
                "memory_used_mib": _number(used),
                "memory_total_mib": _number(total),
                "utilization_pct": _number(util),
                "power_w": _number(power),
            }
        )
    return gpus


def parse_compute_apps(text: str) -> list[dict]:
    """Rows of `--query-compute-apps=pid,process_name,used_gpu_memory`; unreadable memory (WDDM) is None.

    Each row keeps its `raw` line. The process name may itself contain `, `, so only the first and last
    field are split off.
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(", ")
        pid = _number(parts[0])
        rows.append(
            {
                "pid": None if pid is None else int(pid),
                "process_name": ", ".join(parts[1:-1]) if len(parts) > 2 else "",
                "used_memory_mib": _number(parts[-1]) if len(parts) > 1 else None,
                "raw": line,
            }
        )
    return rows


def _windows_processes(run: Runner, windows_smi: str | None = None) -> list[dict] | None:
    """Rows from the Windows `nvidia-smi.exe`, or None if it is not reachable or fails."""
    exe = windows_smi or _find_windows_smi()
    if not exe:
        return None
    try:
        return parse_compute_apps(run([exe, *_APPS_QUERY]))
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def gpu_snapshot(run: Runner | None = None, processes: bool = True, windows_smi: str | None = None) -> dict:
    """Card-level state of every GPU, now.

    Returns `{"available": True, "gpus": [{index, name, timestamp, memory_used_mib, memory_total_mib,
    utilization_pct, power_w}, ...], "foreign_processes": [...] | None}`, or `{"available": False,
    "missing": bool, "error": ...}` if `nvidia-smi` is missing (`missing` True: a CPU-only machine) or fails
    (`missing` False: it exists but could not be queried). `foreign_processes` is the Windows-side list from
    `nvidia-smi.exe` (best effort; None if unreachable or `processes=False`). Under WDDM it contains the
    desktop's own graphics processes with `used_memory_mib` None, so it is evidence, not a verdict.
    """
    run = run or _run_cmd
    try:
        gpus = parse_gpu_query(run(["nvidia-smi", *_GPU_QUERY]))
    except (OSError, subprocess.SubprocessError, ValueError) as err:
        missing = isinstance(err, FileNotFoundError)
        return {"available": False, "missing": missing, "error": f"{type(err).__name__}: {err}"}
    if not gpus:
        return {"available": False, "missing": False, "error": "nvidia-smi listed no GPUs"}
    foreign = _windows_processes(run, windows_smi) if processes else None
    return {"available": True, "gpus": gpus, "foreign_processes": foreign}


def _first_gpu(snapshot: dict) -> dict:
    """The sample of GPU 0 (the one torch uses by default) in the fields the idle check and monitor use."""
    gpu = snapshot["gpus"][0]
    return {k: gpu[k] for k in ("timestamp", "memory_used_mib", "utilization_pct", "power_w")}


def check_idle(
    run: Runner | None = None,
    samples: int = 4,
    interval_s: float = 1.0,
    thresholds: dict | None = None,
    sleep: Callable[[float], None] | None = None,
    windows_smi: str | None = None,
) -> dict:
    """Sample GPU 0 `samples` times, `interval_s` apart, and compare the median of each quantity to `thresholds`.

    `thresholds` overrides entries of `DEFAULT_THRESHOLDS`. Returns `{"available", "idle", "reason",
    "samples", "median", "thresholds", "foreign_processes", "gpu_missing"}`: `idle` is True/False, or None
    when the GPU cannot be read (then `reason` says why, and `gpu_missing` tells a machine without
    `nvidia-smi` from one whose query failed). The Windows process list is read once, after the samples.
    The median keeps one transient spike from failing a run, while a sustained load still shows.
    Power or utilization that `nvidia-smi` does not report (None) is not checked.
    """
    if samples < 1:
        raise ValueError(f"samples must be >= 1, got {samples}")
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    unknown = set(limits) - set(DEFAULT_THRESHOLDS)
    if unknown:
        raise ValueError(f"Unknown idle threshold(s) {sorted(unknown)}, expected {sorted(DEFAULT_THRESHOLDS)}")
    sleep = sleep or time.sleep
    taken = []
    last = None
    for i in range(samples):
        if i:
            sleep(interval_s)
        last = gpu_snapshot(run, processes=False)
        if not last["available"]:
            return {"available": False, "idle": None, "reason": last["error"], "samples": taken, "median": None,
                    "thresholds": limits, "foreign_processes": None, "gpu_missing": last["missing"]}  # fmt: skip
        taken.append(_first_gpu(last))

    def median(key: str) -> float | None:
        values = [s[key] for s in taken if s[key] is not None]
        return statistics.median(values) if values else None

    med = {key: median(key) for key in ("memory_used_mib", "utilization_pct", "power_w")}
    breaches = []
    for key, limit_key, unit in (
        ("memory_used_mib", "max_memory_used_mib", "MiB"),
        ("utilization_pct", "max_utilization_pct", "%"),
        ("power_w", "max_power_w", "W"),
    ):
        if med[key] is not None and med[key] > limits[limit_key]:
            breaches.append(f"{key} {med[key]:g} {unit} > {limits[limit_key]:g}")
    reason = "GPU busy: " + "; ".join(breaches) if breaches else "median sample within all thresholds"
    processes = _windows_processes(run or _run_cmd, windows_smi)
    return {"available": True, "idle": not breaches, "reason": reason, "samples": taken, "median": med,
            "thresholds": limits, "foreign_processes": processes, "gpu_missing": False}  # fmt: skip


def _torch_reserved_mib() -> float | None:
    """This process's `torch.cuda.memory_reserved()` in MiB; None unless torch is loaded and CUDA initialized."""
    torch = sys.modules.get("torch")
    # `is_initialized` first: the sampler thread must never be the one to probe or initialize CUDA.
    if torch is None or not torch.cuda.is_initialized():
        return None
    return torch.cuda.memory_reserved(0) / 2**20


class GpuMonitor:
    """Context manager sampling GPU 0 every `interval_s` seconds into `path` (`gpu_samples.csv`).

    One sample is taken on entry and one on exit, so even a short run has two rows. The thread stops on
    normal exit and on exception alike. `foreign_mib_est` = card memory used - this process's reserved
    memory (never below 0); it includes the desktop baseline and the CUDA context, so compare it with the
    idle-check median rather than reading it as pure interference. A sample that fails for any reason is
    counted in `n_errors` and skipped; the thread keeps going. Without `nvidia-smi` nothing is written
    and `summary` is `{"available": False, ...}`. `reserved` and `clock` are injectable for tests.
    """

    def __init__(
        self,
        path: str | Path,
        interval_s: float = 5.0,
        run: Runner | None = None,
        reserved: Callable[[], float | None] = _torch_reserved_mib,
        clock: Callable[[], datetime] = datetime.now,
    ):
        if interval_s <= 0:
            raise ValueError(f"interval_s must be > 0, got {interval_s}")
        self.path = Path(path)
        self.interval_s = interval_s
        self._run = run
        self._reserved = reserved
        self._clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._file = None
        self._writer = None
        self.rows: list[dict] = []
        self.n_errors = 0

    def sample(self) -> dict | None:
        """Take one sample, append it to the csv and to `rows`; None (and `n_errors` += 1) if anything failed."""
        try:
            snapshot = gpu_snapshot(self._run, processes=False)
            if not snapshot["available"]:
                raise RuntimeError(snapshot["error"])
            gpu = _first_gpu(snapshot)
            reserved = self._reserved()
            used = gpu["memory_used_mib"]
            row = {
                "timestamp": self._clock().isoformat(timespec="seconds"),
                "memory_used_mib": used,
                "utilization_pct": gpu["utilization_pct"],
                "power_w": gpu["power_w"],
                "torch_reserved_mib": reserved,
                "foreign_mib_est": None if used is None or reserved is None else max(used - reserved, 0.0),
            }
            if self._writer is not None:
                self._writer.writerow({k: "" if v is None else v for k, v in row.items()})
                self._file.flush()
        except Exception:
            self.n_errors += 1
            return None
        self.rows.append(row)
        return row

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sample()

    def __enter__(self) -> GpuMonitor:
        if not gpu_snapshot(self._run, processes=False)["available"]:
            return self  # CPU-only machine: no file, no thread
        self._file = open(self.path, "w", newline="")
        try:
            self._writer = csv.DictWriter(self._file, fieldnames=SAMPLE_COLUMNS)
            self._writer.writeheader()
            self.sample()
            self._thread = threading.Thread(target=self._loop, name="gpu-monitor", daemon=True)
            self._thread.start()
        except BaseException:
            self._file.close()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        try:
            if self._thread is not None:
                self._thread.join(timeout=_JOIN_TIMEOUT_S)
                if not self._thread.is_alive():  # else it is stuck in a query; leave the file to it
                    self.sample()
        finally:
            if self._file is not None and (self._thread is None or not self._thread.is_alive()):
                self._file.close()

    @property
    def summary(self) -> dict:
        """Max / mean utilization, max power, max card memory, estimated max foreign memory, sample count."""

        def column(key: str) -> list[float]:
            return [r[key] for r in self.rows if r[key] is not None]

        util, power, used, foreign = (
            column(k) for k in ("utilization_pct", "power_w", "memory_used_mib", "foreign_mib_est")
        )
        return {
            "available": bool(self.rows),
            "n_samples": len(self.rows),
            "n_errors": self.n_errors,
            "interval_s": self.interval_s,
            "max_utilization_pct": max(util, default=None),
            "mean_utilization_pct": statistics.fmean(util) if util else None,
            "max_power_w": max(power, default=None),
            "max_memory_used_mib": max(used, default=None),
            "max_foreign_mib_est": max(foreign, default=None),
        }


_POLICIES = ("off", "warn", "fail")
_CONFIG_DEFAULTS = {
    "require_idle": "warn",
    "monitor": True,
    "interval_s": 5.0,
    "idle_samples": 4,
    "idle_interval_s": 1.0,
    **DEFAULT_THRESHOLDS,
}
_CONFIG_KEYS = frozenset(_CONFIG_DEFAULTS)


def parse_gpu_check(spec: dict | None) -> dict:
    """Validate a `gpu_check` config block and fill in defaults.

    Keys: `require_idle` (off | warn | fail), `monitor` (bool), `interval_s`, `idle_samples`,
    `idle_interval_s` and the three `max_*` thresholds of `DEFAULT_THRESHOLDS`. Unknown keys raise.
    """
    if spec is None:
        spec = {}
    if not isinstance(spec, dict):
        raise TypeError(f"'gpu_check' must be a dict, got {type(spec).__name__}")
    check_keys(spec, _CONFIG_KEYS, frozenset())
    cfg = {**_CONFIG_DEFAULTS, **spec}
    if cfg["require_idle"] not in _POLICIES:
        raise ValueError(f"gpu_check.require_idle must be one of {_POLICIES}, got {cfg['require_idle']!r}")
    if not isinstance(cfg["monitor"], bool):
        raise ValueError(f"gpu_check.monitor must be true or false, got {cfg['monitor']!r}")
    for key in _CONFIG_KEYS - {"require_idle", "monitor"}:
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"gpu_check.{key} must be a non-negative number, got {value!r}")
    for key in ("interval_s", "idle_samples"):
        if cfg[key] <= 0:
            raise ValueError(f"gpu_check.{key} must be > 0, got {cfg[key]!r}")
    cfg["idle_samples"] = int(cfg["idle_samples"])
    return cfg


class GpuSession:
    """The `gpu_check` policy for one run: pre-run idle check on entry, monitor from `attach` to exit.

    Used as `with GpuSession(cfg.get("gpu_check")) as gpu:` around a runner; the runner calls
    `gpu.attach(run.path)` once its run directory exists. On exit (normal or exception) the monitor is
    stopped and `gpu.json` is written into the run directory with the config, the idle-check result and the
    monitor summary. `require_idle: fail` raises `GpuNotIdleError` on entry, before any model is loaded;
    `warn` emits a `UserWarning`; `off` switches everything off (no `nvidia-smi` call, no files). A machine
    without `nvidia-smi` is never an error, but one where it exists and cannot be queried is: the run cannot
    be verified idle, so `fail` raises and `warn` warns.
    """

    def __init__(
        self,
        spec: dict | None = None,
        run: Runner | None = None,
        sleep: Callable[[float], None] | None = None,
        reserved: Callable[[], float | None] = _torch_reserved_mib,
    ):
        self.cfg = parse_gpu_check(spec)
        self._run = run
        self._reserved = reserved
        self._sleep = sleep
        self.idle_check: dict | None = None
        self.monitor: GpuMonitor | None = None
        self._run_dir: Path | None = None

    def __enter__(self) -> GpuSession:
        if self.cfg["require_idle"] == "off":
            return self
        self.idle_check = check_idle(
            self._run,
            samples=self.cfg["idle_samples"],
            interval_s=self.cfg["idle_interval_s"],
            thresholds={k: self.cfg[k] for k in DEFAULT_THRESHOLDS},
            sleep=self._sleep,
        )
        message = None
        if self.idle_check["idle"] is False:
            message = f"GPU is not idle before the run ({self.idle_check['reason']}); timings may be inflated"
        elif self.idle_check["idle"] is None and not self.idle_check["gpu_missing"]:
            message = f"Cannot verify the GPU is idle: nvidia-smi query failed ({self.idle_check['reason']})"
        if message is not None:
            if self.cfg["require_idle"] == "fail":
                raise GpuNotIdleError(message)
            warnings.warn(message, stacklevel=2)
        return self

    def attach(self, run_dir: str | Path) -> None:
        """Bind to the run directory: write `gpu.json` and start the monitor (if enabled and a GPU is readable).

        Does nothing under `require_idle: off`, which switches the whole feature off.
        """
        if self.cfg["require_idle"] == "off":
            return
        self._run_dir = Path(run_dir)
        self._write()
        if self.cfg["monitor"]:
            self.monitor = GpuMonitor(
                self._run_dir / "gpu_samples.csv", self.cfg["interval_s"], run=self._run, reserved=self._reserved
            )
            self.monitor.__enter__()

    def __exit__(self, *exc) -> None:
        try:
            if self.monitor is not None:
                self.monitor.__exit__(*exc)
        finally:
            if self._run_dir is not None:
                self._write()

    def _write(self) -> None:
        monitor = None if self.monitor is None else self.monitor.summary
        median = (self.idle_check or {}).get("median")
        baseline = median["memory_used_mib"] if median else None
        raw_max = None if monitor is None else monitor["max_foreign_mib_est"]
        # Non-torch card memory beyond what the card held before the run. It still contains our own CUDA
        # context (~0.3-0.5 GB here), so only values clearly above that point at another process.
        above = None if baseline is None or raw_max is None else max(raw_max - baseline, 0.0)
        data = {
            "gpu_check": self.cfg,
            "idle_check": self.idle_check,
            "monitor": monitor,
            "baseline_memory_used_mib": baseline,
            "non_torch_above_baseline_mib": above,
        }
        with open(self._run_dir / "gpu.json", "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
