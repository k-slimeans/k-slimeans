"""`gpu_monitor`: nvidia-smi parsing (real captured output), the unavailable path, idle thresholds, the
background sampler and the `gpu_check` policy. Everything runs against a fake runner; nothing sleeps or
touches a GPU.
"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from kslimeans import gpu_monitor
from kslimeans.gpu_monitor import (
    DEFAULT_THRESHOLDS,
    GpuMonitor,
    GpuNotIdleError,
    GpuSession,
    check_idle,
    gpu_snapshot,
    parse_compute_apps,
    parse_gpu_check,
    parse_gpu_query,
)

FIXTURES = Path(__file__).parent / "fixtures" / "nvidia_smi"
REAL_GPU = (FIXTURES / "gpu_query.csv").read_text()
REAL_APPS = (FIXTURES / "compute_apps_windows.csv").read_text()
EXE = "nvidia-smi.exe"


def gpu_row(used=1003, util=2, power=42.6, total=12288, index=0):
    return f"{index}, 2026/11/11 09:45:31.398, NVIDIA GeForce RTX 3060, {used}, {total}, {util}, {power}\n"


class FakeSmi:
    """Fake runner: serves a queue of `--query-gpu` outputs (the last one repeats) and an apps listing."""

    def __init__(self, gpu_outputs=(REAL_GPU,), apps=REAL_APPS):
        self.gpu_outputs = list(gpu_outputs)
        self.apps = apps
        self.calls = []

    def __call__(self, cmd):
        self.calls.append(cmd)
        if cmd[0] == EXE:
            if self.apps is None:
                raise OSError("no interop")
            return self.apps
        if len(self.gpu_outputs) > 1:
            return self.gpu_outputs.pop(0)
        return self.gpu_outputs[0]


def missing(cmd):
    raise FileNotFoundError(cmd[0])


@pytest.fixture
def no_sleep():
    return lambda seconds: None


# --- parsing ----------------------------------------------------------------------------------


def test_parse_real_gpu_query():
    (gpu,) = parse_gpu_query(REAL_GPU)
    assert gpu == {
        "index": 0,
        "timestamp": "2026/11/11 09:45:31.398",
        "name": "NVIDIA GeForce RTX 3060",
        "memory_used_mib": 1003.0,
        "memory_total_mib": 12288.0,
        "utilization_pct": 2.0,
        "power_w": 42.62,
    }


def test_parse_gpu_query_handles_crlf_na_and_several_gpus():
    text = "0, t0, A, 10, 100, [N/A], [Not Supported]\r\n1, t1, B, 20, 100, 5, 7.5\r\n"
    first, second = parse_gpu_query(text)
    assert first["utilization_pct"] is None and first["power_w"] is None
    assert (second["index"], second["memory_used_mib"], second["power_w"]) == (1, 20.0, 7.5)


def test_parse_gpu_query_rejects_malformed_rows():
    with pytest.raises(ValueError, match="Unexpected nvidia-smi row"):
        parse_gpu_query("0, only, three\n")


def test_parse_real_windows_compute_apps_keeps_raw_rows():
    rows = parse_compute_apps(REAL_APPS)  # the fixture has CRLF line endings, as nvidia-smi.exe prints them
    assert len(rows) == 8
    assert rows[0] == {"pid": 25056, "process_name": "[Insufficient Permissions]", "used_memory_mib": None,
                       "raw": "25056, [Insufficient Permissions], [N/A]"}  # fmt: skip
    assert rows[3]["process_name"] == "C:\\Windows\\explorer.exe"
    assert all(r["used_memory_mib"] is None for r in rows)  # WDDM does not report per-process memory


def test_parse_compute_apps_keeps_commas_in_names_and_reads_memory():
    (row,) = parse_compute_apps("42, C:\\a, b\\x.exe, 512\n")
    assert (row["pid"], row["process_name"], row["used_memory_mib"]) == (42, "C:\\a, b\\x.exe", 512.0)


# --- snapshot ---------------------------------------------------------------------------------


def test_snapshot_with_windows_processes():
    snap = gpu_snapshot(FakeSmi(), windows_smi=EXE)
    assert snap["available"] is True
    assert snap["gpus"][0]["memory_used_mib"] == 1003.0
    assert len(snap["foreign_processes"]) == 8
    json.dumps(snap)


def test_snapshot_without_windows_exe_or_with_failing_exe():
    assert gpu_snapshot(FakeSmi(), windows_smi=None)["foreign_processes"] is None  # conftest: exe not found
    assert gpu_snapshot(FakeSmi(apps=None), windows_smi=EXE)["foreign_processes"] is None
    assert gpu_snapshot(FakeSmi(), processes=False, windows_smi=EXE)["foreign_processes"] is None


@pytest.mark.parametrize(
    "runner",
    [
        missing,
        lambda cmd: (_ for _ in ()).throw(subprocess.CalledProcessError(9, cmd)),
        lambda cmd: "garbage\n",
        lambda cmd: "",
    ],
)
def test_snapshot_unavailable_never_raises(runner):
    snap = gpu_snapshot(runner)
    assert snap["available"] is False and snap["error"]
    assert "gpus" not in snap


# --- idle check -------------------------------------------------------------------------------


def idle(outputs, **kwargs):
    kwargs.setdefault("sleep", lambda s: None)
    return check_idle(FakeSmi(outputs), windows_smi=EXE, **kwargs)


def test_windows_desktop_baseline_is_idle():
    result = idle([gpu_row(used=1074, util=3, power=43.4)])
    assert result["idle"] is True and result["available"] is True
    assert len(result["samples"]) == 4
    assert result["thresholds"] == DEFAULT_THRESHOLDS
    assert len(result["foreign_processes"]) == 8


def test_load_seen_before_clinc150_is_flagged():
    result = idle([gpu_row(used=1500, util=43, power=85.0)])
    assert result["idle"] is False
    for part in ("memory_used_mib 1500", "utilization_pct 43", "power_w 85"):
        assert part in result["reason"]


@pytest.mark.parametrize(
    "row, key",
    [(gpu_row(used=1400), "memory_used_mib"), (gpu_row(util=30), "utilization_pct"), (gpu_row(power=90), "power_w")],
)
def test_each_threshold_alone_flags(row, key):
    result = idle([row])
    assert result["idle"] is False
    assert key in result["reason"] and result["reason"].count(";") == 0


def test_thresholds_are_configurable():
    assert idle([gpu_row(used=1500)], thresholds={"max_memory_used_mib": 2000})["idle"] is True
    assert idle([gpu_row()], thresholds={"max_memory_used_mib": 500})["idle"] is False
    with pytest.raises(ValueError, match="Unknown idle threshold"):
        idle([gpu_row()], thresholds={"max_gpu": 1})


def test_one_transient_spike_does_not_fail_the_median():
    rows = [gpu_row(util=2), gpu_row(util=90), gpu_row(util=3), gpu_row(util=2)]
    assert idle(rows)["idle"] is True
    sustained = [gpu_row(util=2), gpu_row(util=90), gpu_row(util=80), gpu_row(util=70)]
    assert idle(sustained)["idle"] is False


def test_unreported_power_is_not_checked():
    assert idle(["0, t, A, 1000, 12288, 2, [N/A]\n"])["idle"] is True


def test_samples_are_spaced_by_interval_without_sleeping():
    sleeps = []
    check_idle(FakeSmi(), samples=3, interval_s=2.5, sleep=sleeps.append, windows_smi=EXE)
    assert sleeps == [2.5, 2.5]
    with pytest.raises(ValueError):
        check_idle(FakeSmi(), samples=0)


def test_idle_check_unavailable_is_not_a_verdict():
    result = check_idle(missing, sleep=lambda s: None)
    assert result["available"] is False and result["idle"] is None and result["reason"]
    json.dumps(result)


# --- monitor ----------------------------------------------------------------------------------


class Ticker:
    """Fake clock: each call is one minute later."""

    def __init__(self):
        self.now = datetime(2026, 11, 11, 12, 0, 0)

    def __call__(self):
        self.now += timedelta(minutes=1)
        return self.now


def read_samples(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_sample_writes_columns_and_estimates_foreign_memory(tmp_path):
    smi = FakeSmi([gpu_row(used=3000, util=50, power=100)])
    monitor = GpuMonitor(tmp_path / "s.csv", run=smi, reserved=lambda: 2000.0, clock=Ticker())
    monitor.__enter__()
    monitor._stop.set()  # stop the loop thread from sampling; exercise sample() by hand
    monitor._thread.join()
    row = monitor.sample()
    monitor.__exit__(None, None, None)
    assert row == {
        "timestamp": "2026-11-11T12:02:00",
        "memory_used_mib": 3000.0,
        "utilization_pct": 50.0,
        "power_w": 100.0,
        "torch_reserved_mib": 2000.0,
        "foreign_mib_est": 1000.0,
    }
    rows = read_samples(tmp_path / "s.csv")
    assert list(rows[0]) == list(gpu_monitor.SAMPLE_COLUMNS)
    assert len(rows) == len(monitor.rows)


def test_foreign_estimate_is_blank_without_cuda_and_never_negative(tmp_path):
    monitor = GpuMonitor(tmp_path / "s.csv", run=FakeSmi(), reserved=lambda: None, clock=Ticker())
    with monitor:
        pass
    assert [r["foreign_mib_est"] for r in read_samples(tmp_path / "s.csv")] == ["", ""]
    assert monitor.summary["max_foreign_mib_est"] is None

    over = GpuMonitor(tmp_path / "t.csv", run=FakeSmi([gpu_row(used=100)]), reserved=lambda: 500.0)
    with over:
        pass
    assert over.summary["max_foreign_mib_est"] == 0.0


def test_monitor_samples_on_entry_and_exit_and_summarizes(tmp_path):
    # reads: availability probe, entry sample, exit sample
    smi = FakeSmi([gpu_row(used=1000, util=10), gpu_row(used=1000, util=10), gpu_row(used=2500, util=90, power=120)])
    reserved = iter([100.0, 200.0, 300.0])
    monitor = GpuMonitor(tmp_path / "s.csv", interval_s=1e6, run=smi, reserved=lambda: next(reserved), clock=Ticker())
    with monitor:
        pass  # the loop thread waits a million seconds; stop() wakes it immediately
    assert not monitor._thread.is_alive()
    assert [r["torch_reserved_mib"] for r in monitor.rows] == [100.0, 200.0]  # entry and exit samples only
    assert [r["memory_used_mib"] for r in monitor.rows] == [1000.0, 2500.0]
    assert monitor.summary["max_utilization_pct"] == 90.0


def test_summary_statistics(tmp_path):
    monitor = GpuMonitor(tmp_path / "s.csv", run=FakeSmi(), reserved=lambda: 100.0)
    monitor.rows = [
        {"utilization_pct": 10.0, "power_w": 50.0, "memory_used_mib": 1000.0, "foreign_mib_est": 900.0},
        {"utilization_pct": 30.0, "power_w": 90.0, "memory_used_mib": 2000.0, "foreign_mib_est": 1900.0},
        {"utilization_pct": None, "power_w": None, "memory_used_mib": 1500.0, "foreign_mib_est": None},
    ]
    assert monitor.summary == {
        "available": True,
        "n_samples": 3,
        "n_errors": 0,
        "interval_s": 5.0,
        "max_utilization_pct": 30.0,
        "mean_utilization_pct": 20.0,
        "max_power_w": 90.0,
        "max_memory_used_mib": 2000.0,
        "max_foreign_mib_est": 1900.0,
    }


def test_monitor_stops_cleanly_when_the_body_raises(tmp_path):
    monitor = GpuMonitor(tmp_path / "s.csv", interval_s=1e6, run=FakeSmi(), reserved=lambda: None)
    with pytest.raises(RuntimeError, match="boom"):
        with monitor:
            raise RuntimeError("boom")
    assert not monitor._thread.is_alive()
    assert monitor._file.closed
    assert len(read_samples(tmp_path / "s.csv")) == 2


def test_monitor_survives_a_failing_read_mid_run(tmp_path):
    calls = []

    def flaky(cmd):
        calls.append(cmd)
        if len(calls) == 3:
            raise OSError("transient")
        return REAL_GPU

    monitor = GpuMonitor(tmp_path / "s.csv", interval_s=1e6, run=flaky, reserved=lambda: None)
    with monitor:
        monitor.sample()  # entry probe, entry sample, then this one fails
    assert monitor.n_errors == 1
    assert monitor.summary["n_errors"] == 1


def test_monitor_without_nvidia_smi_writes_nothing(tmp_path):
    monitor = GpuMonitor(tmp_path / "s.csv", run=missing)
    with monitor:
        pass
    assert not (tmp_path / "s.csv").exists()
    assert monitor._thread is None
    assert monitor.summary["available"] is False and monitor.summary["n_samples"] == 0


def test_monitor_rejects_nonpositive_interval(tmp_path):
    with pytest.raises(ValueError):
        GpuMonitor(tmp_path / "s.csv", interval_s=0)


# --- config + session -------------------------------------------------------------------------


def test_parse_gpu_check_defaults_and_overrides():
    cfg = parse_gpu_check(None)
    assert cfg["require_idle"] == "warn" and cfg["monitor"] is True and cfg["interval_s"] == 5.0
    assert {k: cfg[k] for k in DEFAULT_THRESHOLDS} == DEFAULT_THRESHOLDS
    cfg = parse_gpu_check({"require_idle": "fail", "interval_s": 2, "max_power_w": 70})
    assert (cfg["require_idle"], cfg["interval_s"], cfg["max_power_w"]) == ("fail", 2, 70)


@pytest.mark.parametrize(
    "spec, match",
    [
        ({"require_idel": "warn"}, "Unknown config key"),
        ({"require_idle": "maybe"}, "require_idle"),
        ({"monitor": "yes"}, "monitor"),
        ({"interval_s": 0}, "interval_s"),
        ({"interval_s": "fast"}, "interval_s"),
        ({"idle_samples": 0}, "idle_samples"),
        ({"max_power_w": -1}, "max_power_w"),
        ("warn", "must be a dict"),
    ],
)
def test_parse_gpu_check_rejects_bad_specs(spec, match):
    with pytest.raises((ValueError, TypeError), match=match):
        parse_gpu_check(spec)


def test_session_warns_when_busy_and_still_records(tmp_path, no_sleep):
    smi = FakeSmi([gpu_row(used=1500, util=43, power=85)])
    with pytest.warns(UserWarning, match="not idle"):
        with GpuSession({"interval_s": 1e6}, run=smi, sleep=no_sleep) as gpu:
            gpu.attach(tmp_path)
    data = json.loads((tmp_path / "gpu.json").read_text())
    assert data["idle_check"]["idle"] is False
    assert data["monitor"]["n_samples"] >= 2 and data["monitor"]["max_utilization_pct"] == 43.0
    assert data["gpu_check"]["require_idle"] == "warn"
    assert (tmp_path / "gpu_samples.csv").exists()


def test_session_fail_policy_raises_before_attach(tmp_path, no_sleep):
    smi = FakeSmi([gpu_row(used=1500, util=43, power=85)])
    with pytest.raises(GpuNotIdleError, match="not idle"):
        with GpuSession({"require_idle": "fail"}, run=smi, sleep=no_sleep):
            raise AssertionError("body must not run")


def test_session_fail_policy_passes_on_an_idle_gpu_and_ignores_missing_gpu(tmp_path, no_sleep):
    with GpuSession({"require_idle": "fail", "interval_s": 1e6}, run=FakeSmi(), sleep=no_sleep) as gpu:
        gpu.attach(tmp_path)
    assert json.loads((tmp_path / "gpu.json").read_text())["idle_check"]["idle"] is True

    other = tmp_path / "cpu"
    other.mkdir()
    with GpuSession({"require_idle": "fail"}, run=missing, sleep=no_sleep) as gpu:
        gpu.attach(other)
    data = json.loads((other / "gpu.json").read_text())
    assert data["idle_check"]["idle"] is None and data["monitor"]["available"] is False
    assert not (other / "gpu_samples.csv").exists()


def test_session_off_and_monitor_false(tmp_path):
    smi = FakeSmi()
    with GpuSession({"require_idle": "off", "monitor": False}, run=smi) as gpu:
        gpu.attach(tmp_path)
    assert smi.calls == []
    data = json.loads((tmp_path / "gpu.json").read_text())
    assert data["idle_check"] is None and data["monitor"] is None
    assert not (tmp_path / "gpu_samples.csv").exists()


def test_session_writes_gpu_json_when_the_run_raises(tmp_path, no_sleep):
    with pytest.raises(RuntimeError, match="boom"):
        with GpuSession({"interval_s": 1e6}, run=FakeSmi(), sleep=no_sleep) as gpu:
            gpu.attach(tmp_path)
            raise RuntimeError("boom")
    data = json.loads((tmp_path / "gpu.json").read_text())
    assert data["monitor"]["n_samples"] >= 2
