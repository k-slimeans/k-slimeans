"""`scripts/run_streaming.py`'s config typo-guard and `method_name` support.

`run_streaming.run_streaming(cfg)` mirrors `kslimeans.experiment.run_offline(cfg)`:
config validation (`check_keys`) is the very first thing it does, so these tests can
exercise it with a deliberately incomplete config and never touch a real dataset,
embedder, or summarizer. The script lives outside the installed `kslimeans` package, so
it's loaded from its file path via `importlib` (same approach as
`tests/fixtures/make_fixtures.py`'s `_load_module`).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_streaming.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_streaming_module():
    return _load_module("run_streaming_under_test", SCRIPT_PATH)


def _minimal_config(**overrides) -> dict:
    cfg = {
        "name": "stream_run",
        "dataset": {"name": "stackexchange", "limit": 100},
        "years": [2023],
        "embedder": "distilbert",
        "seeds": [0],
        "max_batch_size": 1000,
        "eval_assign": "lloyd1",
        "baselines": {"kmeans": {}},
        "force_context_length": 0,
        "tol": 1.0e-4,
        "schedule": {"type": "fixed", "n_summaries": 5, "max_iter": 120},
        "summarizer": {"type": "lsa", "top_k": 5},
        "cache_dir": "cache",
        "results_dir": "results",
    }
    cfg.update(overrides)
    return cfg


def test_unknown_key_raises_before_touching_any_dataset(run_streaming_module):
    cfg = _minimal_config(bogus_key="oops")
    with pytest.raises(ValueError, match="bogus_key"):
        run_streaming_module.run_streaming(cfg)


def test_missing_required_key_raises(run_streaming_module):
    cfg = _minimal_config()
    del cfg["max_batch_size"]
    with pytest.raises(ValueError, match="max_batch_size"):
        run_streaming_module.run_streaming(cfg)


def test_method_name_is_allowed_but_not_required(run_streaming_module):
    from kslimeans.config import check_keys

    assert "method_name" in run_streaming_module._ALLOWED_KEYS
    assert "method_name" not in run_streaming_module._REQUIRED_KEYS

    allowed, required = run_streaming_module._ALLOWED_KEYS, run_streaming_module._REQUIRED_KEYS
    cfg = _minimal_config()
    assert "method_name" not in cfg
    check_keys(cfg, allowed, required)  # must not raise: method_name is optional

    check_keys(_minimal_config(method_name="k-NLPmeans"), allowed, required)  # explicit value is fine too
