"""YAML config loading with dotted-key overrides, e.g. from a CLI's `--set` flags.

Also home to two tiny helpers shared by the offline (`experiment.py`) and streaming
(`scripts/run_streaming.py`) runners: `check_keys` (a flat typo-guard for top-level config
keys) and `default_method_name` (the `method_name` label to fall back to when a config
doesn't set one explicitly).
"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config file and apply `key.sub=value` overrides on top of it.

    Each override's value is parsed with `yaml.safe_load`, so numbers,
    booleans, lists, and strings all work the same as they would written
    directly in the YAML file (e.g. `"top_k=5"`, `"schedule.final_iter=false"`).
    """
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return apply_overrides(cfg, overrides)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply `key.sub=value` overrides (values parsed with `yaml.safe_load`) to `cfg` in place; returns `cfg`."""
    for override in overrides or []:
        key, sep, raw_value = override.partition("=")
        if not sep:
            raise ValueError(f"Override '{override}' is missing '=value'")
        _set_dotted(cfg, key.split("."), yaml.safe_load(raw_value))

    return cfg


def _set_dotted(cfg: dict, keys: list[str], value) -> None:
    """Set `cfg[keys[0]][keys[1]]...[keys[-1]] = value`, creating missing intermediate
    dicts as needed.

    Raises `ValueError` if an intermediate key already exists but isn't a dict (e.g.
    `dataset: bank77` in the config, then `--set dataset.year=2023`) -- `setdefault`
    would otherwise silently try to subscript that scalar and raise a confusing
    `TypeError` instead.
    """
    node = cfg
    for depth, key in enumerate(keys[:-1]):
        existing = node.get(key)
        if key not in node:
            node[key] = {}
        elif not isinstance(existing, dict):
            collision = ".".join(keys[: depth + 1])
            full_key = ".".join(keys)
            remainder = ".".join(keys[depth + 1 :])
            raise ValueError(
                f"Can't set '{full_key}': '{collision}' is already {existing!r} (not a dict). "
                f"Set the whole value as a dict instead, e.g. "
                f"--set '{collision}={{name: {existing}, {remainder}: {value}}}'"
            )
        node = node[key]
    node[keys[-1]] = value


def check_keys(cfg: dict, allowed: frozenset[str], required: frozenset[str]) -> None:
    """Typo-guard: raise if `cfg` has a key outside `allowed`, or is missing one in `required`.

    Not a schema validator -- nested config blocks (`schedule`, `summarizer`, ...) are
    validated by whatever builds from them (`build_schedule`, `build_summarizer`, ...).
    Shared by `experiment.run_offline` and `scripts/run_streaming.py` so a typo'd
    `--set` override fails immediately instead of being silently ignored.
    """
    unknown = set(cfg) - set(allowed)
    if unknown:
        raise ValueError(
            f"Unknown config key(s) {sorted(unknown)}, expected a subset of {sorted(allowed)} "
            "(typo in a --set override?)"
        )
    missing = set(required) - set(cfg)
    if missing:
        raise ValueError(f"Missing required config key(s) {sorted(missing)}")


def default_method_name(summarizer_cfg: dict) -> str:
    """`method_name` label to use when a config doesn't set one explicitly."""
    return "k-LLMmeans" if summarizer_cfg.get("type") == "hf" else "k-NLPmeans"
