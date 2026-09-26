"""`load_config`'s YAML parsing + dotted-key overrides, and the shared `check_keys`/
`default_method_name` helpers used by both the offline and streaming runners.
"""

from __future__ import annotations

import pytest

from kslimeans.config import check_keys, default_method_name, load_config

# ----------------------------------------------------------------------------
# load_config: --set override parsing (scalar/list/bool/nested), via yaml.safe_load.
# ----------------------------------------------------------------------------


def _write_yaml(tmp_path, text: str):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_load_config_with_no_overrides_returns_the_yaml_as_is(tmp_path):
    path = _write_yaml(tmp_path, "name: run1\ndataset: bank77\n")
    cfg = load_config(path)
    assert cfg == {"name": "run1", "dataset": "bank77"}


def test_load_config_scalar_override_is_parsed_as_yaml(tmp_path):
    path = _write_yaml(tmp_path, "top_k: 5\n")
    cfg = load_config(path, ["top_k=10"])
    assert cfg["top_k"] == 10
    assert isinstance(cfg["top_k"], int)


def test_load_config_string_override_stays_a_string(tmp_path):
    path = _write_yaml(tmp_path, "dataset: bank77\n")
    cfg = load_config(path, ["dataset=clinic"])
    assert cfg["dataset"] == "clinic"


def test_load_config_bool_override_is_parsed_as_a_bool(tmp_path):
    path = _write_yaml(tmp_path, "schedule:\n  final_iter: true\n")
    cfg = load_config(path, ["schedule.final_iter=false"])
    assert cfg["schedule"]["final_iter"] is False


def test_load_config_list_override_is_parsed_as_a_list(tmp_path):
    path = _write_yaml(tmp_path, "seeds: [0]\n")
    cfg = load_config(path, ["seeds=[0, 1, 2]"])
    assert cfg["seeds"] == [0, 1, 2]


def test_load_config_dict_override_is_parsed_as_a_dict(tmp_path):
    path = _write_yaml(tmp_path, "dataset: bank77\n")
    cfg = load_config(path, ["dataset={name: bank77, year: 2023}"])
    assert cfg["dataset"] == {"name": "bank77", "year": 2023}


def test_load_config_override_missing_equals_raises(tmp_path):
    path = _write_yaml(tmp_path, "name: run1\n")
    with pytest.raises(ValueError, match="missing '=value'"):
        load_config(path, ["top_k"])


def test_load_config_creates_missing_nested_keys_as_dicts(tmp_path):
    path = _write_yaml(tmp_path, "name: run1\n")
    cfg = load_config(path, ["schedule.max_iter=50", "schedule.n_summaries=5"])
    assert cfg["schedule"] == {"max_iter": 50, "n_summaries": 5}


def test_load_config_creates_multiple_levels_of_missing_nested_keys(tmp_path):
    path = _write_yaml(tmp_path, "name: run1\n")
    cfg = load_config(path, ["a.b.c=1"])
    assert cfg["a"] == {"b": {"c": 1}}


def test_load_config_empty_yaml_file_is_an_empty_dict(tmp_path):
    path = _write_yaml(tmp_path, "")
    cfg = load_config(path, ["name=run1"])
    assert cfg == {"name": "run1"}


# ----------------------------------------------------------------------------
# _set_dotted: the collision error when an intermediate key is a scalar, not a dict.
# ----------------------------------------------------------------------------


def test_set_dotted_override_on_a_scalar_intermediate_key_raises_a_clear_error(tmp_path):
    """`--set dataset.year=2023` when `dataset: bank77` used to crash with a confusing
    `TypeError: 'str' object does not support item assignment` (`setdefault` returns the
    existing string, then `[keys[-1]] = value` tries to subscript it). It should instead
    name the dotted key and the colliding value, and suggest the dict form."""
    path = _write_yaml(tmp_path, "dataset: bank77\n")
    with pytest.raises(ValueError) as excinfo:
        load_config(path, ["dataset.year=2023"])

    message = str(excinfo.value)
    assert "dataset.year" in message
    assert "bank77" in message
    assert "dataset={name: bank77, year: 2023}" in message


def test_set_dotted_collision_deeper_in_the_path_names_the_colliding_prefix(tmp_path):
    path = _write_yaml(tmp_path, "a:\n  b: 3\n")
    with pytest.raises(ValueError) as excinfo:
        load_config(path, ["a.b.c=1"])

    message = str(excinfo.value)
    assert "a.b.c" in message
    assert "a.b" in message
    assert "3" in message


def test_set_dotted_does_not_raise_when_intermediate_key_already_is_a_dict(tmp_path):
    path = _write_yaml(tmp_path, "dataset: {name: bank77}\n")
    cfg = load_config(path, ["dataset.year=2023"])
    assert cfg["dataset"] == {"name": "bank77", "year": 2023}


# ----------------------------------------------------------------------------
# check_keys: the shared typo-guard used by both experiment.py and run_streaming.py.
# ----------------------------------------------------------------------------


def test_check_keys_passes_for_an_allowed_and_complete_config():
    check_keys({"name": "x", "dataset": "y"}, frozenset({"name", "dataset", "extra"}), frozenset({"name"}))


def test_check_keys_raises_on_an_unknown_key():
    with pytest.raises(ValueError, match="bogus_key"):
        check_keys({"name": "x", "bogus_key": 1}, frozenset({"name"}), frozenset({"name"}))


def test_check_keys_raises_on_a_missing_required_key():
    with pytest.raises(ValueError, match="dataset"):
        check_keys({"name": "x"}, frozenset({"name", "dataset"}), frozenset({"name", "dataset"}))


# ----------------------------------------------------------------------------
# default_method_name
# ----------------------------------------------------------------------------


def test_default_method_name_is_k_llmmeans_for_the_hf_summarizer():
    assert default_method_name({"type": "hf", "model_id": "x"}) == "k-LLMmeans"


@pytest.mark.parametrize("summarizer_type", ["lsa", "centroid", "textrank", "lexrank"])
def test_default_method_name_is_k_nlpmeans_for_extractive_summarizers(summarizer_type):
    assert default_method_name({"type": summarizer_type}) == "k-NLPmeans"
