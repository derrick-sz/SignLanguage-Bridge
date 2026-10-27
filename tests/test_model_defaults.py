"""Task-36: environment-aware default pose model resolution.

Frozen baselines under ``experiments/p1/**`` are never modified by this layer;
it only decides *which* bundle an environment points at:

    dev / test  -> S2 (stage2 pose subset, pose-only equivalent build)
    prod        -> A2 (pinned pose-only stage1)
    unset       -> prod (safe default: S2 must be opted into, never inherited)

The real ``configs/model_defaults.json`` is local-only (gitignored); the
committed ``configs/model_defaults.example.json`` documents the schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "configs" / "model_defaults.example.json"
LOCAL = REPO / "configs" / "model_defaults.json"
STAGE2_BUNDLE = REPO / "experiments" / "p1" / "e01" / "runs" / "stage2-pose-subset-pilot50-20261027" / "pose_bundle"
STAGE1_BUNDLE = REPO / "models" / "external" / "uni_sign" / "model_bundle"
STAGE2_SHA = "951601a39fbbe19f633e0df9690df432056b4286609f75a3ee723c8f001fc2e3"
STAGE1_SHA = "73bea87a9603f719b4f4a481a3a1a5ed24e7e5f7b81afd14ef498c40469c1a6c"


@pytest.fixture(scope="module")
def defaults_module():
    return __import__("slbridge.model_defaults", fromlist=[""])


def _clear_env(monkeypatch) -> None:
    monkeypatch.delenv("SLBRIDGE_ENV", raising=False)


def _models() -> dict:
    return {
        "unisign_stage2_pose": {
            "adapter_id": "uni_sign_pose_slt",
            "bundle": "experiments/p1/e01/runs/stage2-pose-subset-pilot50-20261027/pose_bundle",
            "checkpoint_sha256": STAGE2_SHA,
        },
        "unisign_pose_stage1": {
            "adapter_id": "uni_sign_pose_slt",
            "bundle": "models/external/uni_sign/model_bundle",
            "checkpoint_sha256": STAGE1_SHA,
        },
    }


def _environments() -> dict:
    return {
        "dev": {"pose_default": "unisign_stage2_pose",
                "pose_fallback": ["unisign_pose_stage1"]},
        "test": {"pose_default": "unisign_stage2_pose",
                 "pose_fallback": ["unisign_pose_stage1"]},
        "prod": {"pose_default": "unisign_pose_stage1", "pose_fallback": []},
    }


def _config(tmp_path: Path, *, environments: dict | None = None,
            models: dict | None = None, schema_version: int = 1) -> Path:
    payload = {
        "schema_version": schema_version,
        "environments": _environments() if environments is None else environments,
        "models": _models() if models is None else models,
        "notes": "pytest fixture",
    }
    path = tmp_path / "model_defaults.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# environment -> model mapping
# ---------------------------------------------------------------------------

def test_dev_and_test_select_stage2(defaults_module, tmp_path: Path, monkeypatch):
    _clear_env(monkeypatch)
    defaults = defaults_module.load_model_defaults(_config(tmp_path))
    for environment in ("dev", "test"):
        resolved = defaults_module.resolve_pose_default(environment, defaults=defaults)
        assert resolved.name == "unisign_stage2_pose"
        assert resolved.adapter_id == "uni_sign_pose_slt"
        assert resolved.checkpoint_sha256 == STAGE2_SHA
        assert resolved.bundle == STAGE2_BUNDLE
        assert resolved.bundle.is_absolute()
        assert resolved.environment == environment


def test_prod_keeps_a2(defaults_module, tmp_path: Path, monkeypatch):
    _clear_env(monkeypatch)
    defaults = defaults_module.load_model_defaults(_config(tmp_path))
    resolved = defaults_module.resolve_pose_default("prod", defaults=defaults)
    assert resolved.name == "unisign_pose_stage1"
    assert resolved.checkpoint_sha256 == STAGE1_SHA
    assert resolved.bundle == STAGE1_BUNDLE
    assert resolved.source == "argument"


def test_unknown_environment_raises(defaults_module, tmp_path: Path, monkeypatch):
    _clear_env(monkeypatch)
    defaults = defaults_module.load_model_defaults(_config(tmp_path))
    with pytest.raises(ValueError, match="staging"):
        defaults_module.resolve_pose_default("staging", defaults=defaults)


# ---------------------------------------------------------------------------
# environment source: argument > SLBRIDGE_ENV > safe default "prod"
# ---------------------------------------------------------------------------

def test_unset_environment_variable_falls_back_to_prod(defaults_module, tmp_path: Path, monkeypatch):
    _clear_env(monkeypatch)
    defaults = defaults_module.load_model_defaults(_config(tmp_path))
    resolved = defaults_module.resolve_pose_default(defaults=defaults)
    assert resolved.environment == "prod"
    assert resolved.name == "unisign_pose_stage1"        # S2 is never inherited
    assert resolved.source == "default"


def test_environment_variable_selects_stage2_in_dev(defaults_module, tmp_path: Path, monkeypatch):
    defaults = defaults_module.load_model_defaults(_config(tmp_path))
    monkeypatch.setenv("SLBRIDGE_ENV", "dev")
    resolved = defaults_module.resolve_pose_default(defaults=defaults)
    assert resolved.environment == "dev"
    assert resolved.name == "unisign_stage2_pose"
    assert "SLBRIDGE_ENV" in resolved.source


def test_argument_beats_environment_variable(defaults_module, tmp_path: Path, monkeypatch):
    defaults = defaults_module.load_model_defaults(_config(tmp_path))
    monkeypatch.setenv("SLBRIDGE_ENV", "dev")
    resolved = defaults_module.resolve_pose_default("prod", defaults=defaults)
    assert resolved.name == "unisign_pose_stage1"
    assert resolved.source == "argument"


def test_invalid_environment_variable_raises(defaults_module, tmp_path: Path, monkeypatch):
    defaults = defaults_module.load_model_defaults(_config(tmp_path))
    monkeypatch.setenv("SLBRIDGE_ENV", "staging")
    with pytest.raises(ValueError, match="SLBRIDGE_ENV"):
        defaults_module.resolve_pose_default(defaults=defaults)


def test_override_beats_environment(defaults_module, tmp_path: Path, monkeypatch):
    defaults = defaults_module.load_model_defaults(_config(tmp_path))
    monkeypatch.setenv("SLBRIDGE_ENV", "dev")
    resolved = defaults_module.resolve_pose_default(override="unisign_pose_stage1",
                                                   defaults=defaults)
    assert resolved.name == "unisign_pose_stage1"
    assert resolved.checkpoint_sha256 == STAGE1_SHA
    assert resolved.source == "override"
    assert resolved.environment == "dev"                  # which env it ran under is kept


def test_override_unknown_model_raises(defaults_module, tmp_path: Path, monkeypatch):
    _clear_env(monkeypatch)
    defaults = defaults_module.load_model_defaults(_config(tmp_path))
    with pytest.raises(ValueError, match="nosuch_model"):
        defaults_module.resolve_pose_default("dev", override="nosuch_model", defaults=defaults)


# ---------------------------------------------------------------------------
# fallback chain
# ---------------------------------------------------------------------------

def test_fallback_chain_is_default_first_then_fallbacks(defaults_module, tmp_path: Path, monkeypatch):
    _clear_env(monkeypatch)
    defaults = defaults_module.load_model_defaults(_config(tmp_path))
    chain = defaults_module.resolve_with_fallback("dev", defaults=defaults)
    assert [item.name for item in chain] == ["unisign_stage2_pose", "unisign_pose_stage1"]
    assert [item.role for item in chain] == ["default", "fallback"]


def test_prod_fallback_chain_has_no_stage2(defaults_module, tmp_path: Path, monkeypatch):
    _clear_env(monkeypatch)
    defaults = defaults_module.load_model_defaults(_config(tmp_path))
    chain = defaults_module.resolve_with_fallback("prod", defaults=defaults)
    assert [item.name for item in chain] == ["unisign_pose_stage1"]
    assert all(item.bundle != STAGE2_BUNDLE for item in chain)


def test_fallback_chain_deduplicates(defaults_module, tmp_path: Path, monkeypatch):
    _clear_env(monkeypatch)
    environments = _environments() | {
        "dev": {"pose_default": "unisign_stage2_pose",
                "pose_fallback": ["unisign_pose_stage1", "unisign_stage2_pose"]},
    }
    defaults = defaults_module.load_model_defaults(_config(tmp_path, environments=environments))
    chain = defaults_module.resolve_with_fallback("dev", defaults=defaults)
    assert [item.name for item in chain] == ["unisign_stage2_pose", "unisign_pose_stage1"]


# ---------------------------------------------------------------------------
# configuration validation errors
# ---------------------------------------------------------------------------

def test_missing_configuration_file_raises(defaults_module, tmp_path: Path):
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError, match="model_defaults.example.json"):
        defaults_module.load_model_defaults(missing)


def test_malformed_json_raises(defaults_module, tmp_path: Path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        defaults_module.load_model_defaults(path)


def test_bad_schema_version_raises(defaults_module, tmp_path: Path):
    with pytest.raises(ValueError, match="schema_version"):
        defaults_module.load_model_defaults(_config(tmp_path, schema_version=2))


def test_unknown_model_name_in_environment_raises(defaults_module, tmp_path: Path):
    environments = _environments() | {
        "dev": {"pose_default": "ghost_model", "pose_fallback": []},
    }
    with pytest.raises(ValueError, match="ghost_model"):
        defaults_module.load_model_defaults(_config(tmp_path, environments=environments))


def test_unknown_fallback_model_name_raises(defaults_module, tmp_path: Path):
    environments = _environments() | {
        "dev": {"pose_default": "unisign_stage2_pose", "pose_fallback": ["ghost_fallback"]},
    }
    with pytest.raises(ValueError, match="ghost_fallback"):
        defaults_module.load_model_defaults(_config(tmp_path, environments=environments))


def test_bad_checkpoint_digest_raises(defaults_module, tmp_path: Path):
    models = _models() | {
        "unisign_pose_stage1": {"adapter_id": "uni_sign_pose_slt",
                                "bundle": "models/external/uni_sign/model_bundle",
                                "checkpoint_sha256": "73bea87a"},
    }
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        defaults_module.load_model_defaults(_config(tmp_path, models=models))


def test_missing_environment_definition_raises(defaults_module, tmp_path: Path):
    environments = {key: value for key, value in _environments().items() if key != "prod"}
    with pytest.raises(ValueError, match="prod"):
        defaults_module.load_model_defaults(_config(tmp_path, environments=environments))


# ---------------------------------------------------------------------------
# shipped configuration
# ---------------------------------------------------------------------------

def test_example_configuration_is_valid(defaults_module):
    defaults = defaults_module.load_model_defaults(EXAMPLE)
    assert set(defaults.environments) == {"dev", "test", "prod"}
    assert defaults.schema_version == 1


def test_example_configuration_matches_the_decision(defaults_module, monkeypatch):
    _clear_env(monkeypatch)
    defaults = defaults_module.load_model_defaults(EXAMPLE)
    assert defaults_module.resolve_pose_default("dev", defaults=defaults).name == "unisign_stage2_pose"
    assert defaults_module.resolve_pose_default("test", defaults=defaults).name == "unisign_stage2_pose"
    assert defaults_module.resolve_pose_default("prod", defaults=defaults).name == "unisign_pose_stage1"
    assert defaults_module.resolve_pose_default(defaults=defaults).name == "unisign_pose_stage1"


def test_local_configuration_matches_example_when_present(defaults_module):
    if not LOCAL.is_file():
        pytest.skip("local configs/model_defaults.json not present (gitignored)")
    example = defaults_module.load_model_defaults(EXAMPLE)
    local = defaults_module.load_model_defaults(LOCAL)
    assert local.environments == example.environments
    assert set(local.models) == set(example.models)


def test_example_models_resolve_to_real_bundles(defaults_module):
    """Resolution only: manifests + registered adapter id, no weights are loaded."""
    from slbridge.model_adapters import load_model_adapter, model_adapter_ids

    defaults = defaults_module.load_model_defaults(EXAMPLE)
    for name, spec in defaults.models.items():
        assert spec.adapter_id in model_adapter_ids(), name
        if not spec.bundle.is_dir():
            continue                                   # checkpoint/bundle stays local-only
        manifest = json.loads((spec.bundle / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["backend"] == spec.adapter_id, name
        assert manifest["checkpoint_sha256"] == spec.checkpoint_sha256, name
        adapter = load_model_adapter(spec.bundle)      # construction only: defers GPU loading
        try:
            assert adapter.capabilities.model_id == manifest["run_id"]
        finally:
            adapter.close()


# --- 环境感知默认模型的接线（desktop entry point） -------------------------------

def test_desktop_default_unisign_model_follows_environment(monkeypatch, tmp_path):
    """dev/test -> stage2 bundle, prod/unset -> legacy stage1 bundle."""
    from slbridge import desktop_models

    monkeypatch.delenv("SLBRIDGE_ENV", raising=False)
    unset = desktop_models.default_unisign_model(tmp_path)
    assert unset.name == "model_bundle"          # legacy stage1 bundle
    assert desktop_models.default_unisign_model(tmp_path, environment="dev").name == "unisign_stage2_pose"
    assert desktop_models.default_unisign_model(tmp_path, environment="test").name == "unisign_stage2_pose"
    assert desktop_models.default_unisign_model(tmp_path, environment="prod").name == "model_bundle"


def test_desktop_default_unisign_model_override_wins(monkeypatch, tmp_path):
    from slbridge import desktop_models

    monkeypatch.delenv("SLBRIDGE_ENV", raising=False)
    chosen = desktop_models.default_unisign_model(tmp_path, environment="prod", override="unisign_stage2_pose")
    assert chosen.name == "unisign_stage2_pose"


def test_desktop_default_unisign_model_env_var(monkeypatch, tmp_path):
    from slbridge import desktop_models

    monkeypatch.setenv("SLBRIDGE_ENV", "dev")
    assert desktop_models.default_unisign_model(tmp_path).name == "unisign_stage2_pose"
    monkeypatch.setenv("SLBRIDGE_ENV", "prod")
    assert desktop_models.default_unisign_model(tmp_path).name == "model_bundle"


def test_desktop_default_unisign_model_falls_back_when_config_missing(monkeypatch, tmp_path):
    """A missing/invalid config must not break desktop entry points."""
    from slbridge import desktop_models, model_defaults

    def missing(path=None):
        raise model_defaults.ModelDefaultsError("configs/model_defaults.json not found")

    monkeypatch.setattr(model_defaults, "load_model_defaults", missing)
    assert desktop_models.default_unisign_model(tmp_path) == tmp_path / desktop_models.LEGACY_UNISIGN_BUNDLE

def test_desktop_default_unisign_model_unknown_environment_raises():
    """Unknown environments must fail fast rather than silently pick a model."""
    import pytest as _pytest
    from slbridge import desktop_models

    with _pytest.raises(Exception):
        desktop_models.default_unisign_model(environment="staging")
