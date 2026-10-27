"""Environment-aware default pose-model resolution (task-36).

The user decision this layer implements (2026-10-27):

* **prod** keeps the pinned pose-only stage1 bundle (A2) as the default;
* **dev** / **test** default to the stage2 pose-subset bundle (S2 -- the
  RGB+pose checkpoint minus its mathematically inert RGB branch), with A2 as
  the fallback;
* an unset environment resolves to **prod** (safe default: nobody inherits the
  experimental bundle by accident).

The layer only decides *which* bundle an environment points at.  It never
loads weights, never touches ``experiments/p1/**`` artifacts (frozen
baselines) and never changes inference behaviour in
``slbridge.uni_sign`` / ``slbridge.uni_sign_rgb_pose``.

Configuration lives in ``configs/model_defaults.json`` (local, gitignored) and
is documented by the committed ``configs/model_defaults.example.json``:

    {
      "schema_version": 1,
      "environments": {
        "dev":  {"pose_default": "unisign_stage2_pose",
                 "pose_fallback": ["unisign_pose_stage1"]},
        "test": {"pose_default": "unisign_stage2_pose",
                 "pose_fallback": ["unisign_pose_stage1"]},
        "prod": {"pose_default": "unisign_pose_stage1", "pose_fallback": []}
      },
      "models": {
        "unisign_stage2_pose": {"adapter_id": "uni_sign_pose_slt",
                                "bundle": "experiments/p1/e01/runs/.../pose_bundle",
                                "checkpoint_sha256": "<64 hex>"},
        "unisign_pose_stage1": {"adapter_id": "uni_sign_pose_slt",
                                "bundle": "models/external/uni_sign/model_bundle",
                                "checkpoint_sha256": "<64 hex>"}
      },
      "notes": "..."
    }

Resolution order for the environment: explicit argument > ``SLBRIDGE_ENV`` >
``"prod"``.  Model override (``override=``) wins over the environment while
still reporting which environment it ran under.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 1
ENVIRONMENT_VARIABLE = "SLBRIDGE_ENV"
SAFE_DEFAULT_ENVIRONMENT = "prod"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "model_defaults.json"
EXAMPLE_CONFIG_PATH = REPO_ROOT / "configs" / "model_defaults.example.json"


class ModelDefaultsError(ValueError):
    """Invalid ``model_defaults`` configuration (schema, environment, model name)."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ModelDefaultsError(message)


def _is_digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdefABCDEF" for character in value))


@dataclass(frozen=True)
class ModelSpec:
    """One configured model: which adapter loads which local bundle."""

    name: str
    adapter_id: str
    bundle: Path
    checkpoint_sha256: str

    @property
    def bundle_available(self) -> bool:
        """True when the local bundle (gitignored / model asset) is present."""
        return (self.bundle / "manifest.json").is_file()


@dataclass(frozen=True)
class EnvironmentPolicy:
    """``pose_default`` plus the ordered fallbacks for one environment."""

    name: str
    pose_default: str
    pose_fallback: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedModel:
    """A model spec resolved for one environment (with provenance)."""

    name: str
    adapter_id: str
    bundle: Path
    checkpoint_sha256: str
    environment: str
    source: str                  # "argument" | "env:SLBRIDGE_ENV" | "default" | "override"
    role: str = "default"        # "default" | "fallback"

    @property
    def bundle_available(self) -> bool:
        return (self.bundle / "manifest.json").is_file()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "adapter_id": self.adapter_id,
            "bundle": str(self.bundle),
            "checkpoint_sha256": self.checkpoint_sha256,
            "environment": self.environment,
            "source": self.source,
            "role": self.role,
            "bundle_available": self.bundle_available,
        }


@dataclass(frozen=True)
class ModelDefaults:
    """Parsed and validated ``model_defaults.json``."""

    path: Path
    schema_version: int
    environments: Mapping[str, EnvironmentPolicy]
    models: Mapping[str, ModelSpec]
    notes: str = ""

    # -- lookups ---------------------------------------------------------
    def environment(self, name: str) -> EnvironmentPolicy:
        try:
            return self.environments[name]
        except KeyError:
            raise ModelDefaultsError(
                f"unknown environment {name!r}; configured: {sorted(self.environments)}") from None

    def model(self, name: str) -> ModelSpec:
        try:
            return self.models[name]
        except KeyError:
            raise ModelDefaultsError(
                f"unknown model {name!r}; configured: {sorted(self.models)}") from None

    def resolve_environment(self, environment: str | None = None) -> tuple[EnvironmentPolicy, str]:
        """Return ``(policy, source)``: argument > SLBRIDGE_ENV > safe default."""
        if environment is not None:
            if not isinstance(environment, str) or not environment:
                raise ModelDefaultsError("environment must be a non-empty string")
            return self.environment(environment), "argument"
        configured = os.environ.get(ENVIRONMENT_VARIABLE, "").strip()
        if configured:
            if configured not in self.environments:
                raise ModelDefaultsError(
                    f"{ENVIRONMENT_VARIABLE}={configured!r} is not a configured environment; "
                    f"configured: {sorted(self.environments)}")
            return self.environments[configured], f"env:{ENVIRONMENT_VARIABLE}"
        return self.environments[SAFE_DEFAULT_ENVIRONMENT], "default"

    # -- resolution ------------------------------------------------------
    def resolve_pose_default(self, environment: str | None = None, *,
                            override: str | None = None) -> ResolvedModel:
        policy, source = self.resolve_environment(environment)
        if override is not None:
            spec = self.model(override)
            return ResolvedModel(name=spec.name, adapter_id=spec.adapter_id, bundle=spec.bundle,
                                 checkpoint_sha256=spec.checkpoint_sha256,
                                 environment=policy.name, source="override", role="default")
        spec = self.model(policy.pose_default)
        return ResolvedModel(name=spec.name, adapter_id=spec.adapter_id, bundle=spec.bundle,
                             checkpoint_sha256=spec.checkpoint_sha256,
                             environment=policy.name, source=source, role="default")

    def resolve_with_fallback(self, environment: str | None = None) -> list[ResolvedModel]:
        policy, source = self.resolve_environment(environment)
        chain: list[ResolvedModel] = []
        seen: set[str] = set()
        for name in (policy.pose_default, *policy.pose_fallback):
            if name in seen:
                continue
            seen.add(name)
            spec = self.model(name)
            chain.append(ResolvedModel(
                name=spec.name, adapter_id=spec.adapter_id, bundle=spec.bundle,
                checkpoint_sha256=spec.checkpoint_sha256, environment=policy.name,
                source=source, role="default" if not chain else "fallback"))
        return chain


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_model_defaults(path: str | Path | None = None) -> ModelDefaults:
    """Load and validate ``configs/model_defaults.json`` (or an explicit path)."""
    target = Path(path).expanduser() if path is not None else DEFAULT_CONFIG_PATH
    if not target.is_absolute():
        target = REPO_ROOT / target
    if not target.is_file():
        raise FileNotFoundError(
            f"model defaults configuration not found: {target} -- copy "
            f"{EXAMPLE_CONFIG_PATH.relative_to(REPO_ROOT)} to "
            f"{DEFAULT_CONFIG_PATH.relative_to(REPO_ROOT)} and point it at the local bundles")
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ModelDefaultsError(
            f"model defaults configuration is not valid JSON: {target}: {error}") from error
    _require(isinstance(raw, dict), f"model defaults configuration must be a JSON object: {target}")
    schema_version = raw.get("schema_version")
    _require(schema_version == SCHEMA_VERSION,
             f"schema_version must be {SCHEMA_VERSION}, got {schema_version!r}")
    notes = raw.get("notes", "")
    _require(isinstance(notes, str), "notes must be a string")

    models_raw = raw.get("models")
    _require(isinstance(models_raw, dict) and models_raw, "models must be a non-empty object")
    models: dict[str, ModelSpec] = {}
    for name, spec in models_raw.items():
        _require(isinstance(name, str) and name, "model names must be non-empty strings")
        _require(isinstance(spec, dict), f"model {name!r} must be a JSON object")
        adapter_id = spec.get("adapter_id")
        bundle = spec.get("bundle")
        digest = spec.get("checkpoint_sha256")
        _require(isinstance(adapter_id, str) and adapter_id,
                 f"model {name!r}: adapter_id must be a non-empty string")
        _require(isinstance(bundle, str) and bundle,
                 f"model {name!r}: bundle must be a non-empty path string")
        _require(_is_digest(digest),
                 f"model {name!r}: checkpoint_sha256 must be a 64-character hex digest, got {digest!r}")
        models[name] = ModelSpec(name=name, adapter_id=adapter_id,
                                 bundle=_resolve_path(bundle),
                                 checkpoint_sha256=str(digest).lower())

    environments_raw = raw.get("environments")
    _require(isinstance(environments_raw, dict) and environments_raw,
             "environments must be a non-empty object")
    environments: dict[str, EnvironmentPolicy] = {}
    for name, policy in environments_raw.items():
        _require(isinstance(name, str) and name, "environment names must be non-empty strings")
        _require(isinstance(policy, dict), f"environment {name!r} must be a JSON object")
        pose_default = policy.get("pose_default")
        pose_fallback = policy.get("pose_fallback", [])
        _require(isinstance(pose_default, str) and pose_default,
                 f"environment {name!r}: pose_default must be a configured model name")
        _require(isinstance(pose_fallback, list)
                 and all(isinstance(item, str) and item for item in pose_fallback),
                 f"environment {name!r}: pose_fallback must be a list of model names")
        for model_name in (pose_default, *pose_fallback):
            _require(model_name in models,
                     f"environment {name!r} references unknown model {model_name!r}; "
                     f"configured models: {sorted(models)}")
        environments[name] = EnvironmentPolicy(name=name, pose_default=pose_default,
                                               pose_fallback=tuple(pose_fallback))
    _require(SAFE_DEFAULT_ENVIRONMENT in environments,
             f"environments must define the safe default environment "
             f"{SAFE_DEFAULT_ENVIRONMENT!r} (unset {ENVIRONMENT_VARIABLE} resolves there)")

    return ModelDefaults(path=target, schema_version=SCHEMA_VERSION,
                         environments=environments, models=models, notes=notes)


def resolve_environment(environment: str | None = None, *, defaults: ModelDefaults | None = None
                        ) -> tuple[str, str]:
    """Return ``(environment_name, source)`` without resolving a model."""
    policy, source = (defaults or load_model_defaults()).resolve_environment(environment)
    return policy.name, source


def resolve_pose_default(environment: str | None = None, *, override: str | None = None,
                         defaults: ModelDefaults | None = None) -> ResolvedModel:
    """Resolve the default pose model for ``environment`` (argument > env var > prod)."""
    return (defaults or load_model_defaults()).resolve_pose_default(environment, override=override)


def resolve_with_fallback(environment: str | None = None, *, defaults: ModelDefaults | None = None
                          ) -> list[ResolvedModel]:
    """Resolve ``[default, *fallbacks]`` for ``environment`` (deduped, ordered)."""
    return (defaults or load_model_defaults()).resolve_with_fallback(environment)
