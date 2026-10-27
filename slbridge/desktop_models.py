"""Local model locations shared by the desktop launcher and language selector.

Uni-Sign 的默认 bundle 由环境感知的解析层决定（dev/test → stage2 pose 子集，
prod 及未设环境变量 → stage1 pose-only，见 ``slbridge/model_defaults.py`` 与
ADR-P1-013）。

错误语义刻意区分：
- **配置文件缺失/非法** → 回退历史硬编码路径（保证桌面端在无配置的开发机上仍可用）；
- **环境名/覆盖项非法** → fail fast 抛错（避免静默用错模型）。
"""
from pathlib import Path

LEGACY_UNISIGN_BUNDLE = Path('models/external/uni_sign/model_bundle')


def default_language_models(root: Path | None = None) -> dict[str, Path]:
    root = Path.cwd() if root is None else Path(root)
    return {
        'csl': root / 'artifacts/models/cecsl-tfnet-onnx-20261009',
        'libras': root / 'artifacts/experiments/libras-rf-pilot-20261009/models/libras-rf-pilot-20261009',
    }


def default_unisign_model(root: Path | None = None, *, environment: str | None = None,
                          override: str | None = None) -> Path:
    """Resolve the Uni-Sign bundle for the current environment.

    Precedence: explicit ``override`` > ``environment`` argument > ``SLBRIDGE_ENV``
    > safe default ``prod``. A missing or invalid *configuration file* falls back to
    the legacy bundle; an unknown environment or model name raises.
    """
    from . import model_defaults

    base = Path.cwd() if root is None else Path(root)
    try:
        defaults = model_defaults.load_model_defaults()
    except (OSError, model_defaults.ModelDefaultsError):
        return base / LEGACY_UNISIGN_BUNDLE
    resolved = model_defaults.resolve_pose_default(environment, override=override, defaults=defaults)
    bundle = resolved.bundle
    return bundle if bundle.is_absolute() else base / bundle
