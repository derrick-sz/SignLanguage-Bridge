"""Local model locations shared by the desktop launcher and language selector."""
from pathlib import Path


def default_language_models(root: Path | None = None) -> dict[str, Path]:
    root = Path.cwd() if root is None else Path(root)
    return {
        'csl': root / 'artifacts/models/cecsl-tfnet-onnx-20261009',
        'libras': root / 'artifacts/experiments/libras-rf-pilot-20261009/models/libras-rf-pilot-20261009',
    }


def default_unisign_model(root: Path | None = None) -> Path:
    root = Path.cwd() if root is None else Path(root)
    return root / 'models/external/uni_sign/model_bundle'
