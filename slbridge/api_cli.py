"""Run versioned task APIs over a server-owned catalog and language routes."""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
from pathlib import Path
import re
from typing import Any, Sequence

from .runtime_options import RuntimeOptions

_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate catalog key: {key}")
        result[key] = value
    return result


def load_catalog(path: str | Path) -> dict[str, Path]:
    """Resolve only strict v1 catalog paths, relative to the catalog file."""
    source = Path(path).resolve()
    with source.open(encoding="utf-8-sig") as stream:
        value = json.load(stream, object_pairs_hook=_unique_object)
    if (not isinstance(value, dict) or set(value) != {"schema_version", "models"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1):
        raise ValueError("Catalog requires schema_version: 1 and models, with no other fields")
    if not isinstance(value["models"], dict) or not value["models"]:
        raise ValueError("Catalog models must be a nonempty object")
    result = {}
    for alias, model_path in value["models"].items():
        if not _ALIAS.fullmatch(alias):
            raise ValueError(f"Invalid model alias: {alias}")
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError(f"Invalid model path for alias: {alias}")
        result[alias] = (source.parent / model_path).resolve()
    return result


def _loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve SignLanguage Bridge's versioned inference API")
    parser.add_argument("--catalog", type=Path, required=True, help="Strict v1 JSON model catalog")
    parser.add_argument("--model", help="Initial model alias; omit to start without a loaded model")
    parser.add_argument("--route", action="append", default=[], metavar="LANGUAGE=MODEL",
                        help="Independent language model route; repeat once per language")
    parser.add_argument("--max-caption-sessions", type=int, default=64)
    parser.add_argument("--inference-timeout-seconds", type=float, default=120.)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--intra-op-threads", type=int, default=1)
    parser.add_argument("--gpu-mem-limit-mb", type=int, default=2048)
    parser.add_argument("--cuda-dll-dir", type=Path, action="append", default=[],
                        help="Server-owned CUDA DLL directory; repeat for multiple directories")
    parser.add_argument("--max-upload-mb", type=int, default=128)
    parser.add_argument("--token-env", default="SLBRIDGE_API_TOKEN",
                        help="Environment variable with a bearer token; required for non-loopback binds")
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if args.max_upload_mb <= 0:
            raise ValueError("max-upload-mb must be positive")
        token = os.environ.get(args.token_env)
        if not _loopback(args.host) and not token:
            raise ValueError("A bearer token environment variable is required for non-loopback hosts")
        options = RuntimeOptions(device=args.device, device_id=args.device_id,
                                 intra_op_threads=args.intra_op_threads, gpu_mem_limit_mb=args.gpu_mem_limit_mb,
                                 cuda_dll_dirs=tuple(str(path.resolve()) for path in args.cuda_dll_dir))
        catalog = load_catalog(args.catalog)
        routes = {}
        for entry in args.route:
            language, separator, alias = entry.partition("=")
            if not separator or not language.strip() or not alias or language in routes:
                raise ValueError("Each --route must be a unique LANGUAGE=MODEL mapping")
            if alias not in catalog:
                raise ValueError(f"Unknown route model alias: {alias}")
            routes[language] = alias
        # Keep --help available in a core-only environment without HTTP extras.
        from .api import create_app
        import uvicorn
        app = create_app(catalog, initial_model=args.model, options=options,
                         max_upload_bytes=args.max_upload_mb * 1024 * 1024, bearer_token=token,
                         routes=routes, max_sessions=args.max_caption_sessions,
                         inference_timeout_seconds=args.inference_timeout_seconds)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    except ModuleNotFoundError:
        parser.error("HTTP API dependencies are missing; install the project's api extra")
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
