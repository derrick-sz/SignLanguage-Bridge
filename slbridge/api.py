"""A bounded, local-first HTTP transport over the shared model manager."""
from __future__ import annotations

from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
import hmac
import logging
from pathlib import Path
import re
import tempfile
from typing import Annotated, Any, Literal, Mapping
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi.security import HTTPBearer
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import parse_options_header
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .api_schemas import ErrorResponse, ModelsResponse, RecognitionResponse, StatusResponse, SwitchRequest
from .runtime import InferenceRequest, ModelManager, ModelUnavailable, RuntimeBusy
from .runtime_options import RuntimeOptions

_LOGGER = logging.getLogger(__name__)
_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv", ".npy", ".npz"})
_FORM_OVERHEAD_BYTES = 64 * 1024
_COPY_CHUNK_BYTES = 64 * 1024


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code, self.code, self.message = status_code, code, message
        super().__init__(message)


class _BodyTooLarge(MultiPartException):
    """Also lets Starlette close multipart spool files before propagating."""


def _error(request_id: str, status: int, code: str, message: str, **kwargs: Any) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"schema_version": 1, "request_id": request_id, "error": {"code": code, "message": message}},
        **kwargs,
    )


class _BoundedRequestMiddleware:
    """Enforce bytes as ASGI receives them, before multipart spooling occurs."""

    def __init__(self, app: ASGIApp, *, limit: int, bearer_token: str | None) -> None:
        self.app, self.limit, self.bearer_token = app, limit, bearer_token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        state = scope.setdefault("state", {})
        state["request_id"] = str(uuid4())
        headers = dict(scope.get("headers", []))
        if self.bearer_token is not None:
            expected = f"Bearer {self.bearer_token}".encode("utf-8")
            if not hmac.compare_digest(headers.get(b"authorization", b""), expected):
                await _error(state["request_id"], 401, "unauthorized", "A valid bearer token is required",
                             headers={"WWW-Authenticate": "Bearer"})(scope, receive, send)
                return
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                size = int(declared)
                if size < 0:
                    raise ValueError
            except ValueError:
                await _error(state["request_id"], 400, "invalid_request", "Invalid Content-Length")(scope, receive, send)
                return
            if size > self.limit:
                await _error(state["request_id"], 413, "upload_too_large", "Request exceeds the upload limit")(
                    scope, receive, send)
                return
        received = 0

        async def bounded_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.limit:
                    state["body_too_large"] = True
                    raise _BodyTooLarge("Request exceeds the upload limit")
            return message

        await self.app(scope, bounded_receive, send)


class _ClosingMultipartParser(MultiPartParser):
    async def parse(self):
        try:
            return await super().parse()
        except BaseException:
            # Starlette closes files for MultiPartException; native parser
            # errors and disconnects also need deterministic spool cleanup.
            for file in self._files_to_close_on_error:
                file.close()
            raise


class _BoundedMultipartRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()

        async def route(request: Request):
            if request.url.path == "/v1/recognitions":
                media_type, _ = parse_options_header(request.headers.get("content-type", ""))
                if media_type != b"multipart/form-data":
                    raise ApiError(422, "invalid_request", "A multipart/form-data input file is required")
                parser = _ClosingMultipartParser(request.headers, request.stream(),
                                                max_files=1, max_fields=3, max_part_size=4096)
                try:
                    form = await parser.parse()
                except MultiPartException as exc:
                    raise HTTPException(400, exc.message) from exc
                except MultipartParseError as exc:
                    raise HTTPException(400, "Malformed multipart body") from exc
                # FastAPI's File/Form handler consumes Request's cached form
                # and registers its close callback on the request exit stack.
                # Covered against the pinned FastAPI/Starlette versions.
                request._form = form
            return await original(request)

        return route


def _catalog(value: Mapping[str, Path]) -> dict[str, Path]:
    if not isinstance(value, Mapping):
        raise ValueError("model_catalog must be a mapping of aliases to paths")
    result = {}
    for alias, path in value.items():
        if not isinstance(alias, str) or not _ALIAS.fullmatch(alias):
            raise ValueError("Model aliases must contain only letters, digits, dots, underscores and hyphens")
        if not isinstance(path, (str, Path)) or not str(path).strip():
            raise ValueError(f"Invalid catalog path for alias {alias}")
        result[alias] = Path(path).resolve()
    return result


def _save_upload(upload: UploadFile, suffix: str, limit: int) -> Path:
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="slbridge-input-", suffix=suffix, delete=False) as target:
            path = Path(target.name)
            total = 0
            while chunk := upload.file.read(_COPY_CHUNK_BYTES):
                total += len(chunk)
                if total > limit:
                    raise ApiError(413, "upload_too_large", "File exceeds the upload limit")
                target.write(chunk)
            if not total:
                raise ApiError(422, "invalid_input", "The input file is empty")
        return path
    except BaseException:
        if path is not None:
            path.unlink(missing_ok=True)
        raise


def _path_variants(values) -> tuple[str, ...]:
    variants = set()
    for value in values:
        if value:
            path = Path(value)
            for spelling in (str(path), path.as_posix()):
                variants.add(spelling)
                variants.add(spelling.replace("\\", "\\\\"))
    return tuple(sorted(variants, key=len, reverse=True))


def _redact(value: Any, sensitive_paths: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        for path in sensitive_paths:
            value = re.sub(re.escape(path), "[server path]", value, flags=re.IGNORECASE)
        return value
    if isinstance(value, dict):
        return {key: _redact(item, sensitive_paths) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item, sensitive_paths) for item in value]
    return value


def _public_runtime(value: dict[str, Any], sensitive_paths: tuple[str, ...]) -> dict[str, Any]:
    runtime = deepcopy(value)
    paths = _path_variants(runtime.pop("cuda_dll_dirs", ())) + sensitive_paths
    return _redact(runtime, paths)


def _public_status(value: dict[str, Any], sensitive_paths: tuple[str, ...]) -> dict[str, Any]:
    status = deepcopy(value)
    for name in ("active", "previous"):
        descriptor = status.get(name)
        if descriptor is not None:
            paths = descriptor.get("options", {}).pop("cuda_dll_dirs", ())
            sensitive_paths += _path_variants(paths)
            descriptor["runtime"] = _public_runtime(descriptor.get("runtime", {}), sensitive_paths)
    status["error"] = _redact(status.get("error"), sensitive_paths)
    return status


def create_app(
    model_catalog: Mapping[str, Path],
    initial_model: str | None = None,
    options: RuntimeOptions | None = None,
    manager: ModelManager | None = None,
    max_upload_bytes: int = 128 * 1024 * 1024,
    *,
    bearer_token: str | None = None,
) -> FastAPI:
    """Create one app with one shared manager, owned by its lifespan.

    Catalog paths are server configuration. Requests can select their aliases,
    and can supply a single input file, never a filesystem path or model weight.
    """
    catalog = _catalog(model_catalog)
    if initial_model is not None and initial_model not in catalog:
        raise ValueError("initial_model must be an alias in model_catalog")
    if type(max_upload_bytes) is not int or max_upload_bytes <= 0:
        raise ValueError("max_upload_bytes must be a positive integer")
    if bearer_token is not None and (not bearer_token or not bearer_token.isascii()
                                     or any(c.isspace() for c in bearer_token)):
        raise ValueError("bearer_token must be a nonempty ASCII token without whitespace")
    defaults = options or RuntimeOptions()
    sensitive_paths = _path_variants((*catalog.values(), *defaults.cuda_dll_dirs))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        shared = manager if manager is not None else ModelManager(options=defaults)
        app.state.manager = shared
        try:
            if initial_model is not None:
                await run_in_threadpool(shared.switch, catalog[initial_model], options=defaults)
            yield
        finally:
            await run_in_threadpool(shared.close)

    error_responses = {code: {"model": ErrorResponse} for code in (400, 401, 404, 409, 413, 422, 500, 503)}
    app = FastAPI(title="SignLanguage Bridge API", version="1.0.0", lifespan=lifespan,
                  responses=error_responses,
                  dependencies=[Depends(HTTPBearer(auto_error=False))] if bearer_token else [])
    app.router.route_class = _BoundedMultipartRoute
    app.add_middleware(_BoundedRequestMiddleware, limit=max_upload_bytes + _FORM_OVERHEAD_BYTES,
                       bearer_token=bearer_token)

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError):
        return _error(request.state.request_id, exc.status_code, exc.code, _redact(exc.message, sensitive_paths))

    @app.exception_handler(_BodyTooLarge)
    async def oversized_body(request: Request, exc: _BodyTooLarge):
        return _error(request.state.request_id, 413, "upload_too_large", "Request exceeds the upload limit")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        messages = []
        for detail in exc.errors():
            field = ".".join(str(value) for value in detail["loc"])
            messages.append(f"{field}: {detail['msg']}")
        return _error(request.state.request_id, 422, "invalid_request", "; ".join(messages))

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        if getattr(request.state, "body_too_large", False):
            return _error(request.state.request_id, 413, "upload_too_large", "Request exceeds the upload limit")
        return _error(request.state.request_id, exc.status_code, "invalid_request", str(exc.detail), headers=exc.headers)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception):
        _LOGGER.error("API request failed: %s", request.state.request_id, exc_info=exc)
        return _error(request.state.request_id, 500, "internal_error", "Internal server error")

    def shared(request: Request) -> ModelManager:
        return request.app.state.manager

    async def operation(request: Request, name: str, *args: Any, **kwargs: Any):
        try:
            result = await run_in_threadpool(getattr(shared(request), name), *args, **kwargs)
            return _public_status(result, sensitive_paths) if name in {"status", "switch", "rollback"} else result
        except RuntimeBusy as exc:
            raise ApiError(409, "runtime_busy", str(exc)) from exc
        except ModelUnavailable as exc:
            raise ApiError(503, "model_unavailable", str(exc)) from exc
        except (ValueError, OSError) as exc:
            code = "invalid_input" if name == "infer_file" else "invalid_model"
            paths = sensitive_paths + (_path_variants((args[0],)) if name == "infer_file" else ())
            raise ApiError(422, code, _redact(str(exc), paths)) from exc

    @app.get("/v1/health", response_model=StatusResponse)
    async def health(request: Request):
        return {"schema_version": 1, "request_id": request.state.request_id,
                "status": await operation(request, "status")}

    @app.get("/v1/models", response_model=ModelsResponse)
    async def models(request: Request):
        status = await operation(request, "status")
        return {"schema_version": 1, "request_id": request.state.request_id,
                "models": [{"model_id": alias} for alias in sorted(catalog)], "active": status["active"]}

    @app.post("/v1/models/switch", response_model=StatusResponse)
    async def switch(body: SwitchRequest, request: Request):
        if body.model_id not in catalog:
            raise ApiError(404, "unknown_model", "The model alias is not in the server catalog")
        # Deployment-only options (such as CUDA DLL paths) stay server-owned.
        selected = replace(defaults, **body.model_dump(exclude={"model_id"}, exclude_none=True))
        status = await operation(request, "switch", catalog[body.model_id], options=selected)
        return {"schema_version": 1, "request_id": request.state.request_id, "status": status}

    @app.post("/v1/models/rollback", response_model=StatusResponse)
    async def rollback(request: Request):
        return {"schema_version": 1, "request_id": request.state.request_id,
                "status": await operation(request, "rollback")}

    @app.post("/v1/recognitions", response_model=RecognitionResponse)
    async def recognize(
        request: Request,
        file: Annotated[UploadFile, File(description="A video, .npy sequence, or .npz landmarks input")],
        request_id: Annotated[str | None, Form(min_length=1, max_length=128,
                                             pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")] = None,
        representation_id: Annotated[Literal["hand63_v1", "pose_hands225_legacy", "mediapipe225_rgb_v1"] | None,
                                      Form()] = None,
        sign_language: Annotated[str | None, Form(min_length=1, max_length=64)] = None,
    ):
        path: Path | None = None
        try:
            if request_id is not None:
                request.state.request_id = request_id
            for name, value in (("request_id", request_id), ("representation_id", representation_id),
                                ("sign_language", sign_language)):
                if value is not None and not value.strip():
                    raise ApiError(422, "invalid_request", f"{name} must be nonempty")
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in _SUFFIXES:
                raise ApiError(422, "unsupported_input", "Supported files: .mp4, .avi, .mov, .mkv, .npy, .npz")
            inference_request = InferenceRequest(request_id=request.state.request_id,
                                                 representation_id=representation_id, sign_language=sign_language)
            path = await run_in_threadpool(_save_upload, file, suffix, max_upload_bytes)
            response = await operation(request, "infer_file", path, inference_request)
            body = response.to_dict()
            body["runtime"] = _public_runtime(body["runtime"], sensitive_paths + _path_variants((path,)))
            return body
        finally:
            if path is not None:
                await run_in_threadpool(path.unlink, missing_ok=True)
            await file.close()

    return app
