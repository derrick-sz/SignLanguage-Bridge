"""A bounded, local-first HTTP transport over the shared model manager."""
from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
from copy import deepcopy
from dataclasses import dataclass, field, replace
import hashlib
import hmac
import json
import logging
from pathlib import Path
import re
import tempfile
import threading
import time
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

from .api_schemas import (ErrorResponse, ModelsResponse, RecognitionResponse,
                          SequenceRecognitionResponse, StatusResponse, SwitchRequest,
                          UnifiedRecognitionResponse, NegotiationRequest, RouteSwitchRequest,
                          CaptionActionRequest, CaptionSessionRequest)
from .runtime import InferenceRequest, ModelManager, ModelUnavailable, RuntimeBusy
from .runtime_options import RuntimeOptions

_LOGGER = logging.getLogger(__name__)
_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv", ".npy", ".npz"})
_FORM_OVERHEAD_BYTES = 64 * 1024
_COPY_CHUNK_BYTES = 64 * 1024


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str,
                 contract_code: str | None = None) -> None:
        self.status_code, self.code, self.message = status_code, code, message
        # T-A4-2: on V1 contract endpoints the error body uses the contract
        # uppercase enum as ``error.code`` plus the legacy snake_case under
        # ``error.legacy_code``; legacy endpoints leave this None (unchanged).
        self.contract_code = contract_code
        super().__init__(message)


class _BodyTooLarge(MultiPartException):
    """Also lets Starlette close multipart spool files before propagating."""


def _error(request_id: str, status: int, code: str, message: str, **kwargs: Any) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"schema_version": 1, "request_id": request_id, "error": {"code": code, "message": message}},
        **kwargs,
    )


def _contract_error(request_id: str, status: int, legacy_code: str, message: str) -> JSONResponse:
    """V1-contract error body: contract uppercase code + legacy_code field."""
    from .api_contract_v1 import map_error_code
    return JSONResponse(
        status_code=status,
        content={"schema_version": 1, "request_id": request_id,
                 "error": {"code": map_error_code(legacy_code), "legacy_code": legacy_code,
                           "message": message}},
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
            if request.url.path in {"/v1/recognitions", "/v2/recognitions", "/v3/recognitions"}:
                media_type, _ = parse_options_header(request.headers.get("content-type", ""))
                if media_type != b"multipart/form-data":
                    raise ApiError(422, "invalid_request", "A multipart/form-data input file is required")
                parser = _ClosingMultipartParser(request.headers, request.stream(),
                                                max_files=1, max_fields=18 if request.url.path.startswith("/v3/") else 3,
                                                max_part_size=4096)
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
                if request.url.path == "/v3/recognitions":
                    names = [key for key, _ in form.multi_items()]
                    allowed = {"file", "request_id", "representation_id", "sign_language", "task_type", "model_id",
                               "expected_generation", "preprocessing_version", "session_id", "segment_id", "sequence",
                               "source_epoch", "start_time", "end_time", "source_id", "timeout_ms", "mode"}
                    if len(names) != len(set(names)) or set(names) - allowed:
                        await form.close()
                        raise ApiError(422, "invalid_request", "Duplicate or unsupported multipart fields")
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


@dataclass
class _RecognitionRecord:
    fingerprint: str
    cancellation: Any
    event_id: str
    segment_id: str
    body: dict[str, Any] | None = None
    error: tuple[int, str, str] | None = None


@dataclass
class _CaptionSession:
    ledger: Any
    sign_language: str
    records: dict[str, _RecognitionRecord] = field(default_factory=dict)
    actions: dict[str, tuple[dict[str, Any], dict[str, Any]]] = field(default_factory=dict)
    lock: Any = field(default_factory=threading.RLock)
    closed: bool = False

    def cancel_pending(self) -> None:
        for record in self.records.values():
            if record.body is None and record.error is None:
                record.cancellation.cancel()
        self.ledger.cancel()

    def close(self) -> None:
        with self.lock:
            self.closed = True
            self.cancel_pending()


def _fingerprint(path: Path, binding: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_COPY_CHUNK_BYTES):
            digest.update(chunk)
    return hashlib.sha256((digest.hexdigest() + json.dumps(binding, sort_keys=True, separators=(",", ":")))
                          .encode("utf-8")).hexdigest()


def create_app(
    model_catalog: Mapping[str, Path],
    initial_model: str | None = None,
    options: RuntimeOptions | None = None,
    manager: ModelManager | None = None,
    max_upload_bytes: int = 128 * 1024 * 1024,
    *,
    bearer_token: str | None = None,
    router: Any = None,
    routes: Mapping[str, str] | None = None,
    max_sessions: int = 64,
    max_session_requests: int = 1024,
    max_session_actions: int = 1024,
    inference_timeout_seconds: float = 120.,
    caption_catalog: Any = None,
    job_run_directory: Path | str | None = None,
    job_run_id: str = "p0a-jobs-v1",
    job_max_store: int = 1024,
    job_timeout_seconds: float | None = 120.,
    job_runner_factory: Any = None,
    job_upload_ttl_seconds: float | None = 300.,
    job_media_probe: Any = None,
) -> FastAPI:
    """Create an app owning a manager or independent language runtimes.

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
    if router is not None and manager is not None:
        raise ValueError("Pass either manager or router")
    route_models = dict(routes or {})
    if route_models and manager is not None:
        raise ValueError("Language routes require independent managers")
    if route_models and initial_model is not None:
        raise ValueError("Use routes or initial_model, not both")
    for language, alias in route_models.items():
        if not isinstance(language, str) or not language.strip() or len(language) > 64 or alias not in catalog:
            raise ValueError("Routes require a nonempty language and a server catalog alias")
    for name, value in (("max_sessions", max_sessions), ("max_session_requests", max_session_requests),
                        ("max_session_actions", max_session_actions)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(inference_timeout_seconds) not in {float, int} or not 0 < inference_timeout_seconds <= 3600:
        raise ValueError("inference_timeout_seconds must be between 0 and 3600")
    if job_run_directory is not None and not isinstance(job_run_directory, (Path, str)):
        raise ValueError("job_run_directory must be a path or null")
    if not isinstance(job_run_id, str) or not job_run_id.strip():
        raise ValueError("job_run_id must be a nonempty string")
    if type(job_max_store) is not int or job_max_store <= 0:
        raise ValueError("job_max_store must be a positive integer")
    if job_timeout_seconds is not None and (
            type(job_timeout_seconds) not in {float, int} or not 0 < job_timeout_seconds <= 3600):
        raise ValueError("job_timeout_seconds must be between 0 and 3600, or null")
    if job_upload_ttl_seconds is not None and (
            type(job_upload_ttl_seconds) not in {float, int} or job_upload_ttl_seconds < 0):
        raise ValueError("job_upload_ttl_seconds must be nonnegative or null")
    sensitive_paths = _path_variants((*catalog.values(), *defaults.cuda_dll_dirs))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from .runtime import ModelRouter
        shared = router if router is not None else (ModelRouter() if route_models else
                                                     manager if manager is not None else ModelManager(options=defaults))
        app.state.manager = shared
        app.state.sessions = {}
        app.state.jobs = set()
        app.state.session_lock = threading.RLock()
        try:
            for language, alias in route_models.items():
                try:
                    routed = shared.manager_for(language)
                except (ValueError, ModelUnavailable):
                    routed = ModelManager(options=defaults)
                    shared.register(language, routed)
                await run_in_threadpool(routed.switch, catalog[alias], options=defaults)
                if routed.describe()["sign_language"] != language:
                    raise ValueError("Route language does not match its model")
            if initial_model is not None:
                await run_in_threadpool(shared.switch, catalog[initial_model], options=defaults)
            # P0-A V1 job layer (T-A2-2/T-A2-3): store + optional recorder.
            from .job_runner import JobStore
            app.state.job_store = JobStore(max_jobs=job_max_store)
            app.state.job_runner = None
            app.state.job_run_directory = (Path(job_run_directory)
                                           if job_run_directory is not None else None)
            app.state.job_run_id = job_run_id
            app.state.job_timeout_seconds = job_timeout_seconds
            app.state.job_runner_factory = job_runner_factory
            app.state.job_upload_ttl_seconds = job_upload_ttl_seconds
            app.state.job_media_probe = job_media_probe
            if app.state.job_run_directory is not None:
                app.state.job_store.persistence_path = app.state.job_run_directory / "jobs.jsonl"
            yield
        finally:
            for session in app.state.sessions.values():
                session.close()
            await run_in_threadpool(shared.close)
            if app.state.jobs:
                await asyncio.gather(*app.state.jobs, return_exceptions=True)

    error_responses = {code: {"model": ErrorResponse} for code in (400, 401, 404, 409, 413, 422, 429, 500, 503, 504)}
    app = FastAPI(title="SignLanguage Bridge API", version="3.0.0", lifespan=lifespan,
                  responses=error_responses,
                  dependencies=[Depends(HTTPBearer(auto_error=False))] if bearer_token else [])
    app.router.route_class = _BoundedMultipartRoute
    app.add_middleware(_BoundedRequestMiddleware, limit=max_upload_bytes + _FORM_OVERHEAD_BYTES,
                       bearer_token=bearer_token)

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError):
        if exc.contract_code is not None:
            return _contract_error(request.state.request_id, exc.status_code, exc.code,
                                   _redact(exc.message, sensitive_paths))
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

    def selected_manager(request: Request, language: str | None):
        runtime = shared(request)
        if hasattr(runtime, "manager_for"):
            if language is None:
                raise ApiError(422, "language_required", "sign_language is required for a routed server")
            try:
                return runtime.manager_for(language)
            except (ModelUnavailable, ValueError) as exc:
                raise ApiError(404, "unknown_language", str(exc)) from exc
        return runtime

    async def operation(request: Request, name: str, *args: Any, **kwargs: Any):
        try:
            target = kwargs.pop("_manager", None) or shared(request)
            result = await run_in_threadpool(getattr(target, name), *args, **kwargs)
            return _public_status(result, sensitive_paths) if name in {"status", "switch", "rollback"} else result
        except TimeoutError as exc:
            raise ApiError(504, "inference_timeout", "The inference deadline expired") from exc
        except RuntimeBusy as exc:
            raise ApiError(409, "runtime_busy", str(exc)) from exc
        except ModelUnavailable as exc:
            if type(exc).__name__ == "ModelGenerationConflict":
                raise ApiError(409, "model_generation_conflict", str(exc)) from exc
            raise ApiError(503, "model_unavailable", str(exc)) from exc
        except RuntimeError as exc:
            if type(exc).__name__ == "InferenceCancelled":
                raise ApiError(409, "request_cancelled", "The recognition request was cancelled") from exc
            raise
        except (ValueError, OSError) as exc:
            code = "invalid_input" if name in {"infer_file", "negotiate"} else "invalid_model"
            paths = sensitive_paths + (_path_variants((args[0],)) if name == "infer_file" else ())
            raise ApiError(422, code, _redact(str(exc), paths)) from exc

    @app.get("/v1/health", response_model=StatusResponse, response_model_exclude_unset=True)
    async def health(request: Request):
        if hasattr(shared(request), "manager_for"):
            raise ApiError(409, "routed_server", "Use /v3/health on a routed server")
        return {"schema_version": 1, "request_id": request.state.request_id,
                "status": await operation(request, "status")}

    @app.get("/v1/models", response_model=ModelsResponse, response_model_exclude_unset=True)
    async def models(request: Request):
        if hasattr(shared(request), "manager_for"):
            raise ApiError(409, "routed_server", "Use /v3/models on a routed server")
        status = await operation(request, "status")
        return {"schema_version": 1, "request_id": request.state.request_id,
                "models": [{"model_id": alias} for alias in sorted(catalog)], "active": status["active"]}

    @app.post("/v1/models/switch", response_model=StatusResponse, response_model_exclude_unset=True)
    async def switch(body: SwitchRequest, request: Request):
        if hasattr(shared(request), "manager_for"):
            raise ApiError(409, "routed_server", "Use /v3/models/switch with a language")
        if body.model_id not in catalog:
            raise ApiError(404, "unknown_model", "The model alias is not in the server catalog")
        # Deployment-only options (such as CUDA DLL paths) stay server-owned.
        selected = replace(defaults, **body.model_dump(exclude={"model_id"}, exclude_none=True))
        status = await operation(request, "switch", catalog[body.model_id], options=selected)
        return {"schema_version": 1, "request_id": request.state.request_id, "status": status}

    @app.post("/v1/models/rollback", response_model=StatusResponse, response_model_exclude_unset=True)
    async def rollback(request: Request):
        if hasattr(shared(request), "manager_for"):
            raise ApiError(409, "routed_server", "Use /v3/models/rollback with a language")
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
                                                 representation_id=representation_id, sign_language=sign_language,
                                                 expected_task_type="classification")
            path = await run_in_threadpool(_save_upload, file, suffix, max_upload_bytes)
            response = await operation(request, "infer_file", path, inference_request,
                                       _manager=selected_manager(request, sign_language))
            body = response.to_dict()
            body = {key: body[key] for key in RecognitionResponse.model_fields}
            body["runtime"] = _public_runtime(body["runtime"], sensitive_paths + _path_variants((path,)))
            return body
        finally:
            if path is not None:
                await run_in_threadpool(path.unlink, missing_ok=True)
            await file.close()

    @app.post("/v2/recognitions", response_model=SequenceRecognitionResponse)
    async def recognize_sequence(
        request: Request,
        file: Annotated[UploadFile, File(description="A complete RGB sign expression video")],
        request_id: Annotated[str | None, Form(min_length=1, max_length=128,
                                             pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")] = None,
        representation_id: Annotated[Literal["tfnet_rgb_v1"] | None, Form()] = None,
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
            if suffix not in _SUFFIXES - {".npy", ".npz"}:
                raise ApiError(422, "unsupported_input", "Supported sequence inputs: .mp4, .avi, .mov, .mkv")
            inference_request = InferenceRequest(
                request_id=request.state.request_id, representation_id=representation_id,
                sign_language=sign_language, expected_task_type="gloss_sequence")
            path = await run_in_threadpool(_save_upload, file, suffix, max_upload_bytes)
            response = await operation(request, "infer_file", path, inference_request,
                                       _manager=selected_manager(request, sign_language))
            body = response.to_dict()
            body = {key: body[key] for key in SequenceRecognitionResponse.model_fields}
            body["schema_version"] = 2
            body["runtime"] = _public_runtime(body["runtime"], sensitive_paths + _path_variants((path,)))
            return body
        finally:
            if path is not None:
                await run_in_threadpool(path.unlink, missing_ok=True)
            await file.close()

    @app.get("/v3/health")
    async def unified_health(request: Request):
        runtime = shared(request)
        capabilities = await operation(request, "capabilities")
        statuses = ({language: await operation(request, "status", _manager=runtime.manager_for(language))
                     for language in capabilities} if hasattr(runtime, "manager_for") else
                    await operation(request, "status"))
        return {"schema_version": 3, "request_id": request.state.request_id,
                "capabilities": _redact(capabilities, sensitive_paths),
                "status": statuses,
                "caption_sessions": len(request.app.state.sessions), "max_caption_sessions": max_sessions}

    @app.get("/v3/models")
    async def unified_models(request: Request):
        runtime = shared(request)
        capabilities = await operation(request, "capabilities")
        statuses = ({language: await operation(request, "status", _manager=runtime.manager_for(language))
                     for language in capabilities} if hasattr(runtime, "manager_for") else
                    await operation(request, "status"))
        return {"schema_version": 3, "request_id": request.state.request_id,
                "models": [{"model_id": alias} for alias in sorted(catalog)],
                "capabilities": _redact(capabilities, sensitive_paths), "status": statuses}

    @app.post("/v3/models/negotiate")
    async def negotiate(body: NegotiationRequest, request: Request):
        target = selected_manager(request, body.sign_language)
        try:
            contract = InferenceRequest(request_id=request.state.request_id, sign_language=body.sign_language,
                                        representation_id=body.representation_id, expected_task_type=body.task_type,
                                        model_id=body.model_id, preprocessing_version=body.preprocessing_version,
                                        expected_generation=body.expected_generation, expected_mode=body.mode)
        except ValueError as exc:
            raise ApiError(422, "invalid_request", str(exc)) from exc
        result = await operation(request, "negotiate", contract, input_kind="file", _manager=target)
        return {"schema_version": 3, "request_id": request.state.request_id,
                "input_contract": result.to_dict() if hasattr(result, "to_dict") else result,
                "model_generation": target.generation,
                "capabilities": _redact(await operation(request, "capabilities", _manager=target), sensitive_paths)}

    @app.post("/v3/models/switch")
    async def routed_switch(body: RouteSwitchRequest, request: Request):
        if body.model_id not in catalog:
            raise ApiError(404, "unknown_model", "The model alias is not in the server catalog")
        target = selected_manager(request, body.sign_language)
        selected = replace(defaults, **body.model_dump(exclude={"model_id", "sign_language"}, exclude_none=True))
        # Prepare before activation so a wrong-language catalog alias cannot
        # replace a route's active model.
        candidate = await operation(request, "prepare", catalog[body.model_id], options=selected, _manager=target)
        try:
            adapter = getattr(candidate, "adapter", None)
            manifest = getattr(getattr(candidate.recognizer, "bundle", None), "manifest", None)
            language = (adapter.capabilities.sign_language if adapter is not None else
                        getattr(manifest, "sign_language", None))
            if language != body.sign_language:
                raise ApiError(422, "invalid_model", "Model language does not match the selected route")
            status = await operation(request, "activate", candidate, _manager=target)
        finally:
            await operation(request, "discard", candidate, _manager=target)
        return {"schema_version": 3, "request_id": request.state.request_id, "status": _public_status(status, sensitive_paths)}

    @app.post("/v3/models/rollback")
    async def routed_rollback(request: Request, sign_language: str):
        status = await operation(request, "rollback", _manager=selected_manager(request, sign_language))
        return {"schema_version": 3, "request_id": request.state.request_id, "status": status}

    def caption_session(request: Request, session_id: str) -> _CaptionSession:
        with request.app.state.session_lock:
            session = request.app.state.sessions.get(session_id)
        if session is None:
            raise ApiError(404, "unknown_session", "Caption session does not exist")
        return session

    def session_body(session: _CaptionSession) -> dict[str, Any]:
        snapshot = session.ledger.snapshot()
        for event in (*snapshot["events"], *snapshot["candidates"]):
            if "runtime" in event:
                event["runtime"] = _public_runtime(event["runtime"], sensitive_paths)
        return {"schema_version": 3, "sign_language": session.sign_language, "closed": session.closed,
                "snapshot": _redact(snapshot, sensitive_paths)}

    @app.post("/v3/caption-sessions", status_code=201)
    async def open_caption_session(body: CaptionSessionRequest, request: Request):
        from .caption_events import CaptionLedger
        target = selected_manager(request, body.sign_language)
        description = await operation(request, "describe", _manager=target)
        if description is None:
            raise ApiError(503, "model_unavailable", "No model is loaded")
        if description["sign_language"] != body.sign_language:
            raise ApiError(422, "invalid_input", "Session language does not match the model")
        session_id = body.session_id or str(uuid4())
        with request.app.state.session_lock:
            existing = request.app.state.sessions.get(session_id)
            if existing is not None:
                if existing.sign_language == body.sign_language and not existing.closed:
                    if existing.ledger.snapshot()["source_epoch"] == body.source_epoch:
                        return JSONResponse(status_code=200, content=session_body(existing))
                raise ApiError(409, "session_conflict", "Session ID is already bound to another lifecycle")
            if len(request.app.state.sessions) >= max_sessions:
                raise ApiError(429, "session_capacity", "Close and delete a session before creating another")
            ledger = CaptionLedger(session_id, max_seen=max(max_session_requests, max_session_actions))
            ledger.bind(source_epoch=body.source_epoch, model_generation=description["generation"])
            session = _CaptionSession(ledger, body.sign_language)
            request.app.state.sessions[session_id] = session
        return session_body(session)

    @app.get("/v3/caption-sessions/{session_id}")
    async def get_caption_session(session_id: str, request: Request):
        session = caption_session(request, session_id)
        with session.lock:
            return session_body(session)

    @app.delete("/v3/caption-sessions/{session_id}")
    async def delete_caption_session(session_id: str, request: Request):
        with request.app.state.session_lock:
            session = request.app.state.sessions.pop(session_id, None)
        if session is not None:
            session.close()
        return {"schema_version": 3, "session_id": session_id, "closed": True}

    @app.post("/v3/caption-sessions/{session_id}/actions")
    async def caption_action(session_id: str, body: CaptionActionRequest, request: Request):
        session = caption_session(request, session_id)
        payload = body.model_dump()
        with session.lock:
            prior = session.actions.get(body.action_id)
            if prior is not None:
                if prior[0] != payload:
                    raise ApiError(409, "action_conflict", "action_id was used with another action")
                return deepcopy(prior[1])
            if len(session.actions) >= max_session_actions:
                raise ApiError(429, "action_capacity", "Session action history is full; create a new session")
            if session.closed:
                raise ApiError(409, "session_closed", "Caption session is closed")
            if body.action != "invalidate_source" and body.source_epoch is not None:
                raise ApiError(422, "invalid_request", "source_epoch applies only to invalidate_source")
            if body.action == "invalidate_source":
                snapshot = session.ledger.snapshot()
                if body.source_epoch is None or body.source_epoch <= snapshot["source_epoch"]:
                    raise ApiError(409, "stale_source", "New source_epoch must increase")
                session.cancel_pending()
                session.ledger.bind(source_epoch=body.source_epoch, model_generation=snapshot["model_generation"])
            elif body.action == "undo":
                session.ledger.undo(action_id=body.action_id)
            elif body.action == "clear":
                session.cancel_pending()
                session.ledger.clear(action_id=body.action_id)
            elif body.action == "close":
                session.close()
            else:
                session.cancel_pending()
            result = session_body(session)
            result["action_id"] = body.action_id
            session.actions[body.action_id] = (payload, deepcopy(result))
            return result

    @app.post("/v3/recognitions", response_model=UnifiedRecognitionResponse)
    async def unified_recognize(
        request: Request,
        file: Annotated[UploadFile, File()],
        request_id: Annotated[str | None, Form(min_length=1, max_length=128,
                                              pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")] = None,
        representation_id: Annotated[str | None, Form(min_length=1, max_length=128)] = None,
        sign_language: Annotated[str | None, Form(min_length=1, max_length=64)] = None,
        task_type: Annotated[Literal["classification", "gloss_sequence", "text"] | None, Form()] = None,
        model_id: Annotated[str | None, Form(min_length=1, max_length=128)] = None,
        expected_generation: Annotated[int | None, Form(ge=1)] = None,
        preprocessing_version: Annotated[str | None, Form(min_length=1, max_length=128)] = None,
        mode: Annotated[Literal["clip", "stream"], Form()] = "clip",
        session_id: Annotated[str | None, Form(min_length=1, max_length=128)] = None,
        segment_id: Annotated[str | None, Form(min_length=1, max_length=128)] = None,
        sequence: Annotated[int | None, Form(ge=1)] = None,
        source_epoch: Annotated[int | None, Form(ge=0)] = None,
        start_time: Annotated[float | None, Form(ge=0, allow_inf_nan=False)] = None,
        end_time: Annotated[float | None, Form(ge=0, allow_inf_nan=False)] = None,
        source_id: Annotated[str | None, Form(min_length=1, max_length=128)] = None,
        timeout_ms: Annotated[int | None, Form(ge=1, le=3600000)] = None,
    ):
        from .runtime_contracts import CancellationToken
        path: Path | None = None
        task: asyncio.Task | None = None
        session: _CaptionSession | None = None
        record: _RecognitionRecord | None = None
        token = CancellationToken()
        if request_id is not None:
            request.state.request_id = request_id
        try:
            values = {"request_id": request.state.request_id, "representation_id": representation_id,
                      "sign_language": sign_language, "task_type": task_type, "model_id": model_id,
                      "expected_generation": expected_generation, "preprocessing_version": preprocessing_version,
                      "mode": mode, "session_id": session_id, "segment_id": segment_id, "sequence": sequence,
                      "source_epoch": source_epoch, "start_time": start_time, "end_time": end_time,
                      "source_id": source_id, "timeout_ms": timeout_ms}
            for name, value in values.items():
                if isinstance(value, str) and not value.strip():
                    raise ApiError(422, "invalid_request", f"{name} must be nonempty")
            segment_fields = (segment_id, sequence, source_epoch, start_time, end_time, source_id)
            if session_id is None and any(value is not None for value in segment_fields):
                raise ApiError(422, "invalid_request", "Segment metadata requires session_id")
            if session_id is not None:
                if any(value is None for value in segment_fields) or sign_language is None or expected_generation is None:
                    raise ApiError(422, "invalid_request", "Caption requests require all segment metadata, language and expected_generation")
                if end_time < start_time:
                    raise ApiError(422, "invalid_request", "end_time must not precede start_time")
                session = caption_session(request, session_id)
                if session.sign_language != sign_language:
                    raise ApiError(409, "session_conflict", "Request language does not match the caption session")
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in _SUFFIXES:
                raise ApiError(422, "unsupported_input", "Supported files: .mp4, .avi, .mov, .mkv, .npy, .npz")
            target = selected_manager(request, sign_language)
            duration = min(inference_timeout_seconds, timeout_ms / 1000 if timeout_ms is not None else inference_timeout_seconds)
            contract = InferenceRequest(request_id=request.state.request_id, representation_id=representation_id,
                                        sign_language=sign_language, expected_task_type=task_type, model_id=model_id,
                                        preprocessing_version=preprocessing_version, expected_generation=expected_generation,
                                        expected_mode=mode, session_id=session_id, segment_id=segment_id,
                                        source_epoch=source_epoch, cancellation=token,
                                        deadline_monotonic=time.monotonic() + duration)
            path = await run_in_threadpool(_save_upload, file, suffix, max_upload_bytes)
            if session is not None:
                fingerprint = await run_in_threadpool(_fingerprint, path, {**values, "suffix": suffix})
                with session.lock:
                    record = session.records.get(contract.request_id)
                    if record is not None:
                        if record.fingerprint != fingerprint:
                            raise ApiError(409, "request_conflict", "request_id was used with another payload")
                        if record.body is not None:
                            return deepcopy(record.body)
                        if record.error is not None:
                            raise ApiError(*record.error)
                        raise ApiError(409, "request_inflight", "This request is already running")
                    if session.closed:
                        raise ApiError(409, "session_closed", "Caption session is closed")
                    snapshot = session.ledger.snapshot()
                    if source_epoch != snapshot["source_epoch"]:
                        raise ApiError(409, "stale_source", "Source epoch is no longer current")
                    if len(session.records) >= max_session_requests:
                        raise ApiError(429, "request_capacity", "Session request history is full; create a new session")
            await operation(request, "negotiate", contract, input_kind="file", _manager=target)
            if session is not None:
                with session.lock:
                    # Negotiation runs off the event loop. Another request or
                    # source action may have changed the session while it ran.
                    concurrent = session.records.get(contract.request_id)
                    if concurrent is not None:
                        if concurrent.fingerprint != fingerprint:
                            raise ApiError(409, "request_conflict", "request_id was used with another payload")
                        if concurrent.body is not None:
                            return deepcopy(concurrent.body)
                        if concurrent.error is not None:
                            raise ApiError(*concurrent.error)
                        raise ApiError(409, "request_inflight", "This request is already running")
                    if session.closed:
                        raise ApiError(409, "session_closed", "Caption session is closed")
                    if source_epoch != session.ledger.snapshot()["source_epoch"]:
                        raise ApiError(409, "stale_source", "Source epoch is no longer current")
                    if len(session.records) >= max_session_requests:
                        raise ApiError(429, "request_capacity", "Session request history is full; create a new session")
                    if session.ledger.snapshot()["model_generation"] != expected_generation:
                        session.cancel_pending()
                    session.ledger.bind(source_epoch=source_epoch, model_generation=expected_generation)
                    try:
                        event_id = session.ledger.reserve(segment_id=segment_id, sequence=sequence,
                            request_id=contract.request_id, source_epoch=source_epoch, model_generation=expected_generation,
                            start_time=start_time, end_time=end_time, source_id=source_id)
                    except ValueError as exc:
                        raise ApiError(409, "segment_conflict", str(exc)) from exc
                    except RuntimeError as exc:
                        raise ApiError(429, "candidate_capacity", str(exc)) from exc
                    record = _RecognitionRecord(fingerprint, token, event_id, segment_id)
                    session.records[contract.request_id] = record
            native_path, path = path, None

            async def execute():
                try:
                    response = await operation(request, "infer_file", native_path, contract, _manager=target)
                    body = response.to_dict()
                    body["schema_version"] = 3
                    body["runtime"] = _public_runtime(body["runtime"], sensitive_paths + _path_variants((native_path,)))
                    body = _redact(body, sensitive_paths + _path_variants((native_path,)))
                    body = UnifiedRecognitionResponse.model_validate(body).model_dump()
                    if token.cancelled:
                        raise ApiError(409, "request_cancelled", "The recognition request was cancelled")
                    if time.monotonic() >= contract.deadline_monotonic:
                        raise ApiError(504, "inference_timeout", "The inference deadline expired")
                    if session is not None:
                        with session.lock:
                            body["caption_event_id"] = record.event_id
                            public_result = replace(response.result,
                                quality=_redact(response.result.quality, sensitive_paths + _path_variants((native_path,))),
                                reason=_redact(response.result.reason, sensitive_paths + _path_variants((native_path,))))
                            session.ledger.complete(replace(response, result=public_result, runtime=body["runtime"]), caption_catalog)
                            event = session.ledger.lookup(segment_id)
                            body["caption_committed"] = bool(event and event["status"] == "committed")
                            if record.error is not None:
                                raise ApiError(*record.error)
                            record.body = deepcopy(body)
                    return body
                except BaseException as exc:
                    if session is not None:
                        with session.lock:
                            session.ledger.reject(segment_id, exc.code if isinstance(exc, ApiError) else type(exc).__name__)
                            if record.error is None:
                                record.error = ((exc.status_code, exc.code, exc.message) if isinstance(exc, ApiError)
                                                else (500, "internal_error", "Internal server error"))
                    raise
                finally:
                    await run_in_threadpool(native_path.unlink, missing_ok=True)

            task = asyncio.create_task(execute())
            request.app.state.jobs.add(task)
            def finished(job):
                request.app.state.jobs.discard(job)
                if not job.cancelled():
                    job.exception()
            task.add_done_callback(finished)
            while not task.done():
                remaining = contract.deadline_monotonic - time.monotonic()
                if remaining <= 0 or token.cancelled:
                    status, code, message = ((504, "inference_timeout", "The inference deadline expired") if remaining <= 0
                                             else (409, "request_cancelled", "The recognition request was cancelled"))
                    token.cancel()
                    if session is not None:
                        with session.lock:
                            record.error = (status, code, message)
                            session.ledger.reject(segment_id, code)
                    raise ApiError(status, code, message)
                await asyncio.wait({task}, timeout=min(remaining, .05))
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            token.cancel()
            if session is not None and record is not None:
                with session.lock:
                    session.ledger.reject(segment_id, "client_disconnected")
            raise
        except ValueError as exc:
            raise ApiError(422, "invalid_request", str(exc)) from exc
        finally:
            if path is not None:
                await run_in_threadpool(path.unlink, missing_ok=True)
            await file.close()

    # ------------------------------------------------------------------
    # P0-A V1 contract endpoints (T-A2-3): POST /v1/videos/infer and
    # GET /v1/jobs/{job_id}.  These reuse the existing auth middleware,
    # bounded-upload middleware, error mapping and path redaction; existing
    # v1/v2/v3 endpoints above are untouched.
    # ------------------------------------------------------------------
    _JOB_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv", ".json", ".npy", ".npz"})

    def _job_runner(request: Request):
        if request.app.state.job_runner is None:
            from .job_runner import JobRunner
            manager = shared(request)
            description = manager.describe() if hasattr(manager, "describe") else None
            if description is None:
                raise ApiError(503, "model_unavailable", "No model is loaded")
            sign_language = description.get("sign_language") or "csl"
            sign_system = {"csl": "CSL", "libras": "LIBRAS"}.get(sign_language, "UNKNOWN")
            task_type = description.get("task_type", "text")
            task = {"text": "CSLT", "gloss_sequence": "CSLR",
                    "classification": "ISLR"}[task_type]
            run_directory = request.app.state.job_run_directory
            request.app.state.job_runner = JobRunner(
                manager, request.app.state.job_store,
                run_directory=(run_directory if run_directory is not None
                               else Path(tempfile.gettempdir()) / f"slbridge-jobs-{job_run_id}"),
                run_id=request.app.state.job_run_id,
                model_id=description.get("model_id", "unknown"),
                model_version=str(description.get("generation", "1")),
                preprocess_version=description.get("input_spec", {}).get(
                    "representation_id", "unknown")
                if isinstance(description.get("input_spec"), dict) else "unknown",
                task=task, sign_system=sign_system,
                timeout_seconds=request.app.state.job_timeout_seconds)
        return request.app.state.job_runner

    # ------------------------------------------------------------------
    # T-A4-2 health endpoints per the V1 contract: /v1/health/live is
    # process-only liveness; /v1/health/ready reflects model availability
    # (not ready -> 503 with a contract error body).  The legacy /v1/health
    # and /v3/health endpoints above are untouched.
    # ------------------------------------------------------------------
    @app.get("/v1/health/live")
    async def v1_health_live():
        return JSONResponse(status_code=200, content={"status": "alive"})

    @app.get("/v1/health/ready")
    async def v1_health_ready(request: Request):
        workers: dict[str, str] = {}
        ready = True
        runtime = shared(request)
        try:
            if hasattr(runtime, "manager_for"):
                capabilities = await operation(request, "capabilities")
                for language in capabilities or {}:
                    description = await operation(request, "describe",
                                                  _manager=runtime.manager_for(language))
                    workers[language] = "ready" if description else "unavailable"
                    ready = ready and bool(description)
            else:
                description = await operation(request, "describe")
                workers["default"] = "ready" if description else "unavailable"
                ready = ready and bool(description)
        except (ApiError, ModelUnavailable):
            ready = False
            workers["default"] = "unavailable"
        if ready:
            return JSONResponse(status_code=200,
                                 content={"status": "ready", "workers": workers})
        return JSONResponse(
            status_code=503,
            content={"schema_version": 1, "request_id": request.state.request_id,
                     "status": "not_ready", "workers": workers,
                     "error": {"code": "MODEL_UNAVAILABLE",
                               "legacy_code": "model_unavailable",
                               "message": "No model is loaded; the service is not ready"}})

    @app.post("/v1/videos/infer", status_code=202)
    async def submit_video_infer(
        request: Request,
        video: Annotated[UploadFile, File(description="A video, or a replay query JSON document")],
        sign_system: Annotated[Literal["CSL", "LIBRAS", "UNKNOWN"], Form()],
        mode: Annotated[Literal["parallel_fusion", "adaptive_cascade", "single_model"], Form()],
        sample_id: Annotated[str | None, Form(min_length=1, max_length=128)] = None,
        job_id: Annotated[str | None, Form(min_length=1, max_length=128,
                                            pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")] = None,
        output_zh: Annotated[bool, Form()] = True,
        output_pt_br: Annotated[bool, Form()] = True,
    ):
        # T-A4-2: every error on the V1 contract endpoints carries the
        # contract uppercase code (+ legacy_code transition field).
        if mode != "single_model":
            # Legal enum value, intentionally not enabled in P0-A (G2 FAILED:
            # fusion stays out of the main path).  Explicit refusal, never a
            # silent fallback to single_model.
            raise ApiError(422, "invalid_input",
                           f"mode '{mode}' is not enabled in P0-A; only 'single_model' is available",
                           contract_code="INVALID_INPUT")
        if sign_system == "LIBRAS":
            raise ApiError(503, "model_unavailable",
                           "no Libras-trained weights; LIBRAS routing is unavailable",
                           contract_code="MODEL_UNAVAILABLE")
        if sign_system == "UNKNOWN":
            raise ApiError(422, "invalid_input",
                           "sign_system must be explicitly CSL in P0/P1",
                           contract_code="INVALID_INPUT")
        if sample_id is None or not sample_id.strip():
            raise ApiError(422, "invalid_input", "sample_id is required for replay-driven inference",
                           contract_code="INVALID_INPUT")
        runner = _job_runner(request)
        resolved_job_id = job_id or str(uuid4())
        path: Path | None = None
        try:
            suffix = Path(video.filename or "").suffix.lower()
            if suffix not in _JOB_SUFFIXES:
                raise ApiError(422, "unsupported_input",
                               "Supported files: .mp4, .avi, .mov, .mkv, .json, .npy, .npz",
                               contract_code="INVALID_INPUT")
            path = await run_in_threadpool(_save_upload, video, suffix, max_upload_bytes)
            # Register the job (idempotent) synchronously, then execute in the
            # background: the 202 must return with status QUEUED (contract
            # VideoInferAccepted), never block on inference.
            try:
                record = await run_in_threadpool(
                    request.app.state.job_store.create, job_id=resolved_job_id,
                    sign_system="CSL", sample_id=sample_id)
            except ValueError as exc:
                message = str(exc)
                if "bound to another payload" in message:
                    raise ApiError(409, "request_conflict", _redact(message, sensitive_paths),
                                   contract_code="INVALID_INPUT") from exc
                raise ApiError(422, "invalid_input", _redact(message, sensitive_paths),
                               contract_code="INVALID_INPUT") from exc
            except RuntimeError as exc:
                raise ApiError(429, "request_capacity", str(exc),
                               contract_code="INVALID_INPUT") from exc
            native_path, path = path, None

            # T-A3-1: probe media metadata honestly (failures stay null with a
            # recorded reason) before execution, and arm the upload TTL.
            # Idempotent re-submission must not re-probe: the record already
            # carries its (job-bound) metadata and re-probing would tie the
            # record to the new temp file's name.
            import time as _time
            with record.lock:
                fresh = record.media_metadata is None and record.media_probe_error is None
            if fresh:
                probe = request.app.state.job_media_probe
                if probe is None:
                    from .media_probe import probe_media as probe
                probed = await run_in_threadpool(probe, native_path)
                ttl = request.app.state.job_upload_ttl_seconds
                with record.lock:
                    record.media_metadata = probed.get("metadata")
                    record.media_probe_error = probed.get("probe_error")
                    record.upload_path = native_path
                    record.upload_expires_at = (_time.monotonic() + ttl
                                                if ttl is not None else None)
                request.app.state.job_store.journal(record)

            async def execute():
                try:
                    await run_in_threadpool(runner.run, job_id=resolved_job_id,
                                            sample_id=sample_id, query_path=native_path)
                finally:
                    # TTL=0 (or expired) cleans immediately; larger TTLs are
                    # swept on later submits/queries.  Prediction JSONL and
                    # manifests are evidence and are never touched.
                    await run_in_threadpool(
                        request.app.state.job_store.sweep_uploads)
                    await run_in_threadpool(native_path.unlink, missing_ok=True)

            task = asyncio.create_task(execute())
            request.app.state.jobs.add(task)

            def finished(job):
                request.app.state.jobs.discard(job)
                if not job.cancelled():
                    job.exception()

            task.add_done_callback(finished)
            body = {"job_id": resolved_job_id, "status": request.app.state.job_store.get(resolved_job_id).state}
            return JSONResponse(status_code=202, content=body)
        finally:
            if path is not None:
                await run_in_threadpool(path.unlink, missing_ok=True)
            await video.close()

    @app.get("/v1/jobs/{job_id}")
    async def get_v1_job(job_id: str, request: Request):
        record = request.app.state.job_store.get(job_id)
        if record is None:
            raise ApiError(404, "unknown_job", "Job not found", contract_code="INVALID_INPUT")
        payload = record.to_dict()
        # Contract-level shape: JobStatus with an optional final subtitle event.
        final_event = None
        if record.state == "SUCCEEDED" and record.text is not None:
            final_event = {
                "event": "subtitle.final", "schema_version": "1.0",
                "session_id": "offline", "utterance_id": record.job_id, "revision": 1,
                "time_span_ms": {"start": 0, "end": 0},
                "sign_system": record.sign_system, "status": "FINAL",
                "text": {"zh-CN": record.text}, "evidence_ids": [record.job_id],
            }
        body = {
            "job_id": record.job_id, "status": record.state,
            "sign_system": record.sign_system,
            "final_event": final_event,
            "error": payload.get("error"),
            "metrics": payload.get("metrics", {}),
            "media_metadata": record.media_metadata,
            "media_probe_error": record.media_probe_error,
        }
        return JSONResponse(status_code=200, content=body)

    return app
