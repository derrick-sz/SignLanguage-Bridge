"""OpenAI-compatible HTTP backend (real implementation for the wb-proxy endpoint).

Endpoint (user-selected, Lead smoke-verified 2026-10-26):

    provider      wb-proxy
    base_url      http://127.0.0.1:8788/v1
    endpoint      POST {base_url}/chat/completions
    model         deepseek-v4.1-flash
    auth          Authorization: Bearer <key>

Credential handling (security rule): the API key is read **only at runtime, in
memory**, from (in order) the ``P1_OPENAI_API_KEY`` / ``WB_PROXY_API_KEY``
environment variables or the yaml ``refs.<credentials_ref>`` entry of
``P1_CREDENTIALS_FILE`` (default ``C:\\Users\\ADMIN\\.dsh\\.credentials.yaml``).
The key is never printed, never written to any output file/JSONL/manifest/log;
manifests only record a reference string such as
``WB_PROXY_API_KEY@dsh-credentials``.

Real calls are gated by ``P1_LLM_ENABLE_REAL=1`` (and the harness ``--allow-real``
flag) so accidental runs cannot happen.

Implementation notes:
* stdlib ``urllib`` only, timeout 90 s;
* temperature=0, fixed ``max_tokens`` (from the prompt spec, 256);
* retries: up to 2 retries (3 attempts) with exponential backoff, counting every
  attempt; retry on timeout, connection errors, HTTP 429 and HTTP 5xx;
* error records keep only the exception type / HTTP status (truncated), never a
  response body, and never request headers.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from experiments.p1.e02.llm_fusion.backends.base import (
    BackendDisabled,
    BackendResult,
    LLMBackend,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8788/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash"
DEFAULT_CREDENTIALS_FILE = r"C:\Users\ADMIN\.dsh\.credentials.yaml"
DEFAULT_CREDENTIALS_REF = "WB_PROXY_API_KEY"
DEFAULT_TIMEOUT_SECONDS = 90
ENABLE_ENV = "P1_LLM_ENABLE_REAL"
CREDENTIALS_REF_LABEL = "WB_PROXY_API_KEY@dsh-credentials"


def _read_api_key(credentials_file: str | None, credentials_ref: str) -> tuple[str | None, str]:
    """Resolve the key in memory. Returns (key, source_label); never logs the value."""

    for env_name in ("P1_OPENAI_API_KEY", "WB_PROXY_API_KEY"):
        value = os.environ.get(env_name)
        if value:
            return value, f"env:{env_name}"
    path = credentials_file or os.environ.get("P1_CREDENTIALS_FILE") or DEFAULT_CREDENTIALS_FILE
    if not path or not os.path.exists(path):
        return None, "missing-credentials-file"
    try:
        import yaml  # pyyaml available in .conda-env

        with open(path, encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        refs = payload.get("refs") or {}
        value = refs.get(credentials_ref)
        if value:
            return str(value), f"yaml:refs.{credentials_ref}"
    except Exception:  # noqa: BLE001 - credential resolution must never crash a run or leak
        return None, "credentials-read-error"
    return None, "missing-ref"


class OpenAICompatBackend(LLMBackend):
    name = "openai_compat"
    version = "openai-compat-urllib-v1"
    is_stub = False
    is_real = True
    provider = "wb-proxy"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        credentials_file: str | None = None,
        credentials_ref: str = DEFAULT_CREDENTIALS_REF,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        enabled: bool | None = None,
        max_retries: int = 2,
    ) -> None:
        self.base_url = (base_url or os.environ.get("P1_OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("P1_OPENAI_MODEL") or DEFAULT_MODEL
        self.credentials_file = credentials_file or os.environ.get("P1_CREDENTIALS_FILE") or DEFAULT_CREDENTIALS_FILE
        self.credentials_ref = credentials_ref
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.enabled = enabled if enabled is not None else os.environ.get(ENABLE_ENV) == "1"
        self._api_key: str | None = None
        self._key_source: str | None = None
        self._last_error: str | None = None

    # -- credential handling ------------------------------------------------
    def _ensure_key(self) -> str | None:
        if self._api_key is None:
            key, source = _read_api_key(self.credentials_file, self.credentials_ref)
            self._api_key = key
            self._key_source = source
        return self._api_key

    def key_present(self) -> bool:
        return self._ensure_key() is not None

    # -- probes -------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        probe = _probe_endpoint(self.base_url)
        return {
            "name": self.name,
            "version": self.version,
            "provider": self.provider,
            "base_url": self.base_url,
            "endpoint": f"{self.base_url}/chat/completions",
            "model": self.model,
            "is_stub": False,
            "is_real": True,
            "available": bool(probe.get("reachable")) and self.key_present(),
            "endpoint_probe": probe,
            "credentials_ref": CREDENTIALS_REF_LABEL,
            "credentials_present": self.key_present(),
            "credentials_source": self._key_source,
            "enabled": self.enabled,
            "enable_env": ENABLE_ENV,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "status": "REAL_READY" if (probe.get("reachable") and self.key_present() and self.enabled)
            else ("DISABLED" if not self.enabled else "NOT_READY"),
        }

    # -- generation ---------------------------------------------------------
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        group: str,
        evidence: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        top_p: float = 1.0,
    ) -> BackendResult:
        if not self.enabled:
            raise BackendDisabled(
                f"real LLM calls are disabled; set {ENABLE_ENV}=1 after the endpoint/model is approved"
            )
        api_key = self._ensure_key()
        if not api_key:
            raise BackendDisabled(
                f"no credentials resolved (ref={self.credentials_ref}, source={self._key_source}); "
                "set WB_PROXY_API_KEY or provide the dsh credentials yaml"
            )
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        url = f"{self.base_url}/chat/completions"
        attempts = 0
        last_error: str | None = None
        started = time.perf_counter()
        while attempts <= self.max_retries:
            attempts += 1
            try:
                request = urllib.request.Request(
                    url,
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read()
                    http_status = getattr(response, "status", 200)
                data = json.loads(raw.decode("utf-8"))
                text = _extract_text(data)
                usage = data.get("usage") or {}
                response_model = data.get("model") or self.model
                latency_ms = (time.perf_counter() - started) * 1000.0
                finish_reason = _first_choice(data).get("finish_reason")
                status = "ok" if text.strip() else "error"
                error = None
                if status != "ok":
                    error = (f"empty completion text (finish_reason={finish_reason}; "
                             f"max_tokens={max_tokens})")[:160]
                return BackendResult(
                    text=text,
                    latency_ms=latency_ms,
                    tokens_in=_as_int(usage.get("prompt_tokens")),
                    tokens_out=_as_int(usage.get("completion_tokens")),
                    model_version=str(response_model),
                    status=status,
                    error=error,
                    raw_event_count=None,
                    extra={
                        "provider": self.provider,
                        "endpoint": url,
                        "http_status": http_status,
                        "retries": attempts - 1,
                        "attempts": attempts,
                        "tokens_total": _as_int(usage.get("total_tokens")),
                        "request_model": self.model,
                        "response_model": response_model,
                        "finish_reason": finish_reason,
                        "group": group,
                    },
                )
            except urllib.error.HTTPError as error:
                status = int(getattr(error, "code", 0) or 0)
                last_error = f"HTTPError status={status}"
                if status in (429,) or status >= 500:
                    if attempts <= self.max_retries:
                        time.sleep(_backoff_seconds(attempts))
                        continue
                break
            except urllib.error.URLError as error:
                reason = getattr(error, "reason", None)
                last_error = f"URLError {type(reason).__name__ if reason else 'unknown'}"
                if attempts <= self.max_retries:
                    time.sleep(_backoff_seconds(attempts))
                    continue
                break
            except (TimeoutError, json.JSONDecodeError) as error:
                last_error = f"{type(error).__name__}"
                if attempts <= self.max_retries:
                    time.sleep(_backoff_seconds(attempts))
                    continue
                break
            except Exception as error:  # noqa: BLE001 - record type only, never a body/header
                last_error = f"{type(error).__name__}"
                break
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._last_error = last_error
        return BackendResult(
            text="",
            latency_ms=latency_ms,
            tokens_in=None,
            tokens_out=None,
            model_version=self.model,
            status="error",
            error=(last_error or "unknown backend error")[:160],
            raw_event_count=None,
            extra={
                "provider": self.provider,
                "endpoint": url,
                "http_status": None,
                "retries": attempts - 1,
                "attempts": attempts,
                "request_model": self.model,
                "group": group,
            },
        )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _backoff_seconds(attempt: int) -> float:
    return float(min(2 ** attempt, 8))


def _as_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _first_choice(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def _extract_text(data: dict[str, Any]) -> str:
    choice = _first_choice(data)
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):  # some providers return content parts
        parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "".join(parts).strip()
    return ""


def _probe_endpoint(base_url: str) -> dict[str, Any]:
    """Local reachability probe (no credentials, no model call): /v1/models returns 401/200."""

    url = f"{base_url}/models"
    try:
        urllib.request.urlopen(url, timeout=5)
        return {"reachable": True, "status": "ok"}
    except urllib.error.HTTPError as error:
        return {"reachable": True, "status": f"HTTP {error.code}"}
    except Exception as error:  # noqa: BLE001 - probe must never break a run
        return {"reachable": False, "status": f"{type(error).__name__}"}
