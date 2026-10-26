"""Codex CLI backend (implemented, NOT called in this round).

Wraps the already-logged-in local Codex CLI:

    codex exec --json --ephemeral --sandbox read-only -m <model> -o <tmpfile> -   (prompt on stdin)

The final answer is read from the `-o/--output-last-message` file (stable across
CLI versions); the `--json` event stream is only used for diagnostics and token
usage if present.  Real calls are **disabled by default**: enable with
``P1_LLM_ENABLE_REAL=1`` after the Lead/user chooses the endpoint and model.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from experiments.p1.e02.llm_fusion.backends.base import (
    BackendDisabled,
    BackendResult,
    LLMBackend,
)

DEFAULT_MODEL_ENV = "P1_CODEX_MODEL"
ENABLE_ENV = "P1_LLM_ENABLE_REAL"


def resolve_codex_command() -> dict[str, Any]:
    """Resolve the local Codex CLI shim into a runnable command prefix.

    On Windows `codex` is an npm shim (codex.CMD); CreateProcess cannot execute
    .CMD directly, so route it through ``cmd.exe /c`` (and .ps1 through pwsh).
    """

    executable = shutil.which("codex") or shutil.which("codex.cmd") or shutil.which("codex.exe")
    if not executable:
        return {"available": False, "reason": "codex executable not found on PATH", "command_prefix": None}
    lower = executable.lower()
    if lower.endswith((".cmd", ".bat")):
        prefix = ["cmd.exe", "/c", executable]
    elif lower.endswith(".ps1"):
        prefix = ["pwsh", "-NoProfile", "-Command", executable]
    else:
        prefix = [executable]
    return {"available": True, "executable": executable, "command_prefix": prefix}


class CodexCLIBackend(LLMBackend):
    name = "codex_cli"
    version = "codex-cli-on-request"
    is_stub = False
    is_real = True

    def __init__(self, model: str | None = None, *, timeout_seconds: int = 180,
                 cwd: Path | None = None, enabled: bool | None = None) -> None:
        self.model = model or os.environ.get(DEFAULT_MODEL_ENV) or None
        self.timeout_seconds = timeout_seconds
        self.cwd = Path(cwd) if cwd else None
        self.enabled = enabled if enabled is not None else os.environ.get(ENABLE_ENV) == "1"

    # -- probes (safe: local CLI only, no network / no model call) ----------
    @staticmethod
    def probe() -> dict[str, Any]:
        resolved = resolve_codex_command()
        if not resolved["available"]:
            return {"available": False, "reason": resolved["reason"]}
        try:
            proc = subprocess.run(
                resolved["command_prefix"] + ["--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            return {
                "available": proc.returncode == 0,
                "path": resolved["executable"],
                "version": (proc.stdout or proc.stderr or "").strip()[:200],
                "returncode": proc.returncode,
            }
        except Exception as error:  # noqa: BLE001 - probe must never break a run
            return {"available": False, "reason": f"{type(error).__name__}: {error}"}

    def describe(self) -> dict[str, Any]:
        probe = self.probe()
        return {
            "name": self.name,
            "version": probe.get("version") or self.version,
            "is_stub": False,
            "is_real": True,
            "available": bool(probe.get("available")) and self.enabled,
            "probe": probe,
            "enabled": self.enabled,
            "enable_env": ENABLE_ENV,
            "model": self.model,
            "model_env": DEFAULT_MODEL_ENV,
            "command": "codex exec --json --ephemeral --sandbox read-only -m <model> -o <tmpfile> -",
            "status": "IMPLEMENTED_UNTESTED" if not self.enabled else "ENABLED",
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
        probe = self.probe()
        if not probe.get("available"):
            raise BackendDisabled(f"codex CLI unavailable: {probe}")
        resolved = resolve_codex_command()
        if not resolved["available"]:  # pragma: no cover - probe already checked
            raise BackendDisabled(f"codex CLI unavailable: {resolved}")
        prompt = _flatten_messages(messages)
        cmd = resolved["command_prefix"] + [
            "exec",
            "--json",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "-o",
            "@@OUTFILE@@",
        ]
        if self.model:
            cmd.extend(["-m", self.model])
        cmd.append("-")
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="g2a_codex_") as tmp_dir:
            outfile = Path(tmp_dir) / "last_message.txt"
            cmd = [str(outfile) if part == "@@OUTFILE@@" else part for part in cmd]
            proc = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                cwd=str(self.cwd) if self.cwd else None,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            final_text = outfile.read_text(encoding="utf-8").strip() if outfile.exists() else ""
            events = _parse_jsonl(proc.stdout)
            tokens_in, tokens_out = _extract_usage(events)
        status = "ok" if proc.returncode == 0 and final_text else "error"
        error = None
        if status != "ok":
            error = f"codex exec exit={proc.returncode}; stderr={proc.stderr[-500:]!r}"
        return BackendResult(
            text=final_text,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model_version=self.model or "codex-default",
            status=status,
            error=error,
            raw_event_count=len(events),
            extra={"group": group, "returncode": proc.returncode, "stdout_prefix": proc.stdout[:300]},
        )


def _flatten_messages(messages: list[dict[str, str]]) -> str:
    lines = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            lines.append(f"[系统指令]\n{content}")
        else:
            lines.append(f"[输入]\n{content}")
    return "\n\n".join(lines)


def _parse_jsonl(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _extract_usage(events: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    tokens_in = tokens_out = None
    for event in events:
        usage = event.get("usage") if isinstance(event, dict) else None
        if isinstance(usage, dict):
            if isinstance(usage.get("input_tokens"), int):
                tokens_in = usage["input_tokens"]
            if isinstance(usage.get("output_tokens"), int):
                tokens_out = usage["output_tokens"]
    return tokens_in, tokens_out
