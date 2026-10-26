"""ffprobe-based media metadata for uploaded job inputs (P0-A A3, T-A3-1).

Honesty rules (task card):
* probe failures return ``{"metadata": None, "probe_error": "<reason>"}`` --
  fields are never fabricated, guessed or defaulted;
* non-media uploads (today's replay-query JSON documents) fail ffprobe by
  nature and record the ffprobe error verbatim;
* every metadata value is a JSON primitive so it can ride on job records and
  run manifests unchanged.

The subprocess runner is injectable (``probe_media_with_runner``) so parse
paths stay testable without ffmpeg.  ffprobe is invoked read-only with
``-nostdin``; files are never modified.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, Callable

__all__ = ["probe_media", "probe_media_with_runner"]

_PROBE_TIMEOUT_SECONDS = 30.0
# Note: this Windows ffprobe build rejects -nostdin with a usage banner, so
# stdin is disabled via subprocess.DEVNULL instead (same effect, portable).
_PROBE_ARGS = ("-v", "error", "-print_format", "json",
               "-show_format", "-show_streams")


def _default_runner(path: Path) -> tuple[int, str, str]:
    completed = subprocess.run(["ffprobe", *_PROBE_ARGS, str(path)],
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=_PROBE_TIMEOUT_SECONDS,
                               stdin=subprocess.DEVNULL)
    return completed.returncode, completed.stdout, completed.stderr


def _parse_fps(rate: str) -> float | None:
    if not isinstance(rate, str) or "/" not in rate:
        return None
    try:
        numerator, denominator = rate.split("/", 1)
        numerator, denominator = float(numerator), float(denominator)
    except ValueError:
        return None
    if denominator == 0 or numerator <= 0:
        return None
    return round(numerator / denominator, 3)


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 3) if number > 0 else None


def probe_media_with_runner(path: Path,
                            runner: Callable[[Path], tuple[int, str, str]]) -> dict[str, Any]:
    """Probe one file with an injectable ffprobe runner (see module docstring)."""

    path = Path(path)
    if not path.is_file():
        return {"metadata": None, "probe_error": f"no such file: {path.name}"}
    try:
        code, stdout, stderr = runner(path)
    except FileNotFoundError as error:
        return {"metadata": None, "probe_error": f"ffprobe not available: {error}"}
    except subprocess.TimeoutExpired:
        return {"metadata": None, "probe_error": f"ffprobe timed out after {_PROBE_TIMEOUT_SECONDS}s"}
    if code != 0:
        detail = (stderr or stdout or "unknown ffprobe failure").strip()
        # Never echo server-side temp paths into API-visible error strings
        # (V3.0 sec. 11); report the file's basename only.
        try:
            detail = detail.replace(str(path), path.name)
        except ValueError:
            pass
        return {"metadata": None, "probe_error": f"ffprobe failed (exit {code}): {detail}"[:512]}
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return {"metadata": None, "probe_error": "ffprobe produced non-JSON output"}
    if not isinstance(payload, dict):
        return {"metadata": None, "probe_error": "ffprobe output was not an object"}
    streams = payload.get("streams")
    video = next((entry for entry in (streams or [])
                  if isinstance(entry, dict) and entry.get("codec_type") == "video"), None)
    if video is None:
        return {"metadata": None, "probe_error": "no video stream found (not a media file?)"}
    fmt = payload.get("format") or {}
    # Optional fields stay None when ffprobe omits them -- never fabricated.
    metadata = {
        "container": (fmt.get("format_name") if isinstance(fmt.get("format_name"), str) else None),
        "video_codec": video.get("codec_name") if isinstance(video.get("codec_name"), str) else None,
        "width": _positive_int(video.get("width")),
        "height": _positive_int(video.get("height")),
        "fps": _parse_fps(video.get("r_frame_rate")),
        "frames": _positive_int(video.get("nb_frames")),
        "duration_seconds": _positive_float(fmt.get("duration") or video.get("duration")),
        "audio_codec": (next((entry.get("codec_name") for entry in (streams or [])
                              if isinstance(entry, dict) and entry.get("codec_type") == "audio"), None)),
    }
    if metadata["width"] is None or metadata["height"] is None:
        return {"metadata": None, "probe_error": "video stream lacks usable dimensions"}
    return {"metadata": metadata, "probe_error": None}


def probe_media(path: Path) -> dict[str, Any]:
    """Probe one uploaded file with the real ffprobe (read-only)."""

    return probe_media_with_runner(Path(path), _default_runner)
