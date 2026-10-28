"""Explicit desktop-to-browser draft handoff; never a playback authorization."""
from __future__ import annotations

import json
from urllib.parse import quote, urlsplit

from ..contracts import GlossSequencePrediction, TextPrediction


def handoff_url(result, *, request_id=None, source_epoch=None, model_generation=None,
                base_url="http://127.0.0.1:8874/animation/"):
    parts = urlsplit(base_url)
    if (parts.scheme != "http" or parts.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parts.username or parts.password or parts.query or parts.fragment
            or parts.path != "/animation/"):
        raise ValueError("动画工作台地址必须为本机 HTTP /animation/ 页面")
    if result is None:
        return base_url
    if result.status != "recognized" or str(result.sign_language).lower() != "csl":
        raise ValueError("当前动画库仅接收中国手语的已返回文本或 Gloss 候选")
    if isinstance(result.prediction, TextPrediction):
        text = result.prediction.text
    elif isinstance(result.prediction, GlossSequencePrediction):
        text = result.prediction.text
    else:
        raise ValueError("该识别结果没有可核对的文本或 Gloss")
    if not text.strip() or len(text) > 500:
        raise ValueError("动画输入需要 1–500 个字符")
    payload = {
        "text": text, "confirmed": False, "autoplay": False,
        "origin": {"kind": "desktop_handoff", "raw_text": text,
                   "request_id": request_id or result.sample_id, "model_id": result.model_id,
                   "source_epoch": source_epoch, "model_generation": model_generation,
                   "task_type": result.task_type},
    }
    return base_url + "#" + quote(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), safe="")
