"""Display captions independently of recognition labels and model weights."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ._io import read_json
from .contracts import RecognitionResult


@dataclass(frozen=True)
class CaptionEntry:
    display_text: Mapping[str, str]
    review_status: str


class CaptionCatalog:
    """A reloadable display dictionary; it never restricts predicted labels."""

    def __init__(self, labels: Mapping[str, CaptionEntry] | None = None, *,
                 sign_language: str | None = None, catalog_id: str = "empty") -> None:
        self.labels = MappingProxyType(dict(labels or {}))
        self.sign_language = sign_language
        self.catalog_id = catalog_id

    @classmethod
    def load(cls, path: str | Path) -> CaptionCatalog:
        data = read_json(path)
        required = {"schema_version", "catalog_id", "sign_language", "labels"}
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError("Caption catalog fields do not match schema v1")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("Unsupported caption catalog schema")
        for name in ("catalog_id", "sign_language"):
            if not isinstance(data[name], str) or not data[name].strip():
                raise ValueError(f"Caption catalog requires a nonempty {name}")
        if not isinstance(data["labels"], list):
            raise ValueError("Caption catalog labels must be a list")
        labels = {}
        for entry in data["labels"]:
            if not isinstance(entry, dict) or set(entry) != {"label_id", "display_text", "review_status"}:
                raise ValueError("Invalid caption catalog entry fields")
            label = entry["label_id"]
            if not isinstance(label, str) or not label.strip() or label in labels:
                raise ValueError("Caption catalog label must be unique and nonempty")
            text = entry["display_text"]
            if not isinstance(text, dict) or not text or any(
                not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip()
                for key, value in text.items()
            ):
                raise ValueError("Caption display texts must be nonempty language/text pairs")
            if entry["review_status"] not in {"draft", "reviewed"}:
                raise ValueError("Caption review_status must be draft or reviewed")
            labels[label] = CaptionEntry(MappingProxyType(dict(text)), entry["review_status"])
        return cls(labels, sign_language=data["sign_language"], catalog_id=data["catalog_id"])


@dataclass(frozen=True)
class Caption:
    status: str
    zh: str
    pt: str
    notice: str = ""
    label_id: str | None = None
    score: float | None = None
    request_id: int = 0
    epoch: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


_REJECTIONS = {
    "ctc_all_blank": "未识别出手语词序列，请检查画面并重试",
    "insufficient_hand_frames": "手部未完整入镜，请重新录制",
    "insufficient_anchor_frames": "肩部参照不足，请让上身完整入镜",
    "at_least_two_frames_required": "片段太短，请重新录制",
    "insufficient_observed_coordinates": "有效关键点不足，请重新录制",
    "insufficient_coordinates_after_resampling": "动作关键点缺失过多，请重新录制",
    "below_validation_selected_score_threshold": "识别分数不足，请重试",
}


def caption_from_result(result: RecognitionResult, catalog: CaptionCatalog | None = None, *,
                        request_id: int = 0, epoch: int = 0) -> Caption:
    if result.status == "rejected":
        reason = result.reason or "unknown"
        return Caption("rejected", _REJECTIONS.get(reason, "未识别，请重试"), "", reason,
                       request_id=request_id, epoch=epoch)
    if result.status != "recognized" or result.prediction is None:
        raise ValueError("Caption requires a recognized prediction or rejected result")
    prediction = result.prediction
    if result.task_type == "text":
        language = prediction.language
        zh = prediction.text if language in {"zh", "zh-CN"} else "中文表达待审核"
        pt = prediction.text if language in {"pt", "pt-BR"} else "葡语表达待审核"
        if language not in {"zh", "zh-CN", "pt", "pt-BR"}:
            zh = prediction.text
        notice = f"模型原始文本（{language}）；未执行跨语言翻译"
        if result.quality.get("continuous_candidate"):
            state = result.quality.get("stability_state")
            if state == "confirmed":
                notice += "；多个连续窗口结果一致，但尚未人工确认"
            elif state == "temporary":
                notice += "；临时候选，可能随新窗口修正"
        return Caption("recognized", zh, pt, notice,
                       score=prediction.score, request_id=request_id, epoch=epoch)
    if result.task_type == "gloss_sequence":
        return Caption("recognized", " / ".join(prediction.glosses), "葡语表达待审核",
                       "中文手语词序列；尚未转换为自然语言句子",
                       request_id=request_id, epoch=epoch)
    text = dict(prediction.display_text)
    catalog = catalog or CaptionCatalog()
    entry = catalog.labels.get(prediction.label_id) if catalog.sign_language == result.sign_language else None
    if entry is not None:
        text.update(entry.display_text)
    zh = text.get("zh-CN") or text.get("zh") or "中文释义待补充"
    pt = text.get("pt-BR") or text.get("pt") or prediction.label_id
    notes = []
    if entry is not None and entry.review_status == "draft" and ("zh-CN" in entry.display_text or "zh" in entry.display_text):
        notes.append("中文释义草案，待核验")
    if zh == "中文释义待补充":
        notes.append("缺少中文释义")
    if result.score_type == "uncalibrated_class_probability":
        notes.append("模型分数未经校准")
    return Caption("recognized", zh, pt, "；".join(notes), prediction.label_id,
                   prediction.score, request_id, epoch)
