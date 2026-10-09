"""Readable desktop text without changing recognition or subtitle contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

from .captions import Caption, CaptionCatalog
from .contracts import RecognitionResult

if TYPE_CHECKING:
    from .inference import Recognizer


@dataclass(frozen=True)
class VocabularyWord:
    label_id: str
    zh: str
    pt: str
    draft: bool


@dataclass(frozen=True)
class ModelPresentation:
    title: str
    summary: str
    words: tuple[VocabularyWord, ...]
    details: dict[str, str]


@dataclass(frozen=True)
class ResultPresentation:
    title: str
    zh: str
    pt: str
    notice: str
    details: dict[str, str]


_LANGUAGES = {"libras": "巴西手语", "csl": "中国手语"}
_BACKENDS = {"random_forest_v1": "随机森林"}
_CAPABILITIES = {"clip": "单个动作识别"}
_MISSING_ZH = "中文释义待补充"
_MISSING_PT = "葡语释义待补充"


def _display_text(text: Mapping[str, str], label: str) -> tuple[str, str]:
    zh = text.get("zh-CN") or text.get("zh")
    pt = text.get("pt-BR") or text.get("pt")
    return (
        zh if zh and zh != label else _MISSING_ZH,
        pt if pt and pt != label else _MISSING_PT,
    )


def describe_model(recognizer: Recognizer | None,
                   catalog: CaptionCatalog | None = None) -> ModelPresentation:
    """Show model-supported words, overlaying only a matching-language catalog."""
    bundle = getattr(recognizer, "bundle", None)
    if bundle is None:
        return ModelPresentation("尚未加载识别模型", "选择模型后即可开始识别", (), {})

    manifest = bundle.manifest
    language = _LANGUAGES.get(manifest.sign_language, "手语类型未指定")
    capabilities = tuple(manifest.capabilities)
    capability_text = "、".join(_CAPABILITIES.get(value, value) for value in capabilities)
    words = []
    catalog_matches = catalog is not None and catalog.sign_language == manifest.sign_language
    for label, model_text in bundle.labels.items():
        text = dict(model_text)
        entry = catalog.labels.get(label) if catalog_matches else None
        draft = False
        if entry is not None:
            text.update(entry.display_text)
            draft = entry.review_status == "draft" and bool(
                entry.display_text.get("zh-CN") or entry.display_text.get("zh")
            )
        zh, pt = _display_text(text, label)
        words.append(VocabularyWord(label, zh, pt, draft))

    summary = f"{language} · {len(words)} 个词"
    if capability_text:
        summary += f" · {capability_text}"
    details = {
        "模型版本": manifest.run_id,
        "模型算法": _BACKENDS.get(manifest.backend, manifest.backend),
        "输入特征": manifest.input_spec.representation_id,
        "识别能力": capability_text,
    }
    if manifest.sign_language is not None:
        details["手语类型"] = manifest.sign_language
    if catalog_matches:
        details["字幕词典版本"] = catalog.catalog_id
    return ModelPresentation(f"{language}识别模型", summary, tuple(words), details)


def describe_result(result: RecognitionResult, caption: Caption) -> ResultPresentation:
    """Keep raw identifiers and uncalibrated model scores out of primary text."""
    details = {"模型版本": result.model_id, "来源": result.sample_id}
    if result.sign_language is not None:
        details["手语类型"] = result.sign_language
    if result.status == "rejected":
        if result.reason:
            details["未识别原因"] = result.reason
        return ResultPresentation("暂未识别出动作", caption.zh, "", "", details)
    if result.status != "recognized" or result.prediction is None:
        raise ValueError("A presentation requires a recognized or rejected result")

    prediction = result.prediction
    zh = caption.zh if caption.zh and caption.zh != prediction.label_id else _MISSING_ZH
    pt = caption.pt if caption.pt and caption.pt != prediction.label_id else _MISSING_PT
    details["标签编号"] = prediction.label_id
    details["模型匹配分数"] = f"{prediction.score:.3f}"
    if result.score_type == "uncalibrated_class_probability":
        details["分数说明"] = "未经校准；分数用于模型排序，不代表识别正确率"
    else:
        details["分数类型"] = result.score_type
    if result.time_range is not None:
        start, end = result.time_range
        details["片段时间范围"] = f"{start:.3f}–{end:.3f} 秒"

    notices = []
    for note in caption.notice.split("；"):
        note = note.strip()
        if not note or note == "模型分数未经校准":
            continue
        if note == "中文释义草案，待核验":
            note = "中文释义待核验"
        elif note == "缺少中文释义":
            note = _MISSING_ZH
        if note not in notices:
            notices.append(note)
    return ResultPresentation("识别完成", zh, pt, "；".join(notices), details)
