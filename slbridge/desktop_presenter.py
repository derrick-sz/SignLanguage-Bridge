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
_BACKENDS = {"random_forest_v1": "随机森林", "onnx_classifier_v1": "ONNX 动作分类模型",
             "onnx_ctc": "ONNX 连续词序列模型", "uni_sign_pose_slt": "Uni-Sign Pose-only（实验）"}
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
    if getattr(manifest, "task_type", "classification") == "gloss_sequence":
        capability_text = "完整表达词序列识别"
    words = []
    catalog_matches = catalog is not None and catalog.sign_language == manifest.sign_language
    if getattr(manifest, "task_type", "classification") == "gloss_sequence":
        vocabulary = getattr(bundle, "vocabulary", ())
        for index, gloss in enumerate(vocabulary):
            if index != getattr(bundle, "blank_id", 0):
                words.append(VocabularyWord(str(index), gloss, _MISSING_PT, False))
    for label, model_text in getattr(bundle, "labels", {}).items():
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

    unit = "词表项" if getattr(manifest, "task_type", "classification") == "gloss_sequence" else "词"
    summary = f"{language} · {len(words)} 个{unit}"
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
    runtime = getattr(recognizer, "runtime_info", {})
    if runtime.get("providers"):
        details["执行引擎"] = "、".join(runtime["providers"])
    if runtime.get("fallback_reason"):
        details["设备说明"] = runtime["fallback_reason"]
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
        title = "暂未识别出词序列" if result.task_type == "gloss_sequence" else "暂未识别出动作"
        return ResultPresentation(title, caption.zh, "", "", details)
    if result.status != "recognized" or result.prediction is None:
        raise ValueError("A presentation requires a recognized or rejected result")

    prediction = result.prediction
    if result.task_type == "text":
        details["识别任务"] = "中文句子翻译"
        details["文本语言"] = prediction.language
        details["分数类型"] = result.score_type
        return ResultPresentation("中文句子候选完成", caption.zh, caption.pt, caption.notice, details)
    if result.task_type == "gloss_sequence":
        details["识别任务"] = "中文手语词序列"
        details["解码方式"] = prediction.decoder
        details["词数"] = str(len(prediction.tokens))
        details["分数说明"] = "CTC 词概率未经校准；词时间边界尚未标定"
        return ResultPresentation("词序列识别完成", caption.zh, caption.pt, caption.notice, details)
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


def describe_runtime_model(info: dict | None, capabilities: dict | None,
                           catalog: CaptionCatalog) -> ModelPresentation:
    """Present the service contract, including adapters without legacy bundles."""
    if info is None or capabilities is None:
        return describe_model(None, catalog)
    sign_language = info["sign_language"]
    language = _LANGUAGES.get(sign_language, sign_language or "手语类型未指定")
    task = capabilities["task_type"]
    words = []
    if task == "gloss_sequence":
        words = [VocabularyWord(str(i), token, _MISSING_PT, False)
                 for i, token in enumerate(capabilities["vocabulary"]) if i != 0]
    elif task == "classification":
        for label in capabilities["vocabulary"]:
            text = dict(info["labels"].get(label, {}))
            entry = catalog.labels.get(label) if catalog.sign_language == sign_language else None
            if entry:
                text.update(entry.display_text)
            zh, pt = _display_text(text, label)
            words.append(VocabularyWord(label, zh, pt, bool(entry and entry.review_status == "draft")))
    task_text = {"classification": "完整动作分类", "gloss_sequence": "完整表达词序列识别", "text": "中文句子翻译"}[task]
    details = {"模型版本": info["model_id"], "模型generation": str(info["generation"]),
               "识别任务": task, "输入特征": "、".join(c["representation_id"] for c in capabilities["inputs"]),
               "识别能力": "、".join(capabilities["modes"]), "分数类型": capabilities["score_type"]}
    runtime = info["runtime"]
    if runtime.get("providers"):
        details["执行引擎"] = "、".join(runtime["providers"])
    if runtime.get("fallback_reason"):
        details["设备说明"] = runtime["fallback_reason"]
    return ModelPresentation(f"{language}识别模型", f"{language} · {len(words)} 个词 · {task_text}",
                             tuple(words), details)
