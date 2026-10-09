from types import SimpleNamespace

import pytest

from slbridge.captions import Caption, CaptionCatalog, CaptionEntry, caption_from_result
from slbridge.contracts import ClassificationPrediction, RecognitionResult
from slbridge.desktop_presenter import describe_model, describe_result


def model(language="libras", labels=None, capabilities=("clip",)):
    manifest = SimpleNamespace(
        sign_language=language, run_id="libras-rf-pilot-20261009",
        backend="random_forest_v1", capabilities=capabilities,
        input_spec=SimpleNamespace(representation_id="mediapipe225_rgb_v1"),
    )
    labels = labels if labels is not None else {
        "item:105": {"pt-BR": "Árvore"}, "item:94": {"pt-BR": "Serviço"},
    }
    return SimpleNamespace(bundle=SimpleNamespace(manifest=manifest, labels=labels))


def catalog(language="libras", review="draft"):
    return CaptionCatalog({
        "item:105": CaptionEntry({"zh-CN": "树", "pt-BR": "Árvore"}, review),
        "item:999": CaptionEntry({"zh-CN": "未训练的词", "pt-BR": "Unused"}, review),
    }, sign_language=language, catalog_id="display-v1")


def recognized(displays=None, label="item:105"):
    prediction = ClassificationPrediction(label, 0.89, {label: 0.89}, displays or {})
    return RecognitionResult("recognized", "C:/video/IMG_2971.mp4", "model-v1",
                             "libras", prediction, {}, time_range=(0.0, 3.5))


def test_model_words_only_include_trained_labels_and_matching_catalog():
    presentation = describe_model(model(), catalog())
    assert presentation.summary == "巴西手语 · 2 个词 · 单个动作识别"
    assert [word.label_id for word in presentation.words] == ["item:105", "item:94"]
    assert (presentation.words[0].zh, presentation.words[0].pt) == ("树", "Árvore")
    assert presentation.words[0].draft
    assert presentation.words[1].zh == "中文释义待补充"
    assert not presentation.words[1].draft
    assert presentation.details["模型版本"] == "libras-rf-pilot-20261009"
    assert presentation.details["模型算法"] == "随机森林"
    assert presentation.details["输入特征"] == "mediapipe225_rgb_v1"
    assert presentation.details["字幕词典版本"] == "display-v1"
    assert "item:" not in presentation.summary


def test_other_language_catalog_does_not_supply_translation_or_details():
    presentation = describe_model(model(), catalog("csl"))
    assert presentation.words[0].zh == "中文释义待补充"
    assert not presentation.words[0].draft
    assert "字幕词典版本" not in presentation.details


def test_unloaded_model_does_not_invent_words_or_metadata():
    presentation = describe_model(None, catalog())
    assert presentation.words == () and presentation.details == {}
    assert "尚未加载" in presentation.title
    assert "巴西" not in presentation.summary


@pytest.mark.parametrize("displays", [{}, {"zh-CN": "item:105", "pt-BR": "item:105"}])
def test_model_missing_text_does_not_leak_label_id(displays):
    word = describe_model(model(labels={"item:105": displays})).words[0]
    assert word.zh == "中文释义待补充" and word.pt == "葡语释义待补充"


def test_reviewed_dictionary_word_is_not_marked_draft():
    assert not describe_model(model(), catalog(review="reviewed")).words[0].draft


def test_portuguese_only_draft_catalog_does_not_mark_model_chinese_as_draft():
    item = CaptionCatalog({"item:105": CaptionEntry({"pt-BR": "Árvore"}, "draft")},
                          sign_language="libras")
    presentation = describe_model(model(labels={"item:105": {"zh-CN": "树"}}), item)
    assert presentation.words[0].zh == "树" and not presentation.words[0].draft


def test_unknown_language_and_backend_are_not_misrepresented():
    recognizer = model(language="future_language", capabilities=())
    recognizer.bundle.manifest.backend = "future_backend"
    presentation = describe_model(recognizer)
    assert "巴西手语" not in presentation.summary
    assert "单个动作识别" not in presentation.summary
    assert presentation.details["手语类型"] == "future_language"
    assert presentation.details["模型算法"] == "future_backend"


def test_result_keeps_draft_notice_and_score_in_details_only():
    result = recognized({"pt-BR": "Árvore"})
    presentation = describe_result(result, caption_from_result(result, catalog()))
    assert presentation.title == "识别完成"
    assert (presentation.zh, presentation.pt) == ("树", "Árvore")
    assert presentation.notice == "中文释义待核验"
    assert presentation.details["模型匹配分数"] == "0.890"
    assert "未经校准" in presentation.details["分数说明"]
    assert "不代表识别正确率" in presentation.details["分数说明"]
    primary = " ".join((presentation.title, presentation.zh, presentation.pt, presentation.notice))
    assert "置信度" not in primary and "0.890" not in primary and "item:105" not in primary
    assert presentation.details["标签编号"] == "item:105"
    assert presentation.details["来源"] == result.sample_id
    assert presentation.details["片段时间范围"] == "0.000–3.500 秒"


def test_result_missing_subtitles_do_not_display_raw_label_fallback():
    result = recognized()
    caption = caption_from_result(result)
    assert caption.pt == "item:105"  # Core contract still retains its stable identifier.
    presentation = describe_result(result, caption)
    assert presentation.zh == "中文释义待补充" and presentation.pt == "葡语释义待补充"
    assert presentation.details["标签编号"] == "item:105"


def test_result_uses_generic_language_display_fallbacks():
    result = recognized({"zh": "树", "pt": "Árvore"})
    presentation = describe_result(result, caption_from_result(result))
    assert (presentation.zh, presentation.pt) == ("树", "Árvore")


def test_rejected_result_preserves_correction_and_omits_stale_caption_score():
    result = RecognitionResult("rejected", "camera:0", "model-v1", "libras", None,
                               {}, "insufficient_hand_frames")
    caption = caption_from_result(result)
    presentation = describe_result(result, caption)
    assert "未识别" in presentation.title
    assert presentation.zh == caption.zh and "手" in presentation.zh
    assert presentation.pt == "" and presentation.notice == ""
    assert presentation.details["未识别原因"] == "insufficient_hand_frames"
    assert "模型匹配分数" not in presentation.details and "标签编号" not in presentation.details


def test_unrecognized_result_shape_is_rejected():
    result = RecognitionResult("processing", "s", "m", None, None, {})
    with pytest.raises(ValueError, match="recognized or rejected"):
        describe_result(result, Caption("processing", "", ""))
