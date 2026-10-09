import json

import pytest

from slbridge.captions import CaptionCatalog, caption_from_result
from slbridge.contracts import ClassificationPrediction, RecognitionResult


def result(label="item:2", displays=None):
    prediction = ClassificationPrediction(label, 0.7, {label: 0.7}, displays or {"pt-BR": "Sim"})
    return RecognitionResult("recognized", "sample", "model", "libras", prediction, {})


def write_catalog(path, zh="是", review="draft", label="item:2"):
    data = {"schema_version": 1, "catalog_id": "test-v1", "sign_language": "libras", "labels": [
        {"label_id": label, "display_text": {"pt-BR": "Sim", "zh-CN": zh}, "review_status": review}
    ]}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def test_saved_dictionary_edit_changes_display_on_reload(tmp_path):
    path = tmp_path / "词典.json"
    write_catalog(path)
    first = caption_from_result(result(), CaptionCatalog.load(path), request_id=1, epoch=2)
    assert first.zh == "是" and first.pt == "Sim"
    assert "草案" in first.notice and "核验" in first.notice
    write_catalog(path, "是的", "reviewed")
    second = caption_from_result(result(), CaptionCatalog.load(path), request_id=2, epoch=2)
    assert second.zh == "是的" and "草案" not in second.notice
    assert (first.request_id, first.epoch) == (1, 2)
    assert second.to_dict()["label_id"] == "item:2"


def test_missing_translation_retains_model_label_and_portuguese():
    caption = caption_from_result(result(), CaptionCatalog(), request_id=1, epoch=0)
    assert caption.status == "recognized" and caption.label_id == "item:2"
    assert caption.pt == "Sim" and caption.zh == "中文释义待补充"
    assert "未经校准" in caption.notice


def test_missing_all_text_retains_stable_id():
    prediction = ClassificationPrediction("new:label", 0.8, {"new:label": 0.8}, {})
    item = RecognitionResult("recognized", "s", "m", None, prediction, {})
    caption = caption_from_result(item, CaptionCatalog())
    assert caption.pt == "new:label" and caption.label_id == "new:label"


def test_catalog_language_does_not_apply_to_other_sign_language(tmp_path):
    path = tmp_path / "catalog.json"
    write_catalog(path)
    other = RecognitionResult("recognized", "s", "m", "csl", result().prediction, {})
    caption = caption_from_result(other, CaptionCatalog.load(path))
    assert caption.zh == "中文释义待补充"


def test_rejection_has_no_previous_prediction_text():
    rejected = RecognitionResult("rejected", "s", "m", "libras", None, {}, "insufficient_hand_frames")
    caption = caption_from_result(rejected, CaptionCatalog())
    assert caption.status == "rejected" and caption.label_id is None and caption.score is None
    assert caption.pt == "" and "手" in caption.zh


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(schema_version=True),
    lambda d: d.update(unexpected=1),
    lambda d: d["labels"].append(d["labels"][0]),
    lambda d: d["labels"][0].update(review_status="automatic_verified"),
    lambda d: d["labels"][0]["display_text"].update({"zh-CN": ""}),
])
def test_invalid_catalog_rejected(tmp_path, mutation):
    path = tmp_path / "catalog.json"
    data = write_catalog(path)
    mutation(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        CaptionCatalog.load(path)


def test_bundled_catalog_covers_actual_twelve_labels():
    catalog = CaptionCatalog.load("configs/libras_caption_catalog.json")
    assert len(catalog.labels) == 12
    caption = caption_from_result(result("item:92", {"pt-BR": "Energia"}), catalog)
    assert caption.pt == "Energia" and "能源" in caption.zh
    assert "草案" in caption.notice
