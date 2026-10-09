from types import SimpleNamespace
from slbridge.captions import caption_from_result
from slbridge.cecsl import RGBInputSpec
from slbridge.contracts import GlossSequencePrediction, GlossToken, RecognitionResult
from slbridge.desktop_presenter import describe_model, describe_result


def test_sequence_model_without_classification_labels_has_readable_vocabulary():
    manifest = SimpleNamespace(sign_language="csl", task_type="gloss_sequence", capabilities=("clip",),
                               backend="onnx_ctc", run_id="csl-tfnet", input_spec=RGBInputSpec())
    model = SimpleNamespace(bundle=SimpleNamespace(manifest=manifest, vocabulary=(" ", "我", "吃")))
    presentation = describe_model(model)
    assert presentation.title == "中国手语识别模型"
    assert "完整表达词序列" in presentation.summary
    assert [word.zh for word in presentation.words] == ["我", "吃"]


def test_sequence_caption_preserves_repeated_words_without_claiming_translation():
    result = RecognitionResult("recognized", "clip", "csl-tfnet", "csl",
        GlossSequencePrediction((GlossToken(1, "我", None), GlossToken(1, "我", None)), "prefix_beam"),
        {}, task_type="gloss_sequence", score_type="ctc_token_probability")
    caption = caption_from_result(result)
    presentation = describe_result(result, caption)
    assert presentation.zh == "我 / 我" and presentation.pt == "葡语表达待审核"
    assert "尚未转换为自然语言句子" in presentation.notice
