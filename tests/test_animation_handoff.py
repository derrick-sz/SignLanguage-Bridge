import json
from dataclasses import replace
from urllib.parse import unquote, urlsplit

import pytest

from slbridge.contracts import RecognitionResult, TextPrediction, GlossSequencePrediction, GlossToken
from slbridge.animation.handoff import handoff_url


def result(text="认识你很高兴", language="csl"):
    return RecognitionResult("recognized", "sample", "test-model", language,
                             TextPrediction(text, "zh"), {}, task_type="text")


def test_handoff_retains_original_and_lifecycle_without_autoplay():
    raw = result()
    url = handoff_url(raw, request_id="request-9", source_epoch=4, model_generation=2)
    parts = urlsplit(url)
    assert parts.hostname == "127.0.0.1"
    assert not parts.query
    value = json.loads(unquote(parts.fragment))
    assert value["text"] == raw.prediction.text
    assert value["origin"]["raw_text"] == raw.prediction.text
    assert value["origin"]["request_id"] == "request-9"
    assert value["origin"]["source_epoch"] == 4
    assert value["origin"]["model_generation"] == 2
    assert value["confirmed"] is False
    assert value["autoplay"] is False
    assert raw.prediction.text == "认识你很高兴"


def test_gloss_is_identified_and_other_languages_or_remote_urls_are_rejected():
    raw = result()
    raw = replace(raw, prediction=GlossSequencePrediction((GlossToken(1, "你好", None),), "greedy"),
                  task_type="gloss_sequence")
    data = json.loads(unquote(urlsplit(handoff_url(raw)).fragment))
    assert data["origin"]["task_type"] == "gloss_sequence"
    with pytest.raises(ValueError):
        handoff_url(result(language="libras"))
    with pytest.raises(ValueError):
        handoff_url(result(), base_url="https://example.com/animation/")
    with pytest.raises(ValueError):
        handoff_url(result(), base_url="http://user:secret@localhost:8874/animation/")
    with pytest.raises(ValueError):
        handoff_url(result("x" * 501))


def test_empty_handoff_is_only_the_workspace_entry():
    assert handoff_url(None) == "http://127.0.0.1:8874/animation/"


def test_desktop_has_explicit_animation_action(monkeypatch):
    from PySide6.QtWidgets import QApplication
    from slbridge.desktop import SignLanguageApp
    from PySide6.QtGui import QDesktopServices
    application = QApplication.instance() or QApplication([])
    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url.toString()) or True)
    window = SignLanguageApp()
    try:
        window._last_result = result()
        window._animation_origin = {"request_id": "r-desktop", "source_epoch": 3}
        assert not opened
        window.animation_button.click()
        application.processEvents()
        assert len(opened) == 1
        payload = json.loads(unquote(urlsplit(opened[0]).fragment))
        assert payload["origin"]["request_id"] == "r-desktop"
        assert payload["autoplay"] is False
    finally:
        window.on_close()
