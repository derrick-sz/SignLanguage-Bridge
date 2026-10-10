import time

import pytest
from PySide6.QtCore import QPoint, QRect
from PySide6.QtWidgets import QApplication

from slbridge.contracts import RecognitionResult, TextPrediction
from slbridge.desktop import SignLanguageApp
from slbridge.model_adapters import ModelCapabilities, ModelInputContract
from slbridge.session import RecognitionSession
from test_automatic_session import ReplaySource, Observer, ClipModel, CONFIG


def pump(predicate, seconds=4):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        QApplication.instance().processEvents()
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("Mounted automatic control did not reach expected behavior")


@pytest.mark.parametrize("size", [(980, 720), (1280, 820)])
def test_mounted_automatic_history_undo_clear_and_layout(size):
    application = QApplication.instance() or QApplication([])
    session = RecognitionSession(ClipModel(), observer_factory=Observer)
    window = SignLanguageApp(session=session, segment_config=CONFIG)
    source = ReplaySource()
    try:
        window.resize(*size)
        window.show()
        session.open_source(source, kind="camera")
        pump(lambda: source.opened if hasattr(source, "opened") else session.source_active and window.auto_check.isEnabled())
        window.auto_check.click()
        assert session.automatic
        assert not window.record_button.isEnabled()
        source.gate.set()
        pump(lambda: window.caption_history.count() == 2)
        for widget in (window.zh_label, window.pt_label, window.caption_history,
                       window.undo_caption_button, window.clear_history_button, window.auto_check):
            assert widget.isVisible()
            assert window.rect().contains(QRect(widget.mapTo(window, QPoint(0, 0)), widget.size()))
        # Panels cannot cover subtitle rows, including the taller draft notice.
        history_top = window.output_tabs.mapTo(window, QPoint(0, 0)).y()
        pt_bottom = window.pt_label.mapTo(window, QPoint(0, window.pt_label.height())).y()
        assert pt_bottom < history_top
        window.undo_caption_button.click()
        assert window.caption_history.count() == 1
        window.clear_history_button.click()
        assert window.caption_history.count() == 0
        window._poll()
        assert window._last_result is None
    finally:
        source.gate.set()
        window.close()
        application.processEvents()


class BundleFreeTextAdapter:
    capabilities = ModelCapabilities("custom-text", "v1", "libras", "text",
        (ModelInputContract("mediapipe225_rgb_v1", (None, 225), "float32", "demo-v1",
                            ("file", "captured_clip")),), (), "test-fixture", ("cpu",), score_type="none")
    runtime_info = {"device": "cpu", "providers": ["test-fixture"]}

    @property
    def recognizer(self):
        return self

    def warmup(self):
        pass

    def close(self):
        pass

    def infer(self, source, request, *, input_kind):
        return RecognitionResult("recognized", request.request_id, "custom-text", "libras",
                                 TextPrediction("Olá", "pt-BR"), {}, task_type="text", score_type="none"), {}


def test_new_bundle_free_adapter_loads_and_manual_path_uses_uniform_service():
    application = QApplication.instance() or QApplication([])
    window = SignLanguageApp()
    source = ReplaySource()
    try:
        window.show()
        window.session.load_model("injected", loader=lambda _: BundleFreeTextAdapter())
        pump(lambda: not window.session.busy and window.session.recognizer is not None)
        window._poll()
        assert window.model_presentation.details["模型版本"] == "custom-text"
        window.session.open_source(source, kind="video", video_path="injected.mp4")
        pump(lambda: window.recognize_button.isEnabled())
        window.recognize_button.click()
        pump(lambda: window.caption is not None)
        assert window.pt_label.text() == "Olá"
        assert window._last_result.task_type == "text"
    finally:
        source.gate.set()
        window.close()
        application.processEvents()
