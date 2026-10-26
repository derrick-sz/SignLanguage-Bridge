"""Main-screen switching uses the real session transaction and Qt event loop."""
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pytest
from PySide6.QtCore import QPoint, QRect, QTimer, Qt
from PySide6.QtWidgets import QApplication

from slbridge.contracts import ClassificationPrediction, RecognitionResult
from slbridge.desktop import SignLanguageApp
from slbridge.session import RecognitionSession
from test_session import FakeSource


def pump(predicate, seconds=4):
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        QApplication.instance().processEvents()
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError('Language switch did not reach the expected state')


class Model:
    def __init__(self, path, language):
        self.bundle = SimpleNamespace(path=path, manifest=SimpleNamespace(
            run_id=path.name, sign_language=language, backend='fixture',
            input_spec=SimpleNamespace(representation_id='mediapipe225_rgb_v1'),
            capabilities=('clip',)), labels={'word': {'pt-BR': 'Sim'}})
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.fail = False
        self.closed = 0

    def warmup(self):
        self.entered.set()
        assert self.release.wait(8)
        if self.fail:
            raise ValueError('damaged model')

    def close(self):
        self.closed += 1


@pytest.fixture(scope='module')
def application():
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


@pytest.fixture
def workspace(application, monkeypatch, tmp_path):
    models = {}
    for language in ('csl', 'libras'):
        path = tmp_path / language
        path.mkdir()
        (path / 'manifest.json').write_text('{}', encoding='utf-8')
        models[path] = Model(path, language)
    session = RecognitionSession(models[tmp_path / 'libras'])
    original = session.load_model
    monkeypatch.setattr(session, 'load_model', lambda path: original(path, loader=lambda p: models[Path(p)]))
    window = SignLanguageApp(session=session, language_models={k: tmp_path / k for k in ('csl', 'libras')})
    window.show()
    window._poll()
    yield window, models, tmp_path
    for model in models.values():
        model.release.set()
    window.close()
    application.processEvents()
    pump(lambda: not session._worker.is_alive())


def select(window, language):
    window.language_combo.setCurrentIndex(window.language_combo.findData(language))


def test_language_selector_visible_before_settings(application):
    window = SignLanguageApp()
    try:
        window.show()
        application.processEvents()
        assert hasattr(window, 'language_combo'), 'Main screen needs a language selector'
        assert window.language_combo.isVisible()
        assert window.language_combo.findData('csl') >= 0
        assert window.language_combo.findData('libras') >= 0
        assert not window.settings_dialog.isVisible()
    finally:
        window.close()


def test_switch_keeps_active_selection_camera_and_ui_responsive(workspace):
    window, models, root = workspace
    session = window.session
    target = models[root / 'csl']
    target.release.clear()
    source = FakeSource()
    session.open_source(source, kind='camera')
    pump(lambda: source.index >= 4 and window.preview.image is not None)
    select(window, 'csl')
    pump(target.entered.is_set)
    assert window.language_combo.currentData() == 'libras'
    assert not window.language_combo.isEnabled()
    assert '中国手语' in window.status_label.text()
    first = source.index
    ticks = []
    QTimer.singleShot(25, lambda: ticks.append(True))
    pump(lambda: ticks and source.index > first + 3)
    assert session.source is source and not source.closed
    target.release.set()
    pump(lambda: not session.busy and window.language_combo.currentData() == 'csl')
    assert session.recognizer is target
    assert window.language_combo.isEnabled()
    assert session.source is source and session.source_active
    assert window.caption is None and window.zh_label.text() == '等待识别'
    assert '中国手语' in window.language_label.text()


def test_failed_switch_preserves_model_selection_preview_and_retry(workspace, monkeypatch):
    window, models, root = workspace
    source = FakeSource()
    window.session.open_source(source, kind='camera')
    pump(lambda: source.index > 4 and window.preview.image is not None)
    resets = []
    monkeypatch.setattr(window, '_reset_preview', lambda *args: resets.append(args))
    models[root / 'csl'].fail = True
    select(window, 'csl')
    pump(lambda: not window.session.busy and bool(window._error_detail))
    assert window.session.recognizer is models[root / 'libras']
    assert window.language_combo.currentData() == 'libras'
    assert window.language_combo.isEnabled()
    assert window.session.source is source and not source.closed
    assert not resets, 'Loading failure must not replace the camera preview'
    assert '保留' in window.status_label.text()
    models[root / 'csl'].fail = False
    select(window, 'csl')
    pump(lambda: not window.session.busy and window.language_combo.currentData() == 'csl')
    assert not window._error_detail


def test_cancelled_load_never_selects_or_activates_candidate(workspace):
    window, models, root = workspace
    target = models[root / 'csl']
    target.release.clear()
    select(window, 'csl')
    pump(target.entered.is_set)
    window.cancel_button.click()
    assert not window.language_combo.isEnabled()
    target.release.set()
    pump(lambda: not window.session.busy and window.language_combo.isEnabled())
    assert window.language_combo.currentData() == 'libras'
    assert window.session.recognizer is models[root / 'libras']
    assert target.closed == 1
    assert '正在' not in window.state_badge.text()


@pytest.mark.parametrize('state', ['RECORDING', 'PROCESSING'])
def test_recording_and_busy_state_cannot_switch(workspace, state):
    window, models, root = workspace
    window.session.state = state
    window.session._busy = state == 'PROCESSING'
    window._update_controls()
    try:
        assert not window.language_combo.isEnabled()
        generation = window.session.model_manager.generation
        select(window, 'csl')  # A programmatic signal must obey the same guard.
        assert window.language_combo.currentData() == 'libras'
        assert not models[root / 'csl'].entered.is_set()
        assert window.session.model_manager.generation == generation
    finally:
        window.session._busy = False
        window.session.state = 'IDLE'


def test_settings_custom_models_and_rollback_sync_selector(workspace):
    window, models, root = workspace
    custom = root / 'custom-csl'
    custom.mkdir()
    (custom / 'manifest.json').write_text('{}', encoding='utf-8')
    models[custom] = Model(custom, 'csl')
    window.load_model(custom)
    pump(lambda: not window.session.busy and window.language_combo.currentData() == 'csl')
    window.rollback_model()
    window._poll()
    assert window.language_combo.currentData() == 'libras'
    select(window, 'csl')
    pump(lambda: not window.session.busy and window.language_combo.currentData() == 'csl')
    assert window.session.recognizer is models[custom]
    unknown = root / 'future-model'
    unknown.mkdir()
    (unknown / 'manifest.json').write_text('{}', encoding='utf-8')
    models[unknown] = Model(unknown, 'future_language')
    window.load_model(unknown)
    pump(lambda: not window.session.busy and window.session.recognizer is models[unknown]
         and window.language_combo.currentData() is None)
    assert window.language_combo.currentData() is None
    assert '其他' in window.language_combo.currentText()


def test_camera_failure_during_load_is_still_reported_as_source_failure(workspace, monkeypatch):
    window, models, root = workspace
    source = FakeSource()
    window.session.open_source(source, kind='camera')
    pump(lambda: source.index > 4 and window.preview.image is not None)
    target = models[root / 'csl']
    target.release.clear()
    resets = []
    monkeypatch.setattr(window, '_reset_preview', lambda *args: resets.append(args))
    select(window, 'csl')
    pump(target.entered.is_set)
    source.fail = True
    pump(lambda: bool(window._error_detail))
    assert resets, 'Camera failure must clear its stale preview even during model loading'
    assert '切换失败' not in window.result_title.text()
    target.release.set()
    pump(lambda: not window.session.busy and window.language_combo.isEnabled())
    assert window.session.recognizer is models[root / 'libras']


def test_missing_model_is_disabled_and_does_not_start_load(application, tmp_path):
    window = SignLanguageApp(language_models={'csl': tmp_path / 'absent'})
    try:
        window.show()
        application.processEvents()
        index = window.language_combo.findData('csl')
        assert not window.language_combo.model().item(index).isEnabled()
        assert '未配置' in window.language_combo.itemText(index)
        select(window, 'csl')
        assert not window.session.busy and window.session.recognizer is None
        assert window.language_combo.currentData() is None
    finally:
        window.close()


def test_switch_clears_current_caption_and_keeps_committed_history(workspace):
    from test_caption_events import reserve, response
    window, models, root = workspace
    ledger = window.session.captions
    ledger.bind(1, 2)
    reserve(ledger, 1)
    ledger.complete(response(ledger, 1))
    result = RecognitionResult('recognized', 'old', 'old', 'libras',
        ClassificationPrediction('word', .9, {'word': .9}, {'pt-BR': 'Sim'}), {})
    window._display_result(result, window.session.request_id, window.session.epoch)
    assert window.pt_label.text() == 'Sim'
    select(window, 'csl')
    pump(lambda: not window.session.busy and window.language_combo.currentData() == 'csl')
    assert len(ledger.snapshot()['events']) == 1
    assert window.caption_history.count() == 1
    assert window.caption is None and not window.pt_label.text()


def test_language_switch_stops_automatic_segmentation_without_reopening_source(application, tmp_path):
    from test_automatic_session import ClipModel, CONFIG, Observer
    path = tmp_path / 'csl'
    path.mkdir()
    (path / 'manifest.json').write_text('{}', encoding='utf-8')
    session = RecognitionSession(ClipModel(), observer_factory=Observer)
    window = SignLanguageApp(session=session, segment_config=CONFIG, language_models={'csl': path})
    source = FakeSource()
    original = session.load_model
    target = Model(path, 'csl')
    session.load_model = lambda selected: original(selected, loader=lambda _: target)
    try:
        window.show()
        session.open_source(source, kind='camera')
        pump(lambda: source.index > 4 and window.auto_check.isEnabled())
        window.auto_check.click()
        assert session.automatic
        select(window, 'csl')
        pump(lambda: not session.busy and window.language_combo.currentData() == 'csl')
        assert not session.automatic and not window.auto_check.isChecked()
        assert session.source is source and session.source_active and not source.closed
    finally:
        window.close()
        application.processEvents()


@pytest.mark.parametrize('size', [(980, 720), (1280, 820)])
def test_selector_and_input_actions_fit_minimum_window(workspace, size):
    window, _, _ = workspace
    window.resize(*size)
    window._update_controls()
    QApplication.instance().processEvents()
    assert window.width() == size[0]
    controls = [window.language_combo, window.video_button, window.camera_button]
    rects = [QRect(w.mapTo(window, QPoint(0, 0)), w.size()) for w in controls]
    for widget, rectangle in zip(controls, rects):
        assert widget.isVisible()
        assert window.rect().contains(rectangle)
    assert not any(a.intersects(b) for i, a in enumerate(rects) for b in rects[i+1:])
