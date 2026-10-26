import time

import pytest
from PySide6.QtWidgets import QApplication

from slbridge.continuous import ContinuousConfig
from slbridge.desktop import SignLanguageApp
from slbridge.session import RecognitionSession
from test_automatic_session import ReplaySource
from test_continuous_session import MotionObserver, WindowModel


def pump(predicate, seconds=4):
    until = time.perf_counter()+seconds
    while time.perf_counter() < until:
        QApplication.instance().processEvents()
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError('Continuous desktop did not reach expected state')


@pytest.fixture(scope='module')
def application():
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


@pytest.fixture
def gui(application):
    MotionObserver.visible = True
    session = RecognitionSession(WindowModel(), observer_factory=MotionObserver)
    source = ReplaySource(count=300)
    window = SignLanguageApp(session=session, continuous_config=ContinuousConfig(
        minimum_seconds=.1, window_seconds=.35, update_interval_seconds=.1, max_result_age_seconds=.3))
    window.show()
    session.open_source(source, kind='camera')
    pump(lambda: session.source_active and window.continuous_check.isEnabled())
    yield window, source
    source.gate.set()
    session.recognizer.release.set()
    window.close()
    application.processEvents()
    pump(lambda: not session._worker.is_alive() and not session._capture_thread.is_alive()
         and not session._observer_worker.is_alive())
    MotionObserver.visible = True


def test_main_screen_offers_continuous_mode_without_pause(application):
    window = SignLanguageApp()
    try:
        window.show()
        application.processEvents()
        assert hasattr(window, 'continuous_check'), 'Continuous stream control is missing'
        assert '无需停顿' in window.continuous_check.text()
    finally:
        window.close()


def test_live_motion_updates_are_labelled_and_not_saved_as_caption_history(gui):
    window, source = gui
    window.continuous_check.click()
    source.gate.set()
    pump(lambda: window._last_result is not None)
    assert window.session.continuous and not window.session.automatic
    assert '当前窗口' in window.result_title.text()
    assert window.pt_label.text() == 'Olá'
    assert window.caption_history.count() == 0
    assert window.continuous_check.isEnabled()
    assert not window.record_button.isEnabled()
    assert '停顿' not in window.input_hint.text() or '无需停顿' in window.input_hint.text()
    window.continuous_check.click()
    pump(lambda: window._last_result is None and not window.session.busy)
    assert not window.session.continuous


def test_no_hands_clears_window_result_but_keeps_preview(gui):
    window, source = gui
    window.continuous_check.click()
    source.gate.set()
    pump(lambda: window._last_result is not None)
    MotionObserver.visible = False
    pump(lambda: window._last_result is None)
    assert window.preview.image is not None
    assert window.session.continuous
    assert not window.pt_label.text()


def test_stopped_video_does_not_leave_an_old_window_result_on_screen(gui):
    window, source = gui
    window.continuous_check.click()
    source.gate.set()
    pump(lambda: window._last_result is not None)
    source.count = source.index
    pump(lambda: source.closed)
    pump(lambda: window._last_result is None)
    assert any(word in window.status_label.text() for word in ('过期', '等待', '结束'))


def test_language_controls_wait_until_continuous_mode_is_stopped(gui):
    window, source = gui
    window.continuous_check.click()
    window._update_controls()
    assert not window.session.busy
    assert not window.language_combo.isEnabled()
    assert not window.model_button.isEnabled()


def test_continuous_startup_opens_video_after_model_loading(application, monkeypatch):
    calls = []
    monkeypatch.setattr(SignLanguageApp, 'open_video', lambda self, path: calls.append(('video', path)))
    monkeypatch.setattr(SignLanguageApp, 'load_model', lambda self, path: calls.append(('model', path)))
    window = SignLanguageApp(model='model', video='short-video.mp4', continuous=True)
    try:
        assert calls == [('model', 'model')]
        window.session.recognizer = WindowModel()
        window._poll()
        assert calls == [('model', 'model'), ('video', 'short-video.mp4')]
        window._poll()
        assert len(calls) == 2
    finally:
        window.close()


def test_catalog_rerender_preserves_continuous_expiry(gui, tmp_path):
    window, source = gui
    catalog = tmp_path/'catalog.json'
    catalog.write_text('{"schema_version":1,"catalog_id":"test","sign_language":"libras","labels":[]}', encoding='utf-8')
    window.continuous_check.click()
    source.gate.set()
    pump(lambda: window._continuous_update is not None)
    source.gate.clear()
    time.sleep(.05)
    window._poll()
    update = window._continuous_update
    window.load_catalog(catalog)
    assert not window._error_detail
    assert window._continuous_update is update
    assert '当前窗口' in window.result_title.text()
    pump(lambda: window._last_result is None)
