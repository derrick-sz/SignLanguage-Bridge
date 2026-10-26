try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QPoint, QRect, QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QFileDialog, QFrame, QMainWindow
import pytest

from slbridge.desktop import SignLanguageApp
from slbridge.sources import FramePacket
from slbridge.contracts import RecognitionResult, ClassificationPrediction


def result(status='recognized'):
    prediction = ClassificationPrediction(
        'bom-dia', .9, {'bom-dia': .9}, {'pt-BR': 'Bom dia'}
    ) if status == 'recognized' else None
    reason = 'insufficient_hand_frames' if status != 'recognized' else None
    return RecognitionResult(status, 'fake.mp4', 'fake-model', 'libras', prediction, {}, reason=reason)


def pump(window, condition, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.instance().processEvents()
        if condition():
            return
        time.sleep(.01)
    raise AssertionError('Qt action did not complete')


@pytest.fixture(scope='session')
def qt_app():
    application = QApplication.instance() or QApplication(['slbridge-test'])
    application.setQuitOnLastWindowClosed(False)
    return application


@pytest.fixture
def gui(qt_app):
    window = SignLanguageApp()
    window.show()
    qt_app.processEvents()
    yield window
    if not window._closed:
        window.on_close()
    qt_app.processEvents()


def test_help_does_not_import_qt_or_gui():
    script = (
        "from app.desktop_app import main; import sys;\n"
        "try: main(['--help'])\nexcept SystemExit: pass\n"
        "assert 'PySide6' not in sys.modules\n"
        "assert 'tkinter' not in sys.modules\n"
        "assert 'slbridge.desktop' not in sys.modules"
    )
    completed = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, encoding='utf-8')
    assert completed.returncode == 0, completed.stderr
    assert '--model' in completed.stdout and '--catalog' in completed.stdout
    assert '--camera' in completed.stdout and '--video' in completed.stdout


def test_installable_console_target_is_callable():
    config = tomllib.loads((Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text(encoding='utf-8'))
    target = config['project']['scripts']['slbridge-desktop']
    module_name, function_name = target.split(':')
    module = __import__(module_name, fromlist=[function_name])
    assert callable(getattr(module, function_name))
    assert target == 'slbridge.desktop_cli:main'


def test_model_switch_control_disabled_during_recording(gui):
    gui.session.state = 'RECORDING'
    gui._update_controls()
    assert not gui.model_button.isEnabled()
    assert not gui.rollback_model_button.isEnabled()


@pytest.mark.parametrize('invocation', [['-m', 'slbridge.desktop_cli'], ['app/desktop_app.py']])
def test_module_and_script_help_force_utf8_under_gbk(invocation):
    env = dict(os.environ, PYTHONIOENCODING='gbk')
    completed = subprocess.run([sys.executable, *invocation, '--help'], capture_output=True, env=env)
    assert completed.returncode == 0, completed.stderr
    output = completed.stdout.decode('utf-8', errors='strict')
    assert '中文' in output and '摄像头' in output
    assert '--model' in output and '--catalog' in output


def test_installable_help_does_not_import_gui():
    script = (
        "from slbridge.desktop_cli import main; import sys;\n"
        "try: main(['--help'])\nexcept SystemExit: pass\n"
        "assert 'PySide6' not in sys.modules\n"
        "assert 'tkinter' not in sys.modules\n"
        "assert 'slbridge.desktop' not in sys.modules"
    )
    completed = subprocess.run([sys.executable, '-c', script], capture_output=True)
    assert completed.returncode == 0, completed.stderr


def test_real_controls_caption_reload_and_cancel(gui, tmp_path):
    data = {
        'schema_version': 1, 'catalog_id': 'ui-test-v1', 'sign_language': 'libras',
        'labels': [{'label_id': 'bom-dia', 'display_text': {'zh-CN': '早上好', 'pt-BR': 'Bom dia'},
                    'review_status': 'draft'}],
    }
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    gui.load_catalog(path)
    gui._display_result(result(), gui.session.request_id, gui.session.epoch)
    QApplication.instance().processEvents()
    assert gui.zh_label.text() == '早上好'
    assert gui.pt_label.text() == 'Bom dia'
    assert '待核验' in gui.notice_label.text()
    assert not gui.settings_dialog.isVisible()
    gui.settings_button.click()
    QApplication.instance().processEvents()
    assert gui.settings_dialog.isVisible()
    data['labels'][0]['display_text']['zh-CN'] = '早安（已编辑）'
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    gui.reload_catalog_button.click()
    assert gui.zh_label.text() == '早安（已编辑）'
    gui.settings_dialog.close()
    gui.cancel_button.click()
    assert gui.caption is None and gui.pt_label.text() == ''
    assert gui.camera_combo is not None and gui.mirror_check is not None
    assert not gui.record_button.isEnabled()


def test_rejection_and_catalog_error_remove_old_success(gui, tmp_path):
    gui._display_result(result(), 0, 0)
    assert gui.pt_label.text() == 'Bom dia'
    gui._display_result(result('rejected'), 0, 0)
    assert gui.pt_label.text() == ''
    assert '重新录制' in gui.zh_label.text()
    gui._display_result(result(), 0, 0)
    gui.load_catalog(tmp_path / 'missing.json')
    assert gui.pt_label.text() == ''
    assert '重试' in gui.status_label.text() or '错误' in gui.status_label.text()


def test_file_dialog_button_passes_whole_path_and_async_result(gui, monkeypatch):
    class Source:
        closed = False
        def open(self): return self
        def read(self): return None
        def close(self): self.closed = True
    source = Source()
    class Model:
        path = None
        def recognize(self, path):
            self.path = path
            return result()
    model = Model()
    gui.session.recognizer = model
    monkeypatch.setattr('slbridge.desktop.VideoSource', lambda path: source)
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args, **kwargs: ('entire.mp4', ''))
    gui.video_button.click()
    gui._update_controls()
    gui.pause_button.click()
    gui.recognize_button.click()
    pump(gui, lambda: gui.caption is not None)
    assert str(model.path) == 'entire.mp4'
    assert gui.pt_label.text() == 'Bom dia'
    gui.stop_button.click()
    pump(gui, lambda: source.closed)
    assert gui.caption is None


def test_preview_mirror_is_display_only_and_close_releases(gui, monkeypatch):
    class Source:
        closed = False
        i = 0
        def open(self): return self
        def read(self):
            time.sleep(.02)
            if self.closed: return None
            self.i += 1
            image = np.zeros((20, 40, 3), np.uint8)
            image[:, 0, 0] = 255
            return FramePacket(image, self.i * .02, self.i, 'fake-camera')
        def close(self): self.closed = True
    source = Source()
    monkeypatch.setattr('slbridge.desktop.CameraSource', lambda index: source)
    gui.open_camera(2)
    pump(gui, lambda: gui._photo is not None)
    assert isinstance(gui._photo, QImage)
    assert gui._photo.width() / gui._photo.height() == 2
    gui.on_close()
    deadline = time.monotonic() + 2
    while not source.closed and time.monotonic() < deadline:
        time.sleep(.005)
    assert source.closed


def test_preview_owns_pixels_and_camera_mirror_never_changes_input(gui):
    gui.session.source_kind = 'camera'
    image = np.zeros((20, 40, 3), np.uint8)
    image[:, 0, 0] = 255
    image[:, -1, 2] = 255
    packet = FramePacket(image, 0., 0, 'camera-test')
    before = image.copy()
    gui.mirror_check.setChecked(True)
    gui._render_preview(packet)
    assert isinstance(gui.preview.image, QImage)
    shown = gui.preview.image
    assert shown.pixelColor(0, 0).red() == 255
    assert shown.pixelColor(shown.width() - 1, 0).blue() == 255
    np.testing.assert_array_equal(packet.image, before)
    gui.mirror_check.setChecked(False)
    gui._render_preview(packet)
    owned = gui.preview.image
    image[:] = 0
    assert owned.pixelColor(0, 0).blue() == 255
    assert owned.pixelColor(owned.width() - 1, 0).red() == 255


def test_blocked_native_task_cancel_keeps_gui_responsive(gui):
    entered, release = threading.Event(), threading.Event()
    class Model:
        def recognize(self, path):
            entered.set()
            assert release.wait(3)
            return result()
    gui.session.recognizer = Model()
    gui.session.source_kind = 'video'
    gui.session.video_path = Path('fake.mp4')
    gui._update_controls()
    try:
        gui.recognize_button.click()
        assert entered.wait(1)
        gui.cancel_button.click()
        QApplication.instance().processEvents()
        assert gui.caption is None and gui.session.busy
        assert '取消' in gui.status_label.text()
        ticked = []
        QTimer.singleShot(20, lambda: ticked.append(True))
        pump(gui, lambda: bool(ticked))
    finally:
        release.set()
    pump(gui, lambda: not gui.session.busy)
    pump(gui, lambda: gui.recognize_button.isEnabled())
    assert gui.caption is None and gui.pt_label.text() == ''


def test_short_camera_recording_keeps_error_and_recovery_after_poll(gui):
    # Use the real session/buffer/finish path, with one deterministic camera frame.
    gui.session.source_kind = 'camera'
    gui.session.source_active = True
    gui.session.recognizer = object()
    gui.start_recording()
    gui.session.buffer.append(FramePacket(np.zeros((20, 40, 3), np.uint8), 0., 0, 'camera:0'))
    gui._update_controls()
    assert gui.finish_button.isVisible() and gui.finish_button.isEnabled()
    gui.finish_button.click()
    assert gui.session.state == 'ERROR'
    error_detail = gui._error_detail
    assert 'at least two frames' in error_detail
    title = gui.result_title.text()
    assert '未能识别' in title and gui.zh_label.text() == '请重试'

    # finish_recording queues clear before its synchronous error. Subsequent
    # event delivery must preserve the recovery card rather than show idle text.
    for _ in range(3):
        gui._poll()
        QApplication.instance().processEvents()
    assert gui.result_title.text() == title
    assert gui.zh_label.text() == '请重试' and gui.pt_label.text() == ''
    assert '重试' in gui.status_label.text()
    assert '详细信息' in gui.notice_label.text()
    assert gui._error_detail == error_detail
    assert error_detail in gui.technical_details.toPlainText()


def test_failed_model_change_keeps_active_version_words_and_bundle_path(gui, monkeypatch, tmp_path):
    good_path = tmp_path / 'working-model'
    bad_path = tmp_path / 'damaged-model'
    good_path.mkdir()
    bad_path.mkdir()
    model = SimpleNamespace(bundle=SimpleNamespace(
        path=good_path,
        manifest=SimpleNamespace(
            run_id='working-model-v1', sign_language='libras', backend='random_forest_v1',
            input_spec=SimpleNamespace(representation_id='mediapipe225_rgb_v1'), capabilities=('clip',),
        ), labels={'item:105': {'pt-BR': 'Árvore'}, 'item:100': {'pt-BR': 'Cair'}},
    ))
    # The successful load supplies a deterministic bundle; the damaged folder
    # goes through the real manifest validation and fails asynchronously.
    (bad_path / 'manifest.json').write_text(json.dumps({
        'input_spec': {'representation_id': 'mediapipe225_rgb_v1'}, 'capabilities': ['clip'],
    }), encoding='utf-8')
    original_load = gui.session.load_model

    def load_selected(path):
        return original_load(path, loader=lambda _: model) if Path(path) == good_path else original_load(path)

    monkeypatch.setattr(gui.session, 'load_model', load_selected)
    gui.load_model(good_path)
    pump(gui, lambda: gui.model_presentation.details.get('模型版本') == 'working-model-v1')
    active_summary = gui.model_label.text()
    active_words = [gui.vocabulary_list.item(index).text() for index in range(gui.vocabulary_list.count())]
    assert gui._model_path == str(good_path)
    gui.load_model(bad_path)
    pump(gui, lambda: gui.session.state == 'ERROR' and bool(gui._error_detail))
    assert gui.session.recognizer is model
    assert gui.model_presentation.details['模型版本'] == 'working-model-v1'
    assert gui.model_label.text() == active_summary
    assert [gui.vocabulary_list.item(index).text() for index in range(gui.vocabulary_list.count())] == active_words
    assert gui._model_path == str(good_path)
    details = gui.technical_details.toPlainText()
    assert f'模型文件夹：{good_path}' in details
    assert f'模型文件夹：{bad_path}' not in details
    assert '错误详情' in details


@pytest.mark.parametrize('size', [(1280, 820), (980, 720)])
@pytest.mark.parametrize('mode', ['video', 'camera'])
def test_large_preview_keeps_mode_actions_and_subtitles_visible(gui, monkeypatch, size, mode):
    class Source:
        def open(self): return self
        def read(self): return None
        def close(self): pass
    monkeypatch.setattr('slbridge.desktop.VideoSource', lambda path: Source())
    monkeypatch.setattr('slbridge.desktop.CameraSource', lambda index: Source())
    if mode == 'video':
        gui.open_video('layout.mp4')
    else:
        gui.open_camera(0)
    gui.resize(*size)
    gui._display_result(result(), 0, 0)
    gui.notice_label.setText('中文释义待核验')
    QApplication.instance().processEvents()
    packet = FramePacket(np.zeros((1080, 1920, 3), np.uint8), 0., 0, 'large')
    for _ in range(3):
        gui._render_preview(packet)
        QApplication.instance().processEvents()
    common_widgets = (gui.zh_label, gui.pt_label, gui.notice_label, gui.cancel_button,
                      gui.video_button, gui.camera_button, gui.stop_button, gui.settings_button)

    def assert_visible_inside(widget):
        assert widget.isVisible(), widget.objectName()
        geometry = QRect(widget.mapTo(gui, QPoint(0, 0)), widget.size())
        assert gui.rect().contains(geometry), (widget.objectName(), geometry, gui.rect())
        if widget in (gui.zh_label, gui.pt_label):
            assert widget.height() >= widget.fontMetrics().height(), (
                widget.objectName(), widget.height(), widget.fontMetrics().height(),
            )

    def assert_caption_rows_do_not_overlap():
        divider = gui.findChild(QFrame, 'divider')
        assert divider is not None and divider.isVisible()
        zh, rule, pt, notice = (
            QRect(widget.mapTo(gui, QPoint(0, 0)), widget.size())
            for widget in (gui.zh_label, divider, gui.pt_label, gui.notice_label)
        )
        assert zh.bottom() < rule.y(), (zh, rule)
        assert rule.bottom() < pt.y(), (rule, pt)
        assert pt.bottom() < notice.y(), (pt, notice)

    mode_buttons = (gui.recognize_button, gui.pause_button) if mode == 'video' else (gui.record_button,)
    for widget in (*common_widgets, *mode_buttons):
        assert_visible_inside(widget)
    assert_caption_rows_do_not_overlap()
    other_buttons = (gui.record_button, gui.finish_button) if mode == 'video' else (
        gui.recognize_button, gui.pause_button,
    )
    assert all(not button.isVisible() for button in other_buttons)
    if mode == 'camera':
        assert not gui.finish_button.isVisible()
        # Layout-only state transition: this fake camera has no frames to record.
        gui.session.state = 'RECORDING'
        gui._record_started = time.perf_counter()
        gui._update_controls()
        QApplication.instance().processEvents()
        assert not gui.record_button.isVisible()
        for widget in (*common_widgets, gui.finish_button):
            assert_visible_inside(widget)
        assert_caption_rows_do_not_overlap()
        assert gui.finish_button.isEnabled()
    assert not gui.settings_dialog.isVisible()
    assert isinstance(gui, QMainWindow)


def test_packaged_catalog_exists_and_loads_outside_checkout(monkeypatch, tmp_path):
    from importlib import resources
    from slbridge.captions import CaptionCatalog
    from slbridge.desktop_cli import default_catalog, default_model
    monkeypatch.chdir(tmp_path)
    seed = default_catalog()
    assert seed.is_file()
    with resources.as_file(seed) as path:
        catalog = CaptionCatalog.load(path)
    assert len(catalog.labels) == 12
    assert catalog.labels['item:105'].display_text['zh-CN'] == '树'
    assert default_model() is None
    config = tomllib.loads((Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text(encoding='utf-8'))
    assert 'resources/*.json' in config['tool']['setuptools']['package-data']['slbridge']


def test_default_catalog_prefers_editable_cwd_file_and_model_is_external(monkeypatch, tmp_path):
    from slbridge.desktop_cli import default_catalog, default_model
    monkeypatch.chdir(tmp_path)
    user_catalog = tmp_path / 'configs/libras_caption_catalog.json'
    user_catalog.parent.mkdir()
    user_catalog.write_text('{}', encoding='utf-8')
    assert default_catalog() == user_catalog
    model_path = tmp_path / 'artifacts/experiments/libras-rf-pilot-20261009/models/libras-rf-pilot-20261009'
    model_path.mkdir(parents=True)
    assert default_model() == model_path


def test_model_metadata_survives_source_switch_before_model_event_is_polled(gui):
    model = SimpleNamespace(bundle=SimpleNamespace(
        manifest=SimpleNamespace(
            run_id='persisted-model-v3', sign_language='libras', backend='random_forest_v1',
            input_spec=SimpleNamespace(representation_id='mediapipe225_rgb_v1'), capabilities=('clip',),
        ), labels={'item:105': {'pt-BR': 'Árvore'}, 'item:100': {'pt-BR': 'Cair'}},
    ))
    gui.session.load_model('fake', loader=lambda _: model)
    deadline = time.monotonic() + 3
    # Deliberately do not pump Qt while the model completes.
    while gui.session.busy and time.monotonic() < deadline:
        time.sleep(.005)
    assert gui.session.recognizer is model
    assert gui.model_presentation.details == {}
    assert gui.model_presentation.words == ()
    assert any(event.kind == 'model' for event in gui.session._events)
    class Source:
        def open(self): return self
        def read(self): return None
        def close(self): pass
    gui.session.open_source(Source(), kind='video', video_path='next.mp4')
    assert not any(event.kind == 'model' for event in gui.session._events)
    pump(gui, lambda: gui.model_presentation.details.get('模型版本') == 'persisted-model-v3')
    assert gui.model_label.text() == gui.model_presentation.summary
    assert '巴西手语' in gui.model_label.text() and '2 个词' in gui.model_label.text()
    assert 'persisted-model-v3' not in gui.model_label.text()
    words = [gui.vocabulary_list.item(index).text() for index in range(gui.vocabulary_list.count())]
    assert any('Árvore' in text for text in words) and any('Cair' in text for text in words)
    assert all('item:' not in text for text in words)


def test_sequence_metadata_counts_nonblank_vocabulary_items(gui):
    model = SimpleNamespace(bundle=SimpleNamespace(
        manifest=SimpleNamespace(
            run_id='ctc-model-v2', sign_language='csl', backend='onnx_ctc', task_type='gloss_sequence',
            input_spec=SimpleNamespace(representation_id='tfnet_rgb_v1'), capabilities=('clip',),
        ), vocabulary=(' ', '树', '。'), blank_id=0, labels={},
    ))
    gui.session.recognizer = model
    gui._sync_model_metadata()
    assert gui.word_count_label.text() == '2 个词表项'
    assert gui.vocabulary_list.count() == 2
    assert '2 个词表项' in gui.model_label.text()
    assert gui.vocabulary_list.item(0).text().startswith('树')
    assert gui.vocabulary_list.item(1).text().startswith('。')


def test_successful_model_preparation_clears_busy_result_title(gui):
    model = SimpleNamespace(bundle=SimpleNamespace(
        manifest=SimpleNamespace(
            run_id='loaded-ctc-v2', sign_language='csl', backend='onnx_ctc', task_type='gloss_sequence',
            input_spec=SimpleNamespace(representation_id='tfnet_rgb_v1'), capabilities=('clip',),
        ), vocabulary=(' ', '树'), labels={},
    ))
    gui._task_kind = 'model'
    gui.result_title.setText('正在准备识别模型')
    gui.session.load_model('fake', loader=lambda _: model)
    pump(gui, lambda: gui.model_presentation.details.get('模型版本') == 'loaded-ctc-v2')
    assert gui.result_title.text() == '等待一段完整手语'
    assert gui.state_badge.text() == '模型已就绪'
