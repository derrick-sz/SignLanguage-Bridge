import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import tkinter as tk
import numpy as np
import pytest
from slbridge.desktop import SignLanguageApp
from slbridge.session import RecognitionSession
from slbridge.sources import FramePacket
from slbridge.contracts import RecognitionResult, ClassificationPrediction


def result(status='recognized'):
    return RecognitionResult(status, 'fake.mp4', 'fake-model', 'libras', ClassificationPrediction('bom-dia', .9, {'bom-dia': .9}, {'pt-BR': 'Bom dia'}) if status == 'recognized' else None, {}, reason='insufficient_hand_frames' if status != 'recognized' else None)


def pump(root, condition, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        root.update()
        if condition():
            return
        time.sleep(.01)
    raise AssertionError('Tk action did not complete')


@pytest.fixture
def gui():
    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f'Tk display unavailable: {error}')
    app = SignLanguageApp(root)
    root.update()
    yield app
    if not app._closed:
        app.on_close()


def test_help_does_not_import_tkinter_or_gui():
    script = "from app.desktop_app import main; import sys;\ntry: main(['--help'])\nexcept SystemExit: pass\nassert 'tkinter' not in sys.modules\nassert 'slbridge.desktop' not in sys.modules"
    completed = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, encoding='utf-8')
    assert completed.returncode == 0, completed.stderr
    assert '--model' in completed.stdout and '--catalog' in completed.stdout
    assert '--camera' in completed.stdout and '--video' in completed.stdout


def test_installable_console_target_is_callable():
    import tomllib
    config = tomllib.loads((Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text(encoding='utf-8'))
    target = config['project']['scripts']['slbridge-desktop']
    module_name, function_name = target.split(':')
    module = __import__(module_name, fromlist=[function_name])
    assert callable(getattr(module, function_name))
    assert target == 'slbridge.desktop_cli:main'


@pytest.mark.parametrize('invocation', [['-m', 'slbridge.desktop_cli'], ['app/desktop_app.py']])
def test_module_and_script_help_force_utf8_under_gbk(invocation):
    env = dict(os.environ, PYTHONIOENCODING='gbk')
    completed = subprocess.run([sys.executable, *invocation, '--help'], capture_output=True, env=env)
    assert completed.returncode == 0, completed.stderr
    output = completed.stdout.decode('utf-8', errors='strict')
    assert '完整动作片段识别' in output
    assert '摄像头设备编号' in output
    assert '--model' in output and '--catalog' in output


def test_installable_help_does_not_import_gui():
    script = "from slbridge.desktop_cli import main; import sys;\ntry: main(['--help'])\nexcept SystemExit: pass\nassert 'tkinter' not in sys.modules\nassert 'slbridge.desktop' not in sys.modules"
    completed = subprocess.run([sys.executable, '-c', script], capture_output=True)
    assert completed.returncode == 0, completed.stderr


def test_real_controls_caption_reload_and_cancel(gui, tmp_path):
    data = {'schema_version': 1, 'catalog_id': 'ui-test-v1', 'sign_language': 'libras', 'labels': [{'label_id': 'bom-dia', 'display_text': {'zh-CN': '早上好', 'pt-BR': 'Bom dia'}, 'review_status': 'draft'}]}
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    gui.load_catalog(path)
    gui._display_result(result(), gui.session.request_id, gui.session.epoch)
    gui.root.update()
    assert gui.zh_label.cget('text') == '早上好'
    assert gui.pt_label.cget('text') == 'Bom dia'
    assert '待核验' in gui.notice_label.cget('text')
    data['labels'][0]['display_text']['zh-CN'] = '早安（已编辑）'
    path.write_text(json.dumps(data), encoding='utf-8')
    gui.load_catalog(path)
    assert gui.zh_label.cget('text') == '早安（已编辑）'
    gui.cancel_button.invoke()
    assert gui.caption is None and gui.pt_var.get() == ''
    assert gui.camera_entry.winfo_exists()
    assert gui.mirror_check.winfo_exists()
    assert str(gui.record_button.cget('state')) == 'disabled'


def test_rejection_and_catalog_error_remove_old_success(gui, tmp_path):
    gui._display_result(result(), 0, 0)
    assert gui.pt_var.get() == 'Bom dia'
    gui._display_result(result('rejected'), 0, 0)
    assert gui.pt_var.get() == ''
    assert '重新录制' in gui.zh_var.get()
    gui._display_result(result(), 0, 0)
    gui.load_catalog(tmp_path / 'missing.json')
    assert gui.pt_var.get() == '' and '错误' in gui.status_var.get()


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
    monkeypatch.setattr('slbridge.desktop.filedialog.askopenfilename', lambda **kwargs: 'entire.mp4')
    gui.video_button.invoke()
    gui._update_controls()
    gui.pause_button.invoke()
    gui.recognize_button.invoke()
    pump(gui.root, lambda: gui.caption is not None)
    assert str(model.path) == 'entire.mp4'
    assert gui.pt_label.cget('text') == 'Bom dia'
    gui.stop_button.invoke()
    pump(gui.root, lambda: source.closed)
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
    monkeypatch.setattr('slbridge.desktop.CameraSource', lambda i: source)
    gui.open_camera(2)
    pump(gui.root, lambda: gui._photo is not None)
    assert gui._photo.width() / gui._photo.height() == 2
    gui.on_close()
    deadline = time.monotonic() + 2
    while not source.closed and time.monotonic() < deadline:
        time.sleep(.005)
    assert source.closed


def test_blocked_native_task_cancel_keeps_gui_responsive(gui):
    entered, release = threading.Event(), threading.Event()
    class Model:
        def recognize(self, path):
            entered.set()
            assert release.wait(3)
            return result()
    gui.session.recognizer = Model()
    gui.session.video_path = Path('fake.mp4')
    gui._update_controls()
    gui.recognize_button.invoke()
    assert entered.wait(1)
    gui.cancel_button.invoke()
    gui.root.update()
    assert gui.caption is None and '等待' in gui.status_var.get()
    release.set()
    pump(gui.root, lambda: not gui.session.busy)
    pump(gui.root, lambda: str(gui.recognize_button.cget('state')) == 'normal')
    assert gui.caption is None and gui.pt_var.get() == ''


@pytest.mark.parametrize('size', ['1060x780', '720x680'])
def test_large_preview_keeps_subtitles_and_actions_visible(gui, size):
    gui.root.geometry(size)
    gui.vocabulary_label.configure(text='支持词汇：' + '、'.join(f'Palavra de teste ({i})' for i in range(12)))
    gui._display_result(result(), 0, 0)
    gui.notice_var.set('中文释义草案，待核验；模型分数未经校准')
    gui.root.update()
    packet = FramePacket(np.zeros((1080, 1920, 3), np.uint8), 0., 0, 'large')
    for _ in range(3):
        gui._render_preview(packet)
        gui.root.update()
    for widget in (gui.zh_label, gui.pt_label, gui.notice_label, gui.record_button,
                   gui.finish_button, gui.recognize_button, gui.cancel_button,
                   gui.model_button, gui.video_button, gui.camera_button, gui.stop_button,
                   gui.camera_entry, gui.mirror_check):
        assert widget.winfo_ismapped()
        left = widget.winfo_rootx() - gui.root.winfo_rootx()
        top = widget.winfo_rooty() - gui.root.winfo_rooty()
        assert left >= 0 and top >= 0
        assert left + widget.winfo_width() <= gui.root.winfo_width()
        assert top + widget.winfo_height() <= gui.root.winfo_height()


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
    import tomllib
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
        manifest=SimpleNamespace(run_id='persisted-model-v3', input_spec=SimpleNamespace(
            representation_id='mediapipe225_rgb_v1'), capabilities=('clip',)),
        labels={'item:105': {'pt-BR': 'Árvore'}, 'item:100': {'pt-BR': 'Cair'}}))
    gui.session.load_model('fake', loader=lambda _: model)
    deadline = time.monotonic() + 3
    # Deliberately do not pump Tk while the model completes.
    while gui.session.busy and time.monotonic() < deadline:
        time.sleep(.005)
    assert gui.session.recognizer is model
    assert '未加载' in gui.model_var.get()
    assert any(event.kind == 'model' for event in gui.session._events)
    class Source:
        def open(self): return self
        def read(self): return None
        def close(self): pass
    gui.session.open_source(Source(), kind='video', video_path='next.mp4')
    assert not any(event.kind == 'model' for event in gui.session._events)
    pump(gui.root, lambda: 'persisted-model-v3' in gui.model_var.get())
    assert 'Árvore (item:105)' in gui.vocabulary_label.cget('text')
    assert 'Cair (item:100)' in gui.vocabulary_label.cget('text')
