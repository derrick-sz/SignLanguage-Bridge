from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from slbridge import desktop_cli


@pytest.mark.parametrize("args,expected", [
    (["--keypoints"], []),
    (["--keypoints", "--video", "sample.mp4"], [("video", Path("sample.mp4"))]),
    (["--keypoints", "--camera", "2"], [("index", 2), ("camera",)]),
])
def test_keypoint_launcher_bypasses_model_and_catalog_loading(monkeypatch, args, expected):
    calls = []
    class Workbench:
        def __init__(self):
            self.camera_index = SimpleNamespace(setValue=lambda n: calls.append(("index", n)))
        def open_video(self, path):
            calls.append(("video", path))
        def open_camera(self):
            calls.append(("camera",))
        def show(self):
            calls.append(("shown",))
    class Application:
        @staticmethod
        def instance():
            return SimpleNamespace(exec=lambda: 0)
    import PySide6.QtWidgets
    monkeypatch.setattr(PySide6.QtWidgets, "QApplication", Application)
    monkeypatch.setitem(sys.modules, "slbridge.keypoint_desktop", SimpleNamespace(KeypointWorkbench=Workbench))
    def forbidden():
        raise AssertionError("keypoint launcher must not look up a recognizer")
    monkeypatch.setattr(desktop_cli, "default_model", forbidden)
    monkeypatch.setattr(desktop_cli, "default_catalog", forbidden)
    assert desktop_cli.main(args) == 0
    assert calls == expected + [("shown",)]


def test_keypoint_launcher_rejects_recognizer_configuration():
    with pytest.raises(SystemExit) as exc:
        desktop_cli.main(["--keypoints", "--model", "unused-model"])
    assert exc.value.code == 2


@pytest.mark.parametrize("index", [-1, 33, 100])
def test_keypoint_launcher_rejects_camera_index_before_qt_can_clamp_it(index, capsys):
    with pytest.raises(SystemExit) as exc:
        desktop_cli.main(["--keypoints", "--camera", str(index)])
    assert exc.value.code == 2
    assert "0 到 32" in capsys.readouterr().err
