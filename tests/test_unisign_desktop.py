from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from slbridge.desktop import SignLanguageApp


@pytest.fixture(scope='module')
def application():
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


def test_unisign_button_reflects_missing_then_installed_bundle(application, tmp_path, monkeypatch):
    bundle = tmp_path / 'model_bundle'
    window = SignLanguageApp(unisign_model=bundle)
    try:
        window.show()
        window.open_settings()
        application.processEvents()
        assert window.unisign_button.isVisible()
        assert not window.unisign_button.isEnabled()
        assert '未安装实验模型' in window.unisign_label.text()
        bundle.mkdir()
        (bundle / 'manifest.json').write_text('{}', encoding='utf-8')
        window._update_controls()
        assert window.unisign_button.isEnabled()
        assert '实验模型已安装' in window.unisign_label.text()
        calls = []
        monkeypatch.setattr(window, 'load_model', lambda path, display_name=None: calls.append((path, display_name)))
        window.unisign_button.click()
        assert calls == [(bundle.resolve(), 'Uni-Sign Pose-only')]
    finally:
        window.close()
        application.processEvents()


def test_unisign_license_is_visible_in_settings(application, tmp_path):
    window = SignLanguageApp(unisign_model=tmp_path / 'missing')
    try:
        window.open_settings()
        application.processEvents()
        assert 'CC BY-NC 4.0' in window.unisign_button.toolTip()
        assert '非商业研究' in window.unisign_button.toolTip()
    finally:
        window.close()
        application.processEvents()
