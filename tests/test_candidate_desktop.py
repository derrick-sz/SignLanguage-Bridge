import json

from PySide6.QtCore import Qt
from PySide6.QtCore import QPoint, QRect

from test_continuous_desktop import application, gui, pump


def test_candidates_survive_current_result_clear_and_export_full_evidence(gui, monkeypatch, tmp_path):
    window, source = gui
    assert hasattr(window, 'candidate_history'), 'Recognition UI needs retained candidate sequence'
    window.continuous_check.click()
    source.gate.set()
    pump(lambda: window.candidate_history.count() >= 3)
    window.continuous_check.click()
    pump(lambda: not window.session.busy)
    assert window._last_result is None and window.candidate_history.count() >= 3
    assert window.caption_history.count() == 0
    assert '未确认' in window.candidate_history.item(0).text()
    row = window.candidate_history.item(0).data(Qt.ItemDataRole.UserRole)
    assert row['result']['prediction']['label_id'] == 'word'
    assert row['result']['prediction']['scores'] == {'word': .8}
    destination = tmp_path/'中文候选.json'
    monkeypatch.setattr('slbridge.desktop.QFileDialog.getSaveFileName',
                        lambda *args: (str(destination), 'JSON (*.json)'))
    window.export_candidates_button.click()
    exported = json.loads(destination.read_text(encoding='utf-8'))
    assert len(exported['candidates']) == window.candidate_history.count()
    assert exported['candidates'][0]['caption']['pt'] == 'Olá'
    window.candidate_limit.setValue(2)
    assert window.candidate_history.count() == 2
    assert window.session.candidate_context()['dropped_count'] > 0
    window.clear_candidates_button.click()
    assert window.candidate_history.count() == 0
    assert not window.export_candidates_button.isEnabled()
    assert window.session.source is source and window.session.source_active


def test_candidate_export_error_does_not_destroy_preview_or_history(gui, monkeypatch):
    window, source = gui
    assert hasattr(window, 'candidate_history'), 'Candidate history is missing'
    window.continuous_check.click()
    source.gate.set()
    pump(lambda: window.candidate_history.count() >= 1)
    window.continuous_check.click()
    pump(lambda: not window.session.busy)
    before = window.session.candidate_context()
    monkeypatch.setattr('slbridge.desktop.QFileDialog.getSaveFileName', lambda *args: ('bad.json', 'JSON'))
    def denied(path):
        raise OSError('write denied')
    monkeypatch.setattr(window.session, 'export_candidate_context', denied)
    window.export_candidates_button.click()
    assert window.session.candidate_context() == before
    assert window.preview.image is not None and window.session.source_active
    assert '导出失败' in window.candidate_status.text()


def test_candidate_panel_stays_readable_at_minimum_window_size(gui):
    window, source = gui
    window.resize(980, 720)
    window.continuous_check.click()
    source.gate.set()
    pump(lambda: window.candidate_history.count() >= 3)
    def bounds(widget):
        return QRect(widget.mapTo(window, QPoint(0, 0)), widget.size())
    assert not bounds(window.candidate_history).intersects(bounds(window.candidate_status))
    assert not bounds(window.candidate_status).intersects(bounds(window.export_candidates_button))
    assert window.candidate_history.viewport().height() >= 90
