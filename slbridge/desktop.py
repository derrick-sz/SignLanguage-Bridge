"""Qt workspace for local clip recognition; the session owns native resources."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QImage
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
    QFrame, QHBoxLayout, QLabel, QLayout, QListWidget, QListWidgetItem, QMainWindow,
    QPlainTextEdit, QProgressBar, QPushButton, QSizePolicy, QSpinBox,
    QTabWidget, QVBoxLayout, QWidget,
)

from .captions import CaptionCatalog, caption_from_result
from .desktop_presenter import describe_model, describe_result
from .desktop_widgets import APP_STYLE, BrandMark, PreviewWidget, icon
from .session import RecognitionSession
from .sources import CameraSource, VideoSource


def _label(text="", name="", *, wrap=True):
    widget = QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    widget.setWordWrap(wrap)
    if name:
        widget.setObjectName(name)
    return widget


def _button(text, callback, name=None, kind="secondary"):
    widget = QPushButton(text)
    widget.setProperty("kind", kind)
    widget.setCursor(Qt.CursorShape.PointingHandCursor)
    if name:
        widget.setIcon(icon(name, "#FFFFFF" if kind == "primary" else "#536A78"))
    widget.clicked.connect(lambda: callback())
    return widget


def _panel():
    widget = QFrame()
    widget.setObjectName("panel")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(20, 20, 20, 20)
    layout.setSpacing(12)
    return widget, layout


class SignLanguageApp(QMainWindow):
    def __init__(self, *, session=None, catalog=None, model=None, video=None, camera=None, runtime_options=None):
        super().__init__()
        self.session = session if session is not None else RecognitionSession(options=runtime_options)
        self.catalog = catalog if catalog is not None else CaptionCatalog()
        self.caption = None
        self._last_result = None
        self._metadata_model = None
        self._metadata_catalog = None
        self._catalog_path = None
        self._model_path = None
        self._source_path = None
        self._photo = None
        self._closed = False
        self._task_kind = None
        self._cancel_pending = False
        self._record_started = None
        self._error_detail = ""
        self._result_details = {}
        self.model_presentation = describe_model(None, self.catalog)
        self.setWindowTitle("手语桥 | Ponte de Sinais")
        self.resize(1280, 820)
        self.setMinimumSize(980, 720)
        self.setFont(QFont("Microsoft YaHei UI", 10))
        self.setStyleSheet(APP_STYLE)
        self._build_ui()
        self._build_settings()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(30)
        if video:
            self.open_video(video)
        elif camera is not None:
            self.open_camera(camera)
        if model:
            self.load_model(model)
        self._update_controls()

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("appRoot")
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(28, 22, 28, 18)
        outer.setSpacing(20)
        header = QHBoxLayout()
        header.setSpacing(14)
        header.addWidget(BrandMark())
        title = QVBoxLayout()
        title.setSpacing(3)
        title.addWidget(_label("手语桥", "title"))
        title.addWidget(_label("巴西手语识别 / Libras", "muted"))
        header.addLayout(title)
        header.addStretch()
        self.state_badge = _label("等待输入", "statusBadge", wrap=False)
        header.addWidget(self.state_badge)
        self.settings_button = _button("设置", self.open_settings, "settings", "ghost")
        header.addWidget(self.settings_button)
        outer.addLayout(header)

        columns = QHBoxLayout()
        columns.setSpacing(20)
        source_panel, left = _panel()
        inputs = QHBoxLayout()
        inputs.setSpacing(8)
        self.video_button = _button("导入视频", self._choose_video, "video")
        self.camera_button = _button("使用摄像头", self.open_camera, "camera")
        for button, name in ((self.video_button, "inputVideo"), (self.camera_button, "inputCamera")):
            button.setObjectName(name)
            button.setCheckable(True)
            button.setAutoExclusive(False)
            inputs.addWidget(button)
        inputs.addStretch()
        left.addLayout(inputs)
        source_line = QHBoxLayout()
        self.source_label = _label("预览画面", "sourceTitle")
        self.source_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.stop_button = _button("关闭", self.stop_source, "close", "ghost")
        source_line.addWidget(self.source_label, 1)
        source_line.addWidget(self.stop_button)
        left.addLayout(source_line)
        self.preview = PreviewWidget()
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview.set_placeholder("从一个手语动作开始", "导入视频，或使用摄像头录制")
        left.addWidget(self.preview, 1)

        self.camera_options = QWidget()
        camera_options = QHBoxLayout(self.camera_options)
        camera_options.setContentsMargins(0, 0, 0, 0)
        self.camera_combo = QComboBox()
        for index in range(3):
            self.camera_combo.addItem("默认摄像头" if index == 0 else f"摄像头 {index + 1}", index)
        self.camera_combo.setToolTip("选择要打开的设备；列表名称不表示设备已连接。")
        self.mirror_check = QCheckBox("镜像预览")
        self.mirror_check.setChecked(True)
        self.mirror_check.setToolTip("仅翻转显示画面，不改变送入模型的动作。")
        camera_options.addWidget(self.camera_combo)
        camera_options.addStretch()
        camera_options.addWidget(self.mirror_check)
        left.addWidget(self.camera_options)

        self.record_progress = QProgressBar()
        self.record_progress.setRange(0, 150)
        self.record_progress.setTextVisible(False)
        self.record_progress.setFixedHeight(5)
        self.record_time_label = _label("", "muted")
        left.addWidget(self.record_progress)
        left.addWidget(self.record_time_label)
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.pause_button = _button("暂停预览", self.toggle_pause, "pause", "ghost")
        self.recognize_button = _button("识别视频", self.recognize_video, "check", "primary")
        self.record_button = _button("开始录制", self.start_recording, "record", "primary")
        self.finish_button = _button("完成并识别", self.finish_recording, "check", "primary")
        self.cancel_button = _button("清空结果", self.cancel, "cancel")
        for button in (self.pause_button, self.recognize_button, self.record_button, self.finish_button):
            actions.addWidget(button)
        actions.addStretch()
        actions.addWidget(self.cancel_button)
        left.addLayout(actions)
        self.input_hint = _label("每次识别一个完整动作。", "muted")
        left.addWidget(self.input_hint)
        columns.addWidget(source_panel, 2)

        right = QVBoxLayout()
        right.setSpacing(16)
        result_panel, result_layout = _panel()
        result_layout.setSpacing(7)
        result_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        result_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        result_layout.addWidget(_label("识别结果", "sectionTitle"))
        self.result_title = _label("等待一个完整动作", "muted")
        result_layout.addWidget(self.result_title)
        self.busy_progress = QProgressBar()
        self.busy_progress.setRange(0, 0)
        self.busy_progress.setTextVisible(False)
        self.busy_progress.setFixedHeight(5)
        result_layout.addWidget(self.busy_progress)
        result_layout.addSpacing(4)
        result_layout.addWidget(_label("中文释义", "muted"))
        self.zh_label = _label("等待识别", "resultZh")
        self.zh_label.setMinimumHeight(48)
        self.zh_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        result_layout.addWidget(self.zh_label)
        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        rule.setObjectName("divider")
        result_layout.addWidget(rule)
        result_layout.addWidget(_label("Português", "muted"))
        self.pt_label = _label("", "resultPt")
        self.pt_label.setMinimumHeight(36)
        self.pt_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        result_layout.addWidget(self.pt_label)
        result_layout.addStretch()
        self.notice_label = _label("识别后，中葡释义会显示在这里。", "notice")
        result_layout.addWidget(self.notice_label)
        self.details_button = _button("详细信息", lambda: self.open_settings(details=True), "info", "ghost")
        self.details_button.setProperty("size", "small")
        result_layout.addWidget(self.details_button, 0, Qt.AlignmentFlag.AlignRight)
        right.addWidget(result_panel)

        vocabulary_panel, words = _panel()
        word_header = QHBoxLayout()
        word_header.addWidget(_label("支持的手语词", "sectionTitle"))
        word_header.addStretch()
        self.word_count_label = _label("", "muted", wrap=False)
        word_header.addWidget(self.word_count_label)
        words.addLayout(word_header)
        self.model_label = _label("请先加载识别模型", "modelSummary")
        words.addWidget(self.model_label)
        self.vocabulary_list = QListWidget()
        self.vocabulary_list.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.vocabulary_list.setMinimumHeight(64)
        words.addWidget(self.vocabulary_list, 1)
        right.addWidget(vocabulary_panel, 1)
        right_widget = QWidget()
        right_widget.setMinimumWidth(280)
        right_widget.setMaximumWidth(380)
        right_widget.setLayout(right)
        columns.addWidget(right_widget, 1)
        outer.addLayout(columns, 1)
        self.status_label = _label("导入视频，或打开摄像头录制一个手语动作。", "muted")
        outer.addWidget(self.status_label)

    def _build_settings(self):
        self.settings_dialog = QDialog(self)
        self.settings_dialog.setWindowTitle("识别设置 | 手语桥")
        self.settings_dialog.resize(640, 590)
        self.settings_dialog.setMinimumSize(560, 500)
        outer = QVBoxLayout(self.settings_dialog)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(16)
        outer.addWidget(_label("识别设置", "sectionTitle"))
        self.settings_tabs = QTabWidget()
        settings = QWidget()
        form = QVBoxLayout(settings)
        form.setContentsMargins(12, 18, 12, 12)
        form.setSpacing(12)
        form.addWidget(_label("识别模型", "sectionTitle"))
        self.settings_model_label = _label("尚未加载模型", "muted")
        form.addWidget(self.settings_model_label)
        self.model_button = _button("选择识别模型", self._choose_model, "folder")
        form.addWidget(self.model_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.rollback_model_button = _button("恢复上个模型", self.rollback_model, "reload")
        form.addWidget(self.rollback_model_button, 0, Qt.AlignmentFlag.AlignLeft)
        form.addWidget(_label("字幕释义", "sectionTitle"))
        self.catalog_label = _label("中文释义待补充", "muted")
        form.addWidget(self.catalog_label)
        catalog_actions = QHBoxLayout()
        self.catalog_button = _button("更换释义文件", self._choose_catalog, "folder")
        self.reload_catalog_button = _button("重新加载", self.reload_catalog, "reload")
        catalog_actions.addWidget(self.catalog_button)
        catalog_actions.addWidget(self.reload_catalog_button)
        catalog_actions.addStretch()
        form.addLayout(catalog_actions)
        form.addWidget(_label("保存释义文件后，点击重新加载即可更新字幕。", "muted"))
        form.addWidget(_label("摄像头设备", "sectionTitle"))
        camera_form = QFormLayout()
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 99)
        self.camera_spin.setValue(0)
        self.camera_spin.valueChanged.connect(self._select_camera_index)
        camera_form.addRow("设备序号（从 0 开始）", self.camera_spin)
        form.addLayout(camera_form)
        form.addWidget(_label("0 为默认设备。调整后点击主界面的使用摄像头。", "muted"))
        form.addStretch()
        self.settings_tabs.addTab(settings, "设置")
        self.technical_details = QPlainTextEdit()
        self.technical_details.setReadOnly(True)
        self.settings_tabs.addTab(self.technical_details, "详细信息")
        outer.addWidget(self.settings_tabs, 1)
        done = _button("完成", self.settings_dialog.hide, "check", "primary")
        outer.addWidget(done, 0, Qt.AlignmentFlag.AlignRight)

    def open_settings(self, *, details=False):
        self._refresh_details()
        self.settings_tabs.setCurrentIndex(1 if details else 0)
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def _select_camera_index(self, index):
        item = self.camera_combo.findData(index)
        if item < 0:
            self.camera_combo.addItem(f"摄像头 {index + 1}", index)
            item = self.camera_combo.count() - 1
        self.camera_combo.setCurrentIndex(item)

    def _clear_caption(self):
        self.caption = None
        self._last_result = None
        self._result_details = {}
        self.result_title.setText("等待一个完整动作")
        self.zh_label.setText("等待识别")
        self.zh_label.setProperty("state", "")
        self.zh_label.style().polish(self.zh_label)
        self.pt_label.setText("")
        self.notice_label.setText("识别后，中葡释义会显示在这里。")
        self.notice_label.setProperty("tone", "")
        self.notice_label.style().polish(self.notice_label)

    def _error(self, error, context="操作未完成"):
        self._clear_caption()
        self._error_detail = str(error)
        self.result_title.setText(context)
        self.zh_label.setText("请重试")
        self.zh_label.setProperty("state", "error")
        self.zh_label.style().polish(self.zh_label)
        self.notice_label.setText("重新选择输入或检查设置；原因可在详细信息中查看。")
        self.notice_label.setProperty("tone", "warning")
        self.notice_label.style().polish(self.notice_label)
        self.status_label.setText(f"{context}。请重试或查看详细信息。")
        self._refresh_details()

    def _call(self, operation, *, context="操作未完成"):
        try:
            operation()
            self._error_detail = ""
            self._clear_caption()
            self._cancel_pending = False
            self._update_controls()
            return True
        except Exception as error:
            self._error(error, context)
            return False

    def _choose_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入手语视频", "", "视频 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)")
        if path:
            self.open_video(path)
        else:
            self._update_controls()

    def open_video(self, path):
        self._record_started = None
        if self._call(lambda: self.session.open_source(VideoSource(path), kind="video", video_path=path), context="无法打开视频"):
            self._source_path = str(path)
            self.source_label.setText("视频：" + self._short_name(path))
            self.source_label.setToolTip(str(path))
            self._reset_preview("正在打开视频", "预览播放位置不会影响识别")
            self.status_label.setText("正在打开视频…")

    @staticmethod
    def _short_name(path):
        name = Path(path).name
        return name if len(name) <= 48 else name[:28] + "…" + name[-16:]

    def open_camera(self, index=None):
        self._record_started = None
        try:
            index = int(self.camera_combo.currentData()) if index is None else int(index)
            if index < 0:
                raise ValueError("摄像头设备序号必须大于或等于 0")
            self._select_camera_index(index)
            self.camera_spin.setValue(index)
            if self._call(lambda: self.session.open_source(CameraSource(index), kind="camera"), context="无法打开摄像头"):
                self._source_path = f"camera:{index}"
                name = "默认摄像头" if index == 0 else f"摄像头 {index + 1}"
                self.source_label.setText(name)
                self.source_label.setToolTip(f"摄像头设备序号：{index}")
                self._reset_preview("正在连接摄像头", "让双手和上半身保持在画面内")
                self.status_label.setText("正在连接摄像头…")
        except Exception as error:
            self._error(error, "无法打开摄像头")

    def stop_source(self):
        self._record_started = None
        kind = self.session.source_kind
        self._call(self.session.stop_source)
        self._source_path = None
        self.source_label.setText("预览画面")
        self._reset_preview("从一个手语动作开始", "导入视频，或使用摄像头录制")
        self.status_label.setText("摄像头已关闭。" if kind == "camera" else "视频已关闭。")

    def _reset_preview(self, title, hint):
        self._photo = None
        self.preview.set_placeholder(title, hint)

    def _choose_model(self):
        path = QFileDialog.getExistingDirectory(self, "选择识别模型文件夹")
        if path:
            self.load_model(path)

    def load_model(self, path):
        self._task_kind = "model"
        if self._call(lambda: self.session.load_model(path), context="无法加载识别模型"):
            self.status_label.setText("正在准备识别模型…")
            self.result_title.setText("正在准备识别模型")

    def rollback_model(self):
        if self._call(self.session.rollback_model, context="无法恢复上个模型"):
            self.status_label.setText("已恢复上个识别模型。")

    def _choose_catalog(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择字幕释义文件", "", "字幕释义 (*.json)")
        if path:
            self.load_catalog(path)

    def load_catalog(self, path):
        try:
            catalog = CaptionCatalog.load(path)
            self.catalog = catalog
            self._catalog_path = Path(path)
            draft = any(entry.review_status == "draft" for entry in catalog.labels.values())
            self.catalog_label.setText("字幕释义已加载" + ("，中文草案待核验。" if draft else "。"))
            self._metadata_catalog = None
            self._sync_model_metadata()
            if self._last_result is not None:
                self._display_result(self._last_result, self.session.request_id, self.session.epoch)
            self._update_controls()
        except Exception as error:
            self._error(error, "无法加载字幕释义")

    def reload_catalog(self):
        if self._catalog_path is not None:
            self.load_catalog(self._catalog_path)

    def recognize_video(self):
        self._task_kind = "recognition"
        if self._call(self.session.recognize_file, context="无法识别视频"):
            self.result_title.setText("正在识别动作")
            self.status_label.setText("正在识别完整视频，请稍候…")

    def start_recording(self):
        if self._call(self.session.start_recording, context="无法开始录制"):
            self._record_started = time.perf_counter()
            self.status_label.setText("正在录制，完成一个动作后点击完成并识别。")

    def finish_recording(self):
        self._task_kind = "recognition"
        self._record_started = None
        if self._call(self.session.finish_recording, context="录制片段未能识别"):
            self.result_title.setText("正在识别动作")
            self.status_label.setText("录制已完成，正在识别动作…")

    def cancel(self):
        self._record_started = None
        kind = "录制" if self.session.state == "RECORDING" else "识别"
        self._call(self.session.cancel)
        self._cancel_pending = self.session.busy
        self.status_label.setText("已取消识别，正在等待当前任务结束…" if self._cancel_pending else f"已清空结果，可以开始新的{kind}。")

    def toggle_pause(self):
        if self.session.source_kind != "video":
            return
        if not self.session.source_active:
            self.open_video(self.session.video_path)
        else:
            self.session.paused = not self.session.paused
            self.status_label.setText("视频预览已暂停，仍可识别完整视频。" if self.session.paused else "视频预览继续播放。")
        self._update_controls()

    def _update_controls(self):
        busy = self.session.busy
        state = self.session.state
        camera_mode = self.session.source_kind == "camera"
        video = self.session.video_path is not None
        loaded = self.session.recognizer is not None
        recording = state == "RECORDING"
        self.video_button.setChecked(self.session.source_kind == "video")
        self.camera_button.setChecked(camera_mode)
        self.model_button.setEnabled(not busy and not recording)
        self.rollback_model_button.setEnabled(not busy and not recording and
                                              self.session.model_manager.status()["previous"] is not None)
        self.reload_catalog_button.setEnabled(self._catalog_path is not None and self._catalog_path.is_file())
        self.recognize_button.setVisible(not camera_mode)
        self.recognize_button.setEnabled(video and loaded and not busy)
        self.pause_button.setVisible(video)
        self.pause_button.setEnabled(video and (self.session.source_active or not busy))
        self.pause_button.setText("重新播放" if video and not self.session.source_active else "继续预览" if self.session.paused else "暂停预览")
        self.record_button.setVisible(camera_mode and not recording)
        self.record_button.setEnabled(camera_mode and self.session.source_active and loaded and not busy)
        self.finish_button.setVisible(camera_mode and recording)
        self.finish_button.setEnabled(recording and not busy)
        self.camera_options.setVisible(camera_mode)
        self.stop_button.setEnabled(self.session.source is not None or video)
        self.stop_button.setText("关闭摄像头" if camera_mode else "关闭视频" if video else "关闭")
        self.cancel_button.setText("取消录制" if recording else "取消识别" if busy else "清空结果")
        self.cancel_button.setEnabled(recording or busy or self.caption is not None or state == "ERROR")
        self.busy_progress.setVisible(busy)
        self.record_progress.setVisible(recording)
        self.record_time_label.setVisible(recording)
        if recording and self._record_started is not None:
            elapsed = max(0, time.perf_counter() - self._record_started)
            self.record_time_label.setText(f"正在录制  {elapsed:.1f} 秒 / 15 秒")
            self.record_progress.setValue(min(150, int(elapsed * 10)))
        self.input_hint.setText("每次录制一个完整动作，最长 15 秒。" if camera_mode else "识别整段视频，播放位置不影响结果。" if video else "每次识别一个完整动作。")
        badge = "正在录制" if recording else "正在准备模型" if busy and self._task_kind == "model" else "正在识别" if busy else "请重试" if state == "ERROR" else "模型已就绪" if loaded else "请选择识别模型"
        self.state_badge.setText(badge)
        self._refresh_details()

    def _display_result(self, result, request_id, epoch):
        self._last_result = result
        self.caption = caption_from_result(result, self.catalog, request_id=request_id, epoch=epoch)
        presentation = describe_result(result, self.caption)
        self._result_details = presentation.details
        self.result_title.setText(presentation.title)
        self.zh_label.setText(presentation.zh)
        self.zh_label.setProperty("state", "rejected" if result.status == "rejected" else "")
        self.zh_label.style().polish(self.zh_label)
        self.pt_label.setText(presentation.pt)
        self.notice_label.setText(presentation.notice)
        self.notice_label.setProperty("tone", "warning" if presentation.notice else "")
        self.notice_label.style().polish(self.notice_label)
        self.status_label.setText("识别完成，可选择新的视频或录制下一个动作。" if result.status == "recognized" else "本次未能识别，请根据提示重试。")
        self._refresh_details()
        self._update_controls()

    def _render_preview(self, packet):
        rgb = packet.image[:, :, ::-1]
        if self.session.source_kind == "camera" and self.mirror_check.isChecked():
            rgb = rgb[:, ::-1]
        pixels = np.ascontiguousarray(rgb)
        height, width = pixels.shape[:2]
        self._photo = QImage(pixels.data, width, height, pixels.strides[0], QImage.Format.Format_RGB888).copy()
        self.preview.set_image(self._photo)
        self.preview.set_overlay("正在录制" if self.session.state == "RECORDING" else "摄像头预览" if self.session.source_kind == "camera" else "视频预览")

    def _sync_model_metadata(self):
        model = self.session.recognizer
        if model is self._metadata_model and self.catalog is self._metadata_catalog:
            return
        presentation = describe_model(model, self.catalog)
        bundle_path = getattr(getattr(model, "bundle", None), "path", None)
        self._model_path = str(bundle_path) if bundle_path is not None else None
        self.model_presentation = presentation
        self.model_label.setText(presentation.summary)
        self.settings_model_label.setText(presentation.summary)
        self.word_count_label.setText(f"{len(presentation.words)} 个词" if presentation.words else "")
        self.vocabulary_list.clear()
        for word in presentation.words:
            item = QListWidgetItem(f"{word.zh}  /  {word.pt}")
            item.setToolTip("中文释义待核验" if word.draft else "")
            self.vocabulary_list.addItem(item)
        if not presentation.words:
            item = QListWidgetItem("加载模型后显示词汇")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.vocabulary_list.addItem(item)
        self._metadata_model = model
        self._metadata_catalog = self.catalog
        self._refresh_details()

    def _refresh_details(self):
        sections = ["识别模型"]
        sections.extend(f"{key}：{value}" for key, value in self.model_presentation.details.items())
        if self._model_path:
            sections.append(f"模型文件夹：{self._model_path}")
        sections.extend(["", "字幕与输入", f"字幕词典版本：{self.catalog.catalog_id}"])
        if self._catalog_path:
            sections.append(f"字幕文件：{self._catalog_path}")
        if self._source_path:
            sections.append(f"输入来源：{self._source_path}")
        sections.append("录制限制：15 秒 / 600 帧 / 128 MiB，超过上限需重新录制")
        if self._result_details:
            sections.extend(["", "本次识别"])
            sections.extend(f"{key}：{value}" for key, value in self._result_details.items())
        if self._error_detail:
            sections.extend(["", "错误详情", self._error_detail])
        text = "\n".join(sections)
        if self.technical_details.toPlainText() != text:
            self.technical_details.setPlainText(text)

    def _poll(self):
        if self._closed:
            return
        for event in self.session.poll():
            if (event.request_id, event.epoch) != (self.session.request_id, self.session.epoch):
                continue
            try:
                if event.kind == "clear":
                    # A synchronous recording failure queues clear before
                    # raising. Its ERROR state must keep the recovery message.
                    if self.session.state == "ERROR":
                        continue
                    self._clear_caption()
                    if self.session.busy:
                        self.result_title.setText("正在准备识别模型" if self._task_kind == "model" else "正在识别动作")
                elif event.kind == "error":
                    self._error(event.value, "录制已中止，请重新录制" if self._record_started is not None else "识别或输入未能完成")
                    self._record_started = None
                    self._reset_preview("输入未能完成", "重新打开视频或摄像头后再试")
                elif event.kind == "opened":
                    self.status_label.setText("摄像头已就绪，让双手和上半身保持在画面内。" if self.session.source_kind == "camera" else "视频已就绪，点击识别视频获取释义。")
                elif event.kind == "eof":
                    if not self.session.busy and self.caption is None:
                        self.status_label.setText("视频播放结束，仍可点击识别视频。")
                elif event.kind == "result":
                    self._display_result(event.value, event.request_id, event.epoch)
                elif event.kind == "model":
                    self._sync_model_metadata()
                    self.status_label.setText("识别模型已就绪，导入视频或录制一个动作即可开始。")
            except Exception as error:
                self._error(error)
        self._sync_model_metadata()
        packet = self.session.take_preview()
        if packet is not None:
            try:
                self._render_preview(packet)
            except Exception as error:
                self._error(error, "预览未能显示")
        if self._cancel_pending and not self.session.busy:
            self._cancel_pending = False
            self.status_label.setText("已取消识别，可以开始新的动作。")
        self._update_controls()

    def closeEvent(self, event):
        if not self._closed:
            self._closed = True
            self._timer.stop()
            self.settings_dialog.close()
            self.session.close()
        event.accept()

    def on_close(self):
        self.close()


DesktopApp = SignLanguageApp
