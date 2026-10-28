"""Qt workspace for local clip recognition; the session owns native resources."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QFont, QImage, QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
    QFrame, QHBoxLayout, QLabel, QLayout, QListWidget, QListWidgetItem, QMainWindow,
    QPlainTextEdit, QProgressBar, QPushButton, QSizePolicy, QSpinBox,
    QTabWidget, QVBoxLayout, QWidget,
)

from .captions import CaptionCatalog, caption_from_result
from .desktop_presenter import describe_model, describe_result, describe_runtime_model
from .desktop_models import default_language_models, default_unisign_model
from .desktop_widgets import APP_STYLE, BrandMark, PreviewWidget, icon
from .session import RecognitionSession
from .sources import CameraSource, VideoSource


_LANGUAGES = {"csl": "中国手语", "libras": "巴西手语"}


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
    def __init__(self, *, session=None, catalog=None, model=None, video=None, camera=None, runtime_options=None,
                 segment_config=None, language_models=None, unisign_model=None, continuous=False, continuous_config=None,
                 defer_startup_input=False):
        super().__init__()
        self.session = session if session is not None else RecognitionSession(options=runtime_options)
        self.catalog = catalog if catalog is not None else CaptionCatalog()
        self.session.caption_catalog = self.catalog
        self.segment_config = segment_config
        self.continuous_config = continuous_config
        self._continuous_requested = bool(continuous)
        self._continuous_update = None
        self._pending_startup_input = ((video, camera) if (continuous or defer_startup_input) and model
                                     and (video or camera is not None) else None)
        self.language_models = {key: Path(path).resolve() for key, path in
                                (default_language_models() if language_models is None else language_models).items()}
        self.unisign_model = (Path(unisign_model).resolve() if unisign_model is not None
                              else default_unisign_model().resolve())
        self._pending_model_path = None
        self._pending_model_name = None
        self._pending_language = None
        self._history_signature = None
        self._candidate_revision = None
        self.caption = None
        self._last_result = None
        self._animation_origin = {}
        self._metadata_model = None
        self._metadata_catalog = None
        self._catalog_path = None
        self._model_path = None
        self._source_path = None
        self._photo = None
        self._latest_raw_packet = None
        self._preview_packet = None
        self._preview_landmarks = None
        self._closed = False
        self._keypoint_workbench = None
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
        self._render_history(self.session.captions.snapshot())
        self._render_candidates(self.session.candidate_context())
        self._build_settings()
        self._sync_language_selector()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(30)
        if self._pending_startup_input is not None:
            pass  # Keep short startup videos intact while the model warms up.
        elif video:
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
        self.language_label = _label("手语识别 / Ponte de Sinais", "muted")
        title.addWidget(self.language_label)
        header.addLayout(title)
        header.addStretch()
        self.state_badge = _label("等待输入", "statusBadge", wrap=False)
        header.addWidget(self.state_badge)
        self.keypoint_button = _button("关键点工作台", self.open_keypoint_workbench, "video", "ghost")
        self.keypoint_button.setObjectName("openKeypointWorkbench")
        header.addWidget(self.keypoint_button)
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
        self.language_selector_label = _label("识别手语", "muted", wrap=False)
        inputs.addWidget(self.language_selector_label)
        self.language_combo = QComboBox()
        self.language_combo.setObjectName("recognitionLanguage")
        self.language_combo.setAccessibleName("识别手语")
        self.language_combo.addItem("请选择手语", None)
        for language, title in _LANGUAGES.items():
            self.language_combo.addItem(title, language)
        self.language_combo.currentIndexChanged.connect(self._select_language)
        inputs.addWidget(self.language_combo)
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
        display_options = QHBoxLayout()
        self.keypoints_check = QCheckBox("显示关键点")
        self.keypoints_check.setObjectName("showRecognitionKeypoints")
        self.keypoints_check.setToolTip("在当前画面叠加身体与双手骨架；不改变录制或识别输入。")
        self.keypoints_check.toggled.connect(self.toggle_keypoints)
        self.mirror_check.toggled.connect(self._redraw_preview)
        self.keypoints_label = _label("", "muted")
        display_options.addWidget(self.keypoints_check)
        display_options.addWidget(self.keypoints_label,1)
        left.addLayout(display_options)

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
        self.continuous_check = QCheckBox("连续识别视频流（无需停顿）")
        self.continuous_check.setObjectName("continuousRecognition")
        self.continuous_check.toggled.connect(self.toggle_continuous)
        left.addWidget(self.continuous_check)
        self.auto_check = QCheckBox("停顿确认后追加字幕记录")
        self.auto_check.setObjectName("automaticCaptions")
        self.auto_check.toggled.connect(self.toggle_automatic)
        left.addWidget(self.auto_check)
        self.candidate_label = _label("自动端点为可配置启发式，请先用回放或真人验证。", "muted")
        left.addWidget(self.candidate_label)
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
        self.animation_button = _button("在动画工作台核对", self.open_animation)
        self.animation_button.setToolTip("打开本机动画工作台；识别候选不会自动播放")
        result_layout.addWidget(self.animation_button)
        self.details_button = _button("详细信息", lambda: self.open_settings(details=True), "info", "ghost")
        self.details_button.setProperty("size", "small")
        result_layout.addWidget(self.details_button, 0, Qt.AlignmentFlag.AlignRight)
        right.addWidget(result_panel)

        history_panel, history_layout = _panel()
        history_layout.addWidget(_label("字幕记录", "sectionTitle"))
        self.caption_history = QListWidget()
        self.caption_history.setObjectName("captionHistory")
        self.caption_history.setMinimumHeight(80)
        history_layout.addWidget(self.caption_history)
        history_actions = QHBoxLayout()
        self.undo_caption_button = _button("撤销上一条", self.undo_caption)
        self.clear_history_button = _button("清空记录", self.clear_caption_history)
        history_actions.addWidget(self.undo_caption_button)
        history_actions.addWidget(self.clear_history_button)
        history_layout.addLayout(history_actions)

        candidates_panel, candidates_layout = _panel()
        candidates_layout.setContentsMargins(10, 10, 10, 10)
        candidates_layout.setSpacing(6)
        candidates_header = QHBoxLayout()
        candidates_header.addWidget(_label("保留", "muted", wrap=False))
        self.candidate_limit = QSpinBox()
        self.candidate_limit.setRange(1, 1000)
        self.candidate_limit.setValue(200)
        self.candidate_limit.setSuffix(" 条")
        self.candidate_limit.setAccessibleName("候选保留条数")
        self.candidate_limit.setStyleSheet("QSpinBox { padding: 4px 8px; }")
        self.candidate_limit.valueChanged.connect(self.set_candidate_limit)
        candidates_header.addWidget(self.candidate_limit)
        self.candidate_status = _label("", "muted", wrap=False)
        self.candidate_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        candidates_header.addWidget(self.candidate_status, 1)
        candidates_layout.addLayout(candidates_header)
        candidates_layout.addWidget(_label("未确认 · 重叠窗口可能重复", "muted"))
        self.candidate_history = QListWidget()
        self.candidate_history.setObjectName("candidateHistory")
        self.candidate_history.setMinimumHeight(96)
        self.candidate_history.setWordWrap(False)
        self.candidate_history.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.candidate_history.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        candidates_layout.addWidget(self.candidate_history, 1)
        candidates_actions = QHBoxLayout()
        self.clear_candidates_button = _button("清空候选", self.clear_candidate_history)
        self.export_candidates_button = _button("导出上下文", self.export_candidate_history)
        for action in (self.clear_candidates_button, self.export_candidates_button):
            action.setProperty("size", "small")
        candidates_actions.addWidget(self.clear_candidates_button)
        candidates_actions.addWidget(self.export_candidates_button)
        candidates_layout.addLayout(candidates_actions)

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
        self.output_tabs = QTabWidget()
        self.output_tabs.addTab(candidates_panel, "候选序列")
        self.output_tabs.addTab(history_panel, "字幕记录")
        self.output_tabs.addTab(vocabulary_panel, "支持的手语词")
        right.addWidget(self.output_tabs, 1)
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
        self.unisign_button = _button("加载 Uni-Sign（实验）", self.load_unisign_model, "reload")
        self.unisign_button.setObjectName("loadUniSign")
        self.unisign_button.setToolTip("CSL-Daily Pose-only 翻译权重，CC BY-NC 4.0；仅用于非商业研究。")
        form.addWidget(self.unisign_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.unisign_label = _label("", "muted")
        form.addWidget(self.unisign_label)
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

    def open_keypoint_workbench(self):
        if self._keypoint_workbench is None or self._keypoint_workbench._closed:
            self.stop_source()
            from .keypoint_desktop import KeypointWorkbench
            self._keypoint_workbench = KeypointWorkbench(self)
        self._keypoint_workbench.show()
        self._keypoint_workbench.raise_()
        self._keypoint_workbench.activateWindow()

    def _select_camera_index(self, index):
        item = self.camera_combo.findData(index)
        if item < 0:
            self.camera_combo.addItem(f"摄像头 {index + 1}", index)
            item = self.camera_combo.count() - 1
        self.camera_combo.setCurrentIndex(item)

    def _clear_caption(self):
        self._continuous_update = None
        self.caption = None
        self._last_result = None
        self._result_details = {}
        self.result_title.setText("等待一段完整手语")
        self.zh_label.setText("等待识别")
        self.zh_label.setProperty("state", "")
        self.zh_label.style().polish(self.zh_label)
        self.pt_label.setText("")
        self.notice_label.setText("识别结果与可用释义会显示在这里。")
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
        self._latest_raw_packet = self._preview_packet = self._preview_landmarks = None
        self.preview.set_placeholder(title, hint)
        self.keypoints_label.setText("等待画面…" if self.session.show_keypoints else "")

    def _choose_model(self):
        path = QFileDialog.getExistingDirectory(self, "选择识别模型文件夹")
        if path:
            self.load_model(path)

    def _sync_language_selector(self):
        """The selector describes the committed model, never the pending target."""
        capabilities = self.session.model_manager.capabilities()
        bundle = getattr(self.session.recognizer, "bundle", None)
        language = (capabilities["sign_language"] if capabilities is not None else
                    getattr(getattr(bundle, "manifest", None), "sign_language", None))
        path = getattr(bundle, "path", None)
        is_unisign = bool(capabilities and capabilities.get("model_id", "").startswith("uni-sign-"))
        if language in _LANGUAGES and path is not None and not is_unisign:
            self.language_models[language] = Path(path).resolve()
        self.language_combo.blockSignals(True)
        try:
            self.language_combo.setItemText(0, "其他手语模型" if self.session.recognizer is not None else "请选择手语")
            self.language_combo.model().item(0).setEnabled(False)
            for key, title in _LANGUAGES.items():
                index = self.language_combo.findData(key)
                model_path = self.language_models.get(key)
                available = model_path is not None and (model_path / "manifest.json").is_file()
                if key == language and is_unisign:
                    label = f"{title} · Uni-Sign（实验）"
                else:
                    label = title if available or key == language else f"{title}（未配置）"
                self.language_combo.setItemText(index, label)
                self.language_combo.model().item(index).setEnabled(available or key == language)
                if key == language and is_unisign:
                    tooltip = "当前为 Uni-Sign Pose-only 实验模型；不会自动设为默认模型。"
                elif available:
                    tooltip = "切换识别模型；完成一段动作后识别。"
                else:
                    tooltip = "模型未配置，可在设置中选择识别模型文件夹。"
                self.language_combo.setItemData(index, tooltip, Qt.ItemDataRole.ToolTipRole)
            index = self.language_combo.findData(language) if language in _LANGUAGES else 0
            self.language_combo.setCurrentIndex(index)
        finally:
            self.language_combo.blockSignals(False)

    def _select_language(self, index):
        language = self.language_combo.itemData(index)
        self._sync_language_selector()
        if (language not in _LANGUAGES or language == self.language_combo.currentData()
                or self.session.busy or self._pending_model_path is not None or self.session.state == "RECORDING"):
            return
        path = self.language_models.get(language)
        if path is None or not (path / "manifest.json").is_file():
            self.status_label.setText("该手语模型未配置，请在设置中选择识别模型。")
            return
        self.load_model(path)

    def load_unisign_model(self):
        path = self.unisign_model
        if not (path / "manifest.json").is_file():
            self._error(FileNotFoundError(f"Uni-Sign 实验模型未安装：{path}"), "无法加载 Uni-Sign")
            return
        self.load_model(path, display_name="Uni-Sign Pose-only")

    def load_model(self, path, display_name=None):
        self._task_kind = "model"
        self._pending_model_name = display_name
        if not self._call(lambda: self.session.load_model(path), context="无法加载识别模型"):
            self._pending_model_name = None
            self._pending_language = None
            self._update_controls()
            return
        self._pending_model_path = Path(path).resolve()
        self._pending_language = None if display_name else next(
            (key for key, value in self.language_models.items()
             if value == self._pending_model_path and key in _LANGUAGES), None)
        title = display_name or _LANGUAGES.get(self._pending_language, "识别模型")
        self.status_label.setText(f"正在准备{title}，预览保持开启…")
        self.result_title.setText("正在准备识别模型")
        self._update_controls()

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
            self.session.caption_catalog = catalog
            self._catalog_path = Path(path)
            draft = any(entry.review_status == "draft" for entry in catalog.labels.values())
            self.catalog_label.setText("字幕释义已加载" + ("，中文草案待核验。" if draft else "。"))
            self._metadata_catalog = None
            self._sync_model_metadata()
            if self._last_result is not None:
                if self._continuous_update is not None:
                    self._display_continuous(self._continuous_update, self.session.request_id, self.session.epoch)
                else:
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
        self._continuous_requested = False
        self._record_started = None
        kind = "录制" if self.session.state == "RECORDING" else "识别"
        self._call(self.session.cancel)
        self._cancel_pending = self.session.busy
        self.status_label.setText("已取消识别，正在等待当前任务结束…" if self._cancel_pending else f"已清空结果，可以开始新的{kind}。")

    def toggle_continuous(self, enabled):
        self._continuous_requested = False
        operation = (lambda: self.session.enable_continuous(self.continuous_config)) if enabled else self.session.disable_continuous
        if self._call(operation, context="连续识别未能启动"):
            if enabled:
                self.output_tabs.setCurrentWidget(self.candidate_history.parentWidget())
            self.status_label.setText("正在积累画面，结果会随动作更新。" if enabled else "连续识别已停止。")
        self._update_controls()

    def toggle_keypoints(self, enabled):
        self.session.set_show_keypoints(enabled)
        self._preview_landmarks = None
        self.keypoints_label.setText("正在检测身体与双手…" if enabled else "")
        if self._latest_raw_packet is not None:
            self._render_preview(self._latest_raw_packet)
        self._update_controls()

    def _redraw_preview(self, *_):
        if self._preview_packet is not None:
            self._render_preview(self._preview_packet,self._preview_landmarks)

    def toggle_automatic(self, enabled):
        try:
            if enabled:
                self.session.enable_automatic(self.segment_config)
                self.output_tabs.setCurrentWidget(self.caption_history.parentWidget())
                self.status_label.setText("自动字幕已启动，完成一个动作后稍作停顿。")
            else:
                self.session.disable_automatic()
            self._update_controls()
        except Exception as error:
            self.auto_check.blockSignals(True)
            self.auto_check.setChecked(False)
            self.auto_check.blockSignals(False)
            self._error(error, "自动字幕未能启动，可使用手动录制")

    def undo_caption(self):
        self.session.undo_caption()
        self._render_history(self.session.captions.snapshot())

    def clear_caption_history(self):
        self.session.clear_history()
        self._clear_caption()
        self._render_history(self.session.captions.snapshot())

    def _render_history(self, snapshot):
        self._history_signature = (snapshot["revision"], tuple(e["event_id"] for e in snapshot["events"]),
                                   len(snapshot["candidates"]))
        self.caption_history.clear()
        for event in snapshot["events"]:
            caption = event["caption"]
            item = QListWidgetItem(f"{caption['zh']} / {caption['pt']}" +
                                   (f"\n{caption['notice']}" if caption['notice'] else ""))
            item.setData(Qt.ItemDataRole.UserRole, event["event_id"])
            item.setToolTip(f"{event['start_time']:.3f}–{event['end_time']:.3f}s\n"
                            f"{event['source_id']}\n{event['model_id']} generation {event['model_generation']}")
            self.caption_history.addItem(item)
        self.caption_history.scrollToBottom()
        has_events = bool(snapshot["events"])
        self.undo_caption_button.setEnabled(has_events)
        self.clear_history_button.setEnabled(has_events or bool(snapshot["candidates"]))

    def set_candidate_limit(self, limit):
        self.session.set_candidate_limit(limit)
        self._render_candidates(self.session.candidate_context())

    def clear_candidate_history(self):
        self.session.clear_candidates()
        self._render_candidates(self.session.candidate_context())

    def export_candidate_history(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出候选上下文", "recognition-context.json", "JSON (*.json)")
        if not path:
            return
        try:
            snapshot = self.session.export_candidate_context(path)
            self.candidate_status.setText(f"已导出 {len(snapshot['candidates'])} 条")
            self.candidate_status.setToolTip("JSON 上下文已导出；LLM 尚未接入。")
        except Exception as error:
            self.candidate_status.setText("导出失败")
            self.candidate_status.setToolTip("候选已保留，可重新选择位置。\n"+str(error))

    def _render_candidates(self, snapshot):
        self._candidate_revision = snapshot['revision']
        blocked = self.candidate_limit.blockSignals(True)
        self.candidate_limit.setValue(snapshot['max_history'])
        self.candidate_limit.blockSignals(blocked)
        scroll = self.candidate_history.verticalScrollBar()
        at_bottom, position = scroll.value() >= scroll.maximum()-2, scroll.value()
        selected = self.candidate_history.currentItem()
        selected_id = selected.data(Qt.ItemDataRole.UserRole)['candidate_id'] if selected else None
        self.candidate_history.clear()
        for row in snapshot['candidates']:
            language = _LANGUAGES.get(row['sign_language'], row['sign_language'] or '未知手语')
            overlap = ' · 与前窗重叠' if row['overlaps_candidate_id'] else ''
            caption = row['caption']
            text = (f"#{row['sequence']} {caption['zh']} / {caption['pt']}\n"
                    f"{language} · {row['start_time']:.2f}–{row['end_time']:.2f}s · 未确认")
            prediction = row['result']['prediction']
            details = ''
            if row['result']['task_type'] == 'classification':
                scores = sorted(prediction['scores'].items(), key=lambda item: (-item[1], item[0]))[:3]
                details = '\n模型备选：' + ' / '.join(f'{label} {score:.3f}' for label, score in scores)
            elif row['result']['task_type'] == 'gloss_sequence':
                details = '\n词序列：' + ' / '.join(token['gloss'] for token in prediction['tokens'])
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, row)
            item.setToolTip(f"{caption['zh']} / {caption['pt']}{details}\n"
                            f"来源 {row['source_epoch']}{overlap}\n"
                            f"{row['model_id']} · {row['model_version']} · generation {row['model_generation']}\n"
                            f"{row['source_id']} · window revision {row['window_revision']}\n"
                            f"{row['request_id']}\n分数类型：{row['result']['score_type']}\n"
                            f"词典：{row['catalog_id']} · {row['catalog_review_status']}\n"
                            "导出包含完整原始分数/词级信息；候选不代表确认字幕。")
            self.candidate_history.addItem(item)
            if row['candidate_id'] == selected_id:
                self.candidate_history.setCurrentItem(item)
        if at_bottom:
            self.candidate_history.scrollToBottom()
        else:
            scroll.setValue(position)
        count = len(snapshot['candidates'])
        self.candidate_status.setText(f"{count} 条 · 淘汰 {snapshot['dropped_count']}")
        self.candidate_status.setToolTip(f"已保留 {count}/{snapshot['max_history']} 条，累计淘汰 {snapshot['dropped_count']} 条。\n"
            "记录保存在当前会话内；退出前可导出 JSON，供后续 LLM 读取。")
        self.clear_candidates_button.setEnabled(bool(count))
        self.export_candidates_button.setEnabled(bool(count))

    def toggle_pause(self):
        if self.session.source_kind != "video":
            return
        if not self.session.source_active:
            self.open_video(self.session.video_path)
        else:
            self.session.pause_source(not self.session.paused)
            self.status_label.setText("视频预览已暂停，仍可识别完整视频。" if self.session.paused else "视频预览继续播放。")
        self._update_controls()

    def _update_controls(self):
        busy = self.session.busy
        state = self.session.state
        camera_mode = self.session.source_kind == "camera"
        video = self.session.video_path is not None
        loaded = self.session.recognizer is not None
        recording = state == "RECORDING"
        automatic = self.session.automatic
        continuous = self.session.continuous
        self.language_combo.setEnabled(not busy and not recording and not continuous and self._pending_model_path is None)
        self.language_combo.setToolTip("停止连续识别后可切换手语。" if continuous else "录制或识别期间不能切换手语。" if busy or recording else
                                      "选择要识别的手语；系统不会自动判断手语种类。")
        self.video_button.setChecked(self.session.source_kind == "video")
        self.camera_button.setChecked(camera_mode)
        self.model_button.setEnabled(not busy and not recording and not continuous)
        unisign_ready = (self.unisign_model / "manifest.json").is_file()
        self.unisign_button.setEnabled(not busy and not recording and not continuous and unisign_ready)
        self.unisign_label.setText(("实验模型已安装：" if unisign_ready else "未安装实验模型；先运行 scripts/download_uni_sign.py") + str(self.unisign_model))
        self.unisign_label.setToolTip("Uni-Sign Pose-only 仅在独立 PyTorch 环境中运行；权重 CC BY-NC 4.0。")
        self.rollback_model_button.setEnabled(not busy and not recording and not continuous and
                                              self.session.model_manager.status()["previous"] is not None)
        self.reload_catalog_button.setEnabled(self._catalog_path is not None and self._catalog_path.is_file())
        self.recognize_button.setVisible(not camera_mode)
        self.recognize_button.setEnabled(video and loaded and not busy and not continuous)
        self.pause_button.setVisible(video)
        self.pause_button.setEnabled(video and (self.session.source_active or not busy))
        self.pause_button.setText("重新播放" if video and not self.session.source_active else "继续预览" if self.session.paused else "暂停预览")
        self.record_button.setVisible(camera_mode and not recording)
        self.record_button.setEnabled(camera_mode and self.session.source_active and loaded and not busy and not automatic and not continuous)
        self.finish_button.setVisible(camera_mode and recording)
        self.finish_button.setEnabled(recording and not busy)
        self.camera_options.setVisible(camera_mode)
        self.stop_button.setEnabled(self.session.source is not None or video)
        self.stop_button.setText("关闭摄像头" if camera_mode else "关闭视频" if video else "关闭")
        self.cancel_button.setText("取消录制" if recording else "取消识别" if busy else "清空结果")
        self.cancel_button.setEnabled(recording or busy or self.caption is not None or state == "ERROR")
        self.busy_progress.setVisible(busy and not continuous)
        self.record_progress.setVisible(recording)
        self.record_time_label.setVisible(recording)
        self.auto_check.blockSignals(True)
        self.auto_check.setChecked(automatic)
        self.auto_check.blockSignals(False)
        self.auto_check.setEnabled(automatic or (self.session.source_active and loaded and not busy and not recording))
        self.continuous_check.blockSignals(True)
        self.continuous_check.setChecked(continuous)
        self.continuous_check.blockSignals(False)
        self.continuous_check.setEnabled(continuous or (self.session.source_active and loaded and not busy and not recording))
        self.keypoints_check.blockSignals(True)
        self.keypoints_check.setChecked(self.session.show_keypoints)
        self.keypoints_check.blockSignals(False)
        if recording and self._record_started is not None:
            elapsed = max(0, time.perf_counter() - self._record_started)
            self.record_time_label.setText(f"正在录制  {elapsed:.1f} 秒 / 15 秒")
            self.record_progress.setValue(min(150, int(elapsed * 10)))
        capabilities = self.session.model_manager.capabilities()
        task_type = capabilities.get("task_type") if capabilities is not None else getattr(
            getattr(getattr(self.session.recognizer, "bundle", None), "manifest", None), "task_type", None)
        text_model = task_type == "text"
        sequence = task_type == "gloss_sequence"
        unit = "句子" if text_model else "表达" if sequence else "动作"
        if continuous and text_model:
            hint = "持续处理重叠窗口并更新中文句子候选；多窗口一致仍不等于人工确认。"
        elif continuous:
            hint = "持续处理最新画面，无需停顿；停止连续识别后可切换手语。"
        elif automatic:
            hint = "完成一个词或固定表达后稍作停顿；端点未经真人质量验证。"
        elif camera_mode:
            hint = f"每次录制一个完整{unit}，最长 15 秒。"
        elif video:
            hint = "识别整段视频，播放位置不影响结果。"
        else:
            hint = f"每次识别一个完整{unit}。"
        self.input_hint.setText(hint)
        loading = (f"正在准备{self._pending_model_name}" if self._pending_model_name else
                   f"正在准备{_LANGUAGES[self._pending_language]}" if self._pending_language in _LANGUAGES
                   else "正在准备模型")
        badge = "连续识别中" if continuous else "正在录制" if recording else loading if busy and self._task_kind == "model" else "正在识别" if busy else "请重试" if state == "ERROR" else "模型已就绪" if loaded else "请选择识别模型"
        self.state_badge.setText(badge)
        self._refresh_details()

    def open_animation(self):
        import os
        from .animation.handoff import handoff_url
        try:
            url = handoff_url(self._last_result, **self._animation_origin,
                              base_url=os.environ.get("SLBRIDGE_ANIMATION_URL",
                                                      "http://127.0.0.1:8874/animation/"))
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        if QDesktopServices.openUrl(QUrl(url)):
            self.status_label.setText("已打开动画工作台，请核对输入后预览；工作台服务需先启动。")
        else:
            self.status_label.setText("无法打开动画工作台，请先启动 slbridge-animation 服务。")

    def _display_result(self, result, request_id, epoch):
        self._continuous_update = None
        self._last_result = result
        self._animation_origin = {"request_id": request_id, "source_epoch": epoch}
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

    def _display_continuous(self, update, request_id, epoch):
        limit = self.session.continuous_status()['max_result_age_seconds']
        age = update.age_seconds+time.perf_counter()-update.completed_at
        if not self.session.continuous or limit is None or age > limit:
            self._clear_caption()
            self.status_label.setText("当前窗口结果已过期，等待最新画面。")
            return
        result = update.response.result
        self._display_result(result, request_id, epoch)
        self._continuous_update = update
        stability = result.quality.get("stability_state")
        if result.task_type == "text" and stability == "confirmed":
            self.result_title.setText("连续识别 · 多窗口一致候选")
            self.status_label.setText("连续窗口输出一致，但仍是机器候选，尚未人工确认。")
        elif result.task_type == "text":
            self.result_title.setText("连续识别 · 临时候选")
            self.status_label.setText("中文句子随窗口更新；当前窗口不会写入已确认字幕记录。")
        else:
            self.result_title.setText("连续识别 · 当前窗口结果")
            self.status_label.setText("结果随新画面修正；当前窗口不会重复写入字幕记录。")
        self.candidate_label.setText(f"最近 {update.end_time-update.start_time:.1f} 秒画面 · 持续更新")
        self._result_details['窗口结束至推理返回'] = f'{update.age_seconds*1000:.0f} ms'
        if result.quality.get("pose_cache_hit_frames") is not None:
            self._result_details['姿态缓存命中'] = (f"{result.quality['pose_cache_hit_frames']} 帧 / "
                                                f"新提取 {result.quality.get('pose_new_frames', 0)} 帧")

    def _render_preview(self, packet, landmarks=None):
        self._preview_packet = packet
        self._preview_landmarks = landmarks
        rgb = packet.image[:, :, ::-1]
        mirrored = self.session.source_kind == "camera" and self.mirror_check.isChecked()
        if mirrored:
            rgb = rgb[:, ::-1]
        pixels = np.ascontiguousarray(rgb)
        height, width = pixels.shape[:2]
        self._photo = QImage(pixels.data, width, height, pixels.strides[0], QImage.Format.Format_RGB888).copy()
        self.preview.set_image(self._photo,landmarks=landmarks,mirrored=mirrored)
        self.preview.set_overlay("正在录制" if self.session.state == "RECORDING" else "摄像头预览" if self.session.source_kind == "camera" else "视频预览")

    def _sync_model_metadata(self):
        model = self.session.recognizer
        if model is self._metadata_model and self.catalog is self._metadata_catalog:
            return
        self._sync_language_selector()
        capabilities = self.session.model_manager.capabilities()
        if self.session.model_manager.supports_typed_inference:
            presentation = describe_runtime_model(self.session.model_manager.describe(),
                                                   capabilities, self.catalog)
        else:
            presentation = describe_model(model, self.catalog)
        runtime_metadata = getattr(model, "runtime_info", {})
        if runtime_metadata.get("license"):
            presentation.details["许可"] = str(runtime_metadata["license"]) + "；仅非商业研究"
        if runtime_metadata.get("pose_fallback_reason"):
            presentation.details["RTMPose 设备说明"] = str(runtime_metadata["pose_fallback_reason"])
        if runtime_metadata.get("source_commit"):
            presentation.details["上游源码提交"] = str(runtime_metadata["source_commit"])
        self.language_label.setText(presentation.title)
        bundle_path = getattr(getattr(model, "bundle", None), "path", None)
        self._model_path = str(bundle_path) if bundle_path is not None else None
        self.model_presentation = presentation
        self.model_label.setText(presentation.summary)
        self.settings_model_label.setText(presentation.summary)
        task = capabilities.get("task_type") if capabilities is not None else getattr(
            getattr(getattr(model, "bundle", None), "manifest", None), "task_type", None)
        unit = "句子" if task == "text" else "词表项" if task == "gloss_sequence" else "词"
        self.word_count_label.setText(f"{len(presentation.words)} 个{unit}" if presentation.words else "")
        self.vocabulary_list.clear()
        for word in presentation.words:
            item = QListWidgetItem(f"{word.zh}  /  {word.pt}")
            item.setToolTip("中文释义待核验" if word.draft else "")
            self.vocabulary_list.addItem(item)
        if not presentation.words:
            item = QListWidgetItem("Uni-Sign 直接输出中文句子" if task == "text" else "加载模型后显示词汇")
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
                    model_error = event.task_kind == "model"
                    continuous_error = event.task_kind == "continuous"
                    self._error(event.value, "连续识别已停止" if continuous_error else "模型切换失败" if model_error else
                                "录制已中止，请重新录制" if self._record_started is not None else "识别或输入未能完成")
                    self._record_started = None
                    if model_error:
                        self._pending_model_path = None
                        self._pending_language = None
                        self._pending_model_name = None
                        self._task_kind = None
                        self._sync_language_selector()
                        if self.session.recognizer is not None:
                            self.status_label.setText("切换失败，已保留原模型；可重试或在设置中选择模型。")
                    elif not continuous_error:
                        self._reset_preview("输入未能完成", "重新打开视频或摄像头后再试")
                elif event.kind == "opened":
                    self.status_label.setText("摄像头已就绪，让双手和上半身保持在画面内。" if self.session.source_kind == "camera" else "视频已就绪，点击识别视频获取释义。")
                elif event.kind == "eof":
                    if not self.session.busy and self.caption is None:
                        self.status_label.setText("视频播放结束，仍可点击识别视频。")
                elif event.kind == "result":
                    self._display_result(event.value, event.request_id, event.epoch)
                elif event.kind == "auto_result":
                    self._display_result(event.value.result, event.request_id, event.epoch)
                elif event.kind == "continuous_clear":
                    self._clear_caption()
                elif event.kind == "keypoints_error":
                    self._preview_landmarks = None
                    self.keypoints_label.setText(event.value)
                    if self._latest_raw_packet is not None:
                        self._render_preview(self._latest_raw_packet)
                elif event.kind == "continuous_status":
                    self.status_label.setText(event.value)
                    self.candidate_label.setText(event.value)
                elif event.kind == "continuous_result":
                    self._display_continuous(event.value, event.request_id, event.epoch)
                elif event.kind == "captions":
                    self._render_history(event.value)
                elif event.kind == "candidate":
                    labels = {"WAITING": "等待动作", "START_CANDIDATE": "确认动作起点",
                              "CAPTURING": "动作采集中", "END_CANDIDATE": "等待停顿确认",
                              "REARM": "等待新的动作起点"}
                    message = labels.get(event.value["state"], event.value["state"])
                    if event.value.get("reason"):
                        message += " · " + event.value["reason"]
                    self.candidate_label.setText(message)
                elif event.kind == "automatic":
                    self.status_label.setText(event.value)
                elif event.kind == "model":
                    self._pending_model_path = None
                    self._pending_language = None
                    self._pending_model_name = None
                    self._task_kind = None
                    self._clear_caption()
                    self._sync_model_metadata()
                    self.status_label.setText("识别模型已就绪，导入视频或录制一个动作即可开始。")
            except Exception as error:
                self._error(error)
        self._sync_model_metadata()
        snapshot = self.session.captions.snapshot()
        signature = (snapshot["revision"], tuple(e["event_id"] for e in snapshot["events"]),
                     len(snapshot["candidates"]))
        if signature != self._history_signature:
            self._render_history(snapshot)
        candidate_snapshot = self.session.candidate_context(after_revision=self._candidate_revision)
        if candidate_snapshot is not None:
            self._render_candidates(candidate_snapshot)
        packet = self.session.take_preview()
        if packet is not None:
            self._latest_raw_packet = packet
        pair = self.session.take_landmark_preview() if self.session.show_keypoints else None
        if self.session.show_keypoints:
            from .preview_landmarks import MAX_PREVIEW_AGE
            if pair is not None:
                packet,landmarks = pair
            elif self._preview_landmarks is not None and (not self.session.source_active or self.session.paused
                    or self._preview_landmarks.source_epoch!=self.session.epoch
                    or time.perf_counter()-self._preview_landmarks.completed_at>MAX_PREVIEW_AGE):
                self._preview_landmarks = None
                packet,landmarks = self._latest_raw_packet,None
            elif self._preview_landmarks is None:
                landmarks = None
            else:
                packet = None
        else:
            landmarks = None
        if packet is not None:
            try:
                self._render_preview(packet,landmarks)
                if self.session.show_keypoints:
                    self.keypoints_label.setText("身体与双手关键点" if landmarks is not None and any(landmarks.point_observed)
                        else "等待检测到身体或双手…")
            except Exception as error:
                self._error(error, "预览未能显示")
        if self._cancel_pending and not self.session.busy:
            self._cancel_pending = False
            self.status_label.setText("已取消识别，可以开始新的动作。")
        if self._pending_model_path is not None and not self.session.busy:
            # Reconcile cancelled/source-invalidated loads after terminal events,
            # including a loader that failed before the first Qt poll.
            self._pending_model_path = None
            self._pending_language = None
            self._pending_model_name = None
            self._task_kind = None
            self._sync_language_selector()
        if self._continuous_update is not None:
            update = self._continuous_update
            limit = self.session.continuous_status()['max_result_age_seconds']
            if limit is None or update.age_seconds+time.perf_counter()-update.completed_at > limit:
                self._clear_caption()
                self.status_label.setText("当前窗口结果已过期，等待最新画面。")
        if (self._pending_startup_input is not None and not self.session.busy
                and self.session.recognizer is not None):
            video, camera = self._pending_startup_input
            self._pending_startup_input = None
            self.open_video(video) if video else self.open_camera(camera)
        if (self._continuous_requested and not self.session.busy and self.session.recognizer is not None
                and self.session.source_active and self.session.state != "RECORDING"):
            self.toggle_continuous(True)
        self._update_controls()

    def closeEvent(self, event):
        if not self._closed:
            self._closed = True
            self._timer.stop()
            self.settings_dialog.close()
            if self._keypoint_workbench is not None:
                self._keypoint_workbench.close()
            self.session.close()
        event.accept()

    def on_close(self):
        self.close()


DesktopApp = SignLanguageApp
