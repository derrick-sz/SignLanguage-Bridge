"""A capture and replay workspace for landmark samples, without recognition."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QSpinBox,
    QVBoxLayout, QWidget,
)

from .desktop_widgets import APP_STYLE
from .sources import CameraSource, VideoSource


_POSE_EDGES = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23),
    (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31),
    (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
)
_HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7),
    (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13),
    (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
    (17, 18), (18, 19), (19, 20),
)


def _label(text, name=""):
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    if name:
        label.setObjectName(name)
    return label


def _button(text, callback, name, *, primary=False):
    button = QPushButton(text)
    button.setObjectName(name)
    button.setProperty("kind", "primary" if primary else "secondary")
    button.clicked.connect(lambda: callback())
    return button


def _identity(frame):
    return frame.source_id, frame.frame_index


class KeypointPreview(QWidget):
    """Render shared image coordinates; never use body-normalized coordinates."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.packet = None
        self.landmarks = None
        self.skeleton_only = False
        self.setMinimumSize(320, 220)
        self.setAccessibleName("关键点与原始画面预览")

    def sizeHint(self):
        return QSize(800, 400)

    def clear(self):
        self.image = self.packet = self.landmarks = None
        self.update()

    def set_packet(self, packet):
        pixels = np.ascontiguousarray(packet.image[..., ::-1])
        self.image = QImage(pixels.data, pixels.shape[1], pixels.shape[0],
                            pixels.strides[0], QImage.Format.Format_RGB888).copy()
        self.packet = packet
        self.update()

    def set_landmarks(self, frame):
        self.landmarks = frame
        self.update()

    def set_skeleton_only(self, enabled):
        self.skeleton_only = enabled
        self.update()

    def _target(self):
        if self.image is not None and not self.skeleton_only:
            width, height = self.image.width(), self.image.height()
        elif self.landmarks is not None:
            width, height = self.landmarks.image_size
        else:
            return QRectF(self.rect())
        scale = min(self.width() / width, self.height() / height)
        return QRectF((self.width() - width * scale) / 2,
                      (self.height() - height * scale) / 2,
                      width * scale, height * scale)

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), QColor("#E6EDF2"))
            target = self._target()
            painter.setClipRect(target)
            if self.image is not None and not self.skeleton_only:
                painter.drawImage(target, self.image)
            frame = self.landmarks
            matches = self.packet is not None and frame is not None and _identity(self.packet) == _identity(frame)
            if frame is not None and (self.skeleton_only or matches):
                coords = np.asarray(frame.image_xy)
                valid = np.asarray(frame.point_observed, dtype=bool) & np.isfinite(coords).all(axis=1)
                for offset, edges, color in ((0, _POSE_EDGES, "#118C69"),
                                              (33, _HAND_EDGES, "#008BBA"),
                                              (54, _HAND_EDGES, "#CD7016")):
                    painter.setPen(QPen(QColor(color), 2.4))
                    painter.setBrush(QColor(color))
                    for start, end in edges:
                        start, end = start + offset, end + offset
                        if valid[start] and valid[end]:
                            painter.drawLine(self._point(coords[start], target), self._point(coords[end], target))
                    indices = range(33) if offset == 0 else range(offset, offset + 21)
                    for index in indices:
                        if valid[index]:
                            painter.drawEllipse(self._point(coords[index], target), 3, 3)
            elif self.image is None or self.skeleton_only:
                painter.setPen(QColor("#536A78"))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                                 "导入视频、打开摄像头，或回放关键点录制")
        finally:
            painter.end()

    @staticmethod
    def _point(coordinate, target):
        return QPointF(target.left() + float(coordinate[0]) * target.width(),
                       target.top() + float(coordinate[1]) * target.height())


class KeypointWorkbench(QDialog):
    """Own one keypoint session and leave the recognition workspace untouched."""

    def __init__(self, parent=None, *, session=None):
        super().__init__(parent)
        if session is None:
            from .keypoint_session import KeypointSession
            session = KeypointSession()
        self.session = session
        self._closed = False
        self._saved_path = None
        self._latest_landmarks = None
        self.setWindowTitle("关键点采集与回放 | 手语桥")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setStyleSheet(APP_STYLE)
        self.resize(1040, 830)
        self.setMinimumSize(560, 480)
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(30)
        self._update_controls()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 16)
        outer.addWidget(_label("关键点采集与回放", "title"))
        self.scope_label = _label("仅采集关键点，尚未接入手语识别", "notice")
        outer.addWidget(self.scope_label)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        body = QVBoxLayout(content)
        body.setContentsMargins(0, 0, 0, 0)
        inputs = QHBoxLayout()
        self.video_button = _button("导入视频", self._choose_video, "keypointVideo")
        self.camera_button = _button("使用摄像头", self.open_camera, "keypointCamera")
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 32)
        self.camera_index.setPrefix("设备 ")
        self.camera_index.setToolTip("0 为默认摄像头；设备列表不表示已经连接。")
        self.replay_button = _button("打开关键点录制", self._choose_recording, "keypointReplay")
        for widget in (self.video_button, self.camera_button, self.camera_index, self.replay_button):
            inputs.addWidget(widget)
        body.addLayout(inputs)
        self.source_label = _label("等待输入", "sourceTitle")
        body.addWidget(self.source_label)
        self.preview = KeypointPreview()
        body.addWidget(self.preview, 1)
        transport = QHBoxLayout()
        self.pause_button = _button("暂停", self.toggle_pause, "keypointPause")
        self.stop_button = _button("停止输入", self.stop_source, "keypointStop")
        self.reset_button = _button("重置跟踪", self.reset_tracking, "keypointReset")
        self.skeleton_only_check = QCheckBox("仅显示骨架")
        self.skeleton_only_check.toggled.connect(self.preview.set_skeleton_only)
        for widget in (self.pause_button, self.stop_button, self.reset_button, self.skeleton_only_check):
            transport.addWidget(widget)
        transport.addStretch()
        body.addLayout(transport)
        self.quality_label = _label("左手 —/21 · 右手 —/21 · 肩锚 —", "muted")
        self.metrics_label = _label("等待关键点结果", "muted")
        body.addWidget(self.quality_label)
        body.addWidget(self.metrics_label)
        sample = QFrame()
        sample.setObjectName("panel")
        sample_layout = QVBoxLayout(sample)
        sample_layout.addWidget(_label("录制关键点样本", "sectionTitle"))
        form = QFormLayout()
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("可留空；例如样本编号或待核验词条")
        self.language_combo = QComboBox()
        self.language_combo.addItem("未标注", None)
        self.language_combo.addItem("Libras / 巴西手语", "libras")
        self.language_combo.addItem("中国手语", "csl")
        self.signer_edit = QLineEdit()
        self.signer_edit.setPlaceholderText("可留空；建议使用编号")
        form.addRow("样本标签（可选）", self.label_edit)
        form.addRow("手语（可选）", self.language_combo)
        form.addRow("示范者编号（可选）", self.signer_edit)
        sample_layout.addLayout(form)
        actions = QHBoxLayout()
        self.record_button = _button("开始录制关键点", self.start_recording, "keypointRecord", primary=True)
        self.finish_button = _button("完成并保存", self.finish_recording, "keypointSave", primary=True)
        self.cancel_button = _button("取消录制", self.cancel_recording, "keypointCancel")
        for widget in (self.record_button, self.finish_button, self.cancel_button):
            actions.addWidget(widget)
        sample_layout.addLayout(actions)
        self.recording_label = _label("尚未录制", "muted")
        sample_layout.addWidget(self.recording_label)
        sample_layout.addWidget(_label("保存关键点及采集信息，不保存原始视频。样本标签仅为待核验备注。关闭工作台会取消未完成录制。", "notice"))
        body.addWidget(sample)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        self.status_label = _label("选择输入开始采集；录制文件可直接回放骨架。", "notice")
        outer.addWidget(self.status_label)
        self.done_button = _button("关闭工作台", self.close, "keypointClose")
        outer.addWidget(self.done_button, 0, Qt.AlignmentFlag.AlignRight)

    def _run(self, operation, success=None):
        try:
            value = operation()
            if success:
                self.status_label.setText(success)
            self._update_controls()
            return True, value
        except Exception as error:
            self.status_label.setText("操作未完成：" + str(error))
            if not self.session.status().get("recording") and self.recording_label.text().startswith("录制中"):
                self.recording_label.setText("录制未完成，未保存本次样本")
            self._update_controls()
            return False, None

    def _clear_preview(self):
        self.preview.clear()
        self._latest_landmarks = None
        self.quality_label.setText("左手 —/21 · 右手 —/21 · 肩锚 —")

    def _choose_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入采集视频", "", "视频 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)")
        if path:
            self.open_video(path)

    def open_video(self, path):
        ok, _ = self._run(lambda: self.session.open_source(VideoSource(path), kind="video", video_path=path), "正在打开视频…")
        if ok:
            self._clear_preview()
            self.source_label.setText("视频：" + Path(path).name)
            self.source_label.setToolTip(str(path))

    def open_camera(self):
        index = self.camera_index.value()
        ok, _ = self._run(lambda: self.session.open_source(CameraSource(index), kind="camera"), "正在打开摄像头…")
        if ok:
            self._clear_preview()
            self.source_label.setText(f"摄像头设备 {index}")

    def _choose_recording(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开关键点录制", "", "关键点录制 (*.slkp.jsonl);;所有文件 (*)")
        if path:
            self.open_recording(path)

    def open_recording(self, path):
        ok, _ = self._run(lambda: self.session.open_recording(path), "正在读取关键点录制…")
        if ok:
            self._clear_preview()
            self.source_label.setText("骨架回放：" + Path(path).name)
            self.source_label.setToolTip(str(path))
            self.skeleton_only_check.setChecked(True)

    def toggle_pause(self):
        self._run(self.session.toggle_pause)

    def stop_source(self):
        ok, _ = self._run(self.session.stop_source, "输入已停止，未完成录制已取消。")
        if ok:
            self._clear_preview()
            self.source_label.setText("输入已停止")

    def reset_tracking(self):
        ok, _ = self._run(self.session.reset_tracking, "已请求重置跟踪；新的关键点将重新建立轨迹。")
        if ok:
            self._clear_preview()

    def start_recording(self):
        filename = "keypoints-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".slkp.jsonl"
        path, _ = QFileDialog.getSaveFileName(self, "保存关键点样本", filename, "关键点录制 (*.slkp.jsonl)")
        if not path:
            return
        if not path.lower().endswith(".slkp.jsonl"):
            path += ".slkp.jsonl"
        metadata = {"review_status": "draft"}
        if self.label_edit.text().strip():
            metadata["label_id"] = self.label_edit.text().strip()
        if self.language_combo.currentData():
            metadata["sign_language"] = self.language_combo.currentData()
        if self.signer_edit.text().strip():
            metadata["signer_id"] = self.signer_edit.text().strip()
        self._run(lambda: self.session.start_recording(path, metadata=metadata), "正在录制关键点；点击完成保存。")

    def finish_recording(self):
        ok, path = self._run(self.session.finish_recording)
        if ok:
            self._saved_path = Path(path)
            self.status_label.setText("关键点已保存：" + str(self._saved_path))
            self.recording_label.setText("已保存：" + self._saved_path.name)

    def cancel_recording(self):
        ok, _ = self._run(self.session.cancel_recording, "录制已取消，未保存本次样本。")
        if ok:
            self.recording_label.setText("录制已取消")

    def _update_controls(self):
        status = self.session.status()
        active, recording = bool(status.get("source_active")), bool(status.get("recording"))
        replay = status.get("source_kind") == "recording"
        for button in (self.video_button, self.camera_button, self.replay_button):
            button.setEnabled(not recording)
        self.pause_button.setEnabled(active)
        self.pause_button.setText("继续" if status.get("paused") else "暂停")
        self.stop_button.setEnabled(status.get("source_kind") is not None and not recording)
        self.reset_button.setEnabled(active and not replay)
        self.record_button.setEnabled(active and status.get("processed_frames", 0) > 0
                                      and not status.get("paused") and not replay and not recording)
        self.finish_button.setEnabled(recording)
        self.cancel_button.setEnabled(recording)
        for widget in (self.label_edit, self.language_combo, self.signer_edit):
            widget.setEnabled(not recording)
        self.skeleton_only_check.setEnabled(not replay)
        if replay:
            self.skeleton_only_check.setChecked(True)
        if recording:
            self.recording_label.setText(f"录制中 · {status.get('recorded_frames', 0)} 帧")
        self.metrics_label.setText(
            f"已处理 {status.get('processed_frames', 0)} 帧 · 丢弃 {status.get('dropped_frames', 0)} 帧"
            f" · 队列 {status.get('queue_size', 0)} · 提取 {self._measure(status.get('extraction_ms'))} ms"
            f" · 结果年龄 {self._measure(status.get('receipt_age_ms'))} ms"
            f" · 时差 {self._measure(status.get('latest_dt'), 3)} s"
            f" · 时间间断 {self._measure(status.get('latest_gap_seconds'), 3)} s"
        )

    @staticmethod
    def _measure(value, decimals=1):
        return "—" if value is None else f"{value:.{decimals}f}"

    def _poll(self):
        if self._closed:
            return
        for event in self.session.poll():
            if hasattr(self.session, "epoch") and event.epoch != self.session.epoch:
                continue
            if event.kind in ("error", "recording_error"):
                self.status_label.setText("采集或录制未完成：" + str(event.value))
                if event.kind == "error":
                    self._clear_preview()
                else:
                    self.recording_label.setText("录制未完成，未保存本次样本")
            elif event.kind == "opened":
                self.status_label.setText("骨架回放已开始。" if self.session.status().get("source_kind") == "recording" else "输入已就绪，让双手和上半身保持在画面内。")
            elif event.kind == "eof":
                self.status_label.setText("播放结束。正在录制的样本可点击完成保存；重新打开可再次播放。")
            elif event.kind == "paused":
                self.status_label.setText("输入已暂停。" if self.session.status().get("paused") else "输入已继续。")
            elif event.kind == "recording_cancelled":
                self.recording_label.setText("录制已取消，未保存本次样本")
                self.status_label.setText("录制已取消：" + str(event.value or "未保存本次样本"))
            elif event.kind == "tracking_reset":
                self._clear_preview()
                self.status_label.setText("跟踪已重置；录制中的样本将保留重置标记。")
            elif event.kind == "recording_saved":
                self._saved_path = Path(event.value)
                self.recording_label.setText("已保存：" + self._saved_path.name)
                self.status_label.setText("关键点已保存：" + str(self._saved_path))
            elif event.kind == "recording_cleanup_warning":
                self.status_label.setText("关键点已保存；临时文件未能清理：" + str(event.value))
        if hasattr(self.session, "take_view"):
            packet, frame = self.session.take_view()
        else:
            packet, frame = self.session.take_preview(), self.session.take_keypoints()
        if packet is not None:
            self.preview.set_packet(packet)
        if frame is not None:
            self._latest_landmarks = frame
            self.preview.set_landmarks(frame)
            valid = np.asarray(frame.point_observed, dtype=bool)
            self.quality_label.setText(
                f"左手 {np.count_nonzero(valid[33:54])}/21 · 右手 {np.count_nonzero(valid[54:75])}/21"
                f" · 肩锚 {'可用' if frame.anchor_valid else '缺失'}"
            )
        self._update_controls()

    def closeEvent(self, event):
        if not self._closed:
            self._closed = True
            self._timer.stop()
            self.session.close()
        event.accept()

    def reject(self):
        self.close()
