"""Optional SLRT panel. Capture, model loading and inference stay off Qt's thread."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import queue
import threading

from PySide6.QtCore import QTimer,Qt
from PySide6.QtGui import QImage,QPixmap
from PySide6.QtWidgets import (QApplication,QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,
                               QFileDialog,QLineEdit,QSpinBox,QComboBox)


class SLRTWindow(QWidget):
    def __init__(self,config,*,video=None,camera=0,pose_npz=None):
        super().__init__()
        self.config,self.pose_npz = config,pose_npz
        self.backend = self.session = None
        self._loaded = queue.Queue(maxsize=1)
        self._loading = True
        self._closing = False
        self._legacy = None
        self.setWindowTitle('SLRT-CSL · 连续 Gloss 识别')
        self.resize(920,700)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('CSL-Daily 连续词汇识别 · 原始 Gloss；自然语言翻译尚未接入'))
        row = QHBoxLayout()
        self.path = QLineEdit(str(video or ''))
        self.path.setPlaceholderText('本地视频路径；留空使用摄像头')
        browse = QPushButton('选择视频');browse.clicked.connect(self._browse)
        row.addWidget(self.path);row.addWidget(browse);layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel('摄像头'))
        self.camera = QSpinBox();self.camera.setRange(0,32);self.camera.setValue(camera)
        row.addWidget(self.camera)
        self.pose = QComboBox()
        self.pose.addItem('官方 HRNet（需要独立资源）','hrnet')
        self.pose.addItem('HRNet NPZ（绑定视频）','hrnet-npz')
        self.pose.addItem('RTMPose（实验前端）','rtmpose-experimental')
        self.pose.setCurrentIndex(max(0,self.pose.findData(config.pose_backend)))
        row.addWidget(self.pose)
        row.addWidget(QLabel('stride'))
        self.stride = QSpinBox();self.stride.setRange(1,config.window_size);self.stride.setValue(config.stride)
        row.addWidget(self.stride);layout.addLayout(row)
        self.preview = QLabel('等待输入');self.preview.setAlignment(Qt.AlignCenter);self.preview.setMinimumHeight(320)
        layout.addWidget(self.preview)
        self.final = QLabel('已提交 Gloss：');self.final.setWordWrap(True);layout.addWidget(self.final)
        self.partial = QLabel('临时预测：');self.partial.setWordWrap(True);layout.addWidget(self.partial)
        self.status = QLabel('正在后台加载模型…');self.status.setWordWrap(True);layout.addWidget(self.status)
        row = QHBoxLayout()
        for text,callback,name in [('开始',self._start,'start_button'),('暂停',self._pause,'pause_button'),
                                   ('继续',self._resume,'resume_button'),('重置',self._reset,'reset_button'),
                                   ('停止',self._stop,'stop_button'),('导出 JSON',self._export,'export_button')]:
            button=QPushButton(text);button.clicked.connect(callback);row.addWidget(button);setattr(self,name,button)
        layout.addLayout(row)
        legacy = QPushButton('打开现有 Legacy / TFNet / Uni-Sign 工作区');legacy.clicked.connect(self._open_legacy)
        layout.addWidget(legacy)
        self.start_button.setEnabled(False)
        self.timer = QTimer(self);self.timer.timeout.connect(self._poll);self.timer.start(50)
        self._loader = threading.Thread(target=self._load,name='slrt-model-load',daemon=True);self._loader.start()

    def _load(self):
        try:
            from .backend import SLRTBackend
            self._loaded.put((SLRTBackend(self.config),None))
        except Exception as error:
            self._loaded.put((None,f'{type(error).__name__}: {error}'))

    def _browse(self):
        path,_=QFileDialog.getOpenFileName(self,'视频',str(Path.cwd()),'Video (*.mp4 *.avi *.mov *.mkv)')
        if path:self.path.setText(path)

    def _start(self):
        if self.backend is None or (self.session and self.session.is_running):return
        from .capture import SLRTCaptureSession
        self.backend.config = replace(self.config,pose_backend=self.pose.currentData(),stride=self.stride.value())
        path=self.path.text().strip()
        self.session=SLRTCaptureSession(self.backend,camera=self.camera.value(),video=Path(path) if path else None,pose_npz=self.pose_npz)
        self.session.start();self.start_button.setEnabled(False)

    def _pause(self):
        if self.session:self.session.pause()

    def _resume(self):
        if self.session:self.session.resume()

    def _reset(self):
        if self.session:self.session.reset()
        self.final.setText('已提交 Gloss：');self.partial.setText('临时预测：')

    def _stop(self):
        if self.session:self.session.request_stop()

    def _poll(self):
        try:
            backend,error=self._loaded.get_nowait()
            self.backend=backend;self._loading=False
            self.status.setText(error or '权重严格加载完成。窗口右上下文和投票会增加延迟；速度需实测。')
            self.start_button.setEnabled(backend is not None)
        except queue.Empty:
            pass
        if self.session:
            self.session.poll()  # Drain the bounded event queue; render latest snapshot.
            report=self.session.result()
            self.final.setText('已提交 Gloss：'+' '.join(report.get('gloss_sequence',[])))
            tail=report.get('partial')
            self.partial.setText('临时预测：'+(tail['gloss'] if tail else ''))
            self.status.setText(report.get('error') or f"{report['state']} · 采集 {report['frames_captured']} 帧 · 丢弃 {report['queue_dropped']} 帧 · 窗口 {report.get('windows_processed',0)}")
            image=self.session.preview()
            if image is not None:
                rgb=image[...,::-1].copy();h,w=rgb.shape[:2]
                qimage=QImage(rgb.data,w,h,rgb.strides[0],QImage.Format_RGB888).copy()
                self.preview.setPixmap(QPixmap.fromImage(qimage).scaled(self.preview.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
            self.start_button.setEnabled(self.backend is not None and not self.session.is_running and not self._closing)
        if self._closing and not self._loading and (self.session is None or not self.session.is_running):
            if self.backend:self.backend.close();self.backend=None
            self.close()

    def _export(self):
        if not self.session:return
        path,_=QFileDialog.getSaveFileName(self,'导出原始 Gloss','slrt-result.json','JSON (*.json)')
        if path:Path(path).write_text(json.dumps(self.session.result(),ensure_ascii=False,indent=2),encoding='utf-8')

    def _open_legacy(self):
        from ..desktop import SignLanguageApp
        from ..desktop_cli import default_model
        self._legacy=SignLanguageApp(model=default_model());self._legacy.show()

    def closeEvent(self,event):
        if self._loading or (self.session and self.session.is_running):
            self._closing=True
            self._stop();self.status.setText('正在释放采集与推理资源…');event.ignore();return
        if self.backend:self.backend.close();self.backend=None
        event.accept()


def launch(config,*,video=None,camera=0,pose_npz=None):
    app=QApplication.instance() or QApplication([])
    window=SLRTWindow(config,video=video,camera=camera,pose_npz=pose_npz);window.show()
    return app.exec()
