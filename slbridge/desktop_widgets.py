"""Small, reusable visual components for the desktop recognition interface."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QImage,
    QPaintEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QSizePolicy, QWidget


APP_STYLE = """
QWidget {
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI";
    font-size: 14px;
    color: #173042;
}
QMainWindow, QDialog, QWidget#appRoot, QWidget#centralWidget {
    background-color: #F1F4F8;
}
QFrame#panel {
    background-color: #FFFFFF;
    border: 1px solid #DFE7ED;
    border-radius: 12px;
}
QFrame#divider {
    background-color: #E7EDF2;
    border: none;
}
QLabel {
    background-color: transparent;
}
QLabel#title {
    font-size: 25px;
    font-weight: 700;
    color: #173042;
}
QLabel#muted {
    font-size: 13px;
    color: #5C7080;
}
QLabel#sectionTitle {
    font-size: 15px;
    font-weight: 600;
    color: #173042;
}
QLabel#sourceTitle {
    font-size: 16px;
    font-weight: 600;
    color: #173042;
}
QLabel#resultZh {
    font-size: 32px;
    font-weight: 700;
    color: #173042;
}
QLabel#resultPt {
    font-size: 24px;
    font-weight: 500;
    color: #087F8C;
}
QLabel#notice {
    font-size: 12px;
    color: #5C7080;
}
QLabel#resultZh[state="rejected"], QLabel#resultZh[state="error"] {
    font-size: 20px;
    font-weight: 500;
}
QListWidget {
    background-color: #FFFFFF;
    border: none;
    outline: none;
    padding: 0;
}
QListWidget::item {
    min-height: 32px;
    padding: 4px 8px;
    border-bottom: 1px solid #EDF2F6;
}
QListWidget::item:selected {
    color: #076F7A;
    background-color: #E6F3F5;
}
QListWidget::item:focus {
    border: 1px solid #087F8C;
}
QProgressBar {
    border: none;
    border-radius: 2px;
    background-color: #E3EDF1;
}
QProgressBar::chunk {
    background-color: #087F8C;
    border-radius: 2px;
}
QTabWidget::pane {
    background-color: #FFFFFF;
    border: 1px solid #DFE7ED;
    border-radius: 8px;
}
QTabBar::tab {
    background-color: transparent;
    padding: 10px 16px;
    color: #5C7080;
}
QTabBar::tab:selected {
    color: #087F8C;
    border-bottom: 2px solid #087F8C;
}
QPlainTextEdit {
    background-color: #FFFFFF;
    border: none;
    padding: 12px;
    selection-background-color: #087F8C;
}
QLabel#modelSummary {
    font-size: 13px;
    color: #5C7080;
}
QLabel#statusBadge {
    padding: 5px 10px;
    border-radius: 7px;
    background-color: #E9F3F5;
    color: #087582;
    font-size: 12px;
    font-weight: 600;
}
QLabel[tone="warning"] {
    color: #936115;
}
QLabel#notice[tone="warning"] {
    color: #936115;
}
QLabel[tone="success"] {
    color: #087F68;
}
QLabel#statusBadge[tone="warning"] {
    background-color: #FFF3DC;
    color: #936115;
}
QLabel#statusBadge[tone="success"] {
    background-color: #E6F5EF;
    color: #087F68;
}
QPushButton {
    min-height: 20px;
    padding: 10px 14px;
    border-radius: 8px;
    border: 1px solid #D6E0E7;
    background-color: #FFFFFF;
    color: #173042;
    font-weight: 500;
}
QPushButton:hover {
    border-color: #A4BBC7;
    background-color: #F4F8FA;
}
QPushButton:pressed {
    background-color: #E8F1F4;
}
QPushButton:focus {
    border: 2px solid #087F8C;
    padding: 9px 13px;
}
QPushButton[kind="primary"] {
    background-color: #087F8C;
    border-color: #087F8C;
    color: #FFFFFF;
    font-weight: 600;
}
QPushButton[kind="primary"]:hover {
    background-color: #096F7A;
    border-color: #096F7A;
}
QPushButton[kind="primary"]:pressed {
    background-color: #075E69;
    border-color: #075E69;
}
QPushButton[kind="primary"]:focus {
    border-color: #173042;
}
QPushButton[kind="secondary"] {
    background-color: #FFFFFF;
    border-color: #D6E0E7;
}
QPushButton[kind="secondary"]:hover {
    background-color: #F4F8FA;
    border-color: #A4BBC7;
}
QPushButton[kind="ghost"] {
    background-color: transparent;
    border-color: transparent;
    color: #536A78;
}
QPushButton[kind="ghost"]:hover {
    background-color: #E8EEF3;
    color: #173042;
}
QPushButton[kind="ghost"]:focus {
    border-color: #087F8C;
}
QPushButton[size="small"] {
    padding: 5px 8px;
    min-height: 18px;
}
QPushButton#inputVideo, QPushButton#inputCamera {
    background-color: #F7F9FB;
    border: 1px solid #DFE7ED;
    padding: 12px 16px;
    font-weight: 600;
}
QPushButton#inputVideo:checked, QPushButton#inputCamera:checked,
QPushButton[active="true"] {
    background-color: #E6F3F5;
    border-color: #087F8C;
    color: #076F7A;
}
QPushButton:disabled {
    background-color: #EDF1F4;
    border-color: #E2E8ED;
    color: #93A4AE;
}
QPushButton[kind="ghost"]:disabled {
    background-color: transparent;
    border-color: transparent;
    color: #9CAAB3;
}
QLineEdit, QSpinBox, QComboBox {
    background-color: #FFFFFF;
    border: 1px solid #D6E0E7;
    border-radius: 7px;
    padding: 7px 9px;
    min-height: 20px;
    selection-background-color: #087F8C;
}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
    border-color: #087F8C;
}
QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled {
    background-color: #F3F6F8;
    color: #93A4AE;
}
QCheckBox {
    spacing: 8px;
    color: #536A78;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #ADBDC7;
    border-radius: 4px;
    background-color: #FFFFFF;
}
QCheckBox::indicator:checked {
    background-color: #087F8C;
    border-color: #087F8C;
}
QCheckBox::indicator:focus {
    border: 2px solid #173042;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
QScrollBar:vertical {
    background-color: transparent;
    width: 10px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background-color: #C4D0D9;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background-color: transparent;
}
QToolTip {
    background-color: #173042;
    color: #FFFFFF;
    border: none;
    padding: 6px 9px;
}
"""


def _pen(color: str | QColor, width: float = 1.75) -> QPen:
    pen = QPen(QColor(color), width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _draw_icon(painter: QPainter, name: str, color: str | QColor) -> None:
    """Draw a 20 by 20 logical-pixel icon using only Qt primitives."""
    painter.setPen(_pen(color))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    if name == "video":
        painter.drawRoundedRect(QRectF(2, 4, 16, 12), 2, 2)
        path.moveTo(8, 7)
        path.lineTo(13, 10)
        path.lineTo(8, 13)
        path.closeSubpath()
        painter.drawPath(path)
    elif name == "camera":
        path.moveTo(3, 6)
        path.lineTo(6, 6)
        path.lineTo(7.5, 4)
        path.lineTo(12.5, 4)
        path.lineTo(14, 6)
        path.lineTo(17, 6)
        path.quadTo(18, 6, 18, 7)
        path.lineTo(18, 15)
        path.quadTo(18, 16, 17, 16)
        path.lineTo(3, 16)
        path.quadTo(2, 16, 2, 15)
        path.lineTo(2, 7)
        path.quadTo(2, 6, 3, 6)
        path.closeSubpath()
        painter.drawPath(path)
        painter.drawEllipse(QRectF(7, 8, 6, 6))
    elif name == "play":
        path.moveTo(6, 3.5)
        path.lineTo(16, 10)
        path.lineTo(6, 16.5)
        path.closeSubpath()
        painter.drawPath(path)
    elif name == "pause":
        painter.drawLine(QPointF(6.5, 4), QPointF(6.5, 16))
        painter.drawLine(QPointF(13.5, 4), QPointF(13.5, 16))
    elif name == "record":
        painter.drawEllipse(QRectF(3, 3, 14, 14))
        painter.setBrush(QColor(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(6, 6, 8, 8))
    elif name == "check":
        path.moveTo(3, 10)
        path.lineTo(7.5, 14.5)
        path.lineTo(17, 5)
        painter.drawPath(path)
    elif name == "settings":
        painter.drawEllipse(QRectF(5, 5, 10, 10))
        painter.drawEllipse(QRectF(8, 8, 4, 4))
        for angle in range(0, 360, 45):
            radians = math.radians(angle)
            painter.drawLine(
                QPointF(10 + 5 * math.cos(radians), 10 + 5 * math.sin(radians)),
                QPointF(10 + 8 * math.cos(radians), 10 + 8 * math.sin(radians)),
            )
    elif name in {"close", "cancel"}:
        if name == "cancel":
            painter.drawEllipse(QRectF(2, 2, 16, 16))
            start, end = 7, 13
        else:
            start, end = 5, 15
        painter.drawLine(QPointF(start, start), QPointF(end, end))
        painter.drawLine(QPointF(end, start), QPointF(start, end))
    elif name == "folder":
        path.moveTo(2, 6)
        path.lineTo(2, 5)
        path.quadTo(2, 4, 3, 4)
        path.lineTo(8, 4)
        path.lineTo(10, 6)
        path.lineTo(17, 6)
        path.quadTo(18, 6, 18, 7)
        path.lineTo(18, 15)
        path.quadTo(18, 16, 17, 16)
        path.lineTo(3, 16)
        path.quadTo(2, 16, 2, 15)
        path.closeSubpath()
        painter.drawPath(path)
        painter.drawLine(QPointF(2, 8), QPointF(18, 8))
    elif name == "info":
        painter.drawEllipse(QRectF(2, 2, 16, 16))
        painter.drawLine(QPointF(10, 9), QPointF(10, 14))
        painter.drawPoint(QPointF(10, 6))
    elif name == "reload":
        painter.drawArc(QRectF(3.5, 3.5, 13, 13), 35 * 16, 285 * 16)
        path.moveTo(12.5, 3)
        path.lineTo(16.5, 3.5)
        path.lineTo(16, 7.5)
        painter.drawPath(path)
    else:
        raise ValueError(f"Unknown desktop icon: {name}")


def icon(name: str, color: str = "#536A78") -> QIcon:
    """Return a crisp 20-pixel line icon, without fonts or external assets."""
    if not QColor(color).isValid():
        raise ValueError(f"Invalid icon color: {color}")
    pixmap = QPixmap(40, 40)
    pixmap.setDevicePixelRatio(2)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        _draw_icon(painter, name, color)
    finally:
        painter.end()
    return QIcon(pixmap)


class BrandMark(QWidget):
    """A quiet bridge mark that also suggests two people meeting."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(44, 44)
        self.setAccessibleName("手语桥")

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#087F8C"))
            painter.drawRoundedRect(QRectF(0, 0, 44, 44), 12, 12)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(_pen("#FFFFFF", 2.5))
            bridge = QPainterPath()
            bridge.moveTo(9, 30)
            bridge.lineTo(9, 25)
            bridge.cubicTo(9, 14, 35, 14, 35, 25)
            bridge.lineTo(35, 30)
            painter.drawPath(bridge)
            painter.drawLine(QPointF(8, 27), QPointF(36, 27))
            painter.drawLine(QPointF(15, 21), QPointF(15, 27))
            painter.drawLine(QPointF(22, 18), QPointF(22, 27))
            painter.drawLine(QPointF(29, 21), QPointF(29, 27))
        finally:
            painter.end()


class PreviewWidget(QWidget):
    """Letterboxed video preview whose geometry is independent of its frame."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self.landmarks = None
        self._landmarks_mirrored = False
        self._placeholder_title = "在这里预览手语动作"
        self._placeholder_hint = "选择视频或打开摄像头开始"
        self._overlay = ""
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 220)
        self.setAccessibleName("手语视频预览")

    @property
    def image(self) -> QImage | None:
        return self._image

    def set_image(self, image: QImage, *, landmarks=None, mirrored=False) -> None:
        """Own the frame bytes even when the caller used a temporary buffer."""
        self._image = None if image.isNull() else image.copy()
        self.landmarks = landmarks if self._image is not None else None
        self._landmarks_mirrored = bool(mirrored)
        self.update()

    def set_placeholder(self, title: str, hint: str) -> None:
        self._image = None
        self.landmarks = None
        self._overlay = ""
        self._placeholder_title = title
        self._placeholder_hint = hint
        self.update()

    def set_overlay(self, text: str) -> None:
        self._overlay = text
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(640, 360)

    def minimumSizeHint(self) -> QSize:
        return QSize(320, 220)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            bounds = QRectF(self.rect())
            clip = QPainterPath()
            clip.addRoundedRect(bounds, 14, 14)
            painter.setClipPath(clip)
            painter.fillRect(bounds, QColor("#162632"))
            if self._image is not None:
                self._paint_image(painter, bounds)
            else:
                self._paint_placeholder(painter, bounds)
            if self._overlay:
                self._paint_overlay(painter, bounds)
        finally:
            painter.end()

    def _paint_image(self, painter: QPainter, bounds: QRectF) -> None:
        image = self._image
        if image is None or image.isNull():
            return
        ratio = min(bounds.width() / image.width(), bounds.height() / image.height())
        width, height = image.width() * ratio, image.height() * ratio
        target = QRectF(
            (bounds.width() - width) / 2,
            (bounds.height() - height) / 2,
            width,
            height,
        )
        painter.drawImage(target, image)
        if self.landmarks is not None:
            from .preview_landmarks import paint_landmarks
            paint_landmarks(painter,target,self.landmarks,mirrored=self._landmarks_mirrored)

    def _paint_placeholder(self, painter: QPainter, bounds: QRectF) -> None:
        center_x = bounds.center().x()
        center_y = bounds.center().y() - 40
        painter.save()
        painter.translate(center_x - 32, center_y - 32)
        painter.scale(3.2, 3.2)
        _draw_icon(painter, "video", "#728C9D")
        painter.restore()

        title_font = QFont("Microsoft YaHei UI")
        title_font.setPixelSize(17)
        title_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(title_font)
        painter.setPen(QColor("#DDE7EE"))
        title_rect = QRectF(20, center_y + 45, bounds.width() - 40, 30)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignCenter, self._placeholder_title)

        hint_font = QFont("Microsoft YaHei UI")
        hint_font.setPixelSize(13)
        painter.setFont(hint_font)
        painter.setPen(QColor("#A1B4C2"))
        hint_rect = QRectF(20, center_y + 80, bounds.width() - 40, 42)
        painter.drawText(
            hint_rect,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
            self._placeholder_hint,
        )

    def _paint_overlay(self, painter: QPainter, bounds: QRectF) -> None:
        overlay_font = QFont("Microsoft YaHei UI")
        overlay_font.setPixelSize(12)
        overlay_font.setWeight(QFont.Weight.Medium)
        painter.setFont(overlay_font)
        metrics = painter.fontMetrics()
        text = metrics.elidedText(
            self._overlay, Qt.TextElideMode.ElideRight, max(1, int(bounds.width()) - 64)
        )
        pill = QRectF(16, 16, metrics.horizontalAdvance(text) + 24, 29)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(14, 28, 39, 220))
        painter.drawRoundedRect(pill, 7, 7)
        painter.setPen(QColor("#F3F8FB"))
        painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, text)
