"""Display-only image landmarks, separate from all model feature protocols."""
from dataclasses import dataclass
import math
import time


POSE_EDGES = ((11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),
    (23,24),(23,25),(25,27),(27,29),(29,31),(27,31),(24,26),(26,28),
    (28,30),(30,32),(28,32))
HAND_EDGES = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),
    (9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),
    (0,17),(17,18),(18,19),(19,20))
MAX_PREVIEW_AGE = .5


@dataclass(frozen=True)
class PreviewLandmarks:
    source_id: str
    source_epoch: int
    frame_index: int
    timestamp: float
    image_size: tuple[int, int]
    image_xy: tuple[tuple[float, float], ...]
    point_observed: tuple[bool, ...]
    completed_at: float

    @classmethod
    def from_result(cls, packet, result, *, source_epoch, pose_visibility=.5):
        points, observed = [], []
        for name, count in (('pose_landmarks',33),('left_hand_landmarks',21),('right_hand_landmarks',21)):
            group = getattr(getattr(result,name,None),'landmark',())
            group = group if len(group)==count else ()
            for index in range(count):
                point = group[index] if group else None
                try:
                    x,y = float(point.x),float(point.y)
                    valid = math.isfinite(x) and math.isfinite(y) and 0<=x<=1 and 0<=y<=1
                    if count==33:
                        visibility=float(point.visibility)
                        valid = valid and math.isfinite(visibility) and pose_visibility<=visibility<=1
                except (AttributeError,TypeError,ValueError,OverflowError):
                    valid=False
                points.append((x,y) if valid else (0.,0.))
                observed.append(bool(valid))
        return cls(packet.source_id,source_epoch,packet.frame_index,packet.timestamp,
            (packet.image.shape[1],packet.image.shape[0]),tuple(points),tuple(observed),time.perf_counter())


def paint_landmarks(painter, target, frame, *, mirrored=False):
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QColor,QPen
    def point(index):
        x,y=frame.image_xy[index]
        return QPointF(target.left()+(1-x if mirrored else x)*target.width(),target.top()+y*target.height())
    painter.save()
    painter.setClipRect(target)
    for offset,edges,color in ((0,POSE_EDGES,'#118C69'),(33,HAND_EDGES,'#008BBA'),(54,HAND_EDGES,'#CD7016')):
        painter.setPen(QPen(QColor(color),2.4))
        painter.setBrush(QColor(color))
        for start,end in edges:
            start,end=start+offset,end+offset
            if frame.point_observed[start] and frame.point_observed[end]:
                painter.drawLine(point(start),point(end))
        for index in range(offset,offset+(33 if offset==0 else 21)):
            if frame.point_observed[index]: painter.drawEllipse(point(index),3,3)
    painter.restore()
