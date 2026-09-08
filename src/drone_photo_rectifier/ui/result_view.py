"""보정 결과(정사영상) 뷰어 + 실측 도구.

보정된 래스터는 축척이 일정하므로, 화면에서 두 점을 찍으면 그대로 실제
거리가 된다. 사용자가 보정 결과를 스스로 검증하는 가장 직관적인 수단이라
반드시 필요하다(알고 있는 치수를 재 보면 맞는지 바로 안다).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView

from ..core.solver import measure
from ..core.transform import PlaneModel

__all__ = ["ResultView"]

_C_LINE = QColor(0, 230, 255)
_C_AREA = QColor(255, 214, 0)


def _font(size: int = 9) -> QFont:
    f = QFont()
    f.setPointSize(size)
    return f


class ResultView(QGraphicsView):
    """정사영상 표시 + 거리/면적 측정."""

    measured = Signal(str)
    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._image: QImage | None = None
        self._buffer = None
        self.grid = None
        self.result = None                 # 불확도 계산용 조정 결과
        self._model = None                 # 미터 -> 원본 픽셀 역변환용
        self._src_wh = (0, 0)
        self.click_px = 1.0                # 점 찍기 정밀도 [출력 픽셀]
        self._pts: list[QPointF] = []      # 씬(출력 픽셀) 좌표
        self._cursor = QPointF()
        self.mode = "distance"             # 'distance' | 'area'
        self._panning = False
        self._anchor = QPointF()
        self._needs_fit = False            # 아직 사용자가 배율을 안 건드림

        self.setRenderHints(QPainter.RenderHint.Antialiasing
                            | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QBrush(QColor(24, 26, 30)))
        self.setCursor(Qt.CursorShape.CrossCursor)

    # ------------------------------------------------------------------ API
    def set_image(self, bgr: np.ndarray, grid) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self._buffer = np.ascontiguousarray(rgb)
        self._image = QImage(self._buffer.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.grid = grid
        self._scene.clear()
        item = self._scene.addPixmap(QPixmap.fromImage(self._image))
        item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.setSceneRect(QRectF(0, 0, w, h))
        self._pts.clear()
        self._needs_fit = True
        self.fit_to_window()

    def set_context(self, result, params, width: int, height: int) -> None:
        """측정 불확도를 계산하는 데 필요한 것들.

        길이만 보여 주면 그 값을 어디까지 믿을지 알 수 없다. 보정
        파라미터의 불확도와 점 찍기 오차를 함께 전파해 ± 로 보여 준다.
        """
        self.result = result
        self._src_wh = (int(width), int(height))
        self._model = PlaneModel(params, width, height) if params is not None else None

    def clear_measure(self) -> None:
        self._pts.clear()
        self.viewport().update()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._pts.clear()
        self.viewport().update()

    def has_image(self) -> bool:
        return self._image is not None

    # -------------------------------------------------------------- 좌표계
    def _to_view(self, p: QPointF) -> QPointF:
        return self.viewportTransform().map(p)

    def _to_scene(self, p: QPointF) -> QPointF:
        inv, ok = self.viewportTransform().inverted()
        return inv.map(p) if ok else QPointF()

    def scene_to_meters(self, p: QPointF) -> tuple[float, float]:
        """출력 픽셀 -> 미터(내부 좌표계)."""
        if self.grid is None:
            return (0.0, 0.0)
        return (self.grid.xmin + p.x() * self.grid.gsd,
                self.grid.ymin + p.y() * self.grid.gsd)

    # ----------------------------------------------------------------- 확대
    def fit_to_window(self) -> None:
        r = self._scene.sceneRect()
        if r.width() <= 0:
            return
        self.resetTransform()
        s = min(self.viewport().width() / r.width(),
                self.viewport().height() / r.height()) * 0.98
        self.scale(s, s)
        self.centerOn(r.center())
        self.viewport().update()

    def showEvent(self, event) -> None:
        """탭이 처음 드러날 때 배율을 다시 맞춘다.

        ``fit_to_window`` 는 viewport 크기를 쓰는데, 숨어 있는 탭은 배치가
        끝나지 않아 크기가 확정되지 않는다. 그 상태에서 맞추면 그림이
        한 귀퉁이에 작게 박힌 채로 남는다. 사용자가 아직 손대지 않았을
        때만 다시 맞춘다.
        """
        super().showEvent(event)
        if self._needs_fit:
            self.fit_to_window()
            self._needs_fit = False

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._needs_fit:
            self.fit_to_window()

    def wheelEvent(self, event) -> None:
        self._needs_fit = False
        if self._image is None:
            return
        anchor = self._to_scene(QPointF(event.position()))
        f = 1.0015 ** event.angleDelta().y()
        cur = float(self.transform().m11())
        new = max(0.01, min(100.0, cur * f))
        self.scale(new / cur, new / cur)
        after = self._to_view(anchor)
        d = after - QPointF(event.position())
        self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() + d.x()))
        self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() + d.y()))
        self.viewport().update()

    # ---------------------------------------------------------------- 입력
    def mousePressEvent(self, event) -> None:
        if self._image is None:
            return
        pos = QPointF(event.position())
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._anchor = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() == Qt.MouseButton.RightButton:
            if self._pts:
                self._pts.pop()
                self.viewport().update()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._pts.append(self._to_scene(pos))
        if self.mode == "distance" and len(self._pts) > 2:
            self._pts = self._pts[-1:]
        self._emit_measure()
        self.viewport().update()

    def mouseMoveEvent(self, event) -> None:
        pos = QPointF(event.position())
        if self._panning:
            d = pos - self._anchor
            self._anchor = pos
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - d.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - d.y()))
            self.viewport().update()
            return
        self._cursor = self._to_scene(pos)
        if self.grid is not None:
            x, y = self.scene_to_meters(self._cursor)
            self.statusMessage.emit(f"X {x:.3f} m,  Y {y:.3f} m")
        self.viewport().update()

    def mouseReleaseEvent(self, event) -> None:
        if self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.CrossCursor)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.clear_measure()
            return
        super().keyPressEvent(event)

    # ---------------------------------------------------------------- 계산
    def _polyline_m(self) -> np.ndarray:
        return np.array([self.scene_to_meters(p) for p in self._pts], dtype=float)

    def _measure(self, m_xy: np.ndarray, mode: str):
        """미터 좌표 폴리라인을 불확도까지 붙여 잰다.

        ``measure`` 는 원본 사진 픽셀을 받으므로 역변환해서 넘긴다.
        """
        if self._model is None or self._src_wh[0] <= 0:
            return None
        pix = self._model.inverse(m_xy)
        if not np.all(np.isfinite(pix)):
            return None
        sigma_pt = self.grid.gsd * self.click_px if self.grid else 0.0
        return measure(self.result, self._src_wh[0], self._src_wh[1],
                       pix, mode, sigma_pt=sigma_pt)

    def _emit_measure(self) -> None:
        if self.grid is None or len(self._pts) < 2:
            self.measured.emit("")
            return
        m = self._polyline_m()
        seg = np.linalg.norm(np.diff(m, axis=0), axis=1)
        total = float(seg.sum())

        mode = "area" if (self.mode == "area" and len(m) >= 3) else "distance"
        res = self._measure(m, mode)
        if res is None:
            head = (f"면적 {total:.3f} m²" if mode == "area"
                    else f"거리 {total:.3f} m")
            self.measured.emit(head)
            return

        detail = ""
        if math.isfinite(res.sigma_model) and math.isfinite(res.sigma_click):
            # 거리는 mm 가 읽기 좋고, 면적은 mm² 로 바꾸면 자릿수만 커진다.
            if res.unit == "m2":
                detail = (f"   [모델 ±{res.sigma_model:.2f} · "
                          f"클릭 ±{res.sigma_click:.2f} m²]")
            else:
                detail = (f"   [모델 ±{res.sigma_model * 1000:.0f} · "
                          f"클릭 ±{res.sigma_click * 1000:.0f} mm]")
        if mode == "area":
            per = total + float(np.linalg.norm(m[0] - m[-1]))
            self.measured.emit(f"면적 {res.text(2)}   둘레 {per:.3f} m"
                               f"   ({len(m)}점){detail}")
        else:
            self.measured.emit(f"거리 {res.text()}{detail}")

    # -------------------------------------------------------------- 오버레이
    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        if self._image is None or self.grid is None:
            return
        painter.save()
        painter.setWorldMatrixEnabled(False)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._draw_scalebar(painter)
        self._draw_measure(painter)
        painter.setWorldMatrixEnabled(True)
        painter.restore()

    def _draw_measure(self, painter: QPainter) -> None:
        if not self._pts:
            return
        color = _C_LINE if self.mode == "distance" else _C_AREA
        pts = [self._to_view(p) for p in self._pts]
        cur = self._to_view(self._cursor)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(color, 2.0))
        if self.mode == "area" and len(pts) >= 3:
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 45)))
            painter.drawPolygon(QPolygonF(pts))
            painter.setBrush(Qt.BrushStyle.NoBrush)
        elif len(pts) > 1:
            painter.drawPolyline(QPolygonF(pts))
        painter.setPen(QPen(color, 1.4, Qt.PenStyle.DashLine))
        painter.drawLine(pts[-1], cur)

        painter.setFont(_font(9))
        painter.setPen(QPen(color, 2.0))
        for q in pts:
            painter.drawEllipse(q, 4, 4)

        # 각 구간 길이 표시
        m = self._polyline_m()
        for i in range(len(pts) - 1):
            d = float(np.linalg.norm(m[i + 1] - m[i]))
            mid = (pts[i] + pts[i + 1]) / 2.0
            self._chip(painter, mid, f"{d:.3f} m", color)
        if len(pts) >= 1:
            xm, ym = self.scene_to_meters(self._cursor)
            live = float(np.linalg.norm(np.array([xm, ym]) - m[-1]))
            self._chip(painter, (pts[-1] + cur) / 2.0, f"{live:.3f} m", color)

    def _chip(self, painter: QPainter, at: QPointF, text: str, color: QColor) -> None:
        fm = painter.fontMetrics()
        w = fm.horizontalAdvance(text) + 10
        h = fm.height() + 4
        box = QRectF(0, 0, w, h)
        box.moveCenter(at)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 180)))
        painter.drawRoundedRect(box, 3, 3)
        painter.setPen(QPen(color))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_scalebar(self, painter: QPainter) -> None:
        """화면 축척 막대. 보정 결과가 축척을 갖는다는 사실을 눈으로 보여 준다."""
        zoom = float(self.transform().m11())
        if zoom <= 0:
            return
        m_per_screen_px = self.grid.gsd / zoom
        target_px = 180.0
        raw = target_px * m_per_screen_px
        # 1, 2, 5 x 10^n 으로 반올림
        exp = math.floor(math.log10(raw)) if raw > 0 else 0
        base = raw / (10 ** exp)
        nice = 1 if base < 1.5 else 2 if base < 3.5 else 5 if base < 7.5 else 10
        length_m = nice * (10 ** exp)
        px = length_m / m_per_screen_px

        x0 = 16
        y0 = self.viewport().height() - 26
        painter.setPen(QPen(QColor(0, 0, 0, 190), 5))
        painter.drawLine(QPointF(x0, y0), QPointF(x0 + px, y0))
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawLine(QPointF(x0, y0), QPointF(x0 + px, y0))
        for xx in (x0, x0 + px):
            painter.drawLine(QPointF(xx, y0 - 5), QPointF(xx, y0 + 5))
        painter.setFont(_font(9))
        label = f"{length_m:g} m" if length_m >= 1 else f"{length_m*100:g} cm"
        self._chip(painter, QPointF(x0 + px / 2, y0 - 16), label, QColor(255, 255, 255))
