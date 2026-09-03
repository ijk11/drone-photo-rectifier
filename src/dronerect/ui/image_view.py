"""정밀 작업용 사진 뷰어.

이 프로그램의 정확도는 결국 "점을 얼마나 정확히 찍었는가"에서 시작한다.
4000 px 사진에서 1 px 은 현장 2~4 cm 에 해당하므로, 눈대중 클릭으로는
실측 성과를 살릴 수 없다. 그래서 다음을 갖췄다.

* **확대경(loupe)** : 커서 주변을 원본 해상도의 8배로 띄워 보여 준다.
* **서브픽셀 코너 스냅** : 찍은 자리 주변에서 ``cv2.cornerSubPix`` 로
  실제 모서리를 0.01 px 수준까지 찾아 붙여 준다.
* **키보드 미세이동** : 방향키 0.1 px, Shift 1 px, Ctrl 0.01 px.
* **커서 좌표 표시** : 픽셀 좌표와(보정 후에는) 미터 좌표를 항상 표시.

오버레이(점/선/라벨)는 모두 뷰포트 좌표로 직접 그린다. 그래야 확대율과
무관하게 화면상 크기가 일정해서, 100배로 확대해도 마커가 화면을 뒤덮지 않는다.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QBrush,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from ..core.constraints import KIND_DISTANCE
from ..core.solver import OUTLIER_THRESHOLD

__all__ = ["ImageView", "MODE_SELECT", "MODE_ADD", "MODE_PICK"]

MODE_SELECT = "select"
MODE_ADD = "add"
MODE_PICK = "pick"

_HIT_RADIUS = 12.0        # 점 선택 허용 반경 [화면 px]
_LOUPE_SIZE = 190         # 확대경 한 변 [화면 px]
_LOUPE_ZOOM = 8.0

_C_POINT = QColor(255, 255, 255)
_C_POINT_EDGE = QColor(20, 20, 20)
_C_SELECT = QColor(255, 214, 0)
_C_PICK = QColor(0, 230, 255)
_C_SEG = QColor(70, 200, 120)
_C_SEG_WARN = QColor(255, 170, 0)
_C_SEG_BAD = QColor(255, 70, 70)
_C_FEATURE = QColor(120, 170, 255)
_C_HORIZON = QColor(255, 60, 60)


def _font(size: int = 8, bold: bool = False) -> QFont:
    """기본 글꼴 기반 크기 조절 폰트.

    ``QFont("", size)`` 처럼 패밀리를 빈 문자열로 주면 일부 플랫폼에서
    글자가 아예 그려지지 않는다. 항상 애플리케이션 기본 글꼴에서 출발한다.
    """
    f = QFont()
    f.setPointSize(size)
    f.setBold(bold)
    return f


class ImageView(QGraphicsView):
    """사진 표시 + 점 편집 뷰."""

    pointAdded = Signal(float, float)          # u, v (스냅 적용 후)
    pointMoved = Signal(str, float, float)     # point_id, u, v
    pointsPicked = Signal(list)                # [point_id, ...]
    selectionChanged = Signal(list)
    cursorMoved = Signal(float, float)         # u, v
    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._image: QImage | None = None
        self._gray: np.ndarray | None = None

        self.project = None
        self.result = None
        self._model = None

        self.mode = MODE_SELECT
        self.pick_count = 0
        self.pick_allow_create = True
        self._picks: list[str] = []
        self._selected: list[str] = []
        self._hover: str | None = None
        self._cursor_scene = QPointF()
        self._dragging: str | None = None
        self._drag_moved = False
        self._panning = False
        self._pan_anchor = QPointF()
        self.snap_enabled = True
        self.show_labels = True
        self.show_loupe = True

        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setBackgroundBrush(QBrush(QColor(30, 32, 36)))
        self.setCursor(Qt.CursorShape.CrossCursor)

    # ------------------------------------------------------------- 이미지
    def load_image(self, bgr: np.ndarray) -> None:
        """OpenCV BGR 배열을 표시한다."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        # QImage 는 버퍼를 참조만 하므로 복사본을 멤버로 붙들어 둔다.
        self._rgb_buffer = np.ascontiguousarray(rgb)
        self._image = QImage(
            self._rgb_buffer.data, w, h, 3 * w, QImage.Format.Format_RGB888
        )
        self._gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(QPixmap.fromImage(self._image))
        self._pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.setSceneRect(QRectF(0, 0, w, h))
        self.fit_to_window()

    def has_image(self) -> bool:
        return self._image is not None

    # --------------------------------------------------------------- 상태
    def set_project(self, project) -> None:
        self.project = project
        self._model = project.model() if project else None
        self._picks.clear()
        self._selected.clear()
        self.viewport().update()

    def set_result(self, result) -> None:
        self.result = result
        if self.project is not None:
            self._model = self.project.model()
        self.viewport().update()

    def refresh(self) -> None:
        if self.project is not None:
            self._model = self.project.model()
        self.viewport().update()

    def set_mode(self, mode: str, pick_count: int = 0, allow_create: bool = True) -> None:
        self.mode = mode
        self.pick_count = pick_count
        self.pick_allow_create = allow_create
        self._picks.clear()
        self.viewport().update()

    def cancel_pick(self) -> None:
        self._picks.clear()
        self.viewport().update()

    def selected_ids(self) -> list[str]:
        return list(self._selected)

    def set_selected(self, ids: list[str]) -> None:
        self._selected = list(ids)
        self.viewport().update()

    # ------------------------------------------------------------ 좌표 변환
    def _to_view(self, u: float, v: float) -> QPointF:
        return self.viewportTransform().map(QPointF(u, v))

    def _to_scene(self, pos: QPointF) -> QPointF:
        inv, ok = self.viewportTransform().inverted()
        return inv.map(pos) if ok else QPointF()

    def zoom_factor(self) -> float:
        return float(self.transform().m11())

    # ---------------------------------------------------------------- 확대
    def fit_to_window(self) -> None:
        if self._pixmap_item is None:
            return
        self.resetTransform()
        r = self._scene.sceneRect()
        if r.width() <= 0 or r.height() <= 0:
            return
        vw, vh = self.viewport().width(), self.viewport().height()
        s = min(vw / r.width(), vh / r.height()) * 0.98
        self.scale(s, s)
        self.centerOn(r.center())
        self.viewport().update()

    def zoom_to(self, factor: float, center_scene: QPointF | None = None) -> None:
        factor = max(0.02, min(200.0, factor))
        cur = self.zoom_factor()
        if cur <= 0:
            return
        self.scale(factor / cur, factor / cur)
        if center_scene is not None:
            self.centerOn(center_scene)
        self.viewport().update()

    def center_on_point(self, u: float, v: float, factor: float | None = None) -> None:
        if factor:
            self.zoom_to(factor)
        self.centerOn(QPointF(u, v))
        self.viewport().update()

    def wheelEvent(self, event) -> None:
        if self._pixmap_item is None:
            return
        anchor_scene = self._to_scene(QPointF(event.position()))
        step = 1.0015 ** event.angleDelta().y()
        new = max(0.02, min(200.0, self.zoom_factor() * step))
        self.scale(new / self.zoom_factor(), new / self.zoom_factor())
        # 커서 아래 지점이 고정되도록 스크롤 보정
        after = self._to_view(anchor_scene.x(), anchor_scene.y())
        delta = after - QPointF(event.position())
        self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() + delta.x()))
        self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() + delta.y()))
        self.statusMessage.emit(f"확대 {self.zoom_factor()*100:.0f}%")
        self.viewport().update()

    # ------------------------------------------------------------ 점 찾기
    def _point_at(self, view_pos: QPointF) -> str | None:
        if not self.project:
            return None
        best, best_d = None, _HIT_RADIUS
        for p in self.project.points:
            q = self._to_view(p.u, p.v)
            d = math.hypot(q.x() - view_pos.x(), q.y() - view_pos.y())
            if d < best_d:
                best, best_d = p.id, d
        return best

    # ---------------------------------------------------------- 서브픽셀 스냅
    def snap(self, u: float, v: float, win: int = 11) -> tuple[float, float, bool]:
        """주변 모서리로 서브픽셀 정밀 스냅. ``(u, v, 적용여부)``.

        ``cv2.cornerSubPix`` 는 그래디언트가 직교하는 지점을 찾는다. 결과가
        원래 위치에서 너무 멀면(모서리가 없는 평탄한 곳) 사용자의 클릭을
        존중해 스냅하지 않는다.
        """
        if self._gray is None or not self.snap_enabled:
            return u, v, False
        h, w = self._gray.shape[:2]
        if not (win < u < w - win and win < v < h - win):
            return u, v, False
        try:
            pt = np.array([[[float(u), float(v)]]], dtype=np.float32)
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-4)
            cv2.cornerSubPix(self._gray, pt, (win, win), (-1, -1), crit)
            nu, nv = float(pt[0, 0, 0]), float(pt[0, 0, 1])
        except cv2.error:
            return u, v, False
        if not (math.isfinite(nu) and math.isfinite(nv)):
            return u, v, False
        if math.hypot(nu - u, nv - v) > win * 0.7:
            return u, v, False
        return nu, nv, True

    # ------------------------------------------------------------ 마우스
    def mousePressEvent(self, event) -> None:
        pos = QPointF(event.position())
        scene = self._to_scene(pos)
        btn = event.button()

        if btn == Qt.MouseButton.MiddleButton or (
            btn == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            and self.mode == MODE_SELECT
        ):
            self._panning = True
            self._pan_anchor = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if btn == Qt.MouseButton.RightButton:
            if self._picks:
                self._picks.pop()
                self.viewport().update()
            return

        if btn != Qt.MouseButton.LeftButton or self._pixmap_item is None:
            super().mousePressEvent(event)
            return

        hit = self._point_at(pos)

        if self.mode == MODE_ADD:
            u, v, snapped = self.snap(scene.x(), scene.y())
            self.pointAdded.emit(u, v)
            self.statusMessage.emit(
                f"점 추가 ({u:.2f}, {v:.2f})" + ("  [코너 스냅]" if snapped else "")
            )
            return

        if self.mode == MODE_PICK:
            pid = hit
            if pid is None:
                if not self.pick_allow_create:
                    self.statusMessage.emit("기존 점을 선택하세요.")
                    return
                u, v, _ = self.snap(scene.x(), scene.y())
                self.pointAdded.emit(u, v)
                pid = self.project.points[-1].id if self.project.points else None
                if pid is None:
                    return
            self._picks.append(pid)
            if self.pick_count > 0 and len(self._picks) >= self.pick_count:
                picked = list(self._picks)
                self._picks.clear()
                self.pointsPicked.emit(picked)
            self.viewport().update()
            return

        # MODE_SELECT
        if hit is not None:
            add = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            if add:
                if hit in self._selected:
                    self._selected.remove(hit)
                else:
                    self._selected.append(hit)
            elif hit not in self._selected:
                self._selected = [hit]
            self._dragging = hit
            self._drag_moved = False
            self.selectionChanged.emit(list(self._selected))
        else:
            self._selected = []
            self.selectionChanged.emit([])
        self.viewport().update()

    def mouseMoveEvent(self, event) -> None:
        pos = QPointF(event.position())
        if self._panning:
            d = pos - self._pan_anchor
            self._pan_anchor = pos
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - d.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - d.y()))
            self.viewport().update()
            return

        scene = self._to_scene(pos)
        self._cursor_scene = scene
        self.cursorMoved.emit(scene.x(), scene.y())

        if self._dragging and self.project is not None:
            p = self.project.point_by_id(self._dragging)
            if p is not None:
                p.u, p.v = float(scene.x()), float(scene.y())
                self._drag_moved = True
        else:
            hv = self._point_at(pos)
            if hv != self._hover:
                self._hover = hv
        self.viewport().update()

    def mouseReleaseEvent(self, event) -> None:
        if self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        if self._dragging is not None:
            pid = self._dragging
            self._dragging = None
            if self._drag_moved and self.project is not None:
                p = self.project.point_by_id(pid)
                if p is not None:
                    u, v, _ = self.snap(p.u, p.v)
                    p.u, p.v = u, v
                    self.pointMoved.emit(pid, u, v)
            self.viewport().update()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if self.mode == MODE_SELECT and self._point_at(QPointF(event.position())) is None:
            self.fit_to_window()

    # ------------------------------------------------------------ 키보드
    def keyPressEvent(self, event) -> None:
        key = event.key()
        mods = event.modifiers()

        if key == Qt.Key.Key_Escape:
            self._picks.clear()
            self.viewport().update()
            self.statusMessage.emit("취소")
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.mode == MODE_PICK and self.pick_count <= 0 and self._picks:
                picked = list(self._picks)
                self._picks.clear()
                self.pointsPicked.emit(picked)
                return

        step = 0.1
        if mods & Qt.KeyboardModifier.ShiftModifier:
            step = 1.0
        elif mods & Qt.KeyboardModifier.ControlModifier:
            step = 0.01
        delta = {
            Qt.Key.Key_Left: (-step, 0.0),
            Qt.Key.Key_Right: (step, 0.0),
            Qt.Key.Key_Up: (0.0, -step),
            Qt.Key.Key_Down: (0.0, step),
        }.get(key)
        if delta and self._selected and self.project is not None:
            for pid in self._selected:
                p = self.project.point_by_id(pid)
                if p is not None:
                    p.u += delta[0]
                    p.v += delta[1]
                    self.pointMoved.emit(pid, p.u, p.v)
            self.statusMessage.emit(f"미세이동 {step} px")
            self.viewport().update()
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------ 오버레이
    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        if self.project is None or self._pixmap_item is None:
            return
        painter.save()
        painter.setWorldMatrixEnabled(False)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        stats = {}
        if self.result is not None:
            stats = {s.obs_id: s for s in self.result.obs_stats}

        self._draw_horizon(painter)
        self._draw_features(painter)
        self._draw_observations(painter, stats)
        self._draw_pending(painter)
        self._draw_points(painter)
        if self.show_loupe and self.underMouse():
            self._draw_loupe(painter)

        painter.setWorldMatrixEnabled(True)
        painter.restore()

    # -- 개별 요소 --------------------------------------------------------
    def _draw_horizon(self, painter: QPainter) -> None:
        if self._model is None or not getattr(self.project, "solved", False):
            return
        poly = self._model.vanishing_line_pixels()
        if poly is None or len(poly) < 2:
            return
        pts = [self._to_view(float(x), float(y)) for x, y in poly]
        pen = QPen(_C_HORIZON, 1.5, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawPolyline(QPolygonF(pts))
        painter.setFont(_font(8))
        painter.drawText(pts[len(pts) // 2] + QPointF(6, -6), "소실선 (이 근처는 신뢰 불가)")

    def _draw_features(self, painter: QPainter) -> None:
        pen = QPen(_C_FEATURE, 2.0)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for f in self.project.features:
            pts = []
            for pid in f.point_ids:
                p = self.project.point_by_id(pid)
                if p is not None:
                    pts.append(self._to_view(p.u, p.v))
            if len(pts) < 2:
                continue
            poly = QPolygonF(pts)
            if f.closed:
                painter.drawPolygon(poly)
            else:
                painter.drawPolyline(poly)

    def _draw_observations(self, painter: QPainter, stats: dict) -> None:
        painter.setFont(_font(8))
        for o in self.project.observations:
            if o.kind != KIND_DISTANCE or len(o.points) < 2:
                continue
            a = self.project.point_by_id(o.points[0])
            b = self.project.point_by_id(o.points[1])
            if a is None or b is None:
                continue
            qa, qb = self._to_view(a.u, a.v), self._to_view(b.u, b.v)
            color = _C_SEG
            st = stats.get(o.id)
            if st is not None and math.isfinite(st.w_test):
                aw = abs(st.w_test)
                if aw > OUTLIER_THRESHOLD:
                    color = _C_SEG_BAD
                elif aw > 2.0:
                    color = _C_SEG_WARN
            style = Qt.PenStyle.SolidLine if o.enabled else Qt.PenStyle.DotLine
            painter.setPen(QPen(color, 2.0, style))
            painter.drawLine(qa, qb)
            if self.show_labels:
                txt = f"{o.value:.3f} m"
                if st is not None:
                    txt += f"  {st.residual*1000:+.0f}mm"
                # 화면상 선분이 라벨보다 짧으면 글자가 겹쳐 오히려 방해가 된다.
                seg_len = math.hypot(qb.x() - qa.x(), qb.y() - qa.y())
                need = painter.fontMetrics().horizontalAdvance(txt) + 14
                if seg_len > need * 1.15:
                    self._label(painter, (qa + qb) / 2.0, txt, color)

    def _draw_pending(self, painter: QPainter) -> None:
        if self.mode != MODE_PICK or not self._picks:
            return
        pts = []
        for pid in self._picks:
            p = self.project.point_by_id(pid)
            if p is not None:
                pts.append(self._to_view(p.u, p.v))
        if not pts:
            return
        painter.setPen(QPen(_C_PICK, 1.6, Qt.PenStyle.DashLine))
        if len(pts) > 1:
            painter.drawPolyline(QPolygonF(pts))
        cur = self._to_view(self._cursor_scene.x(), self._cursor_scene.y())
        painter.drawLine(pts[-1], cur)
        for i, q in enumerate(pts, 1):
            painter.setPen(QPen(_C_PICK, 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(q, 9, 9)
            painter.drawText(q + QPointF(11, -8), str(i))

    def _draw_points(self, painter: QPainter) -> None:
        painter.setFont(_font(8))
        for p in self.project.points:
            q = self._to_view(p.u, p.v)
            if not (-40 <= q.x() <= self.viewport().width() + 40
                    and -40 <= q.y() <= self.viewport().height() + 40):
                continue
            sel = p.id in self._selected
            hov = p.id == self._hover
            painter.setBrush(Qt.BrushStyle.NoBrush)

            painter.setPen(QPen(_C_POINT_EDGE, 3.0))
            painter.drawLine(q + QPointF(-7, 0), q + QPointF(7, 0))
            painter.drawLine(q + QPointF(0, -7), q + QPointF(0, 7))
            painter.setPen(QPen(_C_POINT, 1.2))
            painter.drawLine(q + QPointF(-7, 0), q + QPointF(7, 0))
            painter.drawLine(q + QPointF(0, -7), q + QPointF(0, 7))

            painter.setPen(QPen(_C_POINT_EDGE, 2.4))
            painter.drawEllipse(q, 5, 5)
            painter.setPen(QPen(_C_SELECT if sel else _C_POINT, 1.4))
            painter.drawEllipse(q, 5, 5)
            if sel:
                painter.setPen(QPen(_C_SELECT, 1.6))
                painter.drawEllipse(q, 10, 10)
            if hov and not sel:
                painter.setPen(QPen(_C_PICK, 1.2))
                painter.drawEllipse(q, 10, 10)
            if self.show_labels:
                self._label(painter, q + QPointF(10, 12), p.name, _C_POINT)

    def _label(self, painter: QPainter, at: QPointF, text: str, color: QColor) -> None:
        fm = painter.fontMetrics()
        w = fm.horizontalAdvance(text) + 8
        h = fm.height() + 2
        box = QRectF(0, 0, w, h)
        box.moveCenter(at)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 165)))
        painter.drawRoundedRect(box, 3, 3)
        painter.setPen(QPen(color))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_loupe(self, painter: QPainter) -> None:
        if self._image is None:
            return
        # 커서가 사진 밖이면 확대경은 의미가 없다(검은 사각형만 남는다).
        cs = self._cursor_scene
        if not (0 <= cs.x() <= self._image.width() and 0 <= cs.y() <= self._image.height()):
            return
        cur = self._to_view(cs.x(), cs.y())
        vw, vh = self.viewport().width(), self.viewport().height()
        margin = 12
        # 커서 반대쪽 모서리에 배치해 작업 영역을 가리지 않게 한다.
        x = margin if cur.x() > vw / 2 else vw - _LOUPE_SIZE - margin
        y = margin if cur.y() > vh / 2 else vh - _LOUPE_SIZE - margin
        dst = QRectF(x, y, _LOUPE_SIZE, _LOUPE_SIZE)

        half = _LOUPE_SIZE / (2.0 * _LOUPE_ZOOM)
        src = QRectF(
            self._cursor_scene.x() - half, self._cursor_scene.y() - half, half * 2, half * 2
        )
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.setBrush(QBrush(QColor(0, 0, 0)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(dst)
        painter.drawImage(dst, self._image, src)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        c = dst.center()
        painter.setPen(QPen(QColor(0, 255, 200, 210), 1.0))
        painter.drawLine(QPointF(c.x(), dst.top()), QPointF(c.x(), dst.bottom()))
        painter.drawLine(QPointF(dst.left(), c.y()), QPointF(dst.right(), c.y()))
        painter.setPen(QPen(QColor(0, 255, 200, 120), 1.0))
        painter.drawEllipse(c, _LOUPE_ZOOM / 2, _LOUPE_ZOOM / 2)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(230, 230, 230), 1.0))
        painter.drawRect(dst)
        painter.setFont(_font(8))
        self._label(
            painter,
            QPointF(dst.center().x(), dst.bottom() + 10),
            f"x{_LOUPE_ZOOM:.0f}  ({self._cursor_scene.x():.2f}, {self._cursor_scene.y():.2f})",
            QColor(220, 220, 220),
        )
