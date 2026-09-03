"""메인 윈도우: 전체 작업 흐름을 엮는다.

작업 순서
    1) 사진 열기
    2) 실측한 구간마다 두 점을 찍고 실제 거리를 입력 (필요하면 직각/평행도 추가)
    3) [보정 실행] - 원근과 렌즈왜곡을 동시에 추정하고 정확도를 진단
    4) 잔차/잉여도를 보고 이상값을 정리한 뒤 재계산
    5) 정사보정 래스터 생성 -> 이미지·DXF·성과표로 내보내기
"""

from __future__ import annotations

import math
import os
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QDialog,
    QDialogButtonBox,
)

from ..core import exporters
from ..core.constraints import KIND_ANGLE, KIND_DISTANCE, KIND_INFO, Observation
from ..core.imageio import focal_length_px, imread, imwrite, read_exif
from ..core.project import FILE_SUFFIX, Feature, Project
from ..core.rectify import plan_grid, rectify_image, scale_stats, suggest_gsd
from ..core.solver import adjust
from ..core.transform import PlaneModel, solve_gauge
from .dialogs import GaugeDialog, ObservationDialog
from .image_view import MODE_ADD, MODE_PICK, MODE_SELECT, ImageView
from .panels import ExportPanel, ObservationTable, PointTable, SolvePanel
from .result_view import ResultView

IMAGE_FILTER = "이미지 (*.jpg *.jpeg *.png *.tif *.tiff *.bmp);;모든 파일 (*.*)"
PROJECT_FILTER = f"drone-rectify 프로젝트 (*{FILE_SUFFIX});;모든 파일 (*.*)"

HELP_HTML = """
<h2>사용법</h2>
<ol>
<li><b>사진 열기</b> (Ctrl+O) — 드론/항공 사진을 불러옵니다.</li>
<li><b>실측 거리 입력</b> — 오른쪽 [관측] 탭에서 <b>실측 거리</b>를 누르고,
    사진에서 실제로 줄자를 댄 두 지점을 클릭한 뒤 잰 값을 입력합니다.
    같은 방식으로 계속 추가할 수 있습니다(Esc 로 중단).</li>
<li><b>기하 구속 추가(권장)</b> — 도로 경계·건물 외곽처럼 실제로 직각이거나
    평행한 곳이 있으면 <b>직각/평행</b>을 추가하세요. 줄자 없이 얻는 정보이며
    해를 크게 안정시킵니다.</li>
<li><b>보정 실행</b> (F5) — 원근(호모그래피)과 렌즈 방사왜곡을 동시에
    추정합니다.</li>
<li><b>진단 확인</b> — 잔차·잉여도·조건수를 봅니다. 빨간 항목은 조대오차
    의심이므로 실측값 오타나 점 위치를 확인하세요.</li>
<li><b>결과 생성</b> — [출력] 탭에서 정사보정 래스터를 만들고, 이미지(+월드파일),
    DXF, 성과표, 리포트로 내보냅니다.</li>
</ol>

<h3>정확도를 높이는 요령</h3>
<ul>
<li>실측 선분을 <b>사진 네 귀퉁이까지 고르게</b> 배치하세요. 한쪽에 몰리면
    조건수가 커져 반대편의 정확도가 급격히 나빠집니다.</li>
<li>방향을 섞으세요. 가로 방향만 재면 세로 축척이 결정되지 않습니다.</li>
<li>길이가 <b>서로 많이 다른</b> 구간을 섞으면 축척과 왜곡이 잘 분리됩니다.</li>
<li>최소 7개(자동 선택 기준 5~9개) 이상, 검증을 위해 그보다 3~5개 더
    잡는 것을 권합니다.</li>
<li>점은 <b>확대해서</b> 찍으세요. 확대경과 코너 스냅이 켜져 있습니다.
    방향키로 0.1 px, Ctrl+방향키로 0.01 px 미세이동합니다.</li>
</ul>

<h3>꼭 알아야 할 전제</h3>
<p>이 보정은 대상이 <b>하나의 평면(지면)</b> 위에 있다고 가정합니다.
건물 벽면·수목·적치물처럼 <b>높이가 있는 것은 기복변위</b> 때문에 보정 후에도
위치가 어긋납니다. 실측 기준선은 반드시 지면에서 잡으세요.
지형이 크게 경사지거나 단차가 있으면 구역을 나눠 각각 보정해야 합니다.</p>

<h3>마우스/키보드</h3>
<table cellpadding="4">
<tr><td>휠</td><td>커서 기준 확대/축소</td></tr>
<tr><td>가운데 버튼 드래그</td><td>이동(팬)</td></tr>
<tr><td>오른쪽 버튼</td><td>선택 중인 점 하나 취소</td></tr>
<tr><td>Esc</td><td>진행 중인 입력 취소</td></tr>
<tr><td>방향키</td><td>선택한 점 0.1 px 이동 (Shift 1 px, Ctrl 0.01 px)</td></tr>
<tr><td>Ctrl+0 / Ctrl+1</td><td>화면 맞춤 / 100%</td></tr>
</table>
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("drone-rectify — 실측 기준 정사보정")
        self.resize(1500, 950)

        self.project = Project()
        self.result = None
        self.image: np.ndarray | None = None
        self.rect_image: np.ndarray | None = None
        self.grid = None
        self._undo: list[dict] = []
        self._redo: list[dict] = []
        self._pending_kind: str | None = None
        self._last_layer = "0"
        self._dirty = False

        self._build_ui()
        self._build_actions()
        self._refresh_all()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.view = ImageView(self)
        self.result_view = ResultView(self)
        self.tabs = QTabWidget(self)
        self.tabs.addTab(self.view, "원본 사진")
        self.tabs.addTab(self.result_view, "보정 결과")
        self.setCentralWidget(self.tabs)

        self.point_table = PointTable(self)
        dock_l = QDockWidget("점 목록", self)
        dock_l.setWidget(self.point_table)
        dock_l.setObjectName("dock_points")
        dock_l.setMinimumWidth(380)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock_l)

        self.obs_table = ObservationTable(self)
        self.solve_panel = SolvePanel(self)
        self.export_panel = ExportPanel(self)
        self.right_tabs = QTabWidget(self)
        self.right_tabs.addTab(self.obs_table, "관측")
        self.right_tabs.addTab(self.solve_panel, "보정")
        self.right_tabs.addTab(self.export_panel, "출력")
        dock_r = QDockWidget("작업", self)
        dock_r.setWidget(self.right_tabs)
        dock_r.setObjectName("dock_work")
        dock_r.setMinimumWidth(520)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_r)

        sb = self.statusBar()
        self.lbl_mode = QLabel("모드: 선택")
        self.lbl_px = QLabel("픽셀 -, -")
        self.lbl_m = QLabel("미터 -")
        self.lbl_measure = QLabel("")
        for w in (self.lbl_mode, self.lbl_px, self.lbl_m, self.lbl_measure):
            w.setMinimumWidth(150)
            sb.addPermanentWidget(w)

        # --- 신호 연결
        self.view.pointAdded.connect(self._on_point_added)
        self.view.pointMoved.connect(self._on_point_moved)
        self.view.pointsPicked.connect(self._on_points_picked)
        self.view.selectionChanged.connect(self.point_table.select_ids)
        self.view.cursorMoved.connect(self._on_cursor)
        self.view.statusMessage.connect(lambda s: self.statusBar().showMessage(s, 4000))

        self.result_view.measured.connect(lambda s: self.lbl_measure.setText(s))
        self.result_view.statusMessage.connect(lambda s: self.lbl_m.setText(s))

        self.point_table.pointSelected.connect(self._on_table_point_selected)
        self.point_table.pointEdited.connect(self._on_point_edited)
        self.point_table.deleteRequested.connect(self._delete_points)

        self.obs_table.addRequested.connect(self._start_observation)
        self.obs_table.deleteRequested.connect(self._delete_observations)
        self.obs_table.valueEdited.connect(self._edit_observation)
        self.obs_table.enabledToggled.connect(self._toggle_observation)
        self.obs_table.zoomRequested.connect(self._zoom_to_observation)

        self.solve_panel.solveRequested.connect(self._solve)
        self.solve_panel.disableOutliersRequested.connect(self._disable_outliers)
        self.solve_panel.gaugeRequested.connect(self._edit_gauge)

        self.export_panel.previewRequested.connect(self._make_rectified)
        self.export_panel.saveRasterRequested.connect(self._save_rectified)
        self.export_panel.exportRequested.connect(self._export)
        self.export_panel.gsdModeChanged.connect(self._update_gsd)

    def _build_actions(self) -> None:
        m_file = self.menuBar().addMenu("파일(&F)")
        self._act(m_file, "사진 열기...", self._open_image, QKeySequence.StandardKey.Open)
        self._act(m_file, "프로젝트 열기...", self._open_project, "Ctrl+Shift+O")
        m_file.addSeparator()
        self._act(m_file, "저장", self._save_project, QKeySequence.StandardKey.Save)
        self._act(m_file, "다른 이름으로 저장...", self._save_project_as, "Ctrl+Shift+S")
        m_file.addSeparator()
        self._act(m_file, "검증용 합성 예제 만들기...", self._make_sample)
        m_file.addSeparator()
        self._act(m_file, "종료", self.close, "Ctrl+Q")

        m_edit = self.menuBar().addMenu("편집(&E)")
        self._act(m_edit, "실행 취소", self._do_undo, QKeySequence.StandardKey.Undo)
        self._act(m_edit, "다시 실행", self._do_redo, QKeySequence.StandardKey.Redo)
        m_edit.addSeparator()
        self._act(m_edit, "선택한 점 삭제", lambda: self._delete_points(self.view.selected_ids()),
                  QKeySequence.StandardKey.Delete)

        m_view = self.menuBar().addMenu("보기(&V)")
        self._act(m_view, "화면에 맞춤", self._fit, "Ctrl+0")
        self._act(m_view, "100%", lambda: self.view.zoom_to(1.0), "Ctrl+1")
        m_view.addSeparator()
        self.act_labels = self._act(m_view, "라벨 표시", self._toggle_labels, checkable=True, checked=True)
        self.act_loupe = self._act(m_view, "확대경 표시", self._toggle_loupe, checkable=True, checked=True)
        self.act_snap = self._act(m_view, "코너 서브픽셀 스냅", self._toggle_snap,
                                  checkable=True, checked=True)

        m_solve = self.menuBar().addMenu("보정(&C)")
        self._act(m_solve, "보정 실행", lambda: self._solve(self.solve_panel.free_names(),
                                                            self.solve_panel.chk_robust.isChecked()), "F5")
        self._act(m_solve, "좌표축 정렬...", self._edit_gauge)

        m_help = self.menuBar().addMenu("도움말(&H)")
        self._act(m_help, "사용법", self._show_help, "F1")
        self._act(m_help, "정보", self._show_about)

        tb = self.addToolBar("도구")
        tb.setIconSize(QSize(18, 18))
        tb.setObjectName("main_toolbar")
        group = QActionGroup(self)
        group.setExclusive(True)
        self.act_select = self._act(tb, "선택/이동", lambda: self._set_mode(MODE_SELECT),
                                    "S", checkable=True, checked=True)
        self.act_add = self._act(tb, "점 추가", lambda: self._set_mode(MODE_ADD),
                                 "A", checkable=True)
        for a in (self.act_select, self.act_add):
            group.addAction(a)
        tb.addSeparator()
        self._act(tb, "실측 거리 추가", lambda: self._start_observation(KIND_DISTANCE), "D")
        self._act(tb, "도면 요소 추가", self._start_feature, "G")
        tb.addSeparator()
        self._act(tb, "보정 실행", lambda: self._solve(self.solve_panel.free_names(),
                                                       self.solve_panel.chk_robust.isChecked()))

        tb2 = self.addToolBar("결과")
        tb2.setObjectName("result_toolbar")
        self.act_measure_dist = self._act(tb2, "결과에서 거리 측정",
                                          lambda: self.result_view.set_mode("distance"),
                                          checkable=True, checked=True)
        self.act_measure_area = self._act(tb2, "결과에서 면적 측정",
                                          lambda: self.result_view.set_mode("area"),
                                          checkable=True)
        g2 = QActionGroup(self)
        g2.setExclusive(True)
        g2.addAction(self.act_measure_dist)
        g2.addAction(self.act_measure_area)

    def _act(self, parent, text, slot, shortcut=None, checkable=False, checked=False):
        a = QAction(text, self)
        a.triggered.connect(lambda *_: slot())
        if shortcut:
            a.setShortcut(shortcut)
        if checkable:
            a.setCheckable(True)
            a.setChecked(checked)
        parent.addAction(a)
        return a

    # -------------------------------------------------------------- 상태 관리
    def _snapshot(self) -> None:
        self._undo.append(self.project.to_dict())
        if len(self._undo) > 80:
            self._undo.pop(0)
        self._redo.clear()
        self._dirty = True

    def _restore(self, data: dict) -> None:
        path = self.project.path
        self.project = Project.from_dict(data)
        self.project.path = path
        self.view.set_project(self.project)
        self._refresh_all()

    def _do_undo(self) -> None:
        if not self._undo:
            return
        self._redo.append(self.project.to_dict())
        self._restore(self._undo.pop())
        self.statusBar().showMessage("실행 취소", 2000)

    def _do_redo(self) -> None:
        if not self._redo:
            return
        self._undo.append(self.project.to_dict())
        self._restore(self._redo.pop())
        self.statusBar().showMessage("다시 실행", 2000)

    def _refresh_all(self) -> None:
        model = self.project.model() if self.project.solved else None
        self.point_table.refresh(self.project, model)
        self.obs_table.refresh(self.project, self.result)
        self.view.refresh()
        self._update_title()
        self._update_gsd(self.export_panel.mode_key())

    def _update_title(self) -> None:
        name = Path(self.project.path).name if self.project.path else "(저장 안 됨)"
        img = Path(self.project.image_path).name if self.project.image_path else "사진 없음"
        star = "*" if self._dirty else ""
        self.setWindowTitle(f"drone-rectify — {name}{star}  [{img}]")

    # ------------------------------------------------------------------ 모드
    def _set_mode(self, mode: str) -> None:
        self._pending_kind = None
        self.view.set_mode(mode)
        self.lbl_mode.setText({"select": "모드: 선택/이동", "add": "모드: 점 추가"}.get(mode, "모드: -"))

    def _start_observation(self, kind: str) -> None:
        if self.image is None:
            self._warn("먼저 사진을 여세요.")
            return
        title, npts, unit, tip = KIND_INFO[kind]
        self._pending_kind = kind
        self.view.set_mode(MODE_PICK, pick_count=npts, allow_create=True)
        self.tabs.setCurrentWidget(self.view)
        self.lbl_mode.setText(f"모드: {title} — 점 {npts}개 클릭")
        self.statusBar().showMessage(
            f"{title}: {tip}  점 {npts}개를 클릭하세요. (오른쪽 클릭=취소, Esc=중단)", 8000
        )
        for a in (self.act_select, self.act_add):
            a.setChecked(False)

    def _start_feature(self) -> None:
        if self.image is None:
            self._warn("먼저 사진을 여세요.")
            return
        self._pending_kind = "__feature__"
        self.view.set_mode(MODE_PICK, pick_count=0, allow_create=True)
        self.tabs.setCurrentWidget(self.view)
        self.lbl_mode.setText("모드: 도면 요소 — 점을 차례로 클릭, Enter 로 완료")
        self.statusBar().showMessage(
            "도면 요소: 점을 차례로 클릭하고 Enter 로 완료합니다. (Esc 취소)", 8000
        )

    # ------------------------------------------------------------- 점 이벤트
    def _on_point_added(self, u: float, v: float) -> None:
        self._snapshot()
        self.project.add_point(u, v)
        self._refresh_all()

    def _on_point_moved(self, pid: str, u: float, v: float) -> None:
        self._dirty = True
        self._refresh_all()

    def _on_point_edited(self, pid: str, u: float, v: float, name: str) -> None:
        p = self.project.point_by_id(pid)
        if p is None:
            return
        if abs(p.u - u) < 1e-9 and abs(p.v - v) < 1e-9 and p.name == name:
            return
        self._snapshot()
        p.u, p.v, p.name = u, v, name
        self._refresh_all()

    def _on_table_point_selected(self, pid: str) -> None:
        p = self.project.point_by_id(pid)
        if p is None:
            return
        self.view.set_selected([pid])
        self.view.center_on_point(p.u, p.v)

    def _delete_points(self, ids: list[str]) -> None:
        if not ids:
            return
        used = self.project.used_point_ids()
        linked = [i for i in ids if i in used]
        if linked:
            names = ", ".join(self.project.name_of(i) for i in linked[:6])
            if QMessageBox.question(
                self, "점 삭제",
                f"{names} 은(는) 관측/도면 요소에 사용 중입니다.\n"
                "삭제하면 해당 관측도 함께 지워집니다. 계속할까요?",
            ) != QMessageBox.StandardButton.Yes:
                return
        self._snapshot()
        for pid in ids:
            self.project.remove_point(pid)
        self.view.set_selected([])
        self._refresh_all()

    # ------------------------------------------------------------ 관측 이벤트
    def _on_points_picked(self, ids: list[str]) -> None:
        kind = self._pending_kind
        if not kind:
            return

        if kind == "__feature__":
            if len(ids) < 2:
                self.statusBar().showMessage("도면 요소는 점이 2개 이상 필요합니다.", 4000)
                return
            layer, ok = QInputDialog.getText(self, "도면 요소", "레이어 이름", text=self._last_layer)
            if not ok:
                return
            self._last_layer = layer or "0"
            closed = QMessageBox.question(
                self, "도면 요소", "닫힌 도형(폴리곤)으로 만들까요?"
            ) == QMessageBox.StandardButton.Yes
            self._snapshot()
            self.project.features.append(
                Feature(point_ids=list(ids), layer=self._last_layer, closed=closed)
            )
            self._refresh_all()
            self._start_feature()
            return

        names = [self.project.name_of(i) for i in ids]
        dlg = ObservationDialog(kind, names, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self._start_observation(kind)   # 같은 종류로 계속 입력
            return
        self._snapshot()
        self.project.observations.append(
            Observation(kind=kind, points=list(ids), value=dlg.value(),
                        sigma=dlg.sigma(), label=dlg.label())
        )
        self._refresh_all()
        self.statusBar().showMessage(
            f"{KIND_INFO[kind][0]} 추가 (총 {len(self.project.observations)}개). "
            "계속 입력하려면 클릭, 중단은 Esc.", 5000
        )
        self._start_observation(kind)

    def _delete_observations(self, ids: list[str]) -> None:
        if not ids:
            return
        self._snapshot()
        self.project.observations = [o for o in self.project.observations if o.id not in ids]
        self._refresh_all()

    def _edit_observation(self, oid: str, value: float, sigma: float) -> None:
        for o in self.project.observations:
            if o.id == oid:
                if o.kind in ("parallel", "perpendicular", KIND_ANGLE):
                    sigma = math.radians(sigma)
                if abs(o.value - value) < 1e-12 and abs((o.sigma or 0) - sigma) < 1e-15:
                    return
                self._snapshot()
                if o.kind in (KIND_DISTANCE, KIND_ANGLE):
                    o.value = value
                o.sigma = sigma
                break
        self._refresh_all()

    def _toggle_observation(self, oid: str, enabled: bool) -> None:
        for o in self.project.observations:
            if o.id == oid and o.enabled != enabled:
                self._snapshot()
                o.enabled = enabled
                break
        self._refresh_all()

    def _zoom_to_observation(self, oid: str) -> None:
        for o in self.project.observations:
            if o.id == oid and o.points:
                pts = [self.project.point_by_id(p) for p in o.points]
                pts = [p for p in pts if p]
                if not pts:
                    return
                cu = sum(p.u for p in pts) / len(pts)
                cv = sum(p.v for p in pts) / len(pts)
                self.tabs.setCurrentWidget(self.view)
                self.view.set_selected([p.id for p in pts])
                self.view.center_on_point(cu, cv)
                return

    # ------------------------------------------------------------------ 보정
    def _solve(self, free_names, robust: bool) -> None:
        if not self.project.points:
            self._warn("먼저 점과 실측값을 입력하세요.")
            return
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            res = adjust(
                self.project.pixel_map(),
                self.project.observations,
                self.project.image_width,
                self.project.image_height,
                free_names=list(free_names) if free_names else None,
                robust=robust,
            )
        except Exception as exc:
            QGuiApplication.restoreOverrideCursor()
            self._error("보정 중 오류", f"{exc}\n\n{traceback.format_exc()}")
            return
        QGuiApplication.restoreOverrideCursor()

        self.result = res
        self.solve_panel.set_result(res)
        if not res.ok and res.n_obs == 0:
            self._warn(res.message)
            self.right_tabs.setCurrentWidget(self.solve_panel)
            return
        if not res.ok:
            self.statusBar().showMessage(f"수렴 실패: {res.message}", 8000)

        self.project.params = res.params
        self.project.free_names = list(res.free_names)
        self.project.solved = True
        self._apply_gauge()
        self._dirty = True
        self.view.set_result(res)
        self._refresh_all()
        self.right_tabs.setCurrentWidget(self.solve_panel)
        if math.isfinite(res.rms_length):
            self.statusBar().showMessage(
                f"보정 완료 — 길이 잔차 RMS {res.rms_length*1000:.1f} mm, "
                f"교차검증 {res.press_rms_length*1000:.1f} mm, 자유도 {res.dof}", 12000
            )

    def _apply_gauge(self) -> None:
        g = self.project.gauge
        if not any(g.get(k) for k in ("origin_point", "axis_from", "axis_to")):
            return
        model = self.project.model()

        def uv(key):
            p = self.project.point_by_id(g.get(key, ""))
            return p.uv if p else None

        self.project.params = solve_gauge(
            model,
            origin_pix=uv("origin_point"),
            axis_from_pix=uv("axis_from"),
            axis_to_pix=uv("axis_to"),
            axis_angle_deg=float(g.get("axis_angle_deg", 0.0)),
        )

    def _edit_gauge(self) -> None:
        if not self.project.points:
            self._warn("점이 없습니다.")
            return
        dlg = GaugeDialog(self.project.points, self.project.gauge, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._snapshot()
        self.project.gauge = dlg.gauge()
        if self.project.solved:
            self._apply_gauge()
        self._refresh_all()

    def _disable_outliers(self) -> None:
        if not self.result:
            return
        bad = {s.obs_id for s in self.result.obs_stats if s.outlier}
        if not bad:
            return
        self._snapshot()
        for o in self.project.observations:
            if o.id in bad:
                o.enabled = False
        self._refresh_all()
        self._solve(self.solve_panel.free_names(), self.solve_panel.chk_robust.isChecked())
        self.statusBar().showMessage(f"{len(bad)}개 관측을 끄고 재계산했습니다.", 8000)

    # ------------------------------------------------------------------ 출력
    def _update_gsd(self, mode: str) -> None:
        if not self.project.solved or self.image is None:
            self.export_panel.set_size_text("보정을 먼저 실행하세요.")
            return
        model = self.project.model()
        if mode != "manual":
            try:
                self.export_panel.set_gsd_m(suggest_gsd(model, mode))
            except Exception:
                pass
        try:
            grid = plan_grid(model, self.export_panel.gsd_m())
            st = scale_stats(model)
            extra = ""
            if st:
                extra = (f"\n원본 지상해상도 {st['min']*1000:.1f} ~ {st['max']*1000:.1f} mm/px"
                         f"  (최대/최소 {st['ratio']:.2f})")
            self.export_panel.set_size_text(grid.describe() + extra)
        except Exception as exc:
            self.export_panel.set_size_text(str(exc))

    def _make_rectified(self, gsd: float, interp: int) -> bool:
        if self.image is None or not self.project.solved:
            self._warn("보정을 먼저 실행하세요.")
            return False
        model = self.project.model()
        try:
            grid = plan_grid(model, gsd)
        except Exception as exc:
            self._warn(str(exc))
            return False

        dlg = QProgressDialog("정사보정 래스터 생성 중...", "취소", 0, 100, self)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(300)
        cancelled = {"v": False}

        def cb(done, total):
            if dlg.wasCanceled():
                cancelled["v"] = True
                raise KeyboardInterrupt
            dlg.setValue(int(100 * done / max(total, 1)))
            QApplication.processEvents()

        try:
            rect, mask = rectify_image(self.image, model, grid, interp, progress=cb)
        except KeyboardInterrupt:
            dlg.close()
            self.statusBar().showMessage("취소되었습니다.", 4000)
            return False
        except Exception as exc:
            dlg.close()
            self._error("보정 실패", str(exc))
            return False
        dlg.setValue(100)

        self.rect_image = rect
        self.grid = grid
        self.result_view.set_image(rect, grid)
        self.tabs.setCurrentWidget(self.result_view)
        self.statusBar().showMessage(
            f"보정 결과 생성 완료 — {grid.width} x {grid.height} px, "
            f"GSD {grid.gsd*1000:.2f} mm/px", 10000
        )
        return True

    def _save_rectified(self, gsd: float, interp: int) -> None:
        if self.rect_image is None or self.grid is None:
            if not self._make_rectified(gsd, interp):
                return
        base = Path(self.project.image_path or "rectified")
        default = str(base.with_name(base.stem + "_rectified.png"))
        path, _ = QFileDialog.getSaveFileName(
            self, "보정 이미지 저장", default, "PNG (*.png);;TIFF (*.tif);;JPEG (*.jpg)"
        )
        if not path:
            return
        try:
            imwrite(path, self.rect_image)
            wf = exporters.write_world_file(path, self.grid)
        except Exception as exc:
            self._error("저장 실패", str(exc))
            return
        QMessageBox.information(
            self, "저장 완료",
            f"{Path(path).name}\n{wf.name} (월드파일)\n\n"
            f"1 픽셀 = {self.grid.gsd*1000:.2f} mm. QGIS/CAD 에서 축척이 유지된 채 열립니다."
        )

    def _export(self, kind: str) -> None:
        if not self.project.solved and kind in ("dxf", "camera"):
            self._warn("보정을 먼저 실행하세요.")
            return
        base = Path(self.project.path or self.project.image_path or "export")
        stem = base.with_suffix("")
        spec = {
            "dxf": ("DXF 저장", str(stem) + ".dxf", "DXF (*.dxf)"),
            "csv": ("성과표 저장", str(stem) + "_measurements.csv", "CSV (*.csv)"),
            "report": ("리포트 저장", str(stem) + "_report.txt", "텍스트 (*.txt)"),
            "camera": ("카메라 파라미터 저장", str(stem) + "_camera.json", "JSON (*.json)"),
        }[kind]
        path, _ = QFileDialog.getSaveFileName(self, spec[0], spec[1], spec[2])
        if not path:
            return
        try:
            if kind == "dxf":
                out = exporters.export_dxf(path, self.project, self.project.model())
            elif kind == "csv":
                out = exporters.export_measurements_csv(path, self.project, self.result)
            elif kind == "report":
                out = exporters.export_report(path, self.project, self.result, self.grid)
            else:
                f = focal_length_px(self.project.exif,
                                    self.project.image_width, self.project.image_height)
                out = exporters.export_camera_json(path, self.project, f)
        except Exception as exc:
            self._error("내보내기 실패", str(exc))
            return
        self.statusBar().showMessage(f"저장 완료: {out}", 8000)

    # ------------------------------------------------------------- 파일 입출력
    def _open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "사진 열기", "", IMAGE_FILTER)
        if not path:
            return
        self._load_image_into_new_project(path)

    def _load_image_into_new_project(self, path: str) -> None:
        try:
            img = imread(path)
        except Exception as exc:
            self._error("사진을 열 수 없습니다", str(exc))
            return
        h, w = img.shape[:2]
        self.image = img
        self.project = Project(image_path=os.path.abspath(path),
                               image_width=w, image_height=h,
                               exif=read_exif(path))
        self.result = None
        self.rect_image = None
        self.grid = None
        self._undo.clear()
        self._redo.clear()
        self._dirty = False
        self.view.load_image(img)
        self.view.set_project(self.project)
        self.view.set_result(None)
        self.solve_panel.set_result(None)
        self._refresh_all()
        self.tabs.setCurrentWidget(self.view)
        self.statusBar().showMessage(
            f"{Path(path).name} — {w} x {h} px. [관측] 탭에서 실측 거리를 추가하세요.", 10000
        )

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "프로젝트 열기", "", PROJECT_FILTER)
        if not path:
            return
        try:
            pr = Project.load(path)
        except Exception as exc:
            self._error("프로젝트를 열 수 없습니다", str(exc))
            return
        if not pr.image_path or not Path(pr.image_path).exists():
            QMessageBox.information(
                self, "사진 없음",
                f"프로젝트가 가리키는 사진을 찾을 수 없습니다:\n{pr.image_path}\n\n"
                "사진 파일을 직접 지정해 주세요."
            )
            ip, _ = QFileDialog.getOpenFileName(self, "사진 지정", "", IMAGE_FILTER)
            if not ip:
                return
            pr.image_path = os.path.abspath(ip)
        try:
            img = imread(pr.image_path)
        except Exception as exc:
            self._error("사진을 열 수 없습니다", str(exc))
            return
        h, w = img.shape[:2]
        if pr.image_width and (w != pr.image_width or h != pr.image_height):
            self._warn(
                f"사진 크기가 프로젝트 기록({pr.image_width}x{pr.image_height})과 "
                f"다릅니다({w}x{h}). 점 좌표가 맞지 않을 수 있습니다."
            )
        pr.image_width, pr.image_height = w, h
        self.image = img
        self.project = pr
        self.result = None
        self.rect_image = None
        self.grid = None
        self._undo.clear()
        self._redo.clear()
        self._dirty = False
        self.view.load_image(img)
        self.view.set_project(pr)
        self.view.set_result(None)
        self.solve_panel.set_result(None)
        self._refresh_all()
        self.statusBar().showMessage(
            f"{Path(path).name} 열기 완료 — 점 {len(pr.points)}, 관측 {len(pr.observations)}", 8000
        )

    def _save_project(self) -> None:
        if not self.project.path:
            self._save_project_as()
            return
        try:
            self.project.save(self.project.path)
        except Exception as exc:
            self._error("저장 실패", str(exc))
            return
        self._dirty = False
        self._update_title()
        self.statusBar().showMessage(f"저장 완료: {self.project.path}", 5000)

    def _save_project_as(self) -> None:
        base = Path(self.project.image_path or "project").with_suffix(FILE_SUFFIX)
        path, _ = QFileDialog.getSaveFileName(self, "프로젝트 저장", str(base), PROJECT_FILTER)
        if not path:
            return
        try:
            self.project.save(path)
        except Exception as exc:
            self._error("저장 실패", str(exc))
            return
        self._dirty = False
        self._update_title()
        self.statusBar().showMessage(f"저장 완료: {self.project.path}", 5000)

    def _make_sample(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "합성 예제를 만들 폴더 선택")
        if not folder:
            return
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            from ..tools.make_synthetic import main as gen
            gen(["--out", folder])
        except Exception as exc:
            QGuiApplication.restoreOverrideCursor()
            self._error("생성 실패", str(exc))
            return
        QGuiApplication.restoreOverrideCursor()
        proj = Path(folder) / "synthetic.drproj"
        if QMessageBox.question(
            self, "완료",
            f"합성 예제를 만들었습니다.\n{proj}\n\n지금 열어 볼까요?"
        ) == QMessageBox.StandardButton.Yes:
            try:
                pr = Project.load(str(proj))
                img = imread(pr.image_path)
                self.image = img
                self.project = pr
                self.result = None
                self._undo.clear()
                self.view.load_image(img)
                self.view.set_project(pr)
                self._refresh_all()
            except Exception as exc:
                self._error("열기 실패", str(exc))

    # --------------------------------------------------------------- 기타 UI
    def _on_cursor(self, u: float, v: float) -> None:
        self.lbl_px.setText(f"픽셀 {u:8.2f}, {v:8.2f}")
        if self.project.solved:
            xy, w = self.project.model().forward(np.array([u, v]))
            if np.all(np.isfinite(xy)):
                self.lbl_m.setText(f"미터 {xy[0]:9.3f}, {xy[1]:9.3f}")
            else:
                self.lbl_m.setText("미터 (소실선 너머)")
        else:
            self.lbl_m.setText("미터 -")

    def _fit(self) -> None:
        if self.tabs.currentWidget() is self.result_view:
            self.result_view.fit_to_window()
        else:
            self.view.fit_to_window()

    def _toggle_labels(self) -> None:
        self.view.show_labels = self.act_labels.isChecked()
        self.view.viewport().update()

    def _toggle_loupe(self) -> None:
        self.view.show_loupe = self.act_loupe.isChecked()
        self.view.viewport().update()

    def _toggle_snap(self) -> None:
        self.view.snap_enabled = self.act_snap.isChecked()

    def _show_help(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("사용법")
        dlg.resize(720, 640)
        tb = QTextBrowser(dlg)
        tb.setHtml(HELP_HTML)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(dlg.reject)
        bb.accepted.connect(dlg.accept)
        lay = QVBoxLayout(dlg)
        lay.addWidget(tb)
        lay.addWidget(bb)
        dlg.exec()

    def _show_about(self) -> None:
        from .. import __version__
        QMessageBox.about(
            self, "drone-rectify",
            f"<b>drone-rectify</b> v{__version__}<br><br>"
            "실측 기준선으로 드론/항공 사진의 원근과 렌즈 왜곡을 동시에 보정해<br>"
            "축척이 일정한 2D 도면 바탕을 만드는 도구입니다.<br><br>"
            "MIT License"
        )

    def _warn(self, msg: str) -> None:
        QMessageBox.warning(self, "확인", msg)

    def _error(self, title: str, msg: str) -> None:
        QMessageBox.critical(self, title, msg)

    def closeEvent(self, event) -> None:
        if self._dirty and self.project.points:
            r = QMessageBox.question(
                self, "종료", "저장하지 않은 변경사항이 있습니다. 저장할까요?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if r == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if r == QMessageBox.StandardButton.Save:
                self._save_project()
        event.accept()
