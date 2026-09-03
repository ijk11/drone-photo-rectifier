"""오른쪽/왼쪽 도킹 패널들: 점 목록, 관측 목록, 보정, 출력."""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.constraints import KIND_ANGLE, KIND_DISTANCE, KIND_INFO
from ..core.params import PARAM_LABELS
from ..core.rectify import INTERPOLATIONS
from ..core.solver import OUTLIER_THRESHOLD

_C_BAD = QColor(255, 120, 120)
_C_WARN = QColor(255, 200, 120)
_C_DIM = QColor(150, 150, 150)


def _ro(text: str) -> QTableWidgetItem:
    it = QTableWidgetItem(text)
    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return it


# ---------------------------------------------------------------- 점 목록
class PointTable(QWidget):
    """점의 픽셀 좌표를 숫자로 직접 확인/수정한다.

    마우스로 못 찍는 정밀도(측량 성과에서 받은 좌표 등)를 입력하거나,
    미세이동 결과를 확인하기 위한 창구.
    """

    pointSelected = Signal(str)
    pointEdited = Signal(str, float, float, str)
    deleteRequested = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        self.table = QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(["이름", "u [px]", "v [px]", "미터 좌표"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                                   | QAbstractItemView.EditTrigger.EditKeyPressed)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self._on_changed)
        self.table.itemSelectionChanged.connect(self._on_selected)

        self.btn_delete = QPushButton("선택 점 삭제")
        self.btn_delete.clicked.connect(
            lambda: self.deleteRequested.emit(self.selected_ids())
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self.table)
        lay.addWidget(self.btn_delete)

    def selected_ids(self) -> list[str]:
        ids = []
        for idx in self.table.selectionModel().selectedRows():
            it = self.table.item(idx.row(), 0)
            if it:
                ids.append(it.data(Qt.ItemDataRole.UserRole))
        return ids

    def refresh(self, project, model=None) -> None:
        self._updating = True
        pts = project.points if project else []
        self.table.setRowCount(len(pts))
        xy = None
        if model is not None and pts:
            import numpy as np
            xy = model.forward_xy(np.array([[p.u, p.v] for p in pts]))
        for r, p in enumerate(pts):
            name = QTableWidgetItem(p.name)
            name.setData(Qt.ItemDataRole.UserRole, p.id)
            self.table.setItem(r, 0, name)
            self.table.setItem(r, 1, QTableWidgetItem(f"{p.u:.2f}"))
            self.table.setItem(r, 2, QTableWidgetItem(f"{p.v:.2f}"))
            if xy is not None and math.isfinite(float(xy[r][0])):
                txt = f"({xy[r][0]:.3f}, {xy[r][1]:.3f}) m"
            else:
                txt = "-"
            self.table.setItem(r, 3, _ro(txt))
        self._updating = False

    def select_ids(self, ids: list[str]) -> None:
        self._updating = True
        self.table.clearSelection()
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it and it.data(Qt.ItemDataRole.UserRole) in ids:
                self.table.selectRow(r)
        self._updating = False

    def _on_selected(self) -> None:
        if self._updating:
            return
        ids = self.selected_ids()
        if ids:
            self.pointSelected.emit(ids[0])

    def _on_changed(self, item: QTableWidgetItem) -> None:
        if self._updating:
            return
        r = item.row()
        pid_item = self.table.item(r, 0)
        if pid_item is None:
            return
        pid = pid_item.data(Qt.ItemDataRole.UserRole)
        try:
            u = float(self.table.item(r, 1).text())
            v = float(self.table.item(r, 2).text())
        except (TypeError, ValueError):
            return
        self.pointEdited.emit(pid, u, v, pid_item.text())


# ---------------------------------------------------------------- 관측 목록
class ObservationTable(QWidget):
    """실측/기하 구속의 목록과 사후 진단 표시."""

    addRequested = Signal(str)          # kind
    deleteRequested = Signal(list)      # obs ids
    valueEdited = Signal(str, float, float)   # obs id, value, sigma
    enabledToggled = Signal(str, bool)
    zoomRequested = Signal(str)

    COLS = ["사용", "종류", "점", "실측값", "σ", "모델값", "잔차", "w", "잉여도"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False

        add_box = QGroupBox("구속 추가")
        grid = QGridLayout(add_box)
        grid.setContentsMargins(6, 6, 6, 6)
        for i, (kind, (title, npts, unit, tip)) in enumerate(KIND_INFO.items()):
            b = QPushButton(title)
            b.setToolTip(f"{tip}\n필요한 점: {npts}개")
            b.clicked.connect(lambda _=False, k=kind: self.addRequested.emit(k))
            grid.addWidget(b, i // 3, i % 3)

        self.table = QTableWidget(0, len(self.COLS), self)
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                                   | QAbstractItemView.EditTrigger.EditKeyPressed)
        hh = self.table.horizontalHeader()
        # 열 폭을 내용에 맞춘다. 폭이 모자라면 가로 스크롤이 생기게 두는 편이
        # 점 이름이 "M..." 으로 잘리는 것보다 낫다.
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setStretchLastSection(False)
        self.table.itemChanged.connect(self._on_changed)
        self.table.itemDoubleClicked.connect(self._on_double)

        self.btn_del = QPushButton("선택 삭제")
        self.btn_del.clicked.connect(lambda: self.deleteRequested.emit(self.selected_ids()))
        self.btn_disable = QPushButton("선택 사용 해제")
        self.btn_disable.clicked.connect(self._disable_selected)
        row = QHBoxLayout()
        row.addWidget(self.btn_del)
        row.addWidget(self.btn_disable)

        self.hint = QLabel(
            "잔차 색: 주황 = 주의(|w|>2), 빨강 = 조대오차 의심(|w|>3.29).\n"
            "잉여도가 0에 가까운 관측은 다른 관측으로 검증되지 않습니다."
        )
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#999;")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(add_box)
        lay.addWidget(self.table, 1)
        lay.addLayout(row)
        lay.addWidget(self.hint)

    def selected_ids(self) -> list[str]:
        ids = []
        for idx in self.table.selectionModel().selectedRows():
            it = self.table.item(idx.row(), 1)
            if it:
                ids.append(it.data(Qt.ItemDataRole.UserRole))
        return ids

    def _disable_selected(self) -> None:
        for oid in self.selected_ids():
            self.enabledToggled.emit(oid, False)

    def refresh(self, project, result=None) -> None:
        self._updating = True
        obs = project.observations if project else []
        stats = {s.obs_id: s for s in (result.obs_stats if result else [])}
        self.table.setRowCount(len(obs))
        for r, o in enumerate(obs):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
                         | Qt.ItemFlag.ItemIsSelectable)
            chk.setCheckState(Qt.CheckState.Checked if o.enabled else Qt.CheckState.Unchecked)
            self.table.setItem(r, 0, chk)

            kind_item = _ro(KIND_INFO[o.kind][0])
            kind_item.setData(Qt.ItemDataRole.UserRole, o.id)
            self.table.setItem(r, 1, kind_item)
            self.table.setItem(r, 2, _ro(" ".join(project.name_of(p) for p in o.points)))

            if o.kind in (KIND_DISTANCE, KIND_ANGLE):
                self.table.setItem(r, 3, QTableWidgetItem(f"{o.value:.4f}"))
            else:
                self.table.setItem(r, 3, _ro("-"))

            sig = o.effective_sigma()
            sig_disp = sig if o.kind not in ("parallel", "perpendicular", "angle") \
                else math.degrees(sig)
            self.table.setItem(r, 4, QTableWidgetItem(f"{sig_disp:.4g}"))

            s = stats.get(o.id)
            if s is None:
                for c in (5, 6, 7, 8):
                    self.table.setItem(r, c, _ro("-"))
            else:
                self.table.setItem(r, 5, _ro(f"{s.computed:.4f}"))
                res_txt = f"{s.residual*1000:+.1f} mm" if s.unit == "m" \
                    else f"{s.residual:+.3f}°"
                res_item = _ro(res_txt)
                self.table.setItem(r, 6, res_item)
                wtxt = f"{s.w_test:+.2f}" if math.isfinite(s.w_test) else "-"
                w_item = _ro(wtxt)
                self.table.setItem(r, 7, w_item)
                red_item = _ro(f"{s.redundancy:.2f}")
                self.table.setItem(r, 8, red_item)
                if math.isfinite(s.w_test) and abs(s.w_test) > OUTLIER_THRESHOLD:
                    for c in (6, 7):
                        self.table.item(r, c).setForeground(QBrush(_C_BAD))
                elif math.isfinite(s.w_test) and abs(s.w_test) > 2.0:
                    for c in (6, 7):
                        self.table.item(r, c).setForeground(QBrush(_C_WARN))
                if s.redundancy < 0.05:
                    red_item.setForeground(QBrush(_C_BAD))

            if not o.enabled:
                for c in range(len(self.COLS)):
                    it = self.table.item(r, c)
                    if it:
                        it.setForeground(QBrush(_C_DIM))
        self._updating = False

    def _on_double(self, item: QTableWidgetItem) -> None:
        if item.column() == 2:
            kind_item = self.table.item(item.row(), 1)
            if kind_item:
                self.zoomRequested.emit(kind_item.data(Qt.ItemDataRole.UserRole))

    def _on_changed(self, item: QTableWidgetItem) -> None:
        if self._updating:
            return
        r = item.row()
        kind_item = self.table.item(r, 1)
        if kind_item is None:
            return
        oid = kind_item.data(Qt.ItemDataRole.UserRole)
        if item.column() == 0:
            self.enabledToggled.emit(oid, item.checkState() == Qt.CheckState.Checked)
            return
        if item.column() in (3, 4):
            try:
                val = float(self.table.item(r, 3).text()) \
                    if self.table.item(r, 3).text() not in ("-", "") else 0.0
                sig = float(self.table.item(r, 4).text())
            except (TypeError, ValueError):
                return
            self.valueEdited.emit(oid, val, sig)


# ------------------------------------------------------------------- 보정
class SolvePanel(QWidget):
    """자유 파라미터 선택과 조정계산 실행, 결과 요약."""

    solveRequested = Signal(list, bool)   # free_names, robust
    disableOutliersRequested = Signal()
    gaugeRequested = Signal()

    ORDER = ["l1", "l2", "b", "log_a", "log_s", "k1", "k2", "k3", "p1", "p2",
             "cx_off", "cy_off"]
    GROUPS = {
        "원근 (필수)": ["l1", "l2"],
        "아핀 / 축척 (필수)": ["b", "log_a", "log_s"],
        "렌즈 방사왜곡": ["k1", "k2", "k3"],
        "렌즈 접선왜곡": ["p1", "p2"],
        "주점 보정": ["cx_off", "cy_off"],
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.boxes: dict[str, QCheckBox] = {}

        self.chk_auto = QCheckBox("관측 수에 맞춰 자동 선택 (권장)")
        self.chk_auto.setChecked(True)
        self.chk_auto.setToolTip(
            "잉여관측이 부족한데 파라미터를 많이 풀면 과적합이 일어납니다.\n"
            "자동 선택은 관측 수를 보고 안전한 조합만 켭니다."
        )
        self.chk_auto.toggled.connect(self._sync_enabled)

        param_box = QGroupBox("자유 파라미터")
        pv = QVBoxLayout(param_box)
        pv.setContentsMargins(6, 6, 6, 6)
        pv.addWidget(self.chk_auto)
        for title, names in self.GROUPS.items():
            gl = QHBoxLayout()
            lbl = QLabel(title)
            lbl.setStyleSheet("color:#9ab;")
            gl.addWidget(lbl)
            gl.addStretch(1)
            pv.addLayout(gl)
            row = QHBoxLayout()
            for n in names:
                cb = QCheckBox(PARAM_LABELS[n][0])
                cb.setChecked(n in ("l1", "l2", "b", "log_a", "log_s", "k1", "k2"))
                self.boxes[n] = cb
                row.addWidget(cb)
            row.addStretch(1)
            pv.addLayout(row)

        self.chk_robust = QCheckBox("로버스트(soft-L1) - 조대오차 탐색용")
        self.chk_robust.setToolTip(
            "이상값의 영향을 줄여 해를 구합니다. 이상값을 찾아낸 뒤에는\n"
            "해당 관측을 끄고 일반 최소제곱으로 다시 계산하는 것이 정석입니다."
        )

        self.btn_solve = QPushButton("보정 실행  (F5)")
        self.btn_solve.setMinimumHeight(34)
        f = QFont()
        f.setBold(True)
        self.btn_solve.setFont(f)
        self.btn_solve.clicked.connect(
            lambda: self.solveRequested.emit(self.free_names(), self.chk_robust.isChecked())
        )
        self.btn_outliers = QPushButton("조대오차 의심 관측 끄고 재계산")
        self.btn_outliers.clicked.connect(self.disableOutliersRequested.emit)
        self.btn_outliers.setEnabled(False)
        self.btn_gauge = QPushButton("좌표축 정렬 (원점/방향 지정)")
        self.btn_gauge.clicked.connect(self.gaugeRequested.emit)

        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setFont(QFont("Consolas", 9))
        self.summary.setPlaceholderText(
            "실측 선분을 만들고 [보정 실행]을 누르면 결과가 여기에 표시됩니다."
        )
        self.summary.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(param_box)
        lay.addWidget(self.chk_robust)
        lay.addWidget(self.btn_solve)
        lay.addWidget(self.btn_outliers)
        lay.addWidget(self.btn_gauge)
        lay.addWidget(self.summary, 1)
        self._sync_enabled()

    def _sync_enabled(self) -> None:
        auto = self.chk_auto.isChecked()
        for n, cb in self.boxes.items():
            cb.setEnabled(not auto)

    def free_names(self) -> list:
        if self.chk_auto.isChecked():
            return []      # 빈 리스트 = 자동
        return [n for n in self.ORDER if self.boxes[n].isChecked()]

    def set_result(self, result) -> None:
        if result is None:
            self.summary.setPlainText("")
            self.btn_outliers.setEnabled(False)
            return
        lines = list(result.summary_lines())
        lines.append("")
        lines.append("[추정 파라미터]")
        for n in result.free_names:
            v = getattr(result.params, n)
            sd = result.param_std.get(n, float("nan"))
            lines.append(f"  {n:7s} {v:+12.6f}   ± {sd:.6f}")
        if result.warnings:
            lines.append("")
            lines.append("[경고]")
            for w in result.warnings:
                lines.append("  · " + w)
        self.summary.setPlainText("\n".join(lines))
        self.btn_outliers.setEnabled(any(s.outlier for s in result.obs_stats))
        if not self.chk_auto.isChecked():
            return
        for n, cb in self.boxes.items():
            cb.blockSignals(True)
            cb.setChecked(n in result.free_names)
            cb.blockSignals(False)


# ------------------------------------------------------------------- 출력
class ExportPanel(QWidget):
    """정사보정 래스터 생성과 각종 내보내기."""

    previewRequested = Signal(float, int)     # gsd, interpolation
    saveRasterRequested = Signal(float, int)
    exportRequested = Signal(str)             # 'dxf' | 'csv' | 'report' | 'camera'
    gsdModeChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        box = QGroupBox("정사보정 래스터")
        form = QFormLayout(box)

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["중앙값 (권장)", "가장 조밀한 곳", "가장 성긴 곳", "직접 입력"])
        self.cmb_mode.currentIndexChanged.connect(self._mode_changed)

        self.spin_gsd = QDoubleSpinBox()
        self.spin_gsd.setRange(0.05, 5000.0)
        self.spin_gsd.setDecimals(2)
        self.spin_gsd.setSuffix(" mm/px")
        self.spin_gsd.setValue(20.0)
        self.spin_gsd.setToolTip(
            "출력 이미지 1 픽셀이 현장에서 몇 mm 인지. 작게 할수록 선명하지만\n"
            "파일이 급격히 커집니다(면적은 제곱으로 증가)."
        )
        self.spin_gsd.valueChanged.connect(lambda _: self.sizeHintChanged())

        self.cmb_interp = QComboBox()
        self.cmb_interp.addItems(list(INTERPOLATIONS.keys()))
        self.cmb_interp.setCurrentText("Lanczos4")

        self.lbl_size = QLabel("-")
        self.lbl_size.setWordWrap(True)
        self.lbl_size.setStyleSheet("color:#9ab;")

        form.addRow("GSD 기준", self.cmb_mode)
        form.addRow("GSD", self.spin_gsd)
        form.addRow("리샘플링", self.cmb_interp)
        form.addRow("출력 크기", self.lbl_size)

        self.btn_preview = QPushButton("보정 결과 생성 / 미리보기")
        self.btn_preview.setMinimumHeight(30)
        self.btn_preview.clicked.connect(
            lambda: self.previewRequested.emit(self.gsd_m(), self.interp())
        )
        self.btn_save = QPushButton("보정 이미지 저장 (+ 월드파일)")
        self.btn_save.clicked.connect(
            lambda: self.saveRasterRequested.emit(self.gsd_m(), self.interp())
        )

        exp = QGroupBox("내보내기")
        ev = QVBoxLayout(exp)
        for key, title, tip in (
            ("dxf", "DXF 도면 (미터 단위)", "점·실측선·도면 요소를 CAD 로 내보냅니다."),
            ("csv", "관측 성과표 (CSV)", "관측별 잔차와 진단값 표."),
            ("report", "검사 리포트 (TXT)", "정확도 요약과 경고를 담은 보고서."),
            ("camera", "카메라 파라미터 (JSON)", "추정된 왜곡계수. OpenCV 규약 변환 포함."),
        ):
            b = QPushButton(title)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=key: self.exportRequested.emit(k))
            ev.addWidget(b)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(box)
        lay.addWidget(self.btn_preview)
        lay.addWidget(self.btn_save)
        lay.addWidget(exp)
        lay.addStretch(1)

    # -- API
    def gsd_m(self) -> float:
        return self.spin_gsd.value() / 1000.0

    def interp(self) -> int:
        return INTERPOLATIONS[self.cmb_interp.currentText()]

    def set_gsd_m(self, gsd: float) -> None:
        self.spin_gsd.blockSignals(True)
        self.spin_gsd.setValue(max(0.05, gsd * 1000.0))
        self.spin_gsd.blockSignals(False)

    def set_size_text(self, text: str) -> None:
        self.lbl_size.setText(text)

    def mode_key(self) -> str:
        return ["median", "finest", "coarsest", "manual"][self.cmb_mode.currentIndex()]

    def _mode_changed(self, idx: int) -> None:
        self.spin_gsd.setEnabled(idx == 3)
        self.gsdModeChanged.emit(self.mode_key())

    def sizeHintChanged(self) -> None:
        self.gsdModeChanged.emit("manual" if self.cmb_mode.currentIndex() == 3
                                 else self.mode_key())
