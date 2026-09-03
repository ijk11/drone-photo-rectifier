"""보조 대화상자."""

from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from ..core.constraints import DEFAULT_SIGMA, KIND_ANGLE, KIND_DISTANCE, KIND_INFO

__all__ = ["ObservationDialog", "GaugeDialog"]


class ObservationDialog(QDialog):
    """실측값과 관측 표준편차를 입력받는다.

    표준편차를 굳이 물어보는 이유: 줄자 실측과 레이저 거리계는 정밀도가
    10배 넘게 다르다. 이를 구분해 주어야 최소제곱이 정확한 관측에 더 큰
    비중을 두고, 잔차 통계도 통계적 의미를 갖는다.
    """

    def __init__(self, kind: str, point_names: list[str], parent=None,
                 value: float = 0.0, sigma: float | None = None):
        super().__init__(parent)
        title, npts, unit, tip = KIND_INFO[kind]
        self.kind = kind
        self.setWindowTitle(f"{title} 입력")
        self.setMinimumWidth(360)

        info = QLabel(f"{tip}\n대상 점: {' - '.join(point_names)}")
        info.setWordWrap(True)
        info.setStyleSheet("color:#9ab;")

        form = QFormLayout()
        self.spin_value = None
        if kind in (KIND_DISTANCE, KIND_ANGLE):
            self.spin_value = QDoubleSpinBox()
            self.spin_value.setDecimals(4)
            if kind == KIND_DISTANCE:
                self.spin_value.setRange(0.0001, 100000.0)
                self.spin_value.setSuffix(" m")
                self.spin_value.setValue(value or 1.0)
                self.spin_value.setToolTip("현장에서 실제로 잰 거리를 미터로 입력합니다.")
            else:
                self.spin_value.setRange(0.01, 179.99)
                self.spin_value.setSuffix(" °")
                self.spin_value.setValue(value or 90.0)
            form.addRow("실측값", self.spin_value)

        angular = kind in ("parallel", "perpendicular", "angle")
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setDecimals(4)
        if angular:
            self.spin_sigma.setRange(0.0001, 30.0)
            self.spin_sigma.setSuffix(" °")
            self.spin_sigma.setValue(
                math.degrees(sigma) if sigma else math.degrees(DEFAULT_SIGMA[kind])
            )
            self.spin_sigma.setToolTip(
                "이 기하 조건이 현실에서 얼마나 정확히 성립하는지.\n"
                "시공 오차를 감안해 0.2~0.5° 정도가 무난합니다."
            )
        else:
            self.spin_sigma.setRange(0.0001, 100.0)
            self.spin_sigma.setSuffix(" m")
            self.spin_sigma.setValue(sigma if sigma else DEFAULT_SIGMA[kind])
            self.spin_sigma.setToolTip(
                "실측 자체의 표준편차. 줄자 0.02 m, 레이저 거리계 0.003 m 정도."
            )
        form.addRow("표준편차 σ", self.spin_sigma)

        self.edit_label = QLineEdit()
        self.edit_label.setPlaceholderText("선택 사항 (예: 옹벽 상단 A-B)")
        form.addRow("메모", self.edit_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addWidget(info)
        lay.addLayout(form)
        lay.addWidget(buttons)
        if self.spin_value is not None:
            self.spin_value.setFocus()
            self.spin_value.selectAll()

    def value(self) -> float:
        return float(self.spin_value.value()) if self.spin_value is not None else 0.0

    def sigma(self) -> float:
        v = float(self.spin_sigma.value())
        return math.radians(v) if self.kind in ("parallel", "perpendicular", "angle") else v

    def label(self) -> str:
        return self.edit_label.text().strip()


class GaugeDialog(QDialog):
    """좌표축 정렬(원점과 X축 방향) 지정.

    실측 거리만으로는 지상 평면의 회전과 원점이 결정되지 않는다. 정확도와는
    무관하지만 도면으로 쓸 때는 기준이 있어야 하므로 사용자가 고르게 한다.
    """

    def __init__(self, points, gauge: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("좌표축 정렬")
        self.setMinimumWidth(400)

        info = QLabel(
            "실측 거리만으로는 도면의 회전과 원점이 정해지지 않습니다(정확도에는 "
            "영향 없음). 도면 작성에 편한 기준을 고르세요."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#9ab;")

        def make_combo(allow_empty=True):
            c = QComboBox()
            if allow_empty:
                c.addItem("(지정 안 함)", "")
            for p in points:
                c.addItem(p.name, p.id)
            return c

        self.cmb_origin = make_combo()
        self.cmb_from = make_combo()
        self.cmb_to = make_combo()
        self.spin_angle = QDoubleSpinBox()
        self.spin_angle.setRange(-360.0, 360.0)
        self.spin_angle.setDecimals(3)
        self.spin_angle.setSuffix(" °")
        self.spin_angle.setToolTip("기준 선분이 향할 방향. 0° 는 화면 오른쪽(+X).")

        for cmb, key in ((self.cmb_origin, "origin_point"),
                         (self.cmb_from, "axis_from"), (self.cmb_to, "axis_to")):
            idx = cmb.findData(gauge.get(key, ""))
            if idx >= 0:
                cmb.setCurrentIndex(idx)
        self.spin_angle.setValue(float(gauge.get("axis_angle_deg", 0.0)))

        form = QFormLayout()
        form.addRow("원점으로 쓸 점", self.cmb_origin)
        form.addRow("기준 선분 시작", self.cmb_from)
        form.addRow("기준 선분 끝", self.cmb_to)
        form.addRow("기준 선분 방위각", self.spin_angle)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addWidget(info)
        lay.addLayout(form)
        lay.addWidget(buttons)

    def gauge(self) -> dict:
        return {
            "origin_point": self.cmb_origin.currentData() or "",
            "axis_from": self.cmb_from.currentData() or "",
            "axis_to": self.cmb_to.currentData() or "",
            "axis_angle_deg": float(self.spin_angle.value()),
        }
