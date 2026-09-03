"""관측(구속조건) 정의와 잔차 계산.

이 프로그램의 입력은 "사진 위 두 점을 잇고 실측값을 적는다"가 기본이지만,
현장에서 줄자를 댈 수 없는 곳이 많으므로 **거리 이외의 기하 정보** 도 함께
쓸 수 있게 했다. 직각·평행·직선성은 인공 구조물(도로 경계, 건물 외곽,
주차구획, 옹벽)에서 공짜로 얻을 수 있는 매우 강한 구속이며, 실측 선분
개수가 부족할 때 해의 안정성을 결정적으로 높인다.

잔차는 모두 **자연 단위(미터 또는 라디안)** 로 계산한 뒤 관측 표준편차
``sigma`` 로 나눠 무차원화한다. 이렇게 해야 단위가 다른 관측을 한 번에
최소제곱으로 묶을 수 있고, 잔차 통계(단위중량 표준편차, 데이터 스누핑)가
통계적 의미를 갖는다.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "KIND_DISTANCE",
    "KIND_PARALLEL",
    "KIND_PERPENDICULAR",
    "KIND_EQUAL_LENGTH",
    "KIND_COLLINEAR",
    "KIND_ANGLE",
    "KINDS",
    "KIND_INFO",
    "DEFAULT_SIGMA",
    "Observation",
    "n_residuals",
    "observation_residual",
    "predicted_value",
    "residual_vector",
]

#: 관측 종류.
KIND_DISTANCE = "distance"
KIND_PARALLEL = "parallel"
KIND_PERPENDICULAR = "perpendicular"
KIND_EQUAL_LENGTH = "equal_length"
KIND_COLLINEAR = "collinear"
KIND_ANGLE = "angle"

KINDS = (
    KIND_DISTANCE,
    KIND_PARALLEL,
    KIND_PERPENDICULAR,
    KIND_EQUAL_LENGTH,
    KIND_COLLINEAR,
    KIND_ANGLE,
)

#: kind -> (표시명, 필요한 점 개수, 값 단위, 설명)
KIND_INFO: dict[str, tuple[str, int, str, str]] = {
    KIND_DISTANCE: ("실측 거리", 2, "m", "두 점 사이의 실제 거리를 입력한다."),
    KIND_PARALLEL: ("평행", 4, "", "선분 A-B 와 C-D 가 실제로 평행하다."),
    KIND_PERPENDICULAR: ("직각", 4, "", "선분 A-B 와 C-D 가 실제로 직교한다."),
    KIND_EQUAL_LENGTH: ("등길이", 4, "", "선분 A-B 와 C-D 의 길이가 같다(값은 몰라도 됨)."),
    KIND_COLLINEAR: ("직선상", 3, "", "점 B 가 직선 A-C 위에 있다."),
    KIND_ANGLE: ("사잇각", 3, "deg", "점 B 를 꼭짓점으로 하는 각 A-B-C 의 실제 각도."),
}

#: 관측 표준편차 기본값. 길이 계열은 미터, 각도 계열은 라디안.
DEFAULT_SIGMA: dict[str, float] = {
    KIND_DISTANCE: 0.02,                    # 줄자 실측 2 cm
    KIND_EQUAL_LENGTH: 0.02,
    KIND_COLLINEAR: 0.02,
    KIND_PARALLEL: math.radians(0.3),       # 시공 오차를 감안한 0.3도
    KIND_PERPENDICULAR: math.radians(0.3),
    KIND_ANGLE: math.radians(0.3),
}

#: 각 관측이 만드는 잔차 방정식 개수(현재는 모두 1).
_N_RES = {k: 1 for k in KINDS}


@dataclass
class Observation:
    """하나의 구속조건."""

    kind: str = KIND_DISTANCE
    points: list[str] = field(default_factory=list)
    value: float = 0.0
    """실측값. 거리는 m, 사잇각은 도(deg). 그 외 종류는 사용하지 않는다."""
    sigma: float | None = None
    """관측 표준편차. ``None`` 이면 종류별 기본값(길이 m, 각도 rad)."""
    enabled: bool = True
    label: str = ""
    note: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    # ----------------------------------------------------------------- 편의
    @property
    def n_points(self) -> int:
        return KIND_INFO[self.kind][1]

    @property
    def unit(self) -> str:
        return KIND_INFO[self.kind][2]

    def effective_sigma(self) -> float:
        if self.sigma is not None and self.sigma > 0:
            return float(self.sigma)
        return DEFAULT_SIGMA[self.kind]

    def is_complete(self) -> bool:
        """점이 모두 지정되었고, 값이 필요한 종류면 값도 들어 있는가."""
        if len(self.points) != self.n_points or any(not p for p in self.points):
            return False
        if self.kind == KIND_DISTANCE and not (self.value > 0):
            return False
        if self.kind == KIND_ANGLE and not (0.0 < self.value < 180.0):
            return False
        return True

    def describe(self, name_of=None) -> str:
        name_of = name_of or (lambda pid: pid)
        nm = [name_of(p) for p in self.points]
        title = KIND_INFO[self.kind][0]
        if self.kind == KIND_DISTANCE:
            return f"{title} {nm[0]}-{nm[1]} = {self.value:.4g} m"
        if self.kind == KIND_ANGLE:
            return f"{title} {nm[0]}-{nm[1]}-{nm[2]} = {self.value:.4g}도"
        if self.kind == KIND_COLLINEAR:
            return f"{title} {nm[1]} in {nm[0]}-{nm[2]}"
        return f"{title} ({nm[0]}-{nm[1]} | {nm[2]}-{nm[3]})"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "points": list(self.points),
            "value": float(self.value),
            "sigma": None if self.sigma is None else float(self.sigma),
            "enabled": bool(self.enabled),
            "label": self.label,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Observation":
        obs = cls(
            kind=d.get("kind", KIND_DISTANCE),
            points=list(d.get("points", [])),
            value=float(d.get("value", 0.0)),
            sigma=None if d.get("sigma") in (None, "") else float(d["sigma"]),
            enabled=bool(d.get("enabled", True)),
            label=d.get("label", ""),
            note=d.get("note", ""),
        )
        if d.get("id"):
            obs.id = str(d["id"])
        return obs


def n_residuals(kind: str) -> int:
    return _N_RES[kind]


# --------------------------------------------------------------------- 계산
def _seg(xy: dict[str, np.ndarray], a: str, b: str) -> np.ndarray:
    return xy[b] - xy[a]


def _norm(v: np.ndarray) -> float:
    return float(math.hypot(float(v[0]), float(v[1])))


def predicted_value(obs: Observation, xy: dict[str, np.ndarray]) -> float:
    """현재 모델이 예측하는 값(표시용).

    거리는 m, 각도 계열은 도(deg), 직선상은 편차 m, 등길이는 길이차 m.
    """
    p = obs.points
    if obs.kind == KIND_DISTANCE:
        return _norm(_seg(xy, p[0], p[1]))
    if obs.kind == KIND_EQUAL_LENGTH:
        return _norm(_seg(xy, p[0], p[1])) - _norm(_seg(xy, p[2], p[3]))
    if obs.kind == KIND_COLLINEAR:
        ac = _seg(xy, p[0], p[2])
        ab = _seg(xy, p[0], p[1])
        n = _norm(ac)
        if n < 1e-12:
            return 0.0
        return float(ac[0] * ab[1] - ac[1] * ab[0]) / n
    d1 = _seg(xy, p[0], p[1])
    if obs.kind == KIND_ANGLE:
        d2 = _seg(xy, p[1], p[2])
        # 꼭짓점 B 기준의 두 변 B->A, B->C
        ba = -d1
        bc = d2
        n1, n2 = _norm(ba), _norm(bc)
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        c = float(np.dot(ba, bc)) / (n1 * n2)
        return math.degrees(math.acos(max(-1.0, min(1.0, c))))
    d2 = _seg(xy, p[2], p[3])
    n1, n2 = _norm(d1), _norm(d2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cross = float(d1[0] * d2[1] - d1[1] * d2[0]) / (n1 * n2)
    dot = float(d1[0] * d2[0] + d1[1] * d2[1]) / (n1 * n2)
    if obs.kind == KIND_PARALLEL:
        return math.degrees(math.asin(max(-1.0, min(1.0, cross))))
    # KIND_PERPENDICULAR: 직각에서 벗어난 각(도)
    return math.degrees(math.asin(max(-1.0, min(1.0, dot))))


def observation_residual(obs: Observation, xy: dict[str, np.ndarray]) -> float:
    """자연 단위(미터 또는 라디안) 잔차 = 모델값 - 관측값."""
    p = obs.points
    if obs.kind == KIND_DISTANCE:
        return _norm(_seg(xy, p[0], p[1])) - float(obs.value)

    if obs.kind == KIND_EQUAL_LENGTH:
        return _norm(_seg(xy, p[0], p[1])) - _norm(_seg(xy, p[2], p[3]))

    if obs.kind == KIND_COLLINEAR:
        ac = _seg(xy, p[0], p[2])
        ab = _seg(xy, p[0], p[1])
        n = _norm(ac)
        if n < 1e-12:
            return 0.0
        # 점 B 의 직선 A-C 에 대한 부호 있는 수직거리 [m]
        return float(ac[0] * ab[1] - ac[1] * ab[0]) / n

    if obs.kind == KIND_ANGLE:
        ba = -_seg(xy, p[0], p[1])
        bc = _seg(xy, p[1], p[2])
        n1, n2 = _norm(ba), _norm(bc)
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        c = float(ba[0] * bc[0] + ba[1] * bc[1]) / (n1 * n2)
        ang = math.acos(max(-1.0, min(1.0, c)))
        return ang - math.radians(float(obs.value))

    d1 = _seg(xy, p[0], p[1])
    d2 = _seg(xy, p[2], p[3])
    n1, n2 = _norm(d1), _norm(d2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    if obs.kind == KIND_PARALLEL:
        # 평행에서 벗어난 각 [rad]. 반대 방향(반평행)도 평행으로 본다.
        cross = float(d1[0] * d2[1] - d1[1] * d2[0]) / (n1 * n2)
        return math.asin(max(-1.0, min(1.0, cross)))
    if obs.kind == KIND_PERPENDICULAR:
        # 직각에서 벗어난 각 [rad].
        dot = float(d1[0] * d2[0] + d1[1] * d2[1]) / (n1 * n2)
        return math.asin(max(-1.0, min(1.0, dot)))
    raise ValueError(f"알 수 없는 관측 종류: {obs.kind}")


def residual_vector(observations, xy: dict[str, np.ndarray]) -> np.ndarray:
    """활성 관측들의 가중 잔차 벡터(무차원)."""
    out = np.empty(len(observations), dtype=np.float64)
    for i, obs in enumerate(observations):
        out[i] = observation_residual(obs, xy) / obs.effective_sigma()
    return out
