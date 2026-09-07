"""보정 모델 파라미터 정의와 벡터 패킹.

파라미터화 근거
---------------
이미지 평면 -> 지상 평면 사상은 일반적으로 8자유도 호모그래피다. 그런데
관측값이 "두 점 사이의 실측 거리"뿐이라면, 지상 평면을 통째로 회전/평행이동
시켜도 모든 거리가 그대로이므로 **3자유도(회전 1 + 평행이동 2)는 원리적으로
관측 불가능(gauge freedom)** 하다. 즉 실제로 결정 가능한 것은 8 - 3 = 5 자유도다.

그래서 호모그래피 9개 원소를 그대로 풀지 않고, 관측 가능한 5개만 최소
파라미터로 직접 다룬다::

    H = T(tx,ty) · R(theta) · A(s, a, b) · P(l1, l2)

* ``P`` : 소실선(vanishing line) 2자유도 - 원근을 제거해 아핀 좌표로 만든다.
      P = [[1,0,0],[0,1,0],[l1,l2,1]]
* ``A`` : 아핀 -> 미터 3자유도 - 축척 s(1) + 종횡비/전단 a,b(2).
      A = [[s, s*b, 0],[0, s*a, 0],[0,0,1]]
* ``R``, ``T`` : gauge(회전·평행이동) 3자유도. 실측 거리만 있을 때는
      관측 불가능하므로 기본적으로 **고정** 하고, 계산이 끝난 뒤 사용자가
      고른 기준선/원점에 맞춰 사후 정렬한다. 측량 좌표(GCP)를 입력하는
      경우에만 자유 파라미터로 풀린다.

여기에 렌즈 왜곡 ``k1, k2, k3, p1, p2`` 와 주점 보정 ``cx_off, cy_off`` 가
선택적으로 붙는다. 기본 자유 파라미터는 ``l1, l2, b, log_a, log_s, k1, k2``
7개이며, 이는 **실측 선분 7개 이상** 이면 풀린다는 뜻이다.

축척 ``s`` 와 종횡비 ``a`` 는 항상 양수여야 하고(음수면 좌우가 뒤집힌 해),
값의 범위가 넓으므로 로그로 다룬다: ``s = exp(log_s)``, ``a = exp(log_a)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, fields, replace
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "PARAM_NAMES",
    "PARAM_LABELS",
    "DEFAULT_FREE",
    "GAUGE_PARAMS",
    "PARAM_BOUNDS",
    "bounds_for",
    "ModelParams",
    "auto_free_params",
]

#: 파라미터 벡터의 정식 순서.
PARAM_NAMES: tuple[str, ...] = (
    "l1", "l2",              # 소실선 (원근)
    "b", "log_a", "log_s",   # 아핀 -> 미터
    "k1", "k2", "k3",        # 방사 왜곡
    "p1", "p2",              # 접선 왜곡
    "cx_off", "cy_off",      # 주점 보정 (정규화 단위)
    "theta", "tx", "ty",     # gauge
)

#: UI 표시용 한글 라벨과 단위.
PARAM_LABELS: dict[str, tuple[str, str]] = {
    "l1": ("소실선 l1", "원근"),
    "l2": ("소실선 l2", "원근"),
    "b": ("전단 b", "아핀"),
    "log_a": ("종횡비 log a", "아핀"),
    "log_s": ("축척 log s", "m/정규화"),
    "k1": ("방사왜곡 k1", "렌즈"),
    "k2": ("방사왜곡 k2", "렌즈"),
    "k3": ("방사왜곡 k3", "렌즈"),
    "p1": ("접선왜곡 p1", "렌즈"),
    "p2": ("접선왜곡 p2", "렌즈"),
    "cx_off": ("주점 dx", "정규화"),
    "cy_off": ("주점 dy", "정규화"),
    "theta": ("회전 theta", "rad"),
    "tx": ("평행이동 tx", "m"),
    "ty": ("평행이동 ty", "m"),
}

#: 실측 거리만 있을 때 기본으로 풀 파라미터(7개).
DEFAULT_FREE: tuple[str, ...] = ("l1", "l2", "b", "log_a", "log_s", "k1", "k2")

#: 거리 관측만으로는 결정되지 않는 gauge 파라미터.
GAUGE_PARAMS: frozenset[str] = frozenset({"theta", "tx", "ty"})

#: 파라미터가 물리적으로 가질 수 있는 범위.
#:
#: 관측이 부족하거나 방향이 치우치면 최소제곱해가 유일하지 않다. 그때
#: 최적화기는 잔차만 줄이면 되므로 ``a = exp(log_a)`` 를 0 으로 보내
#: **지상 평면을 선으로 찌부러뜨리는** 해를 찾아내기도 한다. 잔차는 작지만
#: 물리적으로 무의미하고, 그 모델로 정사영상을 만들면 폭이 수십만 픽셀인
#: 래스터가 나와 cv2.remap 이 죽는다(실제로 겪은 고장이다).
#:
#: 그래서 "실제 카메라라면 이 범위를 벗어날 수 없다"는 선을 걸어 둔다.
#: 넉넉하게 잡아 정상적인 촬영은 건드리지 않으면서 파국적인 해만 막는다.
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    # 소실선. |l| > 1 이면 소실선이 사진 안을 지날 만큼 비스듬한 촬영이다.
    # 그보다 훨씬 넉넉하게 둔다.
    "l1": (-5.0, 5.0),
    "l2": (-5.0, 5.0),
    # 원근을 제거하고 남는 아핀은 실제 카메라라면 닮음에 가깝다. gauge 로
    # 회전을 고정했으므로 전단 b 는 0, 종횡비 a 는 1 근처여야 한다.
    "b": (-2.0, 2.0),
    "log_a": (-1.5, 1.5),        # a = 0.22 ~ 4.5 배
    "log_s": (-20.0, 20.0),      # 축척은 현장 크기에 따라 크게 달라진다
    # 렌즈 왜곡(대각선 절반으로 정규화한 규약). 어안이라도 이 범위를 넘지 않는다.
    "k1": (-2.0, 2.0),
    "k2": (-2.0, 2.0),
    "k3": (-2.0, 2.0),
    "p1": (-0.5, 0.5),
    "p2": (-0.5, 0.5),
    # 주점은 사진 중심 근처다. 정규화 단위로 0.5 면 이미 모서리 쪽이다.
    "cx_off": (-0.5, 0.5),
    "cy_off": (-0.5, 0.5),
    # gauge
    "theta": (-math.pi, math.pi),
    "tx": (-1.0e6, 1.0e6),
    "ty": (-1.0e6, 1.0e6),
}


def bounds_for(names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """자유 파라미터 벡터에 대응하는 (하한, 상한) 배열."""
    lo = np.array([PARAM_BOUNDS[n][0] for n in names], dtype=np.float64)
    hi = np.array([PARAM_BOUNDS[n][1] for n in names], dtype=np.float64)
    return lo, hi


@dataclass
class ModelParams:
    """보정 모델의 전체 파라미터 집합."""

    l1: float = 0.0
    l2: float = 0.0
    b: float = 0.0
    log_a: float = 0.0
    log_s: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    cx_off: float = 0.0
    cy_off: float = 0.0
    theta: float = 0.0
    tx: float = 0.0
    ty: float = 0.0

    # ------------------------------------------------------------------ 편의
    @property
    def s(self) -> float:
        """축척 [m / 정규화 단위]."""
        return math.exp(self.log_s)

    @property
    def a(self) -> float:
        """종횡비(아핀 -> 미터)."""
        return math.exp(self.log_a)

    def copy(self) -> "ModelParams":
        return replace(self)

    def as_dict(self) -> dict[str, float]:
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "ModelParams":
        known = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in d.items() if k in known})

    # ------------------------------------------------------- 벡터 <-> 파라미터
    def to_vector(self, names: Sequence[str]) -> np.ndarray:
        return np.array([getattr(self, n) for n in names], dtype=np.float64)

    def with_vector(self, names: Sequence[str], x: Iterable[float]) -> "ModelParams":
        """``names`` 위치만 ``x`` 로 바꾼 새 파라미터를 반환."""
        out = self.copy()
        for n, v in zip(names, x):
            setattr(out, n, float(v))
        return out

    # --------------------------------------------------------- OpenCV 변환
    def to_opencv(self, focal_px: float, s0: float, width: int, height: int) -> dict:
        """초점거리를 알 때 OpenCV 규약의 내부/왜곡 파라미터로 환산.

        본 모델은 대각선 절반 ``s0`` 로 정규화하지만 OpenCV 는 초점거리
        ``f`` 로 정규화한다. 두 규약의 계수는 ``t = f / s0`` 에 대해
        ``k1_cv = k1*t^2``, ``k2_cv = k2*t^4``, ``k3_cv = k3*t^6``,
        ``p_cv = p*t`` 로 대응된다.
        """
        t = float(focal_px) / float(s0)
        return {
            "image_size": [int(width), int(height)],
            "camera_matrix": [
                [focal_px, 0.0, width / 2.0 + self.cx_off * s0],
                [0.0, focal_px, height / 2.0 + self.cy_off * s0],
                [0.0, 0.0, 1.0],
            ],
            "dist_coeffs": [
                self.k1 * t ** 2,
                self.k2 * t ** 4,
                self.p1 * t,
                self.p2 * t,
                self.k3 * t ** 6,
            ],
            "convention": "OpenCV (k1, k2, p1, p2, k3)",
        }


def auto_free_params(n_obs: int, allow_distortion: bool = True) -> list[str]:
    """관측 수에 맞춰 과적합을 피하는 자유 파라미터 조합을 고른다.

    조정계산에서 자유도(잉여관측) = 관측수 - 미지수 이며, 자유도가 0이면
    잔차가 무조건 0이 되어 정확도를 **평가할 수 없다**. 최소 3 이상의
    잉여를 남기도록 단계적으로 파라미터를 추가한다.
    """
    base = ["l1", "l2", "b", "log_a", "log_s"]   # 원근 + 아핀 (5)
    if not allow_distortion:
        return base
    if n_obs >= 9:
        base.append("k1")
    if n_obs >= 12:
        base.append("k2")
    if n_obs >= 16:
        base += ["p1", "p2"]
    if n_obs >= 22:
        base.append("k3")
    return base
