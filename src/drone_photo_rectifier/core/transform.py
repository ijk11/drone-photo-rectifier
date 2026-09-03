"""이미지 픽셀 <-> 지상 평면(미터) 사상.

사상 순서 (forward)::

    픽셀 (u,v)
      -> 정규화      n = (u-cx)/s0,  m = (v-cy)/s0
      -> 왜곡 제거   (nu, mu) = undistort(n, m)
      -> 원근 제거   w = 1 + l1*nu + l2*mu,  xa = nu/w,  ya = mu/w
      -> 미터 변환   [X;Y] = R(theta) . s*[[1,b],[0,a]] . [xa;ya] + [tx;ty]

``w`` 는 원근 제거의 분모다. ``w = 0`` 인 궤적이 곧 **소실선(수평선)** 이며,
그 근처에서는 배율이 무한대로 발산하므로 측정이 무의미하다. 모델은
``w`` 를 함께 돌려주어 UI/보정 단계에서 유효 영역을 판정할 수 있게 한다.

주의: 이 사상은 대상이 **하나의 평면(지면)** 위에 있다고 가정한다.
높이가 있는 물체(건물 벽, 수목, 적치물 상단)는 기복변위(relief displacement)
때문에 보정 후에도 위치가 어긋난다. 실측 기준선은 반드시 지면 위에서 잡아야 한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .distortion import distort, undistort
from .params import ModelParams

__all__ = ["PlaneModel", "solve_gauge"]


@dataclass
class PlaneModel:
    """파라미터 + 이미지 크기를 묶은 사상 객체."""

    params: ModelParams
    width: int
    height: int

    # ------------------------------------------------------------ 기본 상수
    @property
    def s0(self) -> float:
        """정규화 축척(대각선의 절반) [px]."""
        return 0.5 * math.hypot(self.width, self.height)

    @property
    def center(self) -> tuple[float, float]:
        """주점(principal point) [px]."""
        p = self.params
        s0 = self.s0
        return (self.width / 2.0 + p.cx_off * s0, self.height / 2.0 + p.cy_off * s0)

    # ------------------------------------------------------------- 정규화
    def to_normalized(self, pix: np.ndarray) -> np.ndarray:
        pix = np.asarray(pix, dtype=np.float64)
        cx, cy = self.center
        s0 = self.s0
        return np.stack([(pix[..., 0] - cx) / s0, (pix[..., 1] - cy) / s0], axis=-1)

    def to_pixels(self, norm: np.ndarray) -> np.ndarray:
        norm = np.asarray(norm, dtype=np.float64)
        cx, cy = self.center
        s0 = self.s0
        return np.stack([norm[..., 0] * s0 + cx, norm[..., 1] * s0 + cy], axis=-1)

    # -------------------------------------------------------------- 행렬들
    def affine_matrix(self) -> np.ndarray:
        """R(theta) . s*[[1,b],[0,a]] (2x2)."""
        p = self.params
        s, a, b = p.s, p.a, p.b
        c, sn = math.cos(p.theta), math.sin(p.theta)
        m = np.array([[s, s * b], [0.0, s * a]], dtype=np.float64)
        r = np.array([[c, -sn], [sn, c]], dtype=np.float64)
        return r @ m

    def homography(self) -> np.ndarray:
        """왜곡 제거된 정규화 좌표 -> 미터 좌표의 3x3 호모그래피.

        렌즈 왜곡은 비선형이므로 이 행렬에는 포함되지 않는다
        (왜곡 제거가 끝난 좌표에만 적용된다).
        """
        p = self.params
        a2 = self.affine_matrix()
        h = np.eye(3)
        h[:2, :2] = a2
        h[0, 2] = p.tx
        h[1, 2] = p.ty
        proj = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [p.l1, p.l2, 1.0]])
        return h @ proj

    # -------------------------------------------------------------- forward
    def forward(self, pix: np.ndarray, eps: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
        """픽셀 -> 미터. ``(XY, w)`` 를 반환한다.

        ``|w|`` 가 0에 가까운 점은 소실선 위/너머이며 결과가 의미 없으므로
        NaN 으로 표시한다.
        """
        p = self.params
        nd = self.to_normalized(pix)
        nu = undistort(nd, p.k1, p.k2, p.k3, p.p1, p.p2)
        w = 1.0 + p.l1 * nu[..., 0] + p.l2 * nu[..., 1]
        safe = np.where(np.abs(w) < eps, np.nan, w)
        xa = nu[..., 0] / safe
        ya = nu[..., 1] / safe
        a2 = self.affine_matrix()
        x = a2[0, 0] * xa + a2[0, 1] * ya + p.tx
        y = a2[1, 0] * xa + a2[1, 1] * ya + p.ty
        return np.stack([x, y], axis=-1), w

    def forward_xy(self, pix: np.ndarray) -> np.ndarray:
        """``forward`` 의 좌표 성분만."""
        return self.forward(pix)[0]

    # -------------------------------------------------------------- inverse
    def inverse(self, xy: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        """미터 -> 픽셀 (닫힌 형태).

        원근 역변환은 ``w = 1 / (1 - l1*xa - l2*ya)`` 로 해석적으로 풀린다.
        분모가 0 이하이면 그 미터 좌표는 이미지에 대응하지 않는다(NaN).
        """
        p = self.params
        xy = np.asarray(xy, dtype=np.float64)
        a2 = self.affine_matrix()
        det = a2[0, 0] * a2[1, 1] - a2[0, 1] * a2[1, 0]
        if abs(det) < 1e-18:
            return np.full(xy.shape, np.nan)
        inv = np.array([[a2[1, 1], -a2[0, 1]], [-a2[1, 0], a2[0, 0]]]) / det
        dx = xy[..., 0] - p.tx
        dy = xy[..., 1] - p.ty
        xa = inv[0, 0] * dx + inv[0, 1] * dy
        ya = inv[1, 0] * dx + inv[1, 1] * dy

        denom = 1.0 - p.l1 * xa - p.l2 * ya
        denom = np.where(denom <= eps, np.nan, denom)
        w = 1.0 / denom
        nu = np.stack([xa * w, ya * w], axis=-1)
        nd = distort(nu, p.k1, p.k2, p.k3, p.p1, p.p2)
        return self.to_pixels(nd)

    # ---------------------------------------------------------------- 부가
    def local_scale(self, pix: np.ndarray, h: float = 0.5) -> np.ndarray:
        """해당 픽셀 위치의 지상 해상도 [m/px].

        전방사상 야코비의 행렬식 크기를 중심차분으로 구해 ``sqrt(|det J|)``
        를 반환한다. 같은 사진 안에서도 위치마다 축척이 다르다는 원근의
        본질을 사용자가 수치로 확인할 수 있게 하는 값이다.
        """
        pix = np.asarray(pix, dtype=np.float64)
        ex = np.zeros_like(pix)
        ex[..., 0] = h
        ey = np.zeros_like(pix)
        ey[..., 1] = h
        gx = (self.forward_xy(pix + ex) - self.forward_xy(pix - ex)) / (2 * h)
        gy = (self.forward_xy(pix + ey) - self.forward_xy(pix - ey)) / (2 * h)
        det = np.abs(gx[..., 0] * gy[..., 1] - gx[..., 1] * gy[..., 0])
        return np.sqrt(det)

    def vanishing_line_pixels(self, n: int = 256, span: float = 4.0) -> np.ndarray | None:
        """소실선(w=0)을 이미지 픽셀 좌표의 폴리라인으로 반환.

        화면 근방에 걸치지 않으면 ``None``. UI 에서 "이 선 근처는 신뢰할 수
        없음"을 표시하는 데 쓴다.
        """
        p = self.params
        if abs(p.l1) < 1e-12 and abs(p.l2) < 1e-12:
            return None
        t = np.linspace(-span, span, n)
        if abs(p.l2) >= abs(p.l1):
            nu = t
            mu = -(1.0 + p.l1 * t) / p.l2
        else:
            mu = t
            nu = -(1.0 + p.l2 * t) / p.l1
        nd = distort(np.stack([nu, mu], axis=-1), p.k1, p.k2, p.k3, p.p1, p.p2)
        pix = self.to_pixels(nd)
        inside = (
            (pix[:, 0] > -self.width) & (pix[:, 0] < 2 * self.width)
            & (pix[:, 1] > -self.height) & (pix[:, 1] < 2 * self.height)
        )
        if not np.any(inside):
            return None
        return pix[inside]

    def image_corners(self) -> np.ndarray:
        return np.array(
            [[0.0, 0.0], [self.width, 0.0], [self.width, self.height], [0.0, self.height]]
        )

    def metric_bounds(self, samples: int = 64) -> tuple[float, float, float, float] | None:
        """이미지 전체의 미터 좌표 바운딩 박스 ``(xmin, ymin, xmax, ymax)``.

        소실선을 넘어가는 부분은 제외한다. 유효 영역이 없으면 ``None``.
        """
        w, h = float(self.width), float(self.height)
        t = np.linspace(0.0, 1.0, samples)
        edges = np.concatenate([
            np.stack([t * w, np.zeros_like(t)], axis=-1),
            np.stack([t * w, np.full_like(t, h)], axis=-1),
            np.stack([np.zeros_like(t), t * h], axis=-1),
            np.stack([np.full_like(t, w), t * h], axis=-1),
        ])
        gx, gy = np.meshgrid(np.linspace(0, w, 24), np.linspace(0, h, 24))
        grid = np.stack([gx.ravel(), gy.ravel()], axis=-1)
        pts = np.concatenate([edges, grid])
        xy, wv = self.forward(pts)
        ok = np.isfinite(xy).all(axis=1) & (wv > 1e-6)
        if not np.any(ok):
            return None
        v = xy[ok]
        return (
            float(v[:, 0].min()), float(v[:, 1].min()),
            float(v[:, 0].max()), float(v[:, 1].max()),
        )


def solve_gauge(
    model: PlaneModel,
    origin_pix: np.ndarray | None = None,
    axis_from_pix: np.ndarray | None = None,
    axis_to_pix: np.ndarray | None = None,
    axis_angle_deg: float = 0.0,
) -> ModelParams:
    """gauge(회전.평행이동)를 사후 결정한 파라미터를 반환.

    실측 거리만으로는 지상 평면의 방향과 원점이 정해지지 않는다(관측 불가능).
    따라서 도면 작성에 편하도록 사용자가 고른 기준에 맞춰 준다.

    * ``origin_pix`` : 이 픽셀이 미터 좌표 (0, 0) 이 된다.
    * ``axis_from_pix -> axis_to_pix`` : 이 방향이 ``axis_angle_deg``
      (기본 0도, 즉 +X 축)를 향하도록 회전한다.

    거리와 각도는 모두 불변이므로 보정 정확도에는 영향을 주지 않는다.
    """
    base = model.params.copy()
    base.theta = 0.0
    base.tx = 0.0
    base.ty = 0.0
    m0 = PlaneModel(base, model.width, model.height)

    theta = 0.0
    if axis_from_pix is not None and axis_to_pix is not None:
        pa = m0.forward_xy(np.asarray(axis_from_pix, dtype=float))
        pb = m0.forward_xy(np.asarray(axis_to_pix, dtype=float))
        d = pb - pa
        if np.all(np.isfinite(d)) and float(np.hypot(d[0], d[1])) > 1e-12:
            theta = math.radians(axis_angle_deg) - math.atan2(float(d[1]), float(d[0]))

    out = base.copy()
    out.theta = theta
    if origin_pix is not None:
        m1 = PlaneModel(out, model.width, model.height)
        o = m1.forward_xy(np.asarray(origin_pix, dtype=float))
        if np.all(np.isfinite(o)):
            out.tx = -float(o[0])
            out.ty = -float(o[1])
    return out
