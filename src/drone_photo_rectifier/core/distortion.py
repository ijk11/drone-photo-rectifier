"""브라운-콘라디(Brown-Conrady) 렌즈 왜곡 모델.

좌표계 규약
-----------
* 이미지 픽셀 좌표 ``(u, v)``: 이미지 좌상단 모서리가 ``(0, 0)`` 이고
  픽셀 ``(row=i, col=j)`` 의 *중심* 이 ``(u, v) = (j + 0.5, i + 0.5)``.
* 정규화 좌표 ``(n, m)``: ``n = (u - cx) / s0``, ``m = (v - cy) / s0`` 이며
  ``s0 = 0.5 * hypot(W, H)`` (이미지 대각선의 절반).

OpenCV 는 초점거리 ``f`` 로 정규화하지만, 이 프로그램은 카메라 내부
파라미터를 모르는 상태에서도 동작해야 하므로 ``s0`` 로 정규화한다.
초점거리를 알면 두 규약은 다음으로 상호 변환된다::

    k1_cv = k1 * (f/s0)**2      k2_cv = k2 * (f/s0)**4
    k3_cv = k3 * (f/s0)**6      p_cv  = p  * (f/s0)

왜곡 방향
---------
``distort()`` 는 *이상적(왜곡 없는)* 정규화 좌표를 실제 센서에 맺히는
*관측(왜곡된)* 좌표로 보낸다. 사진에서 클릭한 점은 관측 좌표이므로,
보정 계산에는 그 역함수인 ``undistort()`` 가 쓰인다.
역함수는 닫힌 형태가 없어 고정점 반복으로 푼다(OpenCV 와 동일한 방식).
"""

from __future__ import annotations

import numpy as np

__all__ = ["distort", "undistort", "max_undistort_error"]


def distort(
    xy: np.ndarray,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> np.ndarray:
    """이상 좌표 -> 왜곡 좌표.

    Parameters
    ----------
    xy : (..., 2) 배열
        왜곡 없는 정규화 좌표.
    k1, k2, k3 : float
        방사(radial) 왜곡 계수.
    p1, p2 : float
        접선(tangential) 왜곡 계수.

    Returns
    -------
    (..., 2) 배열
    """
    xy = np.asarray(xy, dtype=np.float64)
    x = xy[..., 0]
    y = xy[..., 1]
    r2 = x * x + y * y
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.stack([xd, yd], axis=-1)


def undistort(
    xyd: np.ndarray,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
    iters: int = 30,
    tol: float = 1e-13,
) -> np.ndarray:
    """왜곡 좌표 -> 이상 좌표 (고정점 반복).

    왜곡이 0이면 즉시 입력을 반환한다. 계수가 매우 큰 경우 반복이
    발산할 수 있으므로 마지막에 유한값 여부를 확인하고, 발산한 성분은
    입력값으로 되돌린다(조정계산 중 잔차 폭발을 막기 위함).
    """
    xyd = np.asarray(xyd, dtype=np.float64)
    if k1 == 0.0 and k2 == 0.0 and k3 == 0.0 and p1 == 0.0 and p2 == 0.0:
        return xyd.copy()

    xd = xyd[..., 0]
    yd = xyd[..., 1]
    x = xd.copy()
    y = yd.copy()

    for _ in range(iters):
        r2 = x * x + y * y
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        # 반경 보정항이 0 근처면 나눗셈이 폭발하므로 하한을 둔다.
        radial = np.where(np.abs(radial) < 1e-6, np.sign(radial) * 1e-6 + 1e-12, radial)
        dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        xn = (xd - dx) / radial
        yn = (yd - dy) / radial
        step = max(float(np.max(np.abs(xn - x), initial=0.0)),
                   float(np.max(np.abs(yn - y), initial=0.0)))
        x, y = xn, yn
        if step < tol:
            break

    bad = ~(np.isfinite(x) & np.isfinite(y))
    if np.any(bad):
        x = np.where(bad, xd, x)
        y = np.where(bad, yd, y)
    return np.stack([x, y], axis=-1)


def max_undistort_error(
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
    radius: float = 1.0,
    n: int = 64,
) -> float:
    """``undistort(distort(x)) == x`` 왕복 오차의 최대값(정규화 단위).

    반복 해가 충분히 수렴했는지 자가 진단하는 용도.
    """
    t = np.linspace(-radius, radius, n)
    gx, gy = np.meshgrid(t, t)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) <= radius]
    back = undistort(distort(pts, k1, k2, k3, p1, p2), k1, k2, k3, p1, p2)
    return float(np.max(np.hypot(*(back - pts).T)))
