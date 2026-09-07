"""브라운-콘라디(Brown-Conrady) 렌즈 왜곡 모델.

좌표계 규약
-----------
* 이미지 픽셀 좌표 ``(u, v)``: 이미지 좌상단 모서리가 ``(0, 0)`` 이고
  픽셀 ``(row=i, col=j)`` 의 *중심* 이 ``(u, v) = (j + 0.5, i + 0.5)``.
* 정규화 좌표 ``(n, m)``: ``n = (u - cx) / s0``, ``m = (v - cy) / s0`` 이며
  ``s0 = 0.5 * hypot(W, H)`` (이미지 대각선의 절반).
  이 규약에서 **이미지 모서리는 정확히 r = 1** 이다.

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

모델이 성립하는 범위
--------------------
방사 왜곡의 반경 사상은 ``g(r) = r * (1 + k1 r^2 + k2 r^4 + k3 r^6)`` 이다.
``g`` 가 단조증가해야만 역함수가 **존재** 한다. ``g'(r) = 1 + 3k1 r^2 +
5k2 r^4 + 7k3 r^6`` 이 어딘가에서 0 이하가 되면 그 반경에서 상이 접히고,
서로 다른 두 점이 같은 곳에 맺힌다. 예를 들어 ``k2=k3=0`` 이면
``k1 <= -1/3`` 에서 이미지 모서리(r=1)가 접힌다.

접힌 영역에서는 어떤 알고리즘도 올바른 역함수를 줄 수 없다. 예전 구현은
고정점 반복이 엉뚱한 값으로 수렴해도 그대로 돌려주었고(0.90 을 넣으면
0.58 이 나왔다), 계수가 크면 발산한 값을 조용히 내놓았다. 지금은
``is_invertible()`` 로 모델 자체가 성립하는지 확인할 수 있고,
``undistort()`` 는 수렴을 검증해 실패한 점을 NaN 으로 표시한다.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "distort",
    "undistort",
    "max_undistort_error",
    "radial_slope_min",
    "is_invertible",
]


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


def radial_slope_min(
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    r_max: float = 1.0,
    n: int = 64,
) -> float:
    """``r <= r_max`` 에서 ``g'(r) = 1 + 3k1 r^2 + 5k2 r^4 + 7k3 r^6`` 의 최소값.

    0 이하이면 그 반경에서 상이 접혀 역함수가 존재하지 않는다.
    ``r_max=1`` 이 이미지 모서리에 해당한다.
    """
    r2 = np.linspace(0.0, float(r_max) ** 2, n)
    slope = 1.0 + r2 * (3.0 * k1 + r2 * (5.0 * k2 + r2 * 7.0 * k3))
    return float(np.min(slope))


def is_invertible(
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    r_max: float = 1.0,
) -> bool:
    """이미지 범위 안에서 왜곡 모델이 뒤집을 수 있는 형태인가."""
    return radial_slope_min(k1, k2, k3, r_max) > 0.0


def undistort(
    xyd: np.ndarray,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
    iters: int = 50,
    tol: float = 1e-12,
) -> np.ndarray:
    """왜곡 좌표 -> 이상 좌표.

    감쇠 뉴턴법으로 ``distort(u) = xyd`` 를 푼다. 야코비가 해석적으로
    구해지고 대칭이라 2x2 역행렬이 닫힌 형태로 나온다. 고정점 반복은
    계수가 조금만 커져도(예: ``k1 = +2``) 발산하지만 뉴턴법은 수렴한다.

    수렴하지 못한 점은 **NaN** 으로 돌려준다. 예전처럼 틀린 값을 조용히
    돌려주면 잔차가 그럴듯해 보여서 잘못된 보정을 알아챌 수 없다.
    상위 코드는 NaN 을 이미 "이 점은 못 쓴다"는 신호로 다룬다.
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
        dradial = k1 + r2 * (2.0 * k2 + r2 * 3.0 * k3)   # dR/d(r^2)

        fx = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) - xd
        fy = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y - yd

        # 야코비 (대칭)
        jxx = radial + 2.0 * x * x * dradial + 2.0 * p1 * y + 6.0 * p2 * x
        jyy = radial + 2.0 * y * y * dradial + 6.0 * p1 * y + 2.0 * p2 * x
        jxy = 2.0 * x * y * dradial + 2.0 * p1 * x + 2.0 * p2 * y

        det = jxx * jyy - jxy * jxy
        with np.errstate(divide="ignore", invalid="ignore"):
            step_x = (jyy * fx - jxy * fy) / det
            step_y = (jxx * fy - jxy * fx) / det
        # 특이점 근처에서는 뉴턴 방향이 무의미하므로 움직이지 않는다.
        step_x = np.where(np.isfinite(step_x), step_x, 0.0)
        step_y = np.where(np.isfinite(step_y), step_y, 0.0)

        x = x - step_x
        y = y - step_y
        if max(float(np.max(np.abs(step_x), initial=0.0)),
               float(np.max(np.abs(step_y), initial=0.0))) < tol:
            break

    # 정말 풀렸는지 정방향으로 되짚어 확인한다.
    check = distort(np.stack([x, y], axis=-1), k1, k2, k3, p1, p2)
    bad = (~np.isfinite(x) | ~np.isfinite(y)
           | (np.abs(check[..., 0] - xd) > 1e-8)
           | (np.abs(check[..., 1] - yd) > 1e-8))
    if np.any(bad):
        x = np.where(bad, np.nan, x)
        y = np.where(bad, np.nan, y)
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

    반복 해가 충분히 수렴했는지 자가 진단하는 용도. 모델이 접혀 있으면
    ``undistort`` 가 NaN 을 돌려주므로 ``inf`` 가 된다.
    """
    t = np.linspace(-radius, radius, n)
    gx, gy = np.meshgrid(t, t)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) <= radius]
    back = undistort(distort(pts, k1, k2, k3, p1, p2), k1, k2, k3, p1, p2)
    err = np.hypot(*(back - pts).T)
    if not np.all(np.isfinite(err)):
        return float("inf")
    return float(np.max(err))
