"""수치·통계의 공학적 정확성 검증.

기하가 맞는지(단위, 부호, 왕복)와 통계량이 실제로 그 뜻인지(잉여도, 교차검증,
불확도, 조대오차 탐지)를 확인한다. 여기 있는 검사는 모두 실제로 결함을 하나씩
찾아낸 것들이다.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from drone_photo_rectifier.core.constraints import (
    KIND_COLLINEAR,
    KIND_DISTANCE,
    KIND_PERPENDICULAR,
    Observation,
    observation_residual,
)
from drone_photo_rectifier.core.distortion import (
    distort,
    is_invertible,
    max_undistort_error,
    radial_slope_min,
    undistort,
)
from drone_photo_rectifier.core.params import PARAM_BOUNDS, ModelParams
from drone_photo_rectifier.core.rectify import plan_grid, world_file_text
from drone_photo_rectifier.core.solver import (
    OUTLIER_THRESHOLD,
    _adjust_once,
    params_at_bound,
)
from drone_photo_rectifier.core.transform import PlaneModel

W, H = 4000, 3000
TRUTH = ModelParams(l1=0.14, l2=-0.08, b=0.03, log_a=0.05, log_s=3.0, k1=-0.06)


def _model(**kw) -> PlaneModel:
    p = TRUTH.copy()
    for k, v in kw.items():
        setattr(p, k, v)
    return PlaneModel(p, W, H)


def _spread_layout(n=14, sigma=0.02, seed=11, blunders=0, blunder_m=0.5):
    """사진 전체에 고르게, 방향을 섞어 배치한 관측."""
    rng = np.random.default_rng(seed)
    tm = _model()
    points, obs = {}, []
    for i in range(n):
        p = np.array([250 + (i % 4) * 1150.0, 250 + (i // 4) * 850.0])
        ang = (i * 37.5) * math.pi / 180
        L = 500 + 400 * ((i * 7) % 5)
        q = np.clip(p + np.array([L * math.cos(ang), L * math.sin(ang)]),
                    30, [W - 30, H - 30])
        points[f"A{i}"], points[f"B{i}"] = p, q
        xy = tm.forward_xy(np.array([p, q]))
        d = float(np.hypot(*(xy[1] - xy[0]))) + rng.normal(0, sigma)
        if i < blunders:
            d += blunder_m
        obs.append(Observation(id=f"o{i}", kind=KIND_DISTANCE,
                               points=[f"A{i}", f"B{i}"], value=d, sigma=sigma))
    return points, obs


# ------------------------------------------------------------------ 왜곡 모델
def test_왜곡_모델이_접히는_한계를_안다():
    """k2=k3=0 이면 g'(1) = 1 + 3k1 이므로 k1 = -1/3 이 한계."""
    assert is_invertible(-0.30)
    assert not is_invertible(-0.34)
    assert radial_slope_min(-1.0 / 3.0) == pytest.approx(0.0, abs=1e-12)
    # k2 가 섞이면 한계가 달라진다 - 단일 계수 범위로는 막을 수 없다
    assert not is_invertible(-0.2, k2=-0.3)


def test_뒤집을_수_있는_범위에서는_왕복이_정확하다():
    for k1, k2 in [(-0.30, 0.0), (-0.1, 0.05), (0.0, 0.0), (0.3, -0.1), (0.6, 0.2)]:
        if not is_invertible(k1, k2):
            continue
        err = max_undistort_error(k1, k2, p1=0.001, p2=-0.001)
        assert err < 1e-10, f"k1={k1}, k2={k2} 에서 왕복오차 {err}"


def test_계수가_커도_발산하지_않는다():
    """고정점 반복은 k1 이 조금만 커져도 발산했다. 뉴턴법은 수렴한다."""
    assert max_undistort_error(2.0) < 1e-10


def test_원상이_없는_점은_NaN_으로_알린다():
    """틀린 값을 조용히 돌려주면 잘못된 보정을 알아챌 수 없다.

    k1 < 0 이면 g(r) = r(1 + k1 r^2) 에 최대값이 있어 그보다 먼 반경은
    어떤 이상점에서도 맺히지 않는다. k1 = -0.3 이면 최대가 0.7027 이다.
    """
    assert np.all(np.isfinite(undistort(np.array([[0.70, 0.0]]), k1=-0.3)))
    assert not np.all(np.isfinite(undistort(np.array([[0.75, 0.0]]), k1=-0.3)))


def test_OpenCV_규약_변환이_실제_OpenCV_와_일치한다():
    s0 = 0.5 * math.hypot(W, H)
    f = 2800.0
    p = ModelParams(k1=-0.07, k2=0.02, k3=-0.004, p1=0.0012, p2=-0.0009)
    cvd = p.to_opencv(f, s0, W, H)
    K = np.array(cvd["camera_matrix"])
    dist = np.array(cvd["dist_coeffs"])

    xy_cv = np.array([[0.3, -0.2], [-0.45, 0.35], [0.05, 0.02]])
    obj = np.concatenate([xy_cv, np.ones((len(xy_cv), 1))], axis=1)
    px_cv = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)

    nd = distort(xy_cv * (f / s0), p.k1, p.k2, p.k3, p.p1, p.p2)
    px_ours = np.stack([nd[:, 0] * s0 + W / 2.0, nd[:, 1] * s0 + H / 2.0], axis=-1)
    assert np.max(np.abs(px_cv - px_ours)) < 1e-6


def test_솔버는_접히는_영역으로_가지_않는다():
    points, obs = _spread_layout(n=16)
    res = _adjust_once(points, obs, W, H)
    assert is_invertible(res.params.k1, res.params.k2, res.params.k3)


# ---------------------------------------------------------------- 기하와 단위
def test_행렬_경로와_직접계산이_일치한다():
    m = _model(k1=0.0, k2=0.0, theta=0.4, tx=12.0, ty=-5.0)
    pts = np.array([[100.0, 200.0], [3900.0, 2800.0], [2000.0, 1500.0]])
    Hm = m.homography()
    nd = m.to_normalized(pts)
    hom = np.concatenate([nd, np.ones((len(nd), 1))], axis=1) @ Hm.T
    assert np.max(np.abs(hom[:, :2] / hom[:, 2:3] - m.forward_xy(pts))) < 1e-9


def test_local_scale_가_해석해와_같다():
    """순수 축척 모델이면 지상해상도는 정확히 s / s0 이다."""
    m = PlaneModel(ModelParams(log_s=math.log(2.5)), W, H)
    got = m.local_scale(np.array([[1234.0, 987.0], [3000.0, 2000.0]]))
    assert np.allclose(got, 2.5 / m.s0, rtol=0, atol=1e-12)


def test_직각_잔차가_라디안_단위의_각오차다():
    xy = {"A": np.array([0.0, 0.0]), "B": np.array([10.0, 0.0]),
          "C": np.array([0.0, 0.0]), "D": np.array([0.0, 10.0])}
    o = Observation(id="x", kind=KIND_PERPENDICULAR, points=["A", "B", "C", "D"])
    assert abs(observation_residual(o, xy)) < 1e-12
    th = math.radians(1.0)
    xy["D"] = np.array([10 * math.sin(th), 10 * math.cos(th)])
    assert abs(abs(observation_residual(o, xy)) - th) < 1e-9


def test_직선상_잔차가_미터_단위의_수직거리다():
    xy = {"A": np.array([0.0, 0.0]), "B": np.array([5.0, 0.3]), "C": np.array([10.0, 0.0])}
    o = Observation(id="y", kind=KIND_COLLINEAR, points=["A", "B", "C"])
    assert abs(abs(observation_residual(o, xy)) - 0.3) < 1e-12


def test_월드파일이_미터좌표와_맞는다():
    m = _model()
    grid = plan_grid(m, 0.05)
    a, _, _, e, c, f = [float(x) for x in world_file_text(grid).split()]
    col, row = 137, 89
    mx = grid.xmin + (col + 0.5) * grid.gsd
    my = grid.ymin + (row + 0.5) * grid.gsd
    assert abs(a * col + c - mx) < 1e-9
    assert abs(e * row + f + my) < 1e-9      # 월드 Y = -미터 Y


# -------------------------------------------------------------------- 통계량
def test_잉여도의_합이_자유도와_같다():
    """trace(I - H) = n - rank. 해트행렬이 맞는지 보는 가장 강한 검사."""
    points, obs = _spread_layout()
    res = _adjust_once(points, obs, W, H,
                       free_names=["l1", "l2", "b", "log_a", "log_s", "k1"])
    assert sum(s.redundancy for s in res.obs_stats) == pytest.approx(res.dof, abs=1e-9)


def test_교차검증_근사가_실제_재계산과_일치한다():
    """press = r/(1-h) 근사가 관측을 진짜로 빼고 다시 푼 결과와 맞는가."""
    free = ["l1", "l2", "b", "log_a", "log_s", "k1"]
    points, obs = _spread_layout()
    res = _adjust_once(points, obs, W, H, free_names=list(free))

    brute, approx = [], []
    for i in range(len(obs)):
        kept = [o for j, o in enumerate(obs) if j != i]
        r2 = _adjust_once(points, kept, W, H, free_names=list(free))
        assert r2.ok
        mm = PlaneModel(r2.params, W, H)
        xy = mm.forward_xy(np.array([points[f"A{i}"], points[f"B{i}"]]))
        brute.append(float(np.hypot(*(xy[1] - xy[0]))) - obs[i].value)
        approx.append(res.obs_stats[i].press)
    brute, approx = np.array(brute), np.array(approx)
    assert np.max(np.abs(approx - brute)) < 0.15 * np.max(np.abs(brute))


def test_조대오차를_실제로_찾아낸다():
    """사후 sigma0 로 나누면 오차가 클수록 분모도 커져 스스로를 가린다."""
    for n_bad in (1, 2, 3, 4):
        points, obs = _spread_layout(blunders=n_bad)
        res = _adjust_once(points, obs, W, H,
                           free_names=["l1", "l2", "b", "log_a", "log_s", "k1"])
        found = [s for s in res.obs_stats[:n_bad] if s.outlier]
        assert len(found) == n_bad, (
            f"{n_bad}개를 넣었는데 {len(found)}개만 탐지 "
            f"(sigma0={res.sigma0:.1f})")


def test_정상_데이터를_조대오차로_몰지_않는다():
    points, obs = _spread_layout(blunders=0)
    res = _adjust_once(points, obs, W, H,
                       free_names=["l1", "l2", "b", "log_a", "log_s", "k1"])
    assert not any(s.outlier for s in res.obs_stats)
    assert all(abs(s.w_test) < OUTLIER_THRESHOLD for s in res.obs_stats)


def test_보고한_불확도가_실제_산포와_맞는다():
    """몬테카를로로 확인. 공분산 = (J^T J)^-1 * sigma0^2 가 맞는지."""
    free = ["l1", "l2", "b", "log_a", "log_s", "k1"]
    N = 60
    got = {n: [] for n in free}
    rep = {n: [] for n in free}
    for k in range(N):
        points, obs = _spread_layout(seed=1000 + k)
        r = _adjust_once(points, obs, W, H, free_names=list(free))
        if not r.ok:
            continue
        for n in free:
            got[n].append(getattr(r.params, n))
            rep[n].append(r.param_std[n])
    for n in free:
        emp = float(np.std(got[n], ddof=1))
        mean_reported = float(np.mean(rep[n]))
        assert 0.6 < mean_reported / emp < 1.6, (
            f"{n}: 보고 {mean_reported:.2e} vs 실제 {emp:.2e}")


def test_경계에_붙은_값에는_표준편차를_붙이지_않는다():
    """관측이 결정해 주지 않은 값에 정밀도를 표시하면 오해를 부른다."""
    points, obs = {}, []
    tm = _model()
    rng = np.random.default_rng(2)
    for i in range(9):                     # 한 구석에 뭉친 짧은 구간들
        a, b = f"A{i}", f"B{i}"
        points[a] = np.array([900.0 + (i % 3) * 90, 1500.0 + (i // 3) * 60])
        points[b] = points[a] + np.array([55.0, 2.0])
        xy = tm.forward_xy(np.array([points[a], points[b]]))
        d = float(np.hypot(*(xy[1] - xy[0]))) + rng.normal(0, 0.02)
        obs.append(Observation(id=f"o{i}", kind=KIND_DISTANCE, points=[a, b],
                               value=d, sigma=0.02))
    res = _adjust_once(points, obs, W, H)
    hit = params_at_bound(res)
    assert hit, "이 배치라면 어딘가는 한계에 붙어야 한다"
    for n in hit:
        assert not math.isfinite(res.param_std[n])
    for n in set(res.free_names) - set(hit):
        assert math.isfinite(res.param_std[n])


def test_파라미터_범위가_물리적으로_타당하다():
    """예전 k1 범위(±2.0)는 실제 렌즈와 무관해서 해가 도망가도 몰랐다."""
    assert PARAM_BOUNDS["k1"][0] > -0.5
    assert PARAM_BOUNDS["k1"][1] < 1.0
    assert abs(PARAM_BOUNDS["p1"][0]) <= 0.1
