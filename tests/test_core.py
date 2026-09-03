"""수학 코어 검증.

정답을 아는 합성 데이터로 "보정이 실제로 참값을 복원하는가"를 확인한다.
UI 없이 돌아가므로 CI 에서 그대로 쓸 수 있다.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dronerect.core.constraints import (
    KIND_ANGLE,
    KIND_COLLINEAR,
    KIND_DISTANCE,
    KIND_PARALLEL,
    KIND_PERPENDICULAR,
    Observation,
    observation_residual,
)
from dronerect.core.distortion import distort, max_undistort_error, undistort
from dronerect.core.params import ModelParams, auto_free_params
from dronerect.core.project import Project
from dronerect.core.rectify import plan_grid, suggest_gsd, world_file_text
from dronerect.core.solver import adjust, distance_with_uncertainty, initial_params
from dronerect.core.transform import PlaneModel, solve_gauge

W, H = 4000, 3000
TRUTH = ModelParams(l1=0.10, l2=-0.24, b=0.035, log_a=0.06,
                    log_s=math.log(30.0), k1=-0.16, k2=0.045)


def make_model(params=TRUTH) -> PlaneModel:
    return PlaneModel(params, W, H)


# ---------------------------------------------------------------- 왜곡 모델
def test_distortion_roundtrip_is_exact():
    assert max_undistort_error(k1=-0.25, k2=0.08, p1=1e-3, p2=-8e-4) < 1e-10


def test_distortion_identity_when_zero():
    pts = np.random.default_rng(0).uniform(-1, 1, (50, 2))
    assert np.allclose(distort(pts), pts)
    assert np.allclose(undistort(pts), pts)


def test_distortion_is_radially_symmetric():
    """방사왜곡만 있으면 원점에서의 방향은 보존되어야 한다."""
    pts = np.array([[0.3, 0.0], [0.0, 0.3], [-0.3, 0.0], [0.2121, 0.2121]])
    d = distort(pts, k1=-0.2, k2=0.05)
    for p, q in zip(pts, d):
        cross = p[0] * q[1] - p[1] * q[0]
        assert abs(cross) < 1e-12


# ------------------------------------------------------------------- 사상
def test_forward_inverse_roundtrip():
    m = make_model()
    rng = np.random.default_rng(1)
    pix = rng.uniform([50, 50], [W - 50, H - 50], (200, 2))
    xy = m.forward_xy(pix)
    assert np.all(np.isfinite(xy))
    back = m.inverse(xy)
    assert np.max(np.abs(back - pix)) < 1e-6


def test_vanishing_line_marks_invalid_region():
    m = make_model(ModelParams(l1=0.0, l2=-1.2, log_s=0.0))
    # w = 1 + l2*mu = 0  ->  mu = 1/1.2 근처의 점은 무효여야 한다
    pix_bad = m.to_pixels(np.array([0.0, 1.0 / 1.2]))
    _, w = m.forward(pix_bad)
    assert abs(float(w)) < 1e-6


def test_local_scale_varies_with_perspective():
    """원근이 있으면 사진 위치마다 지상 해상도가 달라야 한다."""
    m = make_model()
    s_top = float(m.local_scale(np.array([W / 2, 100.0])))
    s_bot = float(m.local_scale(np.array([W / 2, H - 100.0])))
    assert s_top > 0 and s_bot > 0
    assert abs(s_top / s_bot - 1.0) > 0.2


def test_gauge_preserves_distances():
    m = make_model()
    rng = np.random.default_rng(2)
    pix = rng.uniform([100, 100], [W - 100, H - 100], (20, 2))
    before = m.forward_xy(pix)
    g = solve_gauge(m, origin_pix=pix[0], axis_from_pix=pix[0], axis_to_pix=pix[1],
                    axis_angle_deg=30.0)
    after = PlaneModel(g, W, H).forward_xy(pix)

    d0 = np.linalg.norm(before[:, None] - before[None], axis=-1)
    d1 = np.linalg.norm(after[:, None] - after[None], axis=-1)
    assert np.max(np.abs(d0 - d1)) < 1e-9
    assert np.allclose(after[0], [0.0, 0.0], atol=1e-9)
    ang = math.degrees(math.atan2(after[1][1] - after[0][1], after[1][0] - after[0][0]))
    assert abs(ang - 30.0) < 1e-6


# ------------------------------------------------------------------- 구속
def _xy(points):
    return {k: np.array(v, dtype=float) for k, v in points.items()}


def test_residual_distance():
    xy = _xy({"a": (0, 0), "b": (3, 4)})
    o = Observation(kind=KIND_DISTANCE, points=["a", "b"], value=5.0)
    assert abs(observation_residual(o, xy)) < 1e-12
    o.value = 4.9
    assert abs(observation_residual(o, xy) - 0.1) < 1e-12


def test_residual_perpendicular_and_parallel():
    xy = _xy({"a": (0, 0), "b": (1, 0), "c": (0, 0), "d": (0, 1)})
    perp = Observation(kind=KIND_PERPENDICULAR, points=["a", "b", "c", "d"])
    assert abs(observation_residual(perp, xy)) < 1e-12
    par = Observation(kind=KIND_PARALLEL, points=["a", "b", "c", "d"])
    # 직교하는 두 선분은 평행에서 90도 벗어나 있다
    assert abs(observation_residual(par, xy)) == pytest.approx(math.pi / 2)

    xy2 = _xy({"a": (0, 0), "b": (1, 0), "c": (0, 5), "d": (2, 5)})
    par2 = Observation(kind=KIND_PARALLEL, points=["a", "b", "c", "d"])
    assert abs(observation_residual(par2, xy2)) < 1e-12


def test_residual_collinear_is_signed_distance():
    xy = _xy({"a": (0, 0), "b": (1, 0.25), "c": (2, 0)})
    o = Observation(kind=KIND_COLLINEAR, points=["a", "b", "c"])
    assert abs(abs(observation_residual(o, xy)) - 0.25) < 1e-12


def test_residual_angle():
    xy = _xy({"a": (1, 0), "b": (0, 0), "c": (0, 1)})
    o = Observation(kind=KIND_ANGLE, points=["a", "b", "c"], value=90.0)
    assert abs(observation_residual(o, xy)) < 1e-12


# ---------------------------------------------------------------- 조정계산
def _synthetic_case(n_points=18, n_pairs=24, tape_sigma=0.02, pixel_noise=1.0, seed=5):
    rng = np.random.default_rng(seed)
    m = make_model()
    pix = rng.uniform([120, 120], [W - 120, H - 120], (n_points, 2))
    xy = m.forward_xy(pix)
    ids = [f"P{i}" for i in range(n_points)]
    observed = {i: p + rng.normal(0, pixel_noise, 2) for i, p in zip(ids, pix)}

    pairs: set[tuple[int, int]] = set()
    while len(pairs) < n_pairs:
        a, b = rng.integers(0, n_points, 2)
        if a != b:
            pairs.add((int(min(a, b)), int(max(a, b))))
    obs = [
        Observation(
            kind=KIND_DISTANCE,
            points=[ids[a], ids[b]],
            value=float(np.hypot(*(xy[b] - xy[a])) + rng.normal(0, tape_sigma)),
            sigma=tape_sigma,
        )
        for a, b in sorted(pairs)
    ]
    return observed, obs, m, pix, xy


def test_adjust_recovers_truth():
    observed, obs, truth_model, pix, xy = _synthetic_case()
    res = adjust(observed, obs, W, H,
                 free_names=["l1", "l2", "b", "log_a", "log_s", "k1", "k2"])
    assert res.ok
    assert res.dof == len(obs) - 7
    assert res.rank == 7

    # 관측 가중이 현실적이면 sigma0 는 1 근처
    assert 0.4 < res.sigma0 < 2.5
    # 왜곡계수는 gauge 와 무관하게 관측 가능한 값이므로 직접 비교 가능
    assert abs(res.params.k1 - TRUTH.k1) < 0.02
    assert abs(res.params.k2 - TRUTH.k2) < 0.03

    # 조정에 쓰지 않은 새 점쌍의 거리를 얼마나 맞히는가 (상대오차)
    est = PlaneModel(res.params, W, H)
    rng = np.random.default_rng(99)
    q = rng.uniform([200, 200], [W - 200, H - 200], (200, 2))
    a, b = q[0::2], q[1::2]
    dt = np.linalg.norm(truth_model.forward_xy(b) - truth_model.forward_xy(a), axis=1)
    de = np.linalg.norm(est.forward_xy(b) - est.forward_xy(a), axis=1)
    rel = np.abs(de - dt) / dt
    assert float(np.sqrt(np.mean(rel ** 2))) < 0.005      # 0.5 % 이내


def test_adjust_detects_blunder():
    """실측값 하나에 큰 오타를 넣으면 데이터 스누핑이 잡아내야 한다."""
    observed, obs, *_ = _synthetic_case()
    obs[3].value += 2.0                       # 2 m 오타
    res = adjust(observed, obs, W, H,
                 free_names=["l1", "l2", "b", "log_a", "log_s", "k1", "k2"])
    flagged = [s for s in res.obs_stats if s.outlier]
    assert flagged, "조대오차를 하나도 잡지 못했다"
    worst = max(res.obs_stats, key=lambda s: abs(s.w_test) if math.isfinite(s.w_test) else 0)
    assert worst.obs_id == obs[3].id


def test_adjust_refuses_underdetermined():
    observed, obs, *_ = _synthetic_case(n_pairs=4)
    res = adjust(observed, obs, W, H,
                 free_names=["l1", "l2", "b", "log_a", "log_s", "k1", "k2"])
    assert not res.ok
    assert "관측" in res.message


def test_adjust_zero_dof_warns():
    observed, obs, *_ = _synthetic_case(n_pairs=5)
    res = adjust(observed, obs, W, H, free_names=["l1", "l2", "b", "log_a", "log_s"])
    assert res.dof == 0
    assert any("잉여관측이 없습니다" in w for w in res.warnings)


def test_geometric_constraints_reduce_required_measurements():
    """직각/평행 구속이 실측 개수를 대신할 수 있는지."""
    rng = np.random.default_rng(12)
    m = make_model()
    # 지상에서 정확한 직사각형 두 개를 만들고 그 픽셀 좌표를 얻는다
    # 사진에 실제로 찍히는 미터 범위 안에 배치해야 한다(밖이면 외삽이 되어 무의미).
    xmin, ymin, xmax, ymax = m.metric_bounds()
    def rect(fx0, fy0, fx1, fy1):
        x0, x1 = xmin + (xmax - xmin) * fx0, xmin + (xmax - xmin) * fx1
        y0, y1 = ymin + (ymax - ymin) * fy0, ymin + (ymax - ymin) * fy1
        # 지상에서 정확한 직사각형이어야 하므로 변 길이는 축에 평행하게 잡는다
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    rects = [rect(0.10, 0.12, 0.45, 0.45), rect(0.55, 0.55, 0.90, 0.88)]
    ids, observed, xy_true = [], {}, {}
    for r, rect in enumerate(rects):
        for c, xy in enumerate(rect):
            pid = f"R{r}C{c}"
            pixel = m.inverse(xy)
            assert np.all(np.isfinite(pixel))
            ids.append(pid)
            observed[pid] = pixel + rng.normal(0, 0.5, 2)
            xy_true[pid] = xy

    obs = []
    for r in range(2):
        a, b, c, d = (f"R{r}C{i}" for i in range(4))
        obs += [
            Observation(kind=KIND_PERPENDICULAR, points=[a, b, b, c]),
            Observation(kind=KIND_PERPENDICULAR, points=[b, c, c, d]),
            Observation(kind=KIND_PARALLEL, points=[a, b, d, c]),
            Observation(kind=KIND_PARALLEL, points=[a, d, b, c]),
        ]
    # 실측 거리는 4개만
    for a, b in (("R0C0", "R0C1"), ("R0C1", "R0C2"), ("R1C0", "R1C1"), ("R0C0", "R1C2")):
        obs.append(Observation(kind=KIND_DISTANCE, points=[a, b],
                               value=float(np.hypot(*(xy_true[b] - xy_true[a]))),
                               sigma=0.01))

    res = adjust(observed, obs, W, H,
                 free_names=["l1", "l2", "b", "log_a", "log_s", "k1"])
    assert res.ok and res.rank == 6
    est = PlaneModel(res.params, W, H)
    d_true = np.hypot(*(xy_true["R1C2"] - xy_true["R0C3"]))
    p = est.forward_xy(np.array([observed["R0C3"], observed["R1C2"]]))
    d_est = float(np.hypot(*(p[1] - p[0])))
    assert abs(d_est - d_true) / d_true < 0.01


def test_uncertainty_matches_actual_error():
    """전파된 표준불확도가 실제 오차 규모와 같은 자릿수인지."""
    observed, obs, truth_model, *_ = _synthetic_case(seed=21)
    res = adjust(observed, obs, W, H,
                 free_names=["l1", "l2", "b", "log_a", "log_s", "k1", "k2"])
    est = PlaneModel(res.params, W, H)
    rng = np.random.default_rng(7)
    errs, sds = [], []
    for _ in range(25):
        a, b = rng.uniform([300, 300], [W - 300, H - 300], (2, 2))
        d_true = float(np.hypot(*(truth_model.forward_xy(b) - truth_model.forward_xy(a))))
        d, sd = distance_with_uncertainty(res, W, H, a, b)
        if math.isfinite(sd):
            errs.append(abs(d - d_true))
            sds.append(sd)
    rms_err = float(np.sqrt(np.mean(np.square(errs))))
    rms_sd = float(np.sqrt(np.mean(np.square(sds))))
    assert 0.3 < rms_err / rms_sd < 3.0


def test_auto_free_params_scales_with_observations():
    assert auto_free_params(5) == ["l1", "l2", "b", "log_a", "log_s"]
    assert "k1" in auto_free_params(10)
    assert "k2" in auto_free_params(13)
    assert set(auto_free_params(30)) >= {"k1", "k2", "k3", "p1", "p2"}


def test_initial_params_estimates_scale():
    observed, obs, *_ = _synthetic_case()
    p0 = initial_params(observed, obs, W, H)
    assert abs(p0.log_s - TRUTH.log_s) < 0.5


# -------------------------------------------------------------- 정사보정
def test_plan_grid_and_world_file():
    m = make_model()
    gsd = suggest_gsd(m, "median")
    assert gsd > 0
    grid = plan_grid(m, gsd)
    assert grid.width > 100 and grid.height > 100
    lines = world_file_text(grid).strip().splitlines()
    assert len(lines) == 6
    assert abs(float(lines[0]) - gsd) < 1e-9
    assert abs(float(lines[3]) + gsd) < 1e-9      # Y 축은 부호 반전


def test_plan_grid_rejects_absurd_size():
    m = make_model()
    with pytest.raises(ValueError, match="너무 큽니다"):
        plan_grid(m, 0.00002)


# ------------------------------------------------------------------ 프로젝트
def test_project_roundtrip(tmp_path):
    pr = Project(image_path=str(tmp_path / "a.png"), image_width=W, image_height=H)
    a = pr.add_point(10.5, 20.25)
    b = pr.add_point(30.0, 40.0)
    pr.observations.append(
        Observation(kind=KIND_DISTANCE, points=[a.id, b.id], value=12.5, sigma=0.02)
    )
    pr.params = TRUTH.copy()
    pr.solved = True
    path = tmp_path / "p.drproj"
    pr.save(str(path))

    back = Project.load(str(path))
    assert len(back.points) == 2
    assert back.points[0].u == pytest.approx(10.5)
    assert back.observations[0].value == pytest.approx(12.5)
    assert back.params.k1 == pytest.approx(TRUTH.k1)
    assert back.solved


def test_remove_point_cleans_dependents():
    pr = Project(image_width=W, image_height=H)
    a, b, c = (pr.add_point(i * 10.0, i * 10.0) for i in range(1, 4))
    pr.observations.append(Observation(kind=KIND_DISTANCE, points=[a.id, b.id], value=1.0))
    pr.remove_point(b.id)
    assert len(pr.points) == 2
    assert pr.observations == []


def test_auto_point_naming_fills_gaps():
    pr = Project()
    p1 = pr.add_point(0, 0)
    p2 = pr.add_point(1, 1)
    assert (p1.name, p2.name) == ("P1", "P2")
    pr.remove_point(p1.id)
    assert pr.add_point(2, 2).name == "P1"
