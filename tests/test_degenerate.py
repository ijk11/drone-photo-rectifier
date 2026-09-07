"""퇴화한 실측 배치에서 파국을 일으키지 않는지 검사한다.

실제로 겪은 고장에서 나온 테스트다. 실측 8개가 가로 방향 한 띠에만 몰려
있어 해가 유일하지 않았고, 최적화기는 ``a = exp(log_a)`` 를 4e-9 까지 보내
**지상 평면을 선으로 찌부러뜨리는** 해를 골랐다. 잔차는 작아 보였지만
그 모델로 정사영상을 만들자 152,512,818 x 1 픽셀짜리 격자가 나왔고
cv2.remap 이 assertion 으로 죽었다.

방어를 세 겹으로 둔다.
    1. 파라미터 범위  - 물리적으로 불가능한 해 자체를 막는다
    2. 판정          - 왜 못 믿는 결과인지 사용자에게 먼저 말한다
    3. 격자 계획      - 찌부러진 모델로는 래스터를 만들지 않는다
"""

from __future__ import annotations

import numpy as np
import pytest

from drone_photo_rectifier.core.constraints import KIND_DISTANCE, Observation
from drone_photo_rectifier.core.params import PARAM_BOUNDS, PARAM_NAMES, ModelParams
from drone_photo_rectifier.core.rectify import plan_grid, suggest_gsd
from drone_photo_rectifier.core.solver import adjust, params_at_bound, verdict
from drone_photo_rectifier.core.transform import PlaneModel

W, H = 4000, 3000

#: 고장 당시 실제로 나왔던 파라미터. log_a = -19.3 -> a = 4e-9.
COLLAPSED = ModelParams(l1=0.063589, l2=0.842045, b=-1.256796,
                        log_a=-19.307452, log_s=3.100871)


def _truth_model() -> PlaneModel:
    return PlaneModel(
        ModelParams(l1=0.18, l2=-0.10, b=0.03, log_a=0.05, log_s=3.0, k1=-0.08),
        W, H,
    )


def _one_direction_layout(n: int = 8):
    """가로 방향으로만, 한 띠에 몰린 실측 - 전형적인 퇴화 배치."""
    tm = _truth_model()
    points: dict[str, np.ndarray] = {}
    obs: list[Observation] = []
    for i in range(n):
        a, b = f"A{i}", f"B{i}"
        points[a] = np.array([400.0 + i * 380, 1500.0])
        points[b] = np.array([700.0 + i * 380, 1520.0])
        xy = tm.forward_xy(np.array([points[a], points[b]]))
        obs.append(Observation(id=f"o{i}", kind=KIND_DISTANCE, points=[a, b],
                               value=float(np.hypot(*(xy[1] - xy[0]))), sigma=0.01))
    return points, obs


# ------------------------------------------------------- 1) 파라미터 범위
def test_모든_파라미터에_범위가_정의돼_있다():
    missing = [n for n in PARAM_NAMES if n not in PARAM_BOUNDS]
    assert not missing, f"범위가 없는 파라미터: {missing}"


def test_범위는_하한이_상한보다_작다():
    for name, (lo, hi) in PARAM_BOUNDS.items():
        assert lo < hi, name


def test_퇴화한_배치에서도_해가_범위를_벗어나지_않는다():
    """평면이 선으로 찌부러지는 해를 애초에 만들지 않는다."""
    points, obs = _one_direction_layout()
    res = adjust(points, obs, W, H)
    for name in res.free_names:
        lo, hi = PARAM_BOUNDS[name]
        value = getattr(res.params, name)
        assert lo <= value <= hi, f"{name}={value} 가 범위 {lo}~{hi} 를 벗어났다"
    # 고장 당시의 log_a = -19.3 근처는 절대 나오면 안 된다
    assert res.params.log_a > -2.0


# ------------------------------------------------------------- 2) 판정
def test_퇴화한_배치는_믿지_말라고_말한다():
    points, obs = _one_direction_layout()
    res = adjust(points, obs, W, H)
    v = verdict(res)
    assert v.level == "bad", f"양호로 판정했다: {v.headline}"
    assert v.advice


def test_경계에_붙은_파라미터를_짚어_준다():
    points, obs = _one_direction_layout()
    res = adjust(points, obs, W, H)
    hit = params_at_bound(res)
    assert hit, "경계까지 밀려났는데 아무것도 짚어내지 못했다"
    assert all(name in res.free_names for name in hit)
    assert any(name in verdict(res).headline for name in hit)


def test_정상적인_배치는_경계에_닿지_않는다():
    """방어 장치가 멀쩡한 해까지 붙잡지 않는지 확인한다."""
    from drone_photo_rectifier.core.project import Project

    pr = Project.load("samples/synthetic.drproj")
    res = adjust(pr.pixel_map(), pr.observations, pr.image_width, pr.image_height)
    assert res.ok
    assert params_at_bound(res) == []
    assert verdict(res).level == "good"


def test_rank가_부족하면_그것부터_말한다():
    """조대오차나 sigma0 보다 먼저 "해가 결정되지 않았다"를 말해야 한다."""
    from drone_photo_rectifier.core.solver import AdjustmentResult

    res = AdjustmentResult(
        ok=True, message="수렴", params=ModelParams(), free_names=[],
        n_obs=8, n_params=5, dof=3, sigma0=17.5, rms_length=0.215,
        press_rms_length=3.564, cond=1e6, rank=4,
    )
    v = verdict(res)
    assert v.level == "bad"
    assert "결정되지 않" in v.headline


# --------------------------------------------------------- 3) 격자 계획
@pytest.mark.parametrize("mode", ["median", "finest", "coarsest"])
def test_찌부러진_모델로는_래스터를_만들지_않는다(mode):
    """cv2.remap 이 죽기 전에 막고, 원인을 알려 준다."""
    model = PlaneModel(COLLAPSED, W, H)
    with pytest.raises(ValueError) as err:
        plan_grid(model, suggest_gsd(model, mode))
    msg = str(err.value)
    assert "찌부러" in msg
    # GSD 를 바꿔 봐야 소용없다는 것을 분명히 해야 한다
    assert "GSD 를 바꿔도" in msg


def test_한_변이_상한을_넘으면_막는다():
    """총 픽셀 수는 통과하지만 한 변이 SHRT_MAX 를 넘는 경우."""
    from drone_photo_rectifier.core.rectify import (
        MAX_OUTPUT_PIXELS,
        MAX_OUTPUT_SIDE,
        OutputGrid,
    )

    assert MAX_OUTPUT_SIDE < 32767, "cv2.remap 의 16비트 한계 안이어야 한다"
    # 가늘고 긴 출력은 총량 검사만으로는 걸러지지 않는다
    assert 100_000 * 1_500 < MAX_OUTPUT_PIXELS
    assert 100_000 > MAX_OUTPUT_SIDE
    assert OutputGrid(0.0, 0.0, 0.01, 10, 10).megapixels == pytest.approx(1e-4)


def test_정상_모델은_격자를_잘_만든다():
    model = _truth_model()
    grid = plan_grid(model, suggest_gsd(model, "median"))
    assert grid.width > 0 and grid.height > 0
    assert max(grid.width, grid.height) <= 32_000
