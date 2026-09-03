"""도움말이 코드와 어긋나지 않는지 검사한다.

설명 문구는 코드를 고쳐도 조용히 낡는다. 파라미터를 하나 추가했는데 설명이
없거나, 표에 열을 넣었는데 머리글 설명이 빠지는 식이다. 그런 누락을 사람이
알아채기 어려우므로 테스트로 막는다.
"""

from __future__ import annotations

import math

import pytest

from drone_photo_rectifier.core.params import GAUGE_PARAMS, PARAM_NAMES
from drone_photo_rectifier.core.solver import AdjustmentResult, ObsStat, verdict
from drone_photo_rectifier.ui import help_text as H


def _stat(w_test=0.5, redundancy=0.5, outlier=False, enabled=True):
    return ObsStat(
        obs_id="o", kind="distance", description="", measured=1.0, computed=1.0,
        residual=0.0, unit="m", residual_std=0.0, redundancy=redundancy,
        w_test=w_test, press=0.0, outlier=outlier, enabled=enabled,
    )


def _result(**kw):
    base = dict(
        ok=True, message="수렴", params=None, free_names=[], n_obs=20, n_params=7,
        dof=13, sigma0=1.0, rms_length=0.02, press_rms_length=0.03, cond=1e3,
        obs_stats=[_stat()],
    )
    base.update(kw)
    return AdjustmentResult(**base)


# ------------------------------------------------------------------ 문구 누락
def test_모든_파라미터에_설명이_있다():
    """자유 파라미터를 추가하면 설명도 함께 넣도록 강제한다."""
    tunable = [n for n in PARAM_NAMES if n not in GAUGE_PARAMS]
    missing = [n for n in tunable if n not in H.PARAM_HELP]
    assert not missing, f"설명이 없는 파라미터: {missing}"


def test_관측표_모든_열에_설명이_있다():
    from drone_photo_rectifier.ui.panels import ObservationTable

    missing = [c for c in ObservationTable.COLS if c not in H.OBS_COLUMNS_HELP]
    assert not missing, f"설명이 없는 열: {missing}"


def test_메인창이_참조하는_툴팁_키가_모두_존재한다():
    """``H.ACTION_TIPS["..."]`` 로 참조한 키가 실제로 있는지 확인한다.

    없으면 프로그램이 시작하다가 KeyError 로 죽는다.
    """
    import re
    from pathlib import Path

    import drone_photo_rectifier.ui.main_window as mw

    src = Path(mw.__file__).read_text(encoding="utf-8")
    used = set(re.findall(r'ACTION_TIPS\["([^"]+)"\]', src))
    assert used, "툴팁을 하나도 쓰지 않고 있다"
    missing = sorted(used - set(H.ACTION_TIPS))
    assert not missing, f"정의되지 않은 툴팁 키: {missing}"


def test_사용설명서_파일을_찾는다():
    from drone_photo_rectifier.ui.main_window import manual_path

    path = manual_path()
    assert path is not None and path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "사용설명서" in text
    # 목차 번호가 빠지지 않았는지 가볍게 확인
    for n in range(1, 13):
        assert f"## {n}. " in text, f"{n}번 항목이 없다"


# --------------------------------------------------------------------- 판정
def test_보정_전에는_안내만_한다():
    v = verdict(None)
    assert v.level == "warn" and v.advice


def test_실패하면_bad():
    v = verdict(_result(ok=False, message="발산"))
    assert v.level == "bad" and "발산" in v.headline


def test_자유도가_0이면_검증불가로_경고():
    v = verdict(_result(dof=0))
    assert v.level == "warn"
    assert "잉여관측" in v.headline
    assert math.isnan(v.accuracy)


def test_조대오차가_있으면_경고하고_건수를_알려준다():
    v = verdict(_result(obs_stats=[_stat(), _stat(w_test=9.0, outlier=True)]))
    assert v.level == "warn" and "1건" in v.headline


def test_배치가_치우치면_경고():
    v = verdict(_result(cond=1e12))
    assert v.level == "warn" and "치우" in v.headline


def test_양호하면_정확도를_mm로_알려준다():
    v = verdict(_result(press_rms_length=0.027))
    assert v.level == "good"
    assert "27 mm" in v.headline
    assert v.accuracy == pytest.approx(0.027)


def test_교차검증이_없으면_잔차로_대신한다():
    v = verdict(_result(press_rms_length=float("nan"), rms_length=0.011))
    assert v.accuracy == pytest.approx(0.011)


# ------------------------------------------------------------------- 안내판
@pytest.mark.parametrize("state", [
    H.GuideState(),
    H.GuideState(has_image=True),
    H.GuideState(has_image=True, n_distance=3),
    H.GuideState(has_image=True, n_distance=10),
    H.GuideState(has_image=True, n_distance=10, n_geometry=2),
    H.GuideState(has_image=True, n_distance=10, solved=True, verdict_level="bad"),
    H.GuideState(has_image=True, n_distance=10, solved=True,
                 verdict_level="warn", n_outliers=2, dof=4),
    H.GuideState(has_image=True, n_distance=14, solved=True,
                 verdict_level="good", dof=8, has_raster=True),
])
def test_안내판은_어떤_상태에서도_다음_할_일을_보여준다(state):
    html = H.guide_html(state)
    assert "<div class='step" in html
    # 끝난 상태가 아니면 반드시 다음 행동으로 가는 링크가 있어야 한다
    done = state.has_raster and state.verdict_level == "good"
    if not done and state.verdict_level != "bad":
        assert "act:" in html, "다음에 무엇을 눌러야 할지 알려주지 않는다"


def test_안내판_링크가_모두_처리된다():
    """``guide_html`` 이 만드는 링크를 ``_on_guide_action`` 이 전부 다루는지."""
    import re
    from pathlib import Path

    import drone_photo_rectifier.ui.main_window as mw

    states = [
        H.GuideState(),
        H.GuideState(has_image=True, n_distance=1),
        H.GuideState(has_image=True, n_distance=10),
        H.GuideState(has_image=True, n_distance=10, solved=True,
                     verdict_level="warn", n_outliers=1, dof=5),
        H.GuideState(has_image=True, n_distance=10, solved=True,
                     verdict_level="good", dof=5, has_raster=True),
    ]
    emitted = set()
    for st in states:
        emitted |= set(re.findall(r"act:([a-z_]+)", H.guide_html(st)))

    assert emitted, "안내판이 링크를 하나도 만들지 않는다"

    src = Path(mw.__file__).read_text(encoding="utf-8")
    handler = src[src.index("def _on_guide_action"):src.index("def _update_title")]
    handled = set(re.findall(r'key == "([a-z_]+)"', handler))
    assert not emitted - handled, f"처리되지 않는 링크: {sorted(emitted - handled)}"


def test_용어사전은_비어있지_않다():
    html = H.glossary_html()
    for term, _ in H.GLOSSARY:
        assert term in html
