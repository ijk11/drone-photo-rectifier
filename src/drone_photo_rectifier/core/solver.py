"""비선형 최소제곱 조정계산(adjustment)과 정확도 통계.

측량에서 쓰는 방식 그대로다. 관측(실측 거리, 직각, 평행...)마다 표준편차를
주고 가중 잔차를 최소화한 뒤, 남은 잔차로 **해가 얼마나 믿을 만한지** 를
숫자로 돌려준다. 보정 자체보다 이 진단부가 실무에서 더 중요하다.
잔차가 0이라고 정확한 게 아니라, **잉여관측이 있는데도** 잔차가 작아야
정확한 것이기 때문이다.

산출되는 진단값
---------------
* ``sigma0`` : 단위중량 표준편차. 입력한 sigma 가 현실적이면 1 근처가 나온다.
  1보다 훨씬 크면 실측값에 오류가 있거나 모델(단일 평면 가정)이 안 맞는 것이고,
  1보다 훨씬 작으면 sigma 를 지나치게 크게 잡은 것이다.
* ``redundancy`` r_i : 각 관측의 잉여도(0~1). 0에 가까우면 그 관측은 다른
  관측으로 검증되지 않으므로 **틀려도 알 수 없다**. 실측 배치가 좋은지
  판단하는 지표.
* ``w_test`` : Baarda 데이터 스누핑 표준화 잔차. |w| > 3.29 면 조대오차(오타,
  엉뚱한 점 지정, 지면이 아닌 곳 측정) 의심.
* ``press_rms`` : Leave-one-out 예측잔차 RMS. "이 관측을 빼고 맞춘 모델이
  그 관측을 얼마나 틀리게 예측하는가"로, 내부 잔차보다 정직한 정확도 추정치.
* ``cond`` : 야코비 조건수. 크면 실측 선분이 한쪽에 몰려 있거나 방향이
  편중되어 해가 불안정하다는 뜻.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .constraints import (
    KIND_ANGLE,
    KIND_DISTANCE,
    KIND_PARALLEL,
    KIND_PERPENDICULAR,
    Observation,
    observation_residual,
    predicted_value,
)
from .distortion import radial_slope_min
from .params import PARAM_BOUNDS, ModelParams, auto_free_params, bounds_for
from .transform import PlaneModel

__all__ = [
    "ObsStat",
    "AdjustmentResult",
    "initial_params",
    "adjust",
    "distance_with_uncertainty",
    "Measurement",
    "measure",
    "Verdict",
    "verdict",
    "OUTLIER_THRESHOLD",
]

#: Baarda 데이터 스누핑 임계값 (유의수준 0.1%).
OUTLIER_THRESHOLD = 3.29

#: scipy 종료 메시지를 사용자에게 보일 한국어로 바꾼다.
#: 원문을 그대로 두면 "ftol termination condition" 같은 문구가 화면에 뜬다.
_TERMINATION_KO = {
    "ftol": "잔차가 더 줄지 않는 지점까지 수렴했습니다.",
    "xtol": "파라미터가 더 움직이지 않는 지점까지 수렴했습니다.",
    "gtol": "기울기가 0에 가까워 수렴했습니다.",
    "maxfev": "반복 한도에 걸렸습니다. 관측이나 초기값을 확인하세요.",
}


def _termination_ko(message: str) -> str:
    """scipy 종료 사유를 한국어 한 문장으로."""
    low = message.lower()
    for key, text in _TERMINATION_KO.items():
        if key in low:
            return text
    if "maximum number" in low:
        return _TERMINATION_KO["maxfev"]
    return message

#: 원근 분모 w 의 하한. 이보다 작아지면 소실선에 접근한 것으로 보고 벌점.
_W_MIN = 0.05
#: 방사 왜곡 사상 g(r) 의 기울기 하한. 0 이하면 이미지가 접혀 역함수가
#: 존재하지 않으므로, 최적화가 그 영역에 들어가지 못하게 장벽을 세운다.
_SLOPE_MIN = 0.05

#: 경계에 "붙었다"고 볼 여유. 최적화기는 경계에 정확히 착지하지 않고
#: 아주 조금 안쪽에서 멈추므로 범위 폭에 비례한 허용오차로 판정한다.
_BOUND_TOL_FRAC = 1e-3


def _is_at_bound(name: str, value: float) -> bool:
    """파라미터가 물리적 한계에 밀려 붙었는가."""
    lo, hi = PARAM_BOUNDS.get(name, (-math.inf, math.inf))
    if not math.isfinite(value) or not math.isfinite(lo) or not math.isfinite(hi):
        return False
    tol = _BOUND_TOL_FRAC * max(hi - lo, 1e-12)
    return value <= lo + tol or value >= hi - tol
_BARRIER_WEIGHT = 1.0e3

#: 각도 계열 관측(잔차가 라디안 단위).
_ANGULAR = {KIND_PARALLEL, KIND_PERPENDICULAR, KIND_ANGLE}


@dataclass
class ObsStat:
    """관측 하나에 대한 사후 진단."""

    obs_id: str
    kind: str
    description: str
    measured: float | None
    computed: float
    residual: float
    """자연 단위 잔차(길이 m / 각도 deg), 모델값 - 관측값."""
    unit: str
    residual_std: float
    """가중(무차원) 잔차."""
    redundancy: float
    w_test: float
    press: float
    """Leave-one-out 예측잔차(자연 단위)."""
    outlier: bool
    enabled: bool = True


@dataclass
class AdjustmentResult:
    ok: bool
    message: str
    params: ModelParams
    free_names: list[str]
    n_obs: int = 0
    n_params: int = 0
    dof: int = 0
    sigma0: float = float("nan")
    rms_length: float = float("nan")
    max_length: float = float("nan")
    rms_angle: float = float("nan")
    max_angle: float = float("nan")
    press_rms_length: float = float("nan")
    cond: float = float("nan")
    rank: int = 0
    param_std: dict[str, float] = field(default_factory=dict)
    cov: np.ndarray | None = None
    obs_stats: list[ObsStat] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_eval: int = 0

    def summary_lines(self) -> list[str]:
        """UI/리포트용 요약 문자열."""
        out = [
            f"상태        : {'성공' if self.ok else '실패'} - {self.message}",
            f"관측 / 미지수 : {self.n_obs} / {self.n_params}  (자유도 {self.dof})",
        ]
        if math.isfinite(self.sigma0):
            out.append(f"단위중량 표준편차 sigma0 : {self.sigma0:.3f}  (1에 가까울수록 적정)")
        if math.isfinite(self.rms_length):
            out.append(f"길이 잔차 RMS : {self.rms_length * 1000:.1f} mm   (최대 {self.max_length * 1000:.1f} mm)")
        if math.isfinite(self.press_rms_length):
            out.append(f"교차검증 RMS  : {self.press_rms_length * 1000:.1f} mm  (leave-one-out)")
        if math.isfinite(self.rms_angle):
            out.append(f"각도 잔차 RMS : {self.rms_angle:.3f} deg (최대 {self.max_angle:.3f} deg)")
        if math.isfinite(self.cond):
            out.append(f"조건수        : {self.cond:.3g}  (rank {self.rank}/{self.n_params})")
        return out


# ------------------------------------------------------------------ 내부 도구
def _num_jac(fun, x: np.ndarray, base: np.ndarray | None = None) -> np.ndarray:
    """중심차분 야코비. 파라미터 수가 작아 비용이 문제되지 않는다."""
    x = np.asarray(x, dtype=np.float64)
    if base is None:
        base = fun(x)
    m = base.size
    n = x.size
    jac = np.zeros((m, n), dtype=np.float64)
    for j in range(n):
        h = 1e-6 * max(1.0, abs(x[j]))
        xp = x.copy()
        xm = x.copy()
        xp[j] += h
        xm[j] -= h
        jac[:, j] = (fun(xp) - fun(xm)) / (2.0 * h)
    return jac


def _xy_dict(model: PlaneModel, ids: list[str], pix: np.ndarray):
    xy, w = model.forward(pix)
    return {pid: xy[i] for i, pid in enumerate(ids)}, xy, w


# --------------------------------------------------------------------- 초기값
def initial_params(
    points: dict[str, np.ndarray],
    observations: list[Observation],
    width: int,
    height: int,
) -> ModelParams:
    """원근/왜곡 0, 축척만 실측 거리에서 추정한 초기값.

    초기 모델은 항등 사상(``w=1``, ``a=1``, ``b=0``)이므로 정규화 좌표에서의
    직선거리와 실측 거리의 비의 **중앙값** 이 곧 축척 추정치다. 중앙값을
    쓰는 것은 실측값 하나가 틀려도 초기값이 망가지지 않게 하기 위함이다.
    """
    p = ModelParams()
    m = PlaneModel(p, width, height)
    ratios: list[float] = []
    for obs in observations:
        if obs.kind != KIND_DISTANCE or not obs.enabled or not obs.is_complete():
            continue
        a, b = obs.points[0], obs.points[1]
        if a not in points or b not in points:
            continue
        na = m.to_normalized(np.asarray(points[a], dtype=float))
        nb = m.to_normalized(np.asarray(points[b], dtype=float))
        d = float(np.hypot(*(nb - na)))
        if d > 1e-9 and obs.value > 0:
            ratios.append(float(obs.value) / d)
    if ratios:
        p.log_s = float(np.log(np.median(ratios)))
    return p


# ------------------------------------------------------------------ 조정계산
#: 관측이 모자랄 때 뒤에서부터 하나씩 빼 보는 "덤" 파라미터.
#: 원근/축척(l1, l2, b, log_a, log_s)은 이 모델의 뼈대라 뺄 수 없고,
#: 렌즈 왜곡은 관측이 넉넉할 때만 의미가 있으므로 이쪽부터 포기한다.
_OPTIONAL_ORDER: tuple[str, ...] = ("k3", "p2", "p1", "k2", "k1")


def adjust(
    points: dict[str, np.ndarray],
    observations: list[Observation],
    width: int,
    height: int,
    free_names: list[str] | None = None,
    init: ModelParams | None = None,
    robust: bool = False,
    max_nfev: int = 20000,
) -> AdjustmentResult:
    """보정 파라미터를 추정한다.

    파라미터를 자동 선택한 경우, 해가 물리적 한계까지 밀려나면(그 해는
    무의미하다) 렌즈 왜곡부터 하나씩 빼면서 다시 풀어 본다. 관측이 모자란데
    많은 것을 풀려다 실패하는 것보다, 풀 수 있는 것만 푸는 편이 낫다.
    """
    auto = free_names is None
    result = _adjust_once(points, observations, width, height,
                          free_names, init, robust, max_nfev)
    if not auto or not result.ok or not params_at_bound(result):
        return result

    trimmed = list(result.free_names)
    dropped: list[str] = []
    for name in _OPTIONAL_ORDER:
        if name not in trimmed:
            continue
        trimmed.remove(name)
        dropped.append(name)
        alt = _adjust_once(points, observations, width, height,
                           list(trimmed), init, robust, max_nfev)
        if alt.ok and not params_at_bound(alt):
            alt.warnings.insert(
                0,
                f"관측이 부족해 {', '.join(dropped)} 까지는 결정할 수 없었습니다. "
                f"그 값을 빼고 다시 계산한 결과입니다(푼 미지수: "
                f"{', '.join(alt.free_names)}). 렌즈 왜곡까지 잡으려면 실측을 "
                "더 추가하세요."
            )
            return alt

    # 다 빼 봐도 한계에 붙으면 원래 결과를 그대로 돌려준다.
    # 이 경우 문제는 파라미터 개수가 아니라 관측 자체에 있다.
    result.warnings.insert(
        0,
        "렌즈 왜곡을 모두 빼고 다시 풀어 봐도 해가 물리적 한계에 붙습니다. "
        "파라미터 개수가 아니라 실측값이나 점 위치에 문제가 있을 가능성이 "
        "높습니다."
    )
    return result


def _adjust_once(
    points: dict[str, np.ndarray],
    observations: list[Observation],
    width: int,
    height: int,
    free_names: list[str] | None = None,
    init: ModelParams | None = None,
    robust: bool = False,
    max_nfev: int = 20000,
) -> AdjustmentResult:
    """가중 최소제곱으로 보정 파라미터를 추정한다(1회).

    Parameters
    ----------
    points
        점 id -> 픽셀 좌표 ``(u, v)``.
    observations
        구속조건 목록. 비활성/미완성 항목은 자동으로 제외된다.
    free_names
        풀 파라미터 이름 목록. ``None`` 이면 관측 수에 맞춰 자동 선택한다.
    robust
        True 면 soft-L1 손실을 써서 조대오차의 영향을 줄인다. 진단용으로
        먼저 돌려 이상값을 찾고, 실제 채택 해는 이상값을 끈 뒤 일반
        최소제곱으로 다시 구하는 것을 권한다.
    """
    warnings: list[str] = []
    active = [
        o for o in observations
        if o.enabled and o.is_complete() and all(p in points for p in o.points)
    ]
    n_obs = len(active)
    if n_obs == 0:
        return AdjustmentResult(False, "사용 가능한 관측이 없습니다.", init or ModelParams(), [])

    if free_names is None:
        free_names = auto_free_params(n_obs)
        warnings.append(
            f"자유 파라미터를 관측 수({n_obs})에 맞춰 자동 선택했습니다: {', '.join(free_names)}"
        )
    free_names = list(free_names)
    n_par = len(free_names)
    if n_obs < n_par:
        return AdjustmentResult(
            False,
            f"관측 {n_obs}개로는 미지수 {n_par}개를 풀 수 없습니다. "
            f"실측 선분을 최소 {n_par}개 이상 만들거나 자유 파라미터를 줄이세요.",
            init or ModelParams(),
            free_names,
        )
    if n_obs == n_par:
        warnings.append(
            "잉여관측이 없습니다(자유도 0). 잔차가 0으로 나오지만 그것은 정확도의 "
            "증거가 아닙니다. 검증을 위해 관측을 더 추가하세요."
        )

    ids = sorted(points.keys())
    pix = np.array([points[i] for i in ids], dtype=np.float64)
    base = (init or initial_params(points, active, width, height)).copy()
    x0 = base.to_vector(free_names)

    # 물리적으로 불가능한 해(예: 지상 평면이 선으로 찌부러지는 a -> 0)를
    # 최적화기가 골라 버리는 일이 있어 파라미터마다 범위를 건다.
    lo, hi = bounds_for(free_names)
    # trf 는 시작점이 경계 안에 있어야 한다. 경계에 딱 붙으면 그 방향으로
    # 못 움직이므로 아주 조금 안쪽으로 넣는다.
    span = hi - lo
    x0 = np.clip(x0, lo + 1e-6 * span, hi - 1e-6 * span)

    def obs_residuals(x: np.ndarray) -> np.ndarray:
        pr = base.with_vector(free_names, x)
        model = PlaneModel(pr, width, height)
        xy_map, xy, _ = _xy_dict(model, ids, pix)
        if not np.all(np.isfinite(xy)):
            return np.full(n_obs, 1.0e6)
        out = np.empty(n_obs, dtype=np.float64)
        for i, obs in enumerate(active):
            out[i] = observation_residual(obs, xy_map) / obs.effective_sigma()
        return np.where(np.isfinite(out), out, 1.0e6)

    def full_residuals(x: np.ndarray) -> np.ndarray:
        pr = base.with_vector(free_names, x)
        model = PlaneModel(pr, width, height)
        _, w = model.forward(pix)
        # 소실선 접근 방지 장벽. 정상 해에서는 정확히 0이므로 통계에 영향이 없다.
        barrier = _BARRIER_WEIGHT * np.minimum(0.0, np.nan_to_num(w, nan=-1.0) - _W_MIN)
        # 왜곡 모델이 접히지 않게 하는 장벽. 접히면 undistort 가 다른 가지로
        # 수렴해 잔차는 그럴듯한데 기하는 완전히 틀린 해가 나온다.
        slope = radial_slope_min(pr.k1, pr.k2, pr.k3)
        fold = np.array([_BARRIER_WEIGHT * min(0.0, slope - _SLOPE_MIN)])
        return np.concatenate([obs_residuals(x), barrier, fold])

    try:
        res = least_squares(
            full_residuals,
            x0,
            jac="3-point",
            method="trf",
            bounds=(lo, hi),
            x_scale="jac",
            loss="soft_l1" if robust else "linear",
            f_scale=3.0,
            max_nfev=max_nfev,
        )
    except Exception as exc:  # pragma: no cover - 수치적 파국 방어
        return AdjustmentResult(False, f"최적화 실패: {exc}", base, free_names)

    params = base.with_vector(free_names, res.x)
    model = PlaneModel(params, width, height)
    xy_map, xy, w = _xy_dict(model, ids, pix)

    if not np.all(np.isfinite(xy)):
        warnings.append("일부 점이 소실선 너머로 사상되었습니다. 실측 선분 배치를 확인하세요.")
    if np.any(w <= _W_MIN):
        warnings.append(
            "소실선에 매우 가까운 점이 있습니다. 그 영역은 배율이 급격히 커져 "
            "보정 신뢰도가 낮습니다."
        )

    # ----------------------------------------------------------- 통계 계산
    r = obs_residuals(res.x)                     # 가중 잔차
    jac = _num_jac(obs_residuals, res.x, r)      # 관측 행만의 야코비
    dof = n_obs - n_par

    try:
        sv = np.linalg.svd(jac, compute_uv=False)
        tol = max(jac.shape) * (sv[0] if sv.size else 0.0) * np.finfo(float).eps
        rank = int(np.sum(sv > tol))
        cond = float(sv[0] / sv[-1]) if sv.size and sv[-1] > 0 else float("inf")
    except np.linalg.LinAlgError:
        rank, cond = 0, float("inf")

    if rank < n_par:
        warnings.append(
            f"정규방정식의 rank 가 부족합니다({rank}/{n_par}). 실측 선분의 방향이 "
            "한쪽으로 몰려 있거나 서로 중복된 정보만 주고 있습니다. 서로 다른 "
            "방향/위치의 실측을 추가하세요."
        )
    elif cond > 1e8:
        warnings.append(
            f"조건수가 매우 큽니다({cond:.2g}). 해가 수치적으로 불안정합니다. "
            "사진 네 귀퉁이 쪽까지 고르게 실측 선분을 배치하면 크게 개선됩니다."
        )

    ssr = float(np.dot(r, r))
    sigma0 = math.sqrt(ssr / dof) if dof > 0 else float("nan")

    qxx = np.linalg.pinv(jac.T @ jac, rcond=1e-12)
    hat = jac @ qxx @ jac.T
    h = np.clip(np.diag(hat), 0.0, 1.0)
    redundancy = 1.0 - h
    scale = sigma0 if (dof > 0 and math.isfinite(sigma0) and sigma0 > 0) else 1.0
    cov = qxx * (scale ** 2)
    param_std = {n: float(math.sqrt(max(cov[i, i], 0.0))) for i, n in enumerate(free_names)}
    # 경계에서 멈춘 파라미터는 관측이 결정해 준 값이 아니다. 통상 공분산은
    # 제약이 없다고 보고 계산하므로 그 값에 표준편차를 붙이면 "이 정도
    # 정밀도로 구했다"는 뜻으로 오해된다. 결정되지 않았음을 그대로 알린다.
    for name in free_names:
        if _is_at_bound(name, getattr(params, name)):
            param_std[name] = float("inf")

    stats: list[ObsStat] = []
    len_res: list[float] = []
    ang_res: list[float] = []
    press_len: list[float] = []
    for i, obs in enumerate(active):
        sig = obs.effective_sigma()
        nat = observation_residual(obs, xy_map)
        angular = obs.kind in _ANGULAR
        disp_res = math.degrees(nat) if angular else nat
        red = float(redundancy[i])
        # Baarda 데이터 스누핑은 **사전분산**(사용자가 준 sigma, 즉 단위분산 1)
        # 을 기준으로 해야 한다. 추정된 sigma0 로 나누면, 조대오차가 클수록
        # sigma0 도 함께 커져 분모가 부풀고 정작 그 오차가 가려진다(masking).
        # 실제로 0.5 m 짜리 오차를 4개 넣었을 때 예전 방식은 하나도 잡아내지
        # 못했고, 사전분산 기준으로는 4개 모두 잡아냈다.
        wt = float(r[i] / math.sqrt(red)) if red > 1e-6 else float("nan")
        press_std = float(r[i] / red) if red > 1e-6 else float("nan")
        press_nat = press_std * sig
        press_disp = math.degrees(press_nat) if angular else press_nat
        stats.append(
            ObsStat(
                obs_id=obs.id,
                kind=obs.kind,
                description=obs.describe(),
                measured=float(obs.value) if obs.kind in (KIND_DISTANCE, KIND_ANGLE) else None,
                computed=predicted_value(obs, xy_map),
                residual=disp_res,
                unit="deg" if angular else "m",
                residual_std=float(r[i]),
                redundancy=red,
                w_test=wt,
                press=press_disp,
                outlier=bool(math.isfinite(wt) and abs(wt) > OUTLIER_THRESHOLD and red > 0.05),
            )
        )
        if angular:
            ang_res.append(disp_res)
        else:
            len_res.append(nat)
            if math.isfinite(press_nat):
                press_len.append(press_nat)

    def _rms(v):
        return float(np.sqrt(np.mean(np.square(v)))) if v else float("nan")

    def _max(v):
        return float(np.max(np.abs(v))) if v else float("nan")

    weak = [s for s in stats if s.redundancy < 0.05]
    if weak:
        warnings.append(
            f"잉여도가 0에 가까운 관측이 {len(weak)}개 있습니다. 이 관측들은 다른 "
            "관측으로 교차검증되지 않으므로 값이 틀려도 잔차에 드러나지 않습니다."
        )
    outliers = [s for s in stats if s.outlier]
    if outliers:
        warnings.append(
            f"조대오차 의심 관측 {len(outliers)}개: "
            + ", ".join(s.description for s in outliers[:5])
            + (" ..." if len(outliers) > 5 else "")
        )
    if math.isfinite(sigma0) and dof >= 3:
        if sigma0 > 3.0:
            warnings.append(
                f"sigma0={sigma0:.2f} 로 큽니다. 실측값 오류, 점 지정 오류, 또는 대상이 "
                "하나의 평면이 아닐 가능성(경사·단차)을 확인하세요."
            )
        elif sigma0 < 0.3:
            warnings.append(
                f"sigma0={sigma0:.2f} 로 작습니다. 관측 표준편차를 실제보다 크게 "
                "입력했을 수 있습니다(정확도가 과소평가됨)."
            )

    return AdjustmentResult(
        ok=bool(res.success),
        message=_termination_ko(str(res.message)),
        params=params,
        free_names=free_names,
        n_obs=n_obs,
        n_params=n_par,
        dof=dof,
        sigma0=sigma0,
        rms_length=_rms(len_res),
        max_length=_max(len_res),
        rms_angle=_rms(ang_res),
        max_angle=_max(ang_res),
        press_rms_length=_rms(press_len),
        cond=cond,
        rank=rank,
        param_std=param_std,
        cov=cov,
        obs_stats=stats,
        warnings=warnings,
        n_eval=int(res.nfev),
    )


# -------------------------------------------------------------- 사후 불확실도
def distance_with_uncertainty(
    result: AdjustmentResult,
    width: int,
    height: int,
    pa: np.ndarray,
    pb: np.ndarray,
) -> tuple[float, float]:
    """보정 모델로 잰 임의 두 점 사이 거리와 그 표준불확도.

    파라미터 공분산을 델타법(1차 전파)으로 거리에 전파한다. 점을 찍는
    행위 자체의 오차는 포함하지 않으므로, 실제 불확도는 이 값 이상이다.
    """
    pa = np.asarray(pa, dtype=float)
    pb = np.asarray(pb, dtype=float)

    def dist_of(x: np.ndarray) -> float:
        pr = result.params.with_vector(result.free_names, x)
        m = PlaneModel(pr, width, height)
        q = m.forward_xy(np.stack([pa, pb]))
        return float(np.hypot(*(q[1] - q[0])))

    x0 = result.params.to_vector(result.free_names)
    d = dist_of(x0)
    if result.cov is None or not math.isfinite(d):
        return d, float("nan")
    g = np.zeros(x0.size)
    for j in range(x0.size):
        h = 1e-6 * max(1.0, abs(x0[j]))
        xp = x0.copy()
        xm = x0.copy()
        xp[j] += h
        xm[j] -= h
        g[j] = (dist_of(xp) - dist_of(xm)) / (2 * h)
    var = float(g @ result.cov @ g)
    return d, math.sqrt(max(var, 0.0))


# ------------------------------------------------------- 평이한 말로 된 판정
@dataclass
class Verdict:
    """조정 결과를 통계 용어 없이 한 줄로 요약한 것.

    UI 요약창과 검사 리포트가 같은 문장을 쓰도록 코어에 둔다. 숫자만 보고는
    "이 결과를 믿어도 되는가"를 판단하기 어렵다는 사용자 피드백에서 나왔다.
    """

    level: str
    """``good`` / ``warn`` / ``bad``. UI 색상 결정용."""
    headline: str
    """한 줄 판정."""
    accuracy: float
    """실용 정확도 추정치 [m]. 교차검증 RMS 를 우선 쓴다. 없으면 NaN."""
    advice: list[str]
    """다음에 무엇을 하면 되는지."""

    @property
    def icon(self) -> str:
        return {"good": "\u2714", "warn": "\u26a0", "bad": "\u2716"}[self.level]


def params_at_bound(result: "AdjustmentResult") -> list[str]:
    """물리적 한계에 딱 붙어서 멈춘 파라미터 이름들.

    경계에 붙었다는 것은 관측이 그 파라미터를 결정하지 못해 최적화기가
    끝까지 밀어붙였다는 뜻이다. 경계 덕분에 파국은 막았지만 그 해를
    그대로 쓰면 안 된다는 신호다.
    """
    return [n for n in result.free_names
            if _is_at_bound(n, getattr(result.params, n, float("nan")))]


def verdict(result: "AdjustmentResult | None") -> Verdict:
    """조정 결과를 사용자가 바로 이해할 수 있는 판정으로 바꾼다."""
    if result is None:
        return Verdict("warn", "아직 보정하지 않았습니다.", float("nan"),
                       ["실측 거리를 입력한 뒤 [보정 실행]을 누르세요."])
    if not result.ok:
        return Verdict("bad", f"보정에 실패했습니다 - {result.message}", float("nan"),
                       ["점을 엉뚱한 곳에 찍지 않았는지 확인하세요.",
                        "실측값의 단위가 모두 미터인지 확인하세요.",
                        "실측 선분을 더 추가하면 해가 안정됩니다."])

    # 실용 정확도: 자기 자신을 맞춘 잔차보다 교차검증 값이 정직하다.
    acc = result.press_rms_length
    if not math.isfinite(acc):
        acc = result.rms_length

    advice: list[str] = []
    level = "good"

    if result.dof <= 0:
        level = "warn"
        head = "검증할 수 없는 결과입니다 (잉여관측 없음)."
        advice.append(
            f"관측 {result.n_obs}개로 미지수 {result.n_params}개를 풀어 잔차가 "
            "항상 0으로 나옵니다. 맞았는지 틀렸는지 알 수 없습니다.")
        advice.append("실측 선분을 3~5개 더 추가하면 정확도를 검증할 수 있습니다.")
        return Verdict(level, head, float("nan"), advice)

    n_out = sum(1 for s in result.obs_stats if s.outlier)
    blind = sum(1 for s in result.obs_stats if s.enabled and s.redundancy < 0.05)
    at_bound = params_at_bound(result)

    # 아래 판정은 순서가 곧 우선순위다. 해가 아예 결정되지 않았다면
    # 잔차나 조대오차를 따지는 것은 의미가 없으므로 그것부터 말한다.
    if result.rank and result.rank < result.n_params:
        return Verdict(
            "bad",
            f"실측 배치가 부족해 해가 결정되지 않았습니다 "
            f"(결정된 미지수 {result.rank}/{result.n_params}).",
            float("nan"),
            ["실측 선분의 방향이 한쪽으로 몰려 있습니다. "
             "가로만 재지 말고 세로·대각 방향을 섞으세요.",
             "사진의 반대쪽 구석에도 실측을 추가하세요.",
             "직각·평행 구속을 넣으면 줄자 없이도 부족한 정보를 채울 수 있습니다.",
             "이 상태의 결과로 정사영상을 만들면 형태가 크게 일그러집니다."])

    if at_bound:
        labels = ", ".join(at_bound)
        # 배치가 정보를 주고 있는데도 한계에 붙었다면, 부족한 것은 배치가
        # 아니라 값들의 일관성이다. 둘을 구분하지 않으면 엉뚱한 처방을 준다.
        informative = (result.rank >= result.n_params
                       and math.isfinite(result.cond) and result.cond < 1e6)
        advice = ["이 결과로는 정사영상을 만들지 마세요."]
        if informative:
            head = (f"실측값들이 서로 맞지 않아 보정이 한계까지 밀려났습니다 "
                    f"({labels}).")
            advice.insert(0, "배치 자체는 정보를 주고 있습니다(rank 충분, 조건수 양호). "
                             "문제는 값들이 하나의 평면 사진으로 설명되지 않는다는 "
                             "점입니다.")
            advice.insert(1, "줄자 값과 클릭한 두 점이 정말 같은 구간인지 "
                             "하나씩 대조하세요. 특히 선의 어느 가장자리를 "
                             "쟀는지 확인하세요.")
            advice.insert(2, "높이가 다른 면(적치물 위, 턱, 경사면)에서 잰 구간이 "
                             "섞여 있으면 빼세요.")
            if result.dof < 4:
                advice.insert(3, f"지금은 잉여관측이 {result.dof}개뿐이라 어느 것이 "
                                 "틀렸는지 프로그램이 가려낼 수 없습니다. "
                                 "긴 구간을 4~6개 더 넣으면 자동으로 지목됩니다.")
        else:
            head = f"보정값이 물리적 한계까지 밀려났습니다 ({labels})."
            advice.insert(0, "관측이 그 값을 결정하지 못해 최적화가 끝까지 "
                             "밀어붙인 상태입니다.")
            advice.insert(1, "실측 선분을 사진 전체에 고르게, 방향을 섞어 "
                             "추가하세요.")
            advice.insert(2, "점을 엉뚱한 곳에 찍었거나 실측값 단위(미터)가 "
                             "틀리지 않았는지 확인하세요.")
        return Verdict("bad", head, float("nan"), advice)

    n_enabled = sum(1 for s in result.obs_stats if s.enabled)
    widespread = (n_out >= max(2, (n_enabled + 1) // 2)
                  and math.isfinite(result.sigma0) and result.sigma0 > 3.0)

    if widespread:
        # 대부분이 걸리면 개별 조대오차라기보다 sigma 설정이나 전제가 문제다.
        level = "warn"
        head = (f"관측 {n_enabled}개 중 {n_out}개가 입력한 오차범위를 "
                "지키지 못합니다.")
        advice.append("몇 개가 틀렸다기보다 σ 를 너무 작게 잡았을 가능성이 "
                      "큽니다. 짧은 구간일수록 점 찍는 오차가 크게 먹히니 "
                      "σ 를 현실적으로 올리세요.")
        advice.append("대상이 하나의 평면이 아닐 수도 있습니다. 높이가 있는 곳"
                      "(적치물 위, 턱, 경사면)에서 잰 구간이 있으면 빼세요.")
        advice.append("그래도 남으면 잔차가 가장 큰 것부터 점 위치를 확인하세요.")
    elif n_out:
        level = "warn"
        head = f"조대오차가 의심되는 관측이 {n_out}건 있습니다."
        advice.append("[관측] 탭에서 빨간 줄을 확인하세요. 실측값 오타이거나 "
                      "점을 잘못 찍었을 가능성이 큽니다.")
        advice.append("확인이 어려우면 [의심 관측 끄고 다시 계산]을 누르세요.")
    elif math.isfinite(result.sigma0) and result.sigma0 > 3.0:
        level = "warn"
        head = "실측값과 모델이 입력한 오차범위보다 많이 어긋납니다."
        advice.append("실측값 자체의 오차를 너무 작게 잡았거나, 대상이 하나의 "
                      "평면이 아닐 수 있습니다(높이차·경사).")
        advice.append("높이가 있는 곳(건물 상단, 적치물)에서 잰 구간이 있으면 빼세요.")
    elif math.isfinite(result.cond) and result.cond > 1e8:
        level = "warn"
        head = "실측 배치가 한쪽으로 치우쳐 해가 불안정합니다."
        advice.append("사진의 반대편 구석에도 실측 선분을 추가하세요.")
        advice.append("가로 방향만 재지 말고 세로·대각 방향도 섞으세요.")
    else:
        head = "보정 결과가 양호합니다."

    if math.isfinite(acc):
        head += f"  현장 기준 오차 약 \u00b1{acc * 1000:.0f} mm."
    if blind and level == "good":
        advice.append(
            f"다만 검증되지 않는 관측이 {blind}건 있습니다(잉여도 0에 가까움). "
            "그 부근은 틀려도 드러나지 않으니 근처에 실측을 하나 더 넣으면 좋습니다.")
    if level == "good" and not advice:
        advice.append("[출력] 탭에서 정사영상을 만들고 내보내면 됩니다.")
    return Verdict(level, head, acc, advice)


# ------------------------------------------------------- 임의 측정의 불확도
@dataclass
class Measurement:
    """보정 결과 위에서 잰 값과 그 표준불확도.

    불확도는 두 몫으로 나뉜다. 어느 쪽이 큰지 알아야 무엇을 개선할지
    판단할 수 있어서 따로 들고 다닌다.

    * ``sigma_model`` : 보정 파라미터가 완벽하지 않아서 생기는 몫.
      실측을 더 넣으면 줄어든다.
    * ``sigma_click`` : 화면에서 점을 찍는 행위의 몫. 사진 해상도(GSD)가
      정하며, 실측을 아무리 늘려도 줄지 않는다. 원본 해상도를 쓰거나
      확대해서 찍어야 줄어든다.
    """

    value: float
    sigma: float
    sigma_model: float
    sigma_click: float
    unit: str = "m"

    def text(self, decimals: int = 3) -> str:
        """``12.340 ± 0.051 m`` 형태의 표시 문자열."""
        unit = "m²" if self.unit == "m2" else self.unit
        if not math.isfinite(self.sigma):
            return f"{self.value:.{decimals}f} {unit}"
        return f"{self.value:.{decimals}f} ± {self.sigma:.{decimals}f} {unit}"


def _polyline_length(xy: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def _polygon_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _click_sigma(xy: np.ndarray, mode: str, sigma_pt: float) -> float:
    """점 찍기 오차가 측정값에 전파된 몫.

    각 꼭짓점의 위치 오차를 독립이라 보고 1차 전파한다. 예를 들어 두 점
    사이 거리는 양 끝이 각각 흔들리므로 ``sigma * sqrt(2)`` 가 된다.
    """
    if sigma_pt <= 0 or len(xy) < 2:
        return 0.0
    n = len(xy)
    grad = np.zeros_like(xy)
    if mode == "area":
        if n < 3:
            return 0.0
        for i in range(n):
            nxt, prv = (i + 1) % n, (i - 1) % n
            grad[i, 0] = 0.5 * (xy[nxt, 1] - xy[prv, 1])
            grad[i, 1] = 0.5 * (xy[prv, 0] - xy[nxt, 0])
    else:
        seg = np.diff(xy, axis=0)
        ln = np.linalg.norm(seg, axis=1, keepdims=True)
        u = np.divide(seg, ln, out=np.zeros_like(seg), where=ln > 1e-12)
        grad[:-1] -= u
        grad[1:] += u
    return float(sigma_pt * math.sqrt(float(np.sum(grad * grad))))


def measure(
    result: "AdjustmentResult | None",
    width: int,
    height: int,
    pix_pts: np.ndarray,
    mode: str = "distance",
    sigma_pt: float = 0.0,
    params: ModelParams | None = None,
) -> Measurement:
    """보정 모델로 임의의 거리/면적을 재고 불확도까지 돌려준다.

    Parameters
    ----------
    pix_pts
        원본 사진 픽셀 좌표 (n, 2).
    mode
        ``distance`` 면 폴리라인 전체 길이, ``area`` 면 다각형 면적.
    sigma_pt
        꼭짓점 하나의 위치 오차 [m]. 정사영상 위에서 찍었다면
        ``GSD * (클릭 정밀도 픽셀)`` 을 주면 된다.
    """
    pix = np.asarray(pix_pts, dtype=np.float64).reshape(-1, 2)
    unit = "m2" if mode == "area" else "m"
    pars = params if params is not None else (result.params if result else None)
    if pars is None or len(pix) < 2:
        return Measurement(float("nan"), float("nan"), float("nan"), float("nan"), unit)

    def value_of(p: ModelParams) -> float:
        xy = PlaneModel(p, width, height).forward_xy(pix)
        if not np.all(np.isfinite(xy)):
            return float("nan")
        return _polygon_area(xy) if mode == "area" else _polyline_length(xy)

    value = value_of(pars)
    xy0 = PlaneModel(pars, width, height).forward_xy(pix)
    s_click = _click_sigma(xy0, mode, sigma_pt) if np.all(np.isfinite(xy0)) else float("nan")

    s_model = float("nan")
    if result is not None and result.cov is not None and math.isfinite(value):
        x0 = pars.to_vector(result.free_names)
        g = np.zeros(x0.size)
        for j in range(x0.size):
            h = 1e-6 * max(1.0, abs(x0[j]))
            xp, xm = x0.copy(), x0.copy()
            xp[j] += h
            xm[j] -= h
            g[j] = (value_of(pars.with_vector(result.free_names, xp))
                    - value_of(pars.with_vector(result.free_names, xm))) / (2 * h)
        if np.all(np.isfinite(g)):
            s_model = math.sqrt(max(float(g @ result.cov @ g), 0.0))

    parts = [s for s in (s_model, s_click) if math.isfinite(s)]
    total = math.sqrt(sum(s * s for s in parts)) if parts else float("nan")
    return Measurement(value, total, s_model, s_click, unit)
