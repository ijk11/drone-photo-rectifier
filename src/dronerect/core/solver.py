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
from .params import ModelParams, auto_free_params
from .transform import PlaneModel

__all__ = [
    "ObsStat",
    "AdjustmentResult",
    "initial_params",
    "adjust",
    "distance_with_uncertainty",
    "OUTLIER_THRESHOLD",
]

#: Baarda 데이터 스누핑 임계값 (유의수준 0.1%).
OUTLIER_THRESHOLD = 3.29

#: 원근 분모 w 의 하한. 이보다 작아지면 소실선에 접근한 것으로 보고 벌점.
_W_MIN = 0.05
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
    """가중 최소제곱으로 보정 파라미터를 추정한다.

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
        return np.concatenate([obs_residuals(x), barrier])

    try:
        res = least_squares(
            full_residuals,
            x0,
            jac="3-point",
            method="trf",
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
        wt = float(r[i] / (scale * math.sqrt(red))) if red > 1e-6 else float("nan")
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
        message=str(res.message),
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
