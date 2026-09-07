"""정사보정(rectification) 래스터 생성.

보정된 결과물은 "화면에 예쁘게 편 사진"이 아니라 **축척이 일정한 도면 바탕**
이어야 한다. 그래서 출력 격자는 픽셀이 아니라 미터로 정의한다.
격자 간격(GSD, ground sample distance)을 정하면 출력 이미지의 1 픽셀이
정확히 그 미터 크기를 갖고, 이후 CAD/GIS 에서 그대로 실측 가능하다.

리샘플링은 출력 -> 입력 역방향으로 한다(inverse mapping). 출력 픽셀 중심의
미터 좌표를 원본 픽셀 좌표로 되돌린 뒤 보간하므로 구멍이 생기지 않는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .transform import PlaneModel

__all__ = [
    "OutputGrid",
    "INTERPOLATIONS",
    "scale_stats",
    "suggest_gsd",
    "plan_grid",
    "rectify_image",
    "world_file_text",
]

#: 리샘플링 방법. 정밀 작업에는 Lanczos, 미리보기에는 Linear 를 쓴다.
INTERPOLATIONS: dict[str, int] = {
    "최근린 (Nearest)": cv2.INTER_NEAREST,
    "양선형 (Linear)": cv2.INTER_LINEAR,
    "3차 (Cubic)": cv2.INTER_CUBIC,
    "Lanczos4": cv2.INTER_LANCZOS4,
}

#: 출력 픽셀 수 상한(메모리 보호). 초과 시 사용자에게 GSD 조정을 요구한다.
MAX_OUTPUT_PIXELS = 200_000_000

#: 한 변의 픽셀 수 상한.
#:
#: cv2.remap 은 내부적으로 좌표를 16비트로 다루기 때문에 가로/세로가 각각
#: SHRT_MAX(32767) 미만이어야 한다. 총 픽셀 수만 검사하면 "10만 x 1500"
#: 같은 가늘고 긴 출력이 상한을 통과한 뒤 remap 에서 assertion 으로 죽는다.
#: 그런 출력은 어차피 쓸 수 없으므로 격자를 계획하는 단계에서 막는다.
MAX_OUTPUT_SIDE = 32_000

#: 이 비율을 넘는 가로세로 비는 보정이 잘못 풀린 것으로 본다.
#: 원본이 4:3 인 사진을 정사보정해 봐야 10:1 을 넘기 어렵다. 반면 해가
#: 퇴화하면 수천 대 1 이 나온다.
DEGENERATE_ASPECT = 50.0


@dataclass
class OutputGrid:
    """출력 래스터의 지오메트리."""

    xmin: float
    ymin: float
    gsd: float
    width: int
    height: int

    @property
    def xmax(self) -> float:
        return self.xmin + self.width * self.gsd

    @property
    def ymax(self) -> float:
        return self.ymin + self.height * self.gsd

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6

    def describe(self) -> str:
        return (
            f"{self.width} x {self.height} px  ({self.megapixels:.1f} MP)\n"
            f"GSD {self.gsd * 1000:.2f} mm/px\n"
            f"범위 X {self.xmin:.2f} ~ {self.xmax:.2f} m, "
            f"Y {self.ymin:.2f} ~ {self.ymax:.2f} m"
        )


def scale_stats(model: PlaneModel, n: int = 40) -> dict[str, float]:
    """이미지 전체에서 지상 해상도[m/px]의 분포.

    최소/최대 비율이 크다는 것은 곧 원근이 심하다는 뜻이며, 보정 없이 잰
    길이가 위치에 따라 얼마나 달라지는지를 보여 준다.
    """
    gx, gy = np.meshgrid(
        np.linspace(0.02, 0.98, n) * model.width,
        np.linspace(0.02, 0.98, n) * model.height,
    )
    pts = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    sc = model.local_scale(pts)
    sc = sc[np.isfinite(sc) & (sc > 0)]
    if sc.size == 0:
        return {}
    return {
        "min": float(sc.min()),
        "median": float(np.median(sc)),
        "max": float(sc.max()),
        "ratio": float(sc.max() / sc.min()),
    }


def suggest_gsd(model: PlaneModel, mode: str = "median") -> float:
    """권장 GSD.

    * ``finest``  : 가장 조밀한 곳 기준 - 정보 손실이 없지만 파일이 커진다.
    * ``median``  : 중앙값 - 균형(기본값).
    * ``coarsest``: 가장 성긴 곳 기준 - 가볍지만 근경이 뭉개진다.
    """
    st = scale_stats(model)
    if not st:
        return 0.01
    return {"finest": st["min"], "median": st["median"], "coarsest": st["max"]}.get(
        mode, st["median"]
    )


def plan_grid(
    model: PlaneModel,
    gsd: float,
    margin_m: float = 0.0,
    bounds: tuple[float, float, float, float] | None = None,
) -> OutputGrid:
    """출력 격자를 계산한다. ``bounds`` 를 주면 그 범위로 잘라낸다."""
    if gsd <= 0:
        raise ValueError("GSD 는 0보다 커야 합니다.")
    b = bounds or model.metric_bounds()
    if b is None:
        raise ValueError(
            "이미지 전체가 소실선 너머로 사상됩니다. 보정 파라미터를 확인하세요."
        )
    xmin, ymin, xmax, ymax = b
    xmin -= margin_m
    ymin -= margin_m
    xmax += margin_m
    ymax += margin_m
    w = int(math.ceil((xmax - xmin) / gsd))
    h = int(math.ceil((ymax - ymin) / gsd))
    if w < 1 or h < 1:
        raise ValueError("출력 크기가 0입니다. GSD 또는 범위를 확인하세요.")
    # 크기가 크다는 것은 증상이고, 원인은 대개 보정이 잘못 풀린 것이다.
    # 원인을 먼저 말해 주지 않으면 사용자는 GSD 만 계속 키우게 된다.
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > DEGENERATE_ASPECT:
        raise ValueError(
            f"보정 결과가 한 방향으로 찌부러졌습니다 "
            f"(가로세로 비 {aspect:.0f} : 1).\n"
            "실측 배치가 부족해 보정이 잘못 풀린 상태입니다. GSD 를 바꿔도 "
            "해결되지 않습니다. [보정] 탭의 판정과 경고를 먼저 확인하세요."
        )
    if w * h > MAX_OUTPUT_PIXELS:
        raise ValueError(
            f"출력이 너무 큽니다 ({w} x {h} = {w*h/1e6:.0f} MP).\n"
            f"GSD 를 {gsd * math.sqrt(w * h / MAX_OUTPUT_PIXELS) * 1000:.3g} mm/px "
            "이상으로 키우거나 출력 범위를 좁히세요."
        )
    if w > MAX_OUTPUT_SIDE or h > MAX_OUTPUT_SIDE:
        need = gsd * max(w, h) / MAX_OUTPUT_SIDE
        raise ValueError(
            f"출력 한 변이 너무 깁니다 ({w} x {h} px, 한계 {MAX_OUTPUT_SIDE}).\n"
            f"GSD 를 {need * 1000:.3g} mm/px 이상으로 키우세요."
        )
    return OutputGrid(xmin=xmin, ymin=ymin, gsd=gsd, width=w, height=h)


def rectify_image(
    image: np.ndarray,
    model: PlaneModel,
    grid: OutputGrid,
    interpolation: int = cv2.INTER_LANCZOS4,
    background=(0, 0, 0),
    band_rows: int = 512,
    progress=None,
) -> tuple[np.ndarray, np.ndarray]:
    """정사보정 래스터와 유효 마스크를 만든다.

    Returns
    -------
    (rectified, mask)
        ``rectified`` 는 입력과 같은 dtype/채널, ``mask`` 는 원본 화면 안에서
        온 픽셀만 255 인 uint8 마스크.

    큰 출력에서도 메모리를 아끼려고 행 단위(band)로 나눠 처리한다.
    """
    h_out, w_out = grid.height, grid.width
    if image.ndim == 2:
        out = np.zeros((h_out, w_out), dtype=image.dtype)
    else:
        out = np.zeros((h_out, w_out, image.shape[2]), dtype=image.dtype)
    mask = np.zeros((h_out, w_out), dtype=np.uint8)

    xs = grid.xmin + (np.arange(w_out, dtype=np.float64) + 0.5) * grid.gsd
    src_h, src_w = image.shape[:2]

    for y0 in range(0, h_out, band_rows):
        y1 = min(y0 + band_rows, h_out)
        ys = grid.ymin + (np.arange(y0, y1, dtype=np.float64) + 0.5) * grid.gsd
        gx, gy = np.meshgrid(xs, ys)
        xy = np.stack([gx, gy], axis=-1)

        pix = model.inverse(xy)
        # cv2.remap 의 좌표계는 픽셀 인덱스(첫 픽셀 중심이 0) 이므로 0.5 를 뺀다.
        mx = (pix[..., 0] - 0.5).astype(np.float32)
        my = (pix[..., 1] - 0.5).astype(np.float32)

        valid = (
            np.isfinite(mx) & np.isfinite(my)
            & (mx >= -0.5) & (mx <= src_w - 0.5)
            & (my >= -0.5) & (my <= src_h - 0.5)
        )
        mx = np.where(valid, mx, -1.0).astype(np.float32)
        my = np.where(valid, my, -1.0).astype(np.float32)

        try:
            band = cv2.remap(
                image, mx, my, interpolation,
                borderMode=cv2.BORDER_CONSTANT, borderValue=background,
            )
        except cv2.error as exc:  # pragma: no cover - 상한 검사를 통과한 예외 상황
            raise ValueError(
                f"래스터를 만들지 못했습니다 ({w_out} x {h_out} px).\n"
                "GSD 를 키우거나, 보정이 제대로 풀렸는지 [보정] 탭에서 "
                "확인하세요."
            ) from exc
        out[y0:y1] = band
        mask[y0:y1] = (valid * 255).astype(np.uint8)
        if progress is not None:
            progress(y1, h_out)

    return out, mask


def world_file_text(grid: OutputGrid) -> str:
    """ESRI world file(.pgw/.tfw) 내용.

    출력 래스터의 행 방향(아래로)은 미터 Y 증가 방향과 같으므로, 일반적인
    지도 관례(위쪽이 +Y)에 맞추기 위해 월드 Y = -미터 Y 로 부호를 뒤집는다.
    QGIS, AutoCAD RASTER, 각종 GIS 에서 축척이 살아 있는 채로 열린다.
    """
    a = grid.gsd
    e = -grid.gsd
    c = grid.xmin + grid.gsd / 2.0          # 좌상단 픽셀 중심의 X
    f = -(grid.ymin) - grid.gsd / 2.0       # 좌상단 픽셀 중심의 월드 Y
    return "\n".join(
        f"{v:.10f}" for v in (a, 0.0, 0.0, e, c, f)
    ) + "\n"
