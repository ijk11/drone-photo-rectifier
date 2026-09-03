"""내보내기: 월드파일, DXF, 성과표(CSV), 검사 리포트, 카메라 파라미터.

좌표계 주의
-----------
내부 미터 좌표는 이미지와 같은 방향(Y가 아래로 증가)이다. 반면 CAD/GIS 는
Y가 위로 증가하는 것이 관례이므로, 파일로 내보낼 때만 ``Y_world = -Y`` 로
부호를 뒤집는다(``to_world``). 좌우 반전은 일어나지 않는다.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
from pathlib import Path

import numpy as np

from .constraints import KIND_DISTANCE
from .project import Project
from .rectify import OutputGrid, scale_stats, world_file_text
from .solver import AdjustmentResult
from .transform import PlaneModel

__all__ = [
    "to_world",
    "write_world_file",
    "export_dxf",
    "export_measurements_csv",
    "export_report",
    "export_camera_json",
]


def to_world(xy: np.ndarray) -> np.ndarray:
    """내부 미터 좌표(Y 아래로) -> CAD/GIS 관례(Y 위로)."""
    xy = np.asarray(xy, dtype=np.float64)
    out = xy.copy()
    out[..., 1] = -out[..., 1]
    return out


def write_world_file(raster_path: str | Path, grid: OutputGrid) -> Path:
    """래스터 옆에 world file 을 쓴다 (.png -> .pgw, .tif -> .tfw)."""
    p = Path(raster_path)
    ext = p.suffix.lower()
    mapping = {".png": ".pgw", ".tif": ".tfw", ".tiff": ".tfw", ".jpg": ".jgw", ".jpeg": ".jgw"}
    wp = p.with_suffix(mapping.get(ext, ".wld"))
    wp.write_text(world_file_text(grid), encoding="ascii")
    return wp


# --------------------------------------------------------------------- DXF
def export_dxf(
    path: str | Path,
    project: Project,
    model: PlaneModel | None = None,
    include_points: bool = True,
    include_measurements: bool = True,
    text_height: float | None = None,
) -> Path:
    """도면 요소/실측선/점을 미터 단위 DXF(R12 ASCII)로 내보낸다.

    외부 의존성 없이 R12 ASCII 를 직접 쓴다. R12 는 모든 CAD 가 읽을 수 있는
    최소공통분모이며, 이 프로그램이 내보내는 것은 선/점/문자뿐이라 상위
    버전의 기능이 필요 없다.
    """
    model = model or project.model()
    pm = project.pixel_map()
    ids = list(pm.keys())
    if not ids:
        raise ValueError("내보낼 점이 없습니다.")
    xy = model.forward_xy(np.array([pm[i] for i in ids]))
    world = to_world(xy)
    coord = {pid: world[i] for i, pid in enumerate(ids) if np.all(np.isfinite(world[i]))}
    if not coord:
        raise ValueError("보정 모델로 사상되는 점이 없습니다. 먼저 보정을 실행하세요.")

    pts = np.array(list(coord.values()))
    diag = float(np.hypot(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))) or 1.0
    th = text_height if text_height and text_height > 0 else max(diag * 0.012, 1e-3)

    layers = {"0": 7, "MEASURE": 1, "POINTS": 3, "LABEL": 8}
    for f in project.features:
        layers.setdefault(f.layer or "0", 5)

    out: list[str] = []

    def w(code, value):
        out.append(str(code))
        if isinstance(value, float):
            out.append(f"{value:.8f}")
        else:
            out.append(str(value))

    # --- HEADER
    w(0, "SECTION"); w(2, "HEADER")
    w(9, "$ACADVER"); w(1, "AC1009")
    w(9, "$INSUNITS"); w(70, 6)            # 6 = meters
    w(9, "$EXTMIN"); w(10, float(pts[:, 0].min())); w(20, float(pts[:, 1].min())); w(30, 0.0)
    w(9, "$EXTMAX"); w(10, float(pts[:, 0].max())); w(20, float(pts[:, 1].max())); w(30, 0.0)
    w(0, "ENDSEC")

    # --- TABLES (레이어)
    w(0, "SECTION"); w(2, "TABLES")
    w(0, "TABLE"); w(2, "LAYER"); w(70, len(layers))
    for name, color in layers.items():
        w(0, "LAYER"); w(2, name); w(70, 0); w(62, color); w(6, "CONTINUOUS")
    w(0, "ENDTAB")
    w(0, "ENDSEC")

    # --- ENTITIES
    w(0, "SECTION"); w(2, "ENTITIES")

    def polyline(layer: str, verts: list[np.ndarray], closed: bool) -> None:
        w(0, "POLYLINE"); w(8, layer); w(66, 1); w(70, 1 if closed else 0)
        w(10, 0.0); w(20, 0.0); w(30, 0.0)
        for v in verts:
            w(0, "VERTEX"); w(8, layer)
            w(10, float(v[0])); w(20, float(v[1])); w(30, 0.0)
        w(0, "SEQEND"); w(8, layer)

    def line(layer: str, a: np.ndarray, b: np.ndarray) -> None:
        w(0, "LINE"); w(8, layer)
        w(10, float(a[0])); w(20, float(a[1])); w(30, 0.0)
        w(11, float(b[0])); w(21, float(b[1])); w(31, 0.0)

    def text(layer: str, pos: np.ndarray, s: str, height: float) -> None:
        w(0, "TEXT"); w(8, layer)
        w(10, float(pos[0])); w(20, float(pos[1])); w(30, 0.0)
        w(40, float(height)); w(1, s)

    for f in project.features:
        verts = [coord[p] for p in f.point_ids if p in coord]
        if len(verts) >= 2:
            polyline(f.layer or "0", verts, f.closed)

    if include_measurements:
        for o in project.observations:
            if o.kind != KIND_DISTANCE or not o.is_complete():
                continue
            a, b = o.points[0], o.points[1]
            if a in coord and b in coord:
                line("MEASURE", coord[a], coord[b])
                mid = (coord[a] + coord[b]) / 2.0
                text("LABEL", mid, f"{o.value:.3f}m", th)

    if include_points:
        for pid, c in coord.items():
            w(0, "POINT"); w(8, "POINTS")
            w(10, float(c[0])); w(20, float(c[1])); w(30, 0.0)
            text("POINTS", c + np.array([th * 0.4, th * 0.4]), project.name_of(pid), th * 0.8)

    w(0, "ENDSEC")
    w(0, "EOF")

    p = Path(path)
    if p.suffix.lower() != ".dxf":
        p = p.with_suffix(".dxf")
    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    return p


# --------------------------------------------------------------------- CSV
def export_measurements_csv(
    path: str | Path, project: Project, result: AdjustmentResult | None
) -> Path:
    """관측별 성과표. 검사조서로 그대로 붙일 수 있게 진단값을 모두 담는다."""
    p = Path(path)
    if p.suffix.lower() != ".csv":
        p = p.with_suffix(".csv")
    stats = {s.obs_id: s for s in (result.obs_stats if result else [])}
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow([
            "번호", "종류", "내용", "점", "실측값", "모델값", "잔차",
            "단위", "표준편차sigma", "표준화잔차w", "잉여도", "교차검증잔차",
            "조대오차의심", "사용",
        ])
        for i, o in enumerate(project.observations, 1):
            s = stats.get(o.id)
            wr.writerow([
                i,
                o.kind,
                o.describe(project.name_of),
                " ".join(project.name_of(x) for x in o.points),
                f"{o.value:.6g}" if o.kind == KIND_DISTANCE or o.kind == "angle" else "",
                f"{s.computed:.6g}" if s else "",
                f"{s.residual:.6g}" if s else "",
                s.unit if s else "",
                f"{o.effective_sigma():.6g}",
                f"{s.w_test:.3f}" if s and math.isfinite(s.w_test) else "",
                f"{s.redundancy:.3f}" if s else "",
                f"{s.press:.6g}" if s and math.isfinite(s.press) else "",
                "Y" if s and s.outlier else "",
                "Y" if o.enabled else "N",
            ])
    return p


# ------------------------------------------------------------------ 리포트
def export_report(
    path: str | Path,
    project: Project,
    result: AdjustmentResult | None,
    grid: OutputGrid | None = None,
) -> Path:
    """사람이 읽는 검사 리포트(텍스트)."""
    p = Path(path)
    if p.suffix.lower() not in (".txt", ".md"):
        p = p.with_suffix(".txt")
    model = project.model()
    L: list[str] = []
    L.append("드론 사진 정사보정 성과 리포트")
    L.append("=" * 60)
    L.append(f"작성 시각 : {_dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    L.append(f"사진      : {project.image_path}")
    L.append(f"크기      : {project.image_width} x {project.image_height} px")
    if project.exif:
        keep = {k: v for k, v in project.exif.items() if k in
                ("Model", "Make", "FocalLength", "FocalLengthIn35mmFilm", "DateTimeOriginal")}
        if keep:
            L.append("EXIF      : " + ", ".join(f"{k}={v}" for k, v in keep.items()))
    L.append(f"점 개수   : {len(project.points)}")
    L.append(f"관측 개수 : {len(project.observations)} (사용 {len(project.active_observations())})")
    L.append("")

    L.append("[조정계산 결과]")
    if result is None:
        L.append("  아직 보정을 실행하지 않았습니다.")
    else:
        for line in result.summary_lines():
            L.append("  " + line)
        L.append("")
        L.append("[추정 파라미터]")
        for n in result.free_names:
            v = getattr(result.params, n)
            sd = result.param_std.get(n, float("nan"))
            L.append(f"  {n:8s} = {v:+.6f}  (sigma {sd:.6f})")
        fixed = [n for n in ("k1", "k2", "k3", "p1", "p2", "cx_off", "cy_off")
                 if n not in result.free_names]
        if fixed:
            L.append(f"  고정: {', '.join(fixed)} = 0")
    L.append("")

    st = scale_stats(model)
    if st:
        L.append("[지상 해상도 분포]")
        L.append(f"  최소 {st['min']*1000:.2f} mm/px, 중앙 {st['median']*1000:.2f} mm/px, "
                 f"최대 {st['max']*1000:.2f} mm/px")
        L.append(f"  최대/최소 비 = {st['ratio']:.2f}  "
                 "(1에 가까울수록 원근 영향이 작음. 보정 전 단순 축척 측정 시 "
                 f"최대 {abs(st['ratio']-1)*100:.0f}% 수준의 오차가 발생할 수 있음)")
        L.append("")

    if grid is not None:
        L.append("[출력 래스터]")
        for line in grid.describe().splitlines():
            L.append("  " + line)
        L.append("")

    if result and result.obs_stats:
        L.append("[관측별 잔차]")
        L.append(f"  {'내용':<34s}{'실측':>10s}{'모델':>10s}{'잔차':>10s}{'w':>8s}{'잉여도':>8s}")
        for s in result.obs_stats:
            meas = f"{s.measured:.3f}" if s.measured is not None else "-"
            wv = f"{s.w_test:.2f}" if math.isfinite(s.w_test) else "-"
            flag = "  <-- 의심" if s.outlier else ""
            L.append(f"  {s.description[:34]:<34s}{meas:>10s}{s.computed:>10.3f}"
                     f"{s.residual:>10.4f}{wv:>8s}{s.redundancy:>8.2f}{flag}")
        L.append("")

    if result and result.warnings:
        L.append("[경고 및 권고]")
        for wmsg in result.warnings:
            L.append("  - " + wmsg)
        L.append("")

    L.append("[전제 조건]")
    L.append("  - 본 보정은 대상이 하나의 평면(지면) 위에 있다고 가정합니다.")
    L.append("    높이가 있는 물체는 기복변위로 인해 보정 후에도 위치 오차가 남습니다.")
    L.append("  - 실측 기준선은 반드시 지면 위에서 취해야 합니다.")
    L.append("  - 잔차가 작다고 정확한 것이 아니라, 잉여관측이 충분한 상태에서")
    L.append("    잔차가 작아야 정확한 것입니다(자유도와 잉여도를 함께 확인하십시오).")

    if project.notes:
        L.append("")
        L.append("[비고]")
        for line in project.notes.splitlines():
            L.append("  " + line)

    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p


# ------------------------------------------------------------- 카메라 파라미터
def export_camera_json(
    path: str | Path, project: Project, focal_px: float | None = None
) -> Path:
    """추정된 왜곡계수를 저장한다. 초점거리를 알면 OpenCV 규약으로도 함께 기록."""
    p = Path(path)
    if p.suffix.lower() != ".json":
        p = p.with_suffix(".json")
    model = project.model()
    data = {
        "convention": "drone-photo-rectifier (대각선 절반 s0 로 정규화)",
        "image_size": [project.image_width, project.image_height],
        "s0_px": model.s0,
        "principal_point_px": list(model.center),
        "distortion": {k: getattr(project.params, k) for k in ("k1", "k2", "k3", "p1", "p2")},
        "homography_undistorted_to_meters": model.homography().tolist(),
    }
    if focal_px:
        data["opencv"] = project.params.to_opencv(
            float(focal_px), model.s0, project.image_width, project.image_height
        )
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
