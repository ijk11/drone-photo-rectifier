"""정답을 아는 합성 항공사진 생성기.

실제 드론 사진으로는 "보정이 얼마나 정확한가"를 검증할 수 없다. 참값을
모르기 때문이다. 그래서 지상 평면의 도면(격자·마커)을 먼저 만들고, 알려진
원근/렌즈왜곡을 **일부러 넣어서** 사진을 합성한다. 그러면 보정 결과를
참값과 직접 비교할 수 있다.

사용::

    python -m dronerect.tools.make_synthetic --out samples

생성물
    ``synthetic.png``        합성된 "드론 사진"
    ``synthetic_truth.json`` 참 파라미터, 마커의 참 미터/픽셀 좌표
    ``synthetic_plane.png``  참 지상 도면(정답 정사영상)
    ``synthetic.drproj``     마커를 점으로, 참 거리를 실측값으로 채운 프로젝트
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from ..core.constraints import KIND_DISTANCE, KIND_PERPENDICULAR, Observation
from ..core.params import ModelParams
from ..core.project import Project
from ..core.transform import PlaneModel

# 지상 도면 정의 (미터)
PLANE_W, PLANE_H = 110.0, 80.0
PLANE_GSD = 0.02  # 2 cm/px -> 5500 x 4000


def build_plane() -> tuple[np.ndarray, float]:
    """1 m 격자 + 마커가 그려진 지상 도면을 만든다."""
    w = int(PLANE_W / PLANE_GSD)
    h = int(PLANE_H / PLANE_GSD)
    img = np.full((h, w, 3), 88, np.uint8)

    rng = np.random.default_rng(3)
    noise = rng.normal(0, 9, (h, w, 1)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def m2p(x: float, y: float) -> tuple[int, int]:
        return int(round(x / PLANE_GSD)), int(round(y / PLANE_GSD))

    # 1 m 격자(옅게), 5 m 격자(진하게)
    for x in np.arange(0, PLANE_W + 1e-9, 1.0):
        c = (150, 150, 150) if abs(x % 5.0) > 1e-6 else (215, 215, 215)
        t = 1 if abs(x % 5.0) > 1e-6 else 2
        cv2.line(img, m2p(x, 0), m2p(x, PLANE_H), c, t, cv2.LINE_AA)
    for y in np.arange(0, PLANE_H + 1e-9, 1.0):
        c = (150, 150, 150) if abs(y % 5.0) > 1e-6 else (215, 215, 215)
        t = 1 if abs(y % 5.0) > 1e-6 else 2
        cv2.line(img, m2p(0, y), m2p(PLANE_W, y), c, t, cv2.LINE_AA)

    # 현장 느낌: 도로 경계선과 구조물 외곽
    cv2.rectangle(img, m2p(24, 22), m2p(48, 38), (30, 190, 240), 3, cv2.LINE_AA)
    cv2.rectangle(img, m2p(60, 42), m2p(88, 60), (30, 190, 240), 3, cv2.LINE_AA)
    cv2.line(img, m2p(0, 40), m2p(PLANE_W, 40), (255, 255, 255), 4, cv2.LINE_AA)
    return img, PLANE_GSD


#: 마커 위치(미터). 사진 전역에 고르게 퍼지도록 배치.
MARKERS: list[tuple[float, float]] = [
    # 구조물 A 모서리 (직각/평행 구속에 사용)
    (24, 22), (48, 22), (24, 38), (48, 38),
    # 구조물 B 모서리
    (60, 42), (88, 42), (60, 60), (88, 60),
    # 사진 전역에 흩뿌린 검증용 마커
    (18, 18), (40, 16), (70, 18), (94, 20),
    (16, 40), (55, 40), (96, 42),
    (20, 62), (38, 54), (76, 28), (82, 64), (32, 48),
]


def draw_markers(img: np.ndarray) -> None:
    for i, (x, y) in enumerate(MARKERS, 1):
        c = (int(round(x / PLANE_GSD)), int(round(y / PLANE_GSD)))
        cv2.circle(img, c, 11, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, c, 11, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.circle(img, c, 3, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.putText(img, str(i), (c[0] + 14, c[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 40), 2, cv2.LINE_AA)


def render_photo(
    plane: np.ndarray, params: ModelParams, width: int, height: int
) -> np.ndarray:
    """지상 도면을 주어진 왜곡/원근으로 촬영한 것처럼 렌더링."""
    model = PlaneModel(params, width, height)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    pix = np.stack([xs + 0.5, ys + 0.5], axis=-1)
    xy, w = model.forward(pix)
    mx = (xy[..., 0] / PLANE_GSD - 0.5).astype(np.float32)
    my = (xy[..., 1] / PLANE_GSD - 0.5).astype(np.float32)
    bad = ~np.isfinite(mx) | ~np.isfinite(my) | (w <= 0)
    mx = np.where(bad, -1, mx).astype(np.float32)
    my = np.where(bad, -1, my).astype(np.float32)
    return cv2.remap(plane, mx, my, cv2.INTER_LANCZOS4,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 45, 40))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="정답을 아는 합성 항공사진 생성")
    ap.add_argument("--out", default="samples", help="출력 폴더")
    ap.add_argument("--width", type=int, default=4000)
    ap.add_argument("--height", type=int, default=3000)
    ap.add_argument("--k1", type=float, default=-0.16, help="방사왜곡 k1")
    ap.add_argument("--k2", type=float, default=0.045, help="방사왜곡 k2")
    ap.add_argument("--tilt", type=float, default=1.0,
                    help="원근 강도 배율(0이면 정사, 1이 기본, 2면 매우 기울어짐)")
    ap.add_argument("--pixel-noise", type=float, default=1.0,
                    help="프로젝트에 넣을 점 지정 오차 [px]")
    ap.add_argument("--tape-sigma", type=float, default=0.02,
                    help="프로젝트에 넣을 실측 오차 [m]")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    plane, _ = build_plane()
    draw_markers(plane)
    cv2.imwrite(str(out / "synthetic_plane.png"), plane)

    # 사진 -> 지면 사상의 참 파라미터. 축척은 사진이 지면을 거의 채우도록 잡는다.
    params = ModelParams(
        l1=0.10 * args.tilt,
        l2=-0.26 * args.tilt,
        b=0.03,
        log_a=0.05,
        log_s=math.log(50.0),
        k1=args.k1,
        k2=args.k2,
    )
    # 지면 중앙이 사진 중앙에 오도록 평행이동
    m0 = PlaneModel(params, args.width, args.height)
    c = m0.forward_xy(np.array([args.width / 2, args.height / 2]))
    params.tx = PLANE_W / 2 - float(c[0])
    params.ty = PLANE_H / 2 - float(c[1])

    photo = render_photo(plane, params, args.width, args.height)
    cv2.imwrite(str(out / "synthetic.png"), photo)

    model = PlaneModel(params, args.width, args.height)
    marker_xy = np.array(MARKERS, dtype=float)
    marker_pix = model.inverse(marker_xy)
    inside = (
        np.isfinite(marker_pix).all(axis=1)
        & (marker_pix[:, 0] > 5) & (marker_pix[:, 0] < args.width - 5)
        & (marker_pix[:, 1] > 5) & (marker_pix[:, 1] < args.height - 5)
    )

    truth = {
        "params": params.as_dict(),
        "image_size": [args.width, args.height],
        "plane_gsd": PLANE_GSD,
        "markers": [
            {"index": i + 1, "xy_m": list(map(float, marker_xy[i])),
             "pixel": list(map(float, marker_pix[i])), "visible": bool(inside[i])}
            for i in range(len(MARKERS))
        ],
    }
    (out / "synthetic_truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 바로 열어 볼 수 있는 프로젝트 파일
    pr = Project(
        image_path=str((out / "synthetic.png").resolve()),
        image_width=args.width,
        image_height=args.height,
        notes="dronerect.tools.make_synthetic 로 생성한 검증용 합성 데이터입니다.\n"
              "참값은 synthetic_truth.json 에 있습니다.",
    )
    idx_of: dict[int, str] = {}
    for i in np.flatnonzero(inside):
        pxy = marker_pix[i] + rng.normal(0, args.pixel_noise, 2)
        pt = pr.add_point(float(pxy[0]), float(pxy[1]), name=f"M{i+1}")
        idx_of[int(i)] = pt.id

    vis = list(idx_of.keys())
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < min(20, len(vis) * (len(vis) - 1) // 2):
        a, b = rng.choice(vis, 2, replace=False)
        key = (int(min(a, b)), int(max(a, b)))
        if key not in pairs:
            pairs.add(key)
    for a, b in sorted(pairs):
        d = float(np.hypot(*(marker_xy[b] - marker_xy[a])))
        pr.observations.append(Observation(
            kind=KIND_DISTANCE,
            points=[idx_of[a], idx_of[b]],
            value=d + float(rng.normal(0, args.tape_sigma)),
            sigma=args.tape_sigma,
        ))
    # 구조물 모서리의 직각 정보(공짜로 얻는 강한 구속)
    corners = [0, 1, 2]   # 구조물 A 의 세 모서리 -> 직각 구속
    if all(c in idx_of for c in corners):
        pr.observations.append(Observation(
            kind=KIND_PERPENDICULAR,
            points=[idx_of[0], idx_of[1], idx_of[0], idx_of[2]],
        ))
    pr.save(str(out / "synthetic.drproj"))

    print(f"생성 완료: {out.resolve()}")
    print(f"  synthetic.png        {args.width} x {args.height}")
    print(f"  synthetic_plane.png  참 지상 도면 (정답 정사영상)")
    print(f"  synthetic_truth.json 참 파라미터/마커 좌표")
    print(f"  synthetic.drproj     점 {len(pr.points)}개, 관측 {len(pr.observations)}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
