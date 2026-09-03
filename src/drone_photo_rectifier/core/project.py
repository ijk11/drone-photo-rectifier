"""프로젝트 데이터 모델과 파일 입출력(.drproj).

프로젝트 파일은 사람이 읽을 수 있는 JSON 이다. 사진 자체는 넣지 않고
경로만 저장하므로(가능하면 프로젝트 파일 기준 상대경로) 용량이 작고,
diff/버전관리가 가능하다. 실측 성과는 오래 보관해야 하는 자료이므로
바이너리 포맷을 피했다.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .constraints import Observation
from .params import DEFAULT_FREE, ModelParams
from .transform import PlaneModel

__all__ = ["Point", "Feature", "Project", "FILE_SUFFIX", "FORMAT_VERSION", "FORMAT_ID"]

FILE_SUFFIX = ".drproj"
FORMAT_VERSION = 1
FORMAT_ID = "drone-photo-rectifier"
#: 저장소 이름을 바꾸기 전에 저장된 파일도 계속 열 수 있게 한다.
_LEGACY_FORMAT_IDS = ("drone-rectify",)


@dataclass
class Point:
    """사진 위에서 지정한 점. 좌표는 원본 이미지의 픽셀(부동소수)."""

    u: float
    v: float
    name: str = ""
    note: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def uv(self) -> np.ndarray:
        return np.array([self.u, self.v], dtype=np.float64)

    def to_dict(self) -> dict:
        return {"id": self.id, "u": float(self.u), "v": float(self.v),
                "name": self.name, "note": self.note}

    @classmethod
    def from_dict(cls, d: dict) -> "Point":
        p = cls(u=float(d["u"]), v=float(d["v"]),
                name=d.get("name", ""), note=d.get("note", ""))
        if d.get("id"):
            p.id = str(d["id"])
        return p


@dataclass
class Feature:
    """도면 요소(폴리라인/폴리곤). 보정 후 미터 좌표로 CAD 에 내보낸다."""

    point_ids: list[str] = field(default_factory=list)
    layer: str = "0"
    closed: bool = False
    name: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def to_dict(self) -> dict:
        return {"id": self.id, "point_ids": list(self.point_ids),
                "layer": self.layer, "closed": bool(self.closed), "name": self.name}

    @classmethod
    def from_dict(cls, d: dict) -> "Feature":
        f = cls(point_ids=list(d.get("point_ids", [])), layer=d.get("layer", "0"),
                closed=bool(d.get("closed", False)), name=d.get("name", ""))
        if d.get("id"):
            f.id = str(d["id"])
        return f


@dataclass
class Project:
    """하나의 사진에 대한 보정 작업 전체."""

    image_path: str = ""
    image_width: int = 0
    image_height: int = 0
    points: list[Point] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    features: list[Feature] = field(default_factory=list)
    params: ModelParams = field(default_factory=ModelParams)
    free_names: list[str] = field(default_factory=lambda: list(DEFAULT_FREE))
    solved: bool = False
    gauge: dict = field(default_factory=lambda: {
        "origin_point": "", "axis_from": "", "axis_to": "", "axis_angle_deg": 0.0,
    })
    notes: str = ""
    exif: dict = field(default_factory=dict)
    path: str = ""
    """현재 저장 경로(파일에는 기록되지 않음)."""

    # ------------------------------------------------------------------ 조회
    def point_by_id(self, pid: str) -> Point | None:
        for p in self.points:
            if p.id == pid:
                return p
        return None

    def name_of(self, pid: str) -> str:
        p = self.point_by_id(pid)
        return p.name if p else "?"

    def pixel_map(self) -> dict[str, np.ndarray]:
        return {p.id: p.uv for p in self.points}

    def model(self) -> PlaneModel:
        return PlaneModel(self.params, self.image_width, self.image_height)

    # ------------------------------------------------------------------ 편집
    def next_point_name(self) -> str:
        used = set()
        for p in self.points:
            if p.name.startswith("P") and p.name[1:].isdigit():
                used.add(int(p.name[1:]))
        i = 1
        while i in used:
            i += 1
        return f"P{i}"

    def add_point(self, u: float, v: float, name: str = "") -> Point:
        p = Point(u=float(u), v=float(v), name=name or self.next_point_name())
        self.points.append(p)
        return p

    def remove_point(self, pid: str) -> None:
        """점과, 그 점을 참조하는 관측/요소를 함께 지운다."""
        self.points = [p for p in self.points if p.id != pid]
        self.observations = [o for o in self.observations if pid not in o.points]
        for f in self.features:
            f.point_ids = [q for q in f.point_ids if q != pid]
        self.features = [f for f in self.features if len(f.point_ids) >= 2]

    def used_point_ids(self) -> set[str]:
        used: set[str] = set()
        for o in self.observations:
            used.update(o.points)
        for f in self.features:
            used.update(f.point_ids)
        return used

    # ------------------------------------------------------------------ 통계
    def active_observations(self) -> list[Observation]:
        pm = self.pixel_map()
        return [o for o in self.observations
                if o.enabled and o.is_complete() and all(p in pm for p in o.points)]

    # -------------------------------------------------------------------- IO
    def to_dict(self, base_dir: str | None = None) -> dict:
        img = self.image_path
        if base_dir and img:
            try:
                img = os.path.relpath(img, base_dir)
            except ValueError:
                pass  # 다른 드라이브면 절대경로 유지
        return {
            "format": FORMAT_ID,
            "version": FORMAT_VERSION,
            "image": {"path": img, "width": self.image_width, "height": self.image_height},
            "points": [p.to_dict() for p in self.points],
            "observations": [o.to_dict() for o in self.observations],
            "features": [f.to_dict() for f in self.features],
            "params": self.params.as_dict(),
            "free_names": list(self.free_names),
            "solved": bool(self.solved),
            "gauge": dict(self.gauge),
            "notes": self.notes,
            "exif": dict(self.exif),
        }

    @classmethod
    def from_dict(cls, d: dict, base_dir: str | None = None) -> "Project":
        img = d.get("image", {})
        path = img.get("path", "")
        if path and base_dir and not os.path.isabs(path):
            path = os.path.normpath(os.path.join(base_dir, path))
        pr = cls(
            image_path=path,
            image_width=int(img.get("width", 0)),
            image_height=int(img.get("height", 0)),
            points=[Point.from_dict(x) for x in d.get("points", [])],
            observations=[Observation.from_dict(x) for x in d.get("observations", [])],
            features=[Feature.from_dict(x) for x in d.get("features", [])],
            params=ModelParams.from_dict(d.get("params", {})),
            free_names=list(d.get("free_names") or DEFAULT_FREE),
            solved=bool(d.get("solved", False)),
            notes=d.get("notes", ""),
            exif=dict(d.get("exif", {})),
        )
        g = d.get("gauge")
        if isinstance(g, dict):
            pr.gauge.update(g)
        return pr

    def save(self, path: str) -> None:
        p = Path(path)
        if p.suffix.lower() != FILE_SUFFIX:
            p = p.with_suffix(FILE_SUFFIX)
        data = self.to_dict(base_dir=str(p.parent))
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.path = str(p)

    @classmethod
    def load(cls, path: str) -> "Project":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("format") not in (FORMAT_ID, *_LEGACY_FORMAT_IDS):
            raise ValueError("drone-photo-rectifier 프로젝트 파일이 아닙니다.")
        if int(data.get("version", 0)) > FORMAT_VERSION:
            raise ValueError(
                f"이 파일은 더 새로운 버전(v{data.get('version')})입니다. "
                "프로그램을 업데이트하세요."
            )
        pr = cls.from_dict(data, base_dir=str(p.parent))
        pr.path = str(p)
        return pr
