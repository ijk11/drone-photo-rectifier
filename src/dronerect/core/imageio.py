"""이미지 입출력과 EXIF 읽기.

OpenCV 의 ``imread``/``imwrite`` 는 Windows 에서 경로를 시스템 ANSI 코드페이지로
변환하기 때문에 한글이나 특수문자가 섞인 경로에서 조용히 실패한다. 실무
폴더명은 대부분 한글이므로, 파일을 바이트로 직접 읽고 쓰는 방식으로 감쌌다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

__all__ = ["imread", "imwrite", "read_exif", "focal_length_px"]


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """유니코드 경로를 지원하는 ``cv2.imread``."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise OSError(f"파일을 읽을 수 없습니다: {path}")
    img = cv2.imdecode(data, flags)
    if img is None:
        raise OSError(f"이미지를 해석할 수 없습니다(지원하지 않는 형식?): {path}")
    return img


def imwrite(path: str | Path, image: np.ndarray, params: list[int] | None = None) -> None:
    """유니코드 경로를 지원하는 ``cv2.imwrite``."""
    p = Path(path)
    ok, buf = cv2.imencode(p.suffix or ".png", image, params or [])
    if not ok:
        raise OSError(f"이미지를 인코딩할 수 없습니다: {path}")
    buf.tofile(str(p))


def read_exif(path: str | Path) -> dict:
    """EXIF 를 문자열 키의 평범한 dict 로 읽는다. 실패하면 빈 dict."""
    try:
        from PIL import Image, ExifTags
    except Exception:
        return {}
    try:
        with Image.open(str(path)) as im:
            raw = im.getexif()
            if not raw:
                return {}
            names = {v: k for k, v in ExifTags.TAGS.items()}
            del names  # 사용하지 않지만 태그표 로딩 확인용
            out: dict = {}
            for tag_id, value in raw.items():
                key = ExifTags.TAGS.get(tag_id, str(tag_id))
                if isinstance(value, bytes):
                    continue
                out[key] = str(value) if not isinstance(value, (int, float)) else value
            # 렌즈 정보는 하위 IFD 에 있는 경우가 많다
            try:
                for tag_id, value in raw.get_ifd(0x8769).items():
                    key = ExifTags.TAGS.get(tag_id, str(tag_id))
                    if isinstance(value, bytes):
                        continue
                    out.setdefault(key, str(value) if not isinstance(value, (int, float)) else value)
            except Exception:
                pass
            return out
    except Exception:
        return {}


def focal_length_px(exif: dict, width: int, height: int) -> float | None:
    """EXIF 로부터 초점거리를 픽셀 단위로 추정.

    35 mm 환산 초점거리가 있으면 그것을 쓴다(35 mm 판의 대각선 43.267 mm 기준).
    왜곡계수를 OpenCV 규약으로 내보낼 때만 쓰이며, 보정 계산 자체에는
    필요하지 않다.
    """
    try:
        f35 = exif.get("FocalLengthIn35mmFilm")
        if f35:
            f35 = float(f35)
            if f35 > 0:
                diag_px = float(np.hypot(width, height))
                return f35 * diag_px / 43.266615
    except Exception:
        pass
    return None
