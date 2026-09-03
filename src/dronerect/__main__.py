"""진입점: ``python -m dronerect`` 또는 콘솔 스크립트 ``dronerect``."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)

    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    app = QApplication(argv)
    app.setApplicationName("drone-rectify")
    app.setOrganizationName("drone-rectify")
    # 한글 라벨이 많으므로 한글 글꼴을 우선 지정한다(없으면 시스템 기본).
    for family in ("Malgun Gothic", "맑은 고딕", "Noto Sans KR", "Segoe UI"):
        f = QFont(family, 9)
        if f.exactMatch() or family == "Segoe UI":
            app.setFont(f)
            break

    from .ui.main_window import MainWindow

    win = MainWindow()
    win.show()

    # 인자로 프로젝트나 이미지를 주면 바로 연다.
    rest = [a for a in argv[1:] if not a.startswith("-")]
    if rest:
        target = rest[0]
        if target.lower().endswith(".drproj"):
            try:
                from .core.project import Project
                from .core.imageio import imread
                pr = Project.load(target)
                img = imread(pr.image_path)
                win.image = img
                win.project = pr
                win.view.load_image(img)
                win.view.set_project(pr)
                win._refresh_all()
            except Exception as exc:  # pragma: no cover
                print(f"프로젝트를 열 수 없습니다: {exc}", file=sys.stderr)
        else:
            win._load_image_into_new_project(target)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
