@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"

rem 이 PC 에 파이썬이 여러 개 깔려 있을 수 있다. 이름만 보고 고르면
rem 패키지가 없는 쪽을 잡는 일이 생기므로, 실제로 import 가 되는 것을 쓴다.
set "PY="
for %%C in (python py python3) do (
    if not defined PY (
        %%C -c "import PySide6, numpy, scipy, cv2" >nul 2>&1
        if !errorlevel! equ 0 set "PY=%%C"
    )
)

if not defined PY (
    echo.
    echo   필요한 패키지가 설치되어 있지 않습니다.
    echo   아래 명령을 한 번만 실행한 뒤 다시 여세요.
    echo.
    echo       python -m pip install -r requirements.txt
    echo.
    echo   ^(python 이 없다고 나오면 https://www.python.org 에서 먼저 설치하세요^)
    echo.
    pause
    exit /b 1
)

%PY% -m drone_photo_rectifier %*
if errorlevel 1 (
    echo.
    echo   프로그램이 오류로 종료되었습니다. 위 내용을 확인하세요.
    pause
)
endlocal
