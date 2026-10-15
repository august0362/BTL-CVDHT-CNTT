@echo off
rem run_demo.bat - bat demo MA3D-Net: EMOCA that (WSL2) + web Streamlit (127.0.0.1:8599). Khong can quyen quan tri.
rem Chay lai khi da chay: khong bat trung (kiem health truoc). Tat bang stop_demo.bat. Huong dan: documents\docs\RUNBOOK_DEMO.md.
rem Tep chi dung ky tu ASCII va dong CRLF (cmd doc .bat theo code page). Duong dan du an lay tu %~dp0.
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "PY=%USERPROFILE%\venvs\ma3d-web\Scripts\python.exe"
set "DISTRO=Ubuntu-22.04"
set "WSL_USER=viet"
set "EMOCA_HEALTH=http://127.0.0.1:8765/v1/health"
set "WEB_URL=http://127.0.0.1:8599"
set "EMOCA_WAIT_S=180"
set "WEB_WAIT_S=90"
set "EMOCA_OK=0"

echo [%time%] MA3D-Net demo - thu muc: %cd%

rem ---------- 0. kiem dieu kien ----------
if not exist "%PY%" goto :no_venv
where curl.exe >nul 2>&1
if errorlevel 1 goto :no_curl
wsl.exe -d %DISTRO% -u %WSL_USER% -e true >nul 2>&1
if errorlevel 1 goto :no_wsl

rem ---------- 1. EMOCA server trong WSL ----------
call :http_code "%EMOCA_HEALTH%" code
if "!code!"=="200" (
  set "EMOCA_OK=1"
  echo [!time!] EMOCA da chay san ^(health 200^) - khong bat lai.
  goto :web
)
rem Server dang khoi dong/nap mo hinh (health chua 200) => KHONG bat trung, chi cho health. Dau hieu: cong 8765 dang
rem LISTEN, hoac co tien trinh run_emoca_server.sh / server.py cua user trong WSL (mau [x] de pgrep khong khop chinh no).
wsl.exe -d %DISTRO% -u %WSL_USER% -e bash -c "ss -ltnH 'sport = :8765' | grep -q . || pgrep -u %WSL_USER% -f 'run_emoca_server[.]sh|python -u [s]erver[.]py' >/dev/null" >nul 2>&1
if not errorlevel 1 (
  echo [!time!] EMOCA dang khoi dong trong WSL ^(cong 8765 hoac server.py da co^) - khong bat trung, chi cho health.
  call :now_cs t_start
  goto :wait_emoca
)
echo [%time%] Bat EMOCA server trong WSL ^(log: ~/emoca_wsl/out/run_server_bat.log^) ...
rem Cua so rieng thu nho giu phien wsl.exe => server song tiep sau khi .bat nay ket thuc.
start "EMOCA server (WSL)" /min wsl.exe -d %DISTRO% -u %WSL_USER% -e bash -lc "bash ~/emoca_wsl/bin/run_emoca_server.sh >> ~/emoca_wsl/out/run_server_bat.log 2>&1"
call :now_cs t_start

:wait_emoca
call :sleep 5
call :elapsed_s t_start waited
call :http_code "%EMOCA_HEALTH%" code
if "!code!"=="200" goto :emoca_up
echo [%time%]   cho EMOCA ... !waited!/%EMOCA_WAIT_S% s ^(nap mo hinh ~20s^)
if !waited! lss %EMOCA_WAIT_S% goto :wait_emoca
echo [CANH BAO] EMOCA chua len sau %EMOCA_WAIT_S% s - web se chay che do GIA LAP.
echo            Xem log trong WSL: tail -n 50 ~/emoca_wsl/out/run_server_bat.log
goto :web

:emoca_up
call :elapsed_s t_start waited
set "EMOCA_OK=1"
echo [%time%] EMOCA san sang sau !waited! s.

rem ---------- 2. web Streamlit ----------
:web
call :http_code "%WEB_URL%/_stcore/health" code
if "!code!"=="200" (
  echo [!time!] Web 8599 da chay san - khong bat lai. Che do EMOCA/engine la cua lan bat truoc.
  goto :open
)
set "EMOCA_MODE="
set "EMOCA_URL="
set "ENGINE_MODE="
if "%EMOCA_OK%"=="1" (
  set "EMOCA_MODE=real"
  set "EMOCA_URL=http://127.0.0.1:8765"
  set "ENGINE_MODE=real"
  echo [!time!] Bat web: EMOCA that + POSTER/MA3D-Net that ^(CPU^).
) else (
  echo [!time!] Bat web: che do GIA LAP ^(EMOCA khong san sang^).
)
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=2"
start "MA3D web" /min "%PY%" -m streamlit run frontend\app.py --server.address 127.0.0.1 --server.port 8599 --server.headless true --browser.gatherUsageStats false
call :now_cs t_start

:wait_web
call :sleep 2
call :elapsed_s t_start waited
call :http_code "%WEB_URL%/_stcore/health" code
if "!code!"=="200" goto :web_up
if !waited! lss %WEB_WAIT_S% goto :wait_web
echo [LOI] Web chua len sau %WEB_WAIT_S% s. Xem cua so "MA3D web".
goto :fail

:web_up
call :elapsed_s t_start waited
echo [%time%] Web san sang sau !waited! s.

:open
echo [%time%] Mo trinh duyet: %WEB_URL%
if not "%DEMO_NO_BROWSER%"=="1" start "" "%WEB_URL%"
echo Xong. Tat demo: stop_demo.bat
endlocal
exit /b 0

:no_venv
echo [LOI] Khong thay venv web: %PY%
echo       Tao theo documents\docs\RUNBOOK_DEMO.md muc 1.1-1.2 (venv o %%USERPROFILE%%\venvs\ma3d-web, cai environments\web\requirements-real.txt).
goto :fail
:no_curl
echo [LOI] Khong thay curl.exe (co san tu Windows 10 1803).
goto :fail
:no_wsl
echo [LOI] Khong vao duoc WSL distro %DISTRO% voi user %WSL_USER%.
echo       Cai WSL2 + Ubuntu-22.04 va dung EMOCA theo documents\docs\RUNBOOK_DEMO.md muc 4-wsl.
goto :fail

:fail
endlocal
exit /b 1

rem ---------- ham phu ----------
:http_code
rem %1 = url, %2 = ten bien nhan ma HTTP (000 neu khong noi duoc)
set "%~2=000"
for /f "delims=" %%c in ('curl.exe -s -o NUL -m 3 -w "%%{http_code}" "%~1" 2^>nul') do set "%~2=%%c"
exit /b 0

:now_cs
rem %1 = ten bien nhan so phan-tram-giay tu nua dem, doc tu %time% (HH:MM:SS.cc hoac HH:MM:SS,cc; gio < 10 co dau cach)
for /f "tokens=1-4 delims=:.," %%a in ("%time: =0%") do set /a "%~1=(((1%%a-100)*60+(1%%b-100))*60+(1%%c-100))*100+(1%%d-100)"
exit /b 0

:elapsed_s
rem %1 = bien moc (tu :now_cs), %2 = ten bien nhan so giay da troi qua theo DONG HO THAT (qua nua dem: cong 24 h)
call :now_cs _now
set /a "_d=_now-!%~1!"
if !_d! lss 0 set /a "_d+=8640000"
set /a "%~2=_d/100"
exit /b 0

:sleep
rem ping thay cho timeout (timeout loi khi stdin bi chuyen huong); ping -n N ~ N-1 giay
set /a _n=%~1+1
ping -n !_n! 127.0.0.1 >nul 2>&1
exit /b 0
