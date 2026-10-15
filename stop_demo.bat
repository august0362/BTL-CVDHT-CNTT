@echo off
rem stop_demo.bat - tat demo MA3D-Net: web Streamlit cong 8599 (Windows) va EMOCA server cong 8765 (WSL2).
rem Tim tien trinh theo CONG dang nghe (khong pkill theo mau ten). Khong can quyen quan tri.
rem Tep chi dung ky tu ASCII va dong CRLF.
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
set "DISTRO=Ubuntu-22.04"
set "WSL_USER=viet"

rem ---------- 1. web 8599: tien trinh nghe cong + tien trinh cha neu la python.exe (launcher cua venv) ----------
echo [%time%] Tat web 8599 ...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "$c = Get-NetTCPConnection -LocalPort 8599 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1;" ^
  "if (-not $c) { 'web 8599: khong co tien trinh nghe'; exit 0 };" ^
  "$p = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $c.OwningProcess);" ^
  "$par = if ($p) { Get-CimInstance Win32_Process -Filter ('ProcessId=' + $p.ParentProcessId) } else { $null };" ^
  "Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue; 'web 8599: da dung PID ' + $c.OwningProcess;" ^
  "if ($par -and $par.Name -eq 'python.exe' -and $par.CommandLine -match 'streamlit') { Stop-Process -Id $par.ProcessId -Force -ErrorAction SilentlyContinue; 'web 8599: da dung tien trinh cha PID ' + $par.ProcessId }"

rem ---------- 2. EMOCA 8765 trong WSL: PID lay tu ss -ltnp, gui SIGTERM; script run_emoca_server.sh tu ket thuc ----------
echo [%time%] Tat EMOCA server 8765 trong WSL ...
wsl.exe -d %DISTRO% -u %WSL_USER% -e true >nul 2>&1
if errorlevel 1 (
  echo EMOCA: khong vao duoc WSL %DISTRO% - bo qua.
  goto :check
)
wsl.exe -d %DISTRO% -u %WSL_USER% -e bash -c "pid=$(ss -ltnpH 'sport = :8765' | grep -o 'pid=[0-9]*' | head -n1 | cut -d= -f2); if [ -z $pid ]; then echo 'EMOCA 8765: khong co tien trinh nghe'; exit 0; fi; kill -TERM $pid && echo 'EMOCA 8765: da gui SIGTERM toi PID' $pid; for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 $pid 2>/dev/null || { echo 'EMOCA 8765: da dung'; exit 0; }; sleep 1; done; kill -KILL $pid 2>/dev/null && echo 'EMOCA 8765: SIGKILL PID' $pid '(khong dung sau 10 s)'"

:check
echo [%time%] Kiem cong:
powershell.exe -NoProfile -Command ^
  "foreach ($port in 8599, 8765) { $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue; if ($c) { '  ' + $port + ': VAN CON nghe (PID ' + ($c.OwningProcess -join ',') + ')' } else { '  ' + $port + ': trong' } }"
endlocal
exit /b 0
