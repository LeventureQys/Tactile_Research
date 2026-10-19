@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
  echo [FAIL] vcvars64.bat not found
  exit /b 1
)
cd /d "%~dp0"
cl /nologo /O2 /W3 /LD /Fe:dv6_portable.dll dv6_portable_abi.c
if errorlevel 1 (
  echo [FAIL] compile error
  exit /b 1
)
del /q *.obj >nul 2>&1
echo [OK] dv6_portable.dll built
endlocal
