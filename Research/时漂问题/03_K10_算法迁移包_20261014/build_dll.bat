@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
  echo [FAIL] vcvars64.bat not found
  exit /b 1
)
cd /d "%~dp0"
rem /utf-8: every source file here is UTF-8 without BOM. Without the flag MSVC
rem decodes them with the system code page (936/GBK), emits C4819 and may
rem mis-read the Chinese comments. The host product already compiles with
rem /utf-8 globally (CMakeLists.txt add_compile_options("/utf-8")), so the DLL
rem must use the same setting to stay bit-identical with src/domain/drift_v6.
cl /nologo /utf-8 /O2 /W3 /LD /Fe:dv6_portable.dll dv6_portable_abi.c
if errorlevel 1 (
  echo [FAIL] compile error
  exit /b 1
)
del /q *.obj >nul 2>&1
echo [OK] dv6_portable.dll built (utf-8)
endlocal
