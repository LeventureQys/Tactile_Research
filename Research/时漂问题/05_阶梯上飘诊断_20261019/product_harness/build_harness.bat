@echo off
setlocal
rem Build product_harness.exe: drives src/domain/drift_v6/creep_observer.cpp alone
rem over a binary frame stream (ts, v[n]) so it can be cross-checked frame by frame
rem against 03_K10_算法迁移包_20261014/dv6_portable.dll.
rem
rem same_tu_parity.cpp (product + portable in ONE translation unit) is provided as
rem source only: cl.exe under a GBK console cannot take the non-ASCII package path
rem via /I, so build it by hand from a UTF-8 shell, e.g.
rem   powershell:  cl /utf-8 /std:c++17 /O2 /EHsc /I src /I thirdparty\eigen-5.0.0
rem                /I "D:\workshop\Github\Tactile_Research\Research\时漂问题\03_K10_算法迁移包_20261014"
rem                same_tu_parity.cpp src\domain\drift_v6\creep_observer.cpp
rem (verified 2026-10-19: worst frame delta 0.0 over 287374 frames x 57 channels)
set REPO=D:\workshop\Processing\multi-device-cascade-host-cpp
set HERE=%~dp0
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 ( echo [FAIL] vcvars64.bat not found & exit /b 1 )
cd /d "%REPO%"
if not exist "%HERE%bin" mkdir "%HERE%bin"
cl /nologo /utf-8 /std:c++17 /O2 /EHsc /W3 /I src /I thirdparty\eigen-5.0.0 "%HERE%product_harness.cpp" src\domain\drift_v6\creep_observer.cpp /Fo:"%HERE%bin\\" /Fe:"%HERE%bin\product_harness.exe"
if errorlevel 1 ( echo [FAIL] product_harness & exit /b 1 )
echo [OK] %HERE%bin\product_harness.exe
endlocal