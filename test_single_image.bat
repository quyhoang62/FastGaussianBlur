@echo off
REM ================================================================
REM Script test với 1 ảnh cho cả 3 trường hợp
REM ================================================================

set INPUT_IMAGE=test.jpg
set OUTPUT_PREFIX=output
set SIGMA=5.0
set PASSES=3
set BORDER=extend

if "%1" neq "" set INPUT_IMAGE=%1
if "%2" neq "" set OUTPUT_PREFIX=%2
if "%3" neq "" set SIGMA=%3
if "%4" neq "" set PASSES=%4
if "%5" neq "" set BORDER=%5

echo.
echo ================================================================
echo   TEST SINGLE IMAGE - 3 VERSIONS
echo ================================================================
echo.
echo Input image: %INPUT_IMAGE%
echo Sigma: %SIGMA%
echo Passes: %PASSES%
echo Border: %BORDER%
echo.

REM Test với fastblur.exe (Sequential + OpenMP)
if exist fastblur.exe (
    echo ================================================================
    echo   Running fastblur.exe (Sequential + OpenMP comparison)
    echo ================================================================
    echo.
    
    set OUTPUT1=%OUTPUT_PREFIX%_comparison.png
    fastblur.exe %INPUT_IMAGE% %OUTPUT1% %SIGMA% %PASSES% %BORDER%
    
    echo.
) else (
    echo WARNING: fastblur.exe not found!
    echo.
)

REM Test với fastblur_cuda.exe (nếu có)
if exist fastblur_cuda.exe (
    echo ================================================================
    echo   Running fastblur_cuda.exe (CUDA version)
    echo ================================================================
    echo.
    
    set OUTPUT2=%OUTPUT_PREFIX%_cuda.png
    fastblur_cuda.exe %INPUT_IMAGE% %OUTPUT2% %SIGMA% %PASSES% %BORDER%
    
    echo.
) else (
    echo WARNING: fastblur_cuda.exe not found!
    echo Build it with: make fastblur_cuda
    echo.
)

echo ================================================================
echo   TEST COMPLETED!
echo ================================================================
echo.
echo Output files:
if exist %OUTPUT_PREFIX%_comparison.png (
    echo   - %OUTPUT_PREFIX%_comparison.png (from fastblur.exe)
)
if exist %OUTPUT_PREFIX%_cuda.png (
    echo   - %OUTPUT_PREFIX%_cuda.png (from fastblur_cuda.exe)
)
echo.

pause

