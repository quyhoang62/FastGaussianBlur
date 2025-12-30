@echo off
REM ================================================================
REM SCRIPT BUILD CUDA VERSION CỦA FAST GAUSSIAN BLUR
REM ================================================================

echo.
echo ================================================================
echo   BUILDING FAST GAUSSIAN BLUR - CUDA VERSION
echo ================================================================
echo.

REM Kiểm tra nvcc có sẵn không
where nvcc >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: nvcc not found in PATH!
    echo Please install CUDA Toolkit or add it to PATH.
    echo.
    pause
    exit /b 1
)

REM Kiểm tra cl.exe có sẵn không
where cl.exe >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: cl.exe not found in PATH!
    echo Please open "Developer Command Prompt for VS" instead.
    echo Or install Visual Studio Build Tools.
    echo.
    echo Attempting to find Visual Studio...
    
    REM Thử tìm Visual Studio
    if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    ) else (
        echo ERROR: Cannot find Visual Studio installation!
        echo Please install Visual Studio Build Tools or Visual Studio Community.
        echo.
        pause
        exit /b 1
    )
)

REM Kiểm tra lại cl.exe sau khi set environment
where cl.exe >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: cl.exe still not found!
    echo Please open "Developer Command Prompt for VS" and run this script.
    echo.
    pause
    exit /b 1
)

echo [1/3] Checking CUDA...
nvcc --version
echo.

echo [2/3] Checking MSVC compiler...
cl.exe 2>&1 | findstr /C:"Microsoft" /C:"Version"
echo.

REM Xác định compute capability (mặc định sm_86 cho RTX 30xx)
set ARCH=sm_86
echo [3/3] Building with architecture: %ARCH%
echo.

REM Build command
echo Compiling...
nvcc main.cpp fast_gaussian_blur_cuda.cu -o fastblur_cuda.exe ^
     -O3 -std=c++17 -arch=%ARCH% ^
     -Xcompiler "/openmp" -DUSE_CUDA -lcudart

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================
    echo   BUILD SUCCESSFUL!
    echo ================================================================
    echo.
    echo Executable: fastblur_cuda.exe
    echo.
    echo To run:
    echo   fastblur_cuda.exe test.jpg output_cuda.png 5.0
    echo.
) else (
    echo.
    echo ================================================================
    echo   BUILD FAILED!
    echo ================================================================
    echo.
    echo Please check the error messages above.
    echo.
    pause
    exit /b 1
)

pause

