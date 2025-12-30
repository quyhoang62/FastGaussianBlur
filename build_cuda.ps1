# ================================================================
# PowerShell Script để Build CUDA Version
# ================================================================

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  BUILDING FAST GAUSSIAN BLUR - CUDA VERSION" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Kiểm tra nvcc
$nvccPath = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvccPath) {
    Write-Host "ERROR: nvcc not found in PATH!" -ForegroundColor Red
    Write-Host "Please install CUDA Toolkit or add it to PATH." -ForegroundColor Yellow
    exit 1
}

Write-Host "[1/4] Checking CUDA..." -ForegroundColor Green
nvcc --version
Write-Host ""

# Kiểm tra cl.exe
$clPath = Get-Command cl.exe -ErrorAction SilentlyContinue
if (-not $clPath) {
    Write-Host "WARNING: cl.exe not found in PATH!" -ForegroundColor Yellow
    Write-Host "Attempting to find Visual Studio..." -ForegroundColor Yellow
    
    # Tìm Visual Studio
    $vsPaths = @(
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    )
    
    $found = $false
    foreach ($path in $vsPaths) {
        if (Test-Path $path) {
            Write-Host "Found Visual Studio at: $path" -ForegroundColor Green
            Write-Host "Please run this script from Developer Command Prompt for VS" -ForegroundColor Yellow
            Write-Host "Or manually set environment variables." -ForegroundColor Yellow
            $found = $true
            break
        }
    }
    
    if (-not $found) {
        Write-Host "ERROR: Cannot find Visual Studio installation!" -ForegroundColor Red
        Write-Host "Please install Visual Studio Build Tools or Visual Studio Community." -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "[2/4] Checking MSVC compiler..." -ForegroundColor Green
$clVersion = & cl.exe 2>&1 | Select-String "Microsoft" | Select-Object -First 1
if ($clVersion) {
    Write-Host $clVersion -ForegroundColor Cyan
} else {
    Write-Host "WARNING: Could not get compiler version" -ForegroundColor Yellow
}
Write-Host ""

# Kiểm tra GPU và compute capability
Write-Host "[3/4] Checking GPU..." -ForegroundColor Green
$gpuInfo = nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host $gpuInfo -ForegroundColor Cyan
    # Parse compute capability (format: "GPU Name, 8.6")
    $computeCap = ($gpuInfo -split ',')[1].Trim()
    $major = [int]($computeCap -split '\.')[0]
    $minor = [int]($computeCap -split '\.')[1]
    $arch = "sm_$major$minor"
    Write-Host "Using architecture: $arch" -ForegroundColor Cyan
} else {
    Write-Host "WARNING: Could not query GPU info, using default sm_86" -ForegroundColor Yellow
    $arch = "sm_86"
}
Write-Host ""

# Build
Write-Host "[4/4] Building..." -ForegroundColor Green
Write-Host "Command: nvcc main.cpp fast_gaussian_blur_cuda.cu -o fastblur_cuda.exe -O3 -std=c++17 -arch=$arch -Xcompiler `/openmp` -DUSE_CUDA -lcudart" -ForegroundColor Gray
Write-Host ""

$buildCmd = "nvcc main.cpp fast_gaussian_blur_cuda.cu -o fastblur_cuda.exe -O3 -std=c++17 -arch=$arch -Xcompiler `/openmp` -DUSE_CUDA -lcudart"
Invoke-Expression $buildCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host "  BUILD SUCCESSFUL!" -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Executable: fastblur_cuda.exe" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To run:" -ForegroundColor Yellow
    Write-Host "  .\fastblur_cuda.exe test.jpg output_cuda.png 5.0" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Red
    Write-Host "  BUILD FAILED!" -ForegroundColor Red
    Write-Host "================================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please check the error messages above." -ForegroundColor Yellow
    exit 1
}

