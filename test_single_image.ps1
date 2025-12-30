# ================================================================
# Script test với 1 ảnh cho cả 3 trường hợp: Sequential, OpenMP, CUDA
# ================================================================

param(
    [string]$InputImage = "test.jpg",
    [string]$OutputPrefix = "output",
    [float]$Sigma = 5.0,
    [int]$Passes = 3,
    [string]$Border = "extend"
)

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  TEST SINGLE IMAGE - 3 VERSIONS" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Kiểm tra ảnh input
if (-not (Test-Path $InputImage)) {
    Write-Host "ERROR: Input image not found: $InputImage" -ForegroundColor Red
    exit 1
}

Write-Host "Input image: $InputImage" -ForegroundColor Yellow
Write-Host "Sigma: $Sigma" -ForegroundColor Yellow
Write-Host "Passes: $Passes" -ForegroundColor Yellow
Write-Host "Border: $Border" -ForegroundColor Yellow
Write-Host ""

# Test với fastblur.exe (Sequential và OpenMP)
if (Test-Path "fastblur.exe") {
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host "  Running fastblur.exe (Sequential + OpenMP comparison)" -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host ""
    
    $output1 = "${OutputPrefix}_comparison.png"
    & .\fastblur.exe $InputImage $output1 $Sigma $Passes $Border
    
    Write-Host ""
} else {
    Write-Host "WARNING: fastblur.exe not found!" -ForegroundColor Yellow
}

# Test với fastblur_cuda.exe (nếu có)
if (Test-Path "fastblur_cuda.exe") {
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host "  Running fastblur_cuda.exe (CUDA version)" -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host ""
    
    $output2 = "${OutputPrefix}_cuda.png"
    & .\fastblur_cuda.exe $InputImage $output2 $Sigma $Passes $Border
    
    Write-Host ""
} else {
    Write-Host "WARNING: fastblur_cuda.exe not found!" -ForegroundColor Yellow
    Write-Host "Build it with: make fastblur_cuda" -ForegroundColor Gray
}

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  TEST COMPLETED!" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Output files:" -ForegroundColor Yellow
if (Test-Path "${OutputPrefix}_comparison.png") {
    Write-Host "  - ${OutputPrefix}_comparison.png (from fastblur.exe)" -ForegroundColor White
}
if (Test-Path "${OutputPrefix}_cuda.png") {
    Write-Host "  - ${OutputPrefix}_cuda.png (from fastblur_cuda.exe)" -ForegroundColor White
}
Write-Host ""

