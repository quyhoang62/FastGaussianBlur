# ================================================================
# Script để test batch processing với 200 ảnh
# ================================================================

param(
    [string]$InputFolder = "test_images",
    [string]$OutputFolder = "output_batch",
    [float]$Sigma = 5.0,
    [int]$Passes = 3,
    [string]$Border = "extend"
)

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  BATCH PROCESSING TEST - 200 IMAGES" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Kiểm tra folder input
if (-not (Test-Path $InputFolder)) {
    Write-Host "ERROR: Input folder not found: $InputFolder" -ForegroundColor Red
    Write-Host "Creating test folder structure..." -ForegroundColor Yellow
    
    # Tạo folder và copy ảnh test
    New-Item -ItemType Directory -Path $InputFolder -Force | Out-Null
    
    # Copy test.jpg nhiều lần để tạo 200 ảnh (hoặc dùng ảnh có sẵn)
    if (Test-Path "test.jpg") {
        Write-Host "Copying test.jpg to create 200 images..." -ForegroundColor Yellow
        for ($i = 1; $i -le 200; $i++) {
            $dest = Join-Path $InputFolder "test_$($i.ToString('000')).jpg"
            Copy-Item "test.jpg" -Destination $dest -Force
        }
        Write-Host "Created 200 test images in $InputFolder" -ForegroundColor Green
    } else {
        Write-Host "ERROR: test.jpg not found. Please create $InputFolder with 200 images." -ForegroundColor Red
        exit 1
    }
}

# Đếm số ảnh
$imageCount = (Get-ChildItem -Path $InputFolder -Include *.jpg,*.png,*.bmp -Recurse).Count
Write-Host "Found $imageCount images in $InputFolder" -ForegroundColor Cyan
Write-Host ""

if ($imageCount -eq 0) {
    Write-Host "ERROR: No images found in $InputFolder" -ForegroundColor Red
    exit 1
}

# Kiểm tra executable
$batchExe = "batch_main.exe"
if (-not (Test-Path $batchExe)) {
    Write-Host "ERROR: $batchExe not found!" -ForegroundColor Red
    Write-Host "Please build it first: make batch_main" -ForegroundColor Yellow
    exit 1
}

# Chạy batch processing
Write-Host "Running batch processing..." -ForegroundColor Yellow
Write-Host "  Input:  $InputFolder" -ForegroundColor Gray
Write-Host "  Output: $OutputFolder" -ForegroundColor Gray
Write-Host "  Sigma:  $Sigma" -ForegroundColor Gray
Write-Host "  Passes: $Passes" -ForegroundColor Gray
Write-Host "  Border: $Border" -ForegroundColor Gray
Write-Host ""

$startTime = Get-Date

& .\batch_main.exe $InputFolder $OutputFolder $Sigma $Passes $Border

$endTime = Get-Date
$duration = ($endTime - $startTime).TotalSeconds

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  BATCH PROCESSING COMPLETED!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host "Total wall-clock time: $([math]::Round($duration, 2)) seconds" -ForegroundColor Cyan
Write-Host ""

# Kiểm tra output
if (Test-Path $OutputFolder) {
    $seqCount = (Get-ChildItem -Path "$OutputFolder/sequential" -ErrorAction SilentlyContinue).Count
    $ompCount = (Get-ChildItem -Path "$OutputFolder/openmp" -ErrorAction SilentlyContinue).Count
    $cudaCount = (Get-ChildItem -Path "$OutputFolder/cuda" -ErrorAction SilentlyContinue).Count
    
    Write-Host "Output files:" -ForegroundColor Yellow
    if ($seqCount -gt 0) { Write-Host "  Sequential: $seqCount files" -ForegroundColor White }
    if ($ompCount -gt 0) { Write-Host "  OpenMP:     $ompCount files" -ForegroundColor White }
    if ($cudaCount -gt 0) { Write-Host "  CUDA:       $cudaCount files" -ForegroundColor White }
    Write-Host ""
}

