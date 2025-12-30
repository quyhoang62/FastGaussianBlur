# ================================================================
# Script để tạo 200 ảnh test từ ảnh mẫu
# ================================================================

param(
    [string]$SourceImage = "test.jpg",
    [string]$OutputFolder = "test_images",
    [int]$Count = 200
)

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  CREATING TEST IMAGES" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Kiểm tra ảnh nguồn
if (-not (Test-Path $SourceImage)) {
    Write-Host "ERROR: Source image not found: $SourceImage" -ForegroundColor Red
    Write-Host "Please provide a valid image file." -ForegroundColor Yellow
    exit 1
}

# Tạo output folder
if (-not (Test-Path $OutputFolder)) {
    New-Item -ItemType Directory -Path $OutputFolder -Force | Out-Null
    Write-Host "Created folder: $OutputFolder" -ForegroundColor Green
}

# Copy ảnh
Write-Host "Copying $SourceImage to create $Count images..." -ForegroundColor Yellow

$sourceExt = [System.IO.Path]::GetExtension($SourceImage)
$baseName = [System.IO.Path]::GetFileNameWithoutExtension($SourceImage)

for ($i = 1; $i -le $Count; $i++) {
    $dest = Join-Path $OutputFolder "$baseName`_$($i.ToString('000'))$sourceExt"
    Copy-Item $SourceImage -Destination $dest -Force
    
    if ($i % 50 -eq 0) {
        Write-Host "  Created $i images..." -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  SUCCESS!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host "Created $Count images in $OutputFolder" -ForegroundColor Cyan
Write-Host ""

# Verify
$actualCount = (Get-ChildItem -Path $OutputFolder -Include *.* -File).Count
Write-Host "Verified: $actualCount files in folder" -ForegroundColor Cyan
Write-Host ""

