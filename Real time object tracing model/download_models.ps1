# Download Script for MobileNetSSD Model Files
# This script downloads the required model files for object detection

Write-Host "Downloading MobileNetSSD Model Files..." -ForegroundColor Green
Write-Host ""

# Create a temporary directory if needed
$modelDir = "."

# URLs for model files
$prototxtUrl = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt"
$caffemodelUrl = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel"

# Download prototxt file
Write-Host "[1/2] Downloading MobileNetSSD_deploy.prototxt..." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri $prototxtUrl -OutFile "$modelDir\MobileNetSSD_deploy.prototxt"
    Write-Host "✓ Prototxt file downloaded successfully!" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to download prototxt file: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Download caffemodel file (this is a large file ~23MB)
Write-Host "[2/2] Downloading MobileNetSSD_deploy.caffemodel (~23MB)..." -ForegroundColor Yellow
Write-Host "This may take a few moments..." -ForegroundColor Cyan
try {
    Invoke-WebRequest -Uri $caffemodelUrl -OutFile "$modelDir\MobileNetSSD_deploy.caffemodel"
    Write-Host "✓ Caffemodel file downloaded successfully!" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to download caffemodel file: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "All model files downloaded successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "You can now run the object detection script:" -ForegroundColor Cyan
Write-Host "  python object_detection.py" -ForegroundColor White
Write-Host ""
