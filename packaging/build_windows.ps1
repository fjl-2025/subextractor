# Windows 打包脚本（PowerShell 5+）。
# 用法：在仓库根目录执行  powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
# 产物：dist\SubExtractor\（模型不进包，另见该目录下 models\ 放置说明）

$ErrorActionPreference = "Stop"

$PackagingDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $PackagingDir
Set-Location $RepoRoot

Write-Host "==> PyInstaller 打包（onedir）..." -ForegroundColor Cyan
python -m PyInstaller --noconfirm --clean (Join-Path $PackagingDir "subextractor.spec")

$Dist = Join-Path $RepoRoot "dist\SubExtractor"
if (-not (Test-Path (Join-Path $Dist "SubExtractor.exe"))) {
    throw "未找到产物 SubExtractor.exe，打包失败"
}

Write-Host "==> 准备外置模型目录骨架..." -ForegroundColor Cyan
$Models = Join-Path $Dist "models"
New-Item -ItemType Directory -Force -Path (Join-Path $Models "asr") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Models "mt") | Out-Null
Copy-Item (Join-Path $RepoRoot "models\checksums.json") $Models -Force
Copy-Item (Join-Path $PackagingDir "models-README.txt") (Join-Path $Models "放置模型请读我.txt") -Force

Write-Host ""
Write-Host "==> 打包完成：$Dist" -ForegroundColor Green
Write-Host "    GUI：SubExtractor.exe ｜ 命令行：subextractor-cli.exe"
