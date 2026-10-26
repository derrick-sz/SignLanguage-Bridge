param(
    [string]$EnvPrefix = (Join-Path $PSScriptRoot '..\.conda-env'),
    [string]$Proxy = '',
    [switch]$SkipCreate,
    [switch]$SkipTorch
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$EnvPrefix = [System.IO.Path]::GetFullPath($EnvPrefix)
$Python = Join-Path $EnvPrefix 'Scripts\python.exe'

if ($Proxy) {
    $env:HTTP_PROXY = $Proxy
    $env:HTTPS_PROXY = $Proxy
}
$env:PIP_CACHE_DIR = Join-Path $ProjectRoot '.cache\pip'
$env:TMP = Join-Path $ProjectRoot '.cache\tmp'
$env:TEMP = $env:TMP
New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR, $env:TMP | Out-Null

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw 'conda is not available on PATH'
}
if (-not $SkipCreate -and -not (Test-Path -LiteralPath $Python)) {
    & conda create -p $EnvPrefix python=3.10 pip -y --override-channels -c defaults
    if ($LASTEXITCODE -ne 0) { throw "conda create failed with exit code $LASTEXITCODE" }
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Conda environment Python not found: $Python"
}

if (-not $SkipTorch) {
    & $Python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 torchvision==0.24.1
    if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed with exit code $LASTEXITCODE" }
}
& $Python -m pip install -r (Join-Path $ProjectRoot 'requirements-unisign.txt')
if ($LASTEXITCODE -ne 0) { throw "Uni-Sign requirements failed with exit code $LASTEXITCODE" }

$Rtmlib = Join-Path $ProjectRoot 'models\external\uni_sign_source\demo\rtmlib-main'
if (-not (Test-Path -LiteralPath $Rtmlib)) {
    throw "RTMPose source is missing. Run scripts\download_uni_sign.py first: $Rtmlib"
}
& $Python -m pip install --no-deps -e $Rtmlib
if ($LASTEXITCODE -ne 0) { throw "rtmlib install failed with exit code $LASTEXITCODE" }
& $Python -m pip install --no-deps -e $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw "project install failed with exit code $LASTEXITCODE" }

& $Python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); import transformers, rtmlib, PySide6; print('environment ready')"
if ($LASTEXITCODE -ne 0) { throw "environment verification failed with exit code $LASTEXITCODE" }
