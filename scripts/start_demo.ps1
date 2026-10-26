param(
    [ValidateSet('csl', 'libras')][string]$Language = 'csl',
    [ValidateRange(0, 32)][int]$Camera = 0,
    [string]$Video,
    [switch]$Manual,
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [switch]$NoLaunch
)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath $ProjectRoot).ProviderPath
$taskProbe = "import importlib.util,onnxruntime as o,sys; assert all(importlib.util.find_spec(n) is not None for n in ('PySide6','mediapipe','cv2','slbridge')); assert sys.argv[1] != 'gpu' or 'CUDAExecutionProvider' in o.get_available_providers()"
$taskRuntime = $null
foreach ($taskEnvironment in @('gpu', 'libras')) {
    $taskPython = Join-Path $taskRoot ('.venv-' + $taskEnvironment + '\Scripts\python.exe')
    if (-not (Test-Path -LiteralPath $taskPython)) { continue }
    # Windows PowerShell 5 treats native stderr as an error record. A failed
    # dependency probe must fall through to the next environment.
    $taskProbePreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'SilentlyContinue'
        & $taskPython -c $taskProbe $taskEnvironment 2>$null
    } finally { $ErrorActionPreference = $taskProbePreference }
    if ($LASTEXITCODE -eq 0) { $taskRuntime = $taskPython; break }
}
if (-not $taskRuntime) { throw 'No usable desktop Python environment. Install requirements-desktop.txt first.' }
$taskModels = @{
    csl = 'artifacts\models\cecsl-tfnet-onnx-20261009'
    libras = 'artifacts\experiments\libras-rf-pilot-20261009\models\libras-rf-pilot-20261009'
}
$taskModel = Join-Path $taskRoot $taskModels[$Language]
if (-not (Test-Path -LiteralPath (Join-Path $taskModel 'manifest.json'))) { throw 'The selected sign-language model is not configured.' }
$taskArguments = @((Join-Path $PSScriptRoot 'launch_desktop.py'), '--model', $taskModel, '--device', 'auto')
if ($Video) {
    $taskVideo = (Resolve-Path -LiteralPath $Video).ProviderPath
    $taskArguments += @('--video', $taskVideo)
} else {
    $taskArguments += @('--camera', [string]$Camera)
}
if (-not $Manual) { $taskArguments += '--continuous' }
if ($NoLaunch) {
    @{ python = $taskRuntime; arguments = $taskArguments; workingDirectory = $taskRoot } | ConvertTo-Json -Compress
    exit 0
}
# Native argv is quoted explicitly for Start-Process; all path arguments are
# resolved Windows paths (double quotes cannot occur in a Windows filename).
$taskQuoted = $taskArguments | ForEach-Object { '"' + $_ + '"' }
$taskPythonw = Join-Path (Split-Path -Parent $taskRuntime) 'pythonw.exe'
Start-Process -FilePath $taskPythonw -ArgumentList $taskQuoted -WorkingDirectory $taskRoot -WindowStyle Hidden
