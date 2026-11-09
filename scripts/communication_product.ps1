param(
    [Parameter(Mandatory = $true)][string]$Profile,
    [ValidateSet('Start', 'Stop', 'StopProduct', 'RestartProduct')][string]$Action = 'Start'
)
$ErrorActionPreference = 'Stop'
$settings = Get-Content -LiteralPath $Profile -Raw -Encoding UTF8 | ConvertFrom-Json
if ($settings.schema_version -ne 1) { throw 'Unsupported local product profile.' }
$outputRoot = [IO.Path]::GetFullPath($settings.output_root)
[void](New-Item -ItemType Directory -Path $outputRoot -Force)

function Get-OwnedListener($service, $module, $tokens) {
    $listener = @(Get-NetTCPConnection -State Listen -LocalPort $service.port -ErrorAction SilentlyContinue)
    if (-not $listener.Count) { return $null }
    $ownerIds = @($listener.OwningProcess | Select-Object -Unique)
    if ($ownerIds.Count -ne 1) { throw "Port $($service.port) has multiple owners." }
    $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$($ownerIds[0])"
    foreach ($token in @($module) + $tokens) {
        if (-not $owner.CommandLine -or -not $owner.CommandLine.Contains([string]$token)) {
            throw "Port $($service.port) belongs to a different service; it has been left running."
        }
    }
    return $owner
}

function Stop-OwnedTree($owner) {
    if (-not $owner) { return }
    $snapshot = @(Get-CimInstance Win32_Process)
    $ids = [Collections.Generic.List[int]]::new()
    $ids.Add([int]$owner.ProcessId)
    do {
        $added = $false
        foreach ($item in $snapshot) {
            if ($ids.Contains([int]$item.ParentProcessId) -and -not $ids.Contains([int]$item.ProcessId)) {
                $ids.Add([int]$item.ProcessId); $added = $true
            }
        }
    } while ($added)
    for ($index = $ids.Count - 1; $index -ge 0; $index--) {
        $saved = $snapshot | Where-Object ProcessId -EQ $ids[$index]
        $current = Get-CimInstance Win32_Process -Filter "ProcessId=$($ids[$index])" -ErrorAction SilentlyContinue
        if ($current -and $current.CommandLine -eq $saved.CommandLine -and $current.CreationDate -eq $saved.CreationDate) {
            Stop-Process -Id $ids[$index] -ErrorAction Stop
        }
    }
}

function Start-ServiceProcess($service, $arguments, $name, $module, $tokens) {
    $owner = Get-OwnedListener $service $module $tokens
    if ($owner) { return $owner }
    foreach ($path in @($service.python, $service.workdir)) {
        if (-not (Test-Path -LiteralPath $path)) { throw "Required local asset missing: $path" }
    }
    # Quote each Windows argument; profiles are server-owned local files, never request input.
    $quoted = foreach ($argument in $arguments) {
        if ([string]$argument -match '["\r\n]') { throw 'Invalid launch argument.' }
        '"' + [string]$argument + '"'
    }
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $launch = Start-Process -FilePath $service.python -ArgumentList ($quoted -join ' ') -WorkingDirectory $service.workdir -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $outputRoot "$name-$stamp.stdout.log") -RedirectStandardError (Join-Path $outputRoot "$name-$stamp.stderr.log")
    $deadline = (Get-Date).AddSeconds(30)
    do {
        Start-Sleep -Milliseconds 250
        $owner = Get-OwnedListener $service $module $tokens
        $launch.Refresh()
        if ($launch.HasExited -and -not $owner) { throw "$name did not start. See $outputRoot logs." }
    } until ($owner -or (Get-Date) -ge $deadline)
    if (-not $owner) { throw "$name has not started within 30 seconds. See $outputRoot logs." }
    return $owner
}

$recognition = $settings.recognition
$product = $settings.product
$recognitionTokens = @($recognition.catalog, '--route', 'csl=csl-tfnet', 'libras=libras-rf', '--port')
$recognitionUrl = "http://127.0.0.1:$($recognition.port)"
$productTokens = @($product.composition_config, $product.presentation, '--recognition-url', $recognitionUrl, '--port')
if ($Action -in @('Stop', 'StopProduct', 'RestartProduct')) {
    $owner = Get-OwnedListener $product 'slbridge.animation.scene_cli' $productTokens
    if ($Action -eq 'RestartProduct' -and $owner) {
        $generation = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*generate_custom_signspark.py*' -and $_.Name -like 'python*' })
        if ($generation.Count) { throw 'A full animation generation is active; wait or cancel it before restarting.' }
    }
    Stop-OwnedTree $owner
    if ($Action -eq 'StopProduct') {
        Write-Output 'Stopped only the matching product service and its own workers.'
        exit 0
    }
    if ($Action -eq 'Stop') {
        Stop-OwnedTree (Get-OwnedListener $recognition 'slbridge.api_cli' $recognitionTokens)
        Write-Output 'Stopped only the matching local product and recognition services.'
        exit 0
    }
}
$recognitionArgs = @('-u', '-X', 'utf8', '-m', 'slbridge.api_cli', '--catalog', $recognition.catalog, '--route', 'csl=csl-tfnet', '--route', 'libras=libras-rf', '--device', 'cpu', '--intra-op-threads', '4', '--port', [string]$recognition.port)
$recognitionOwner = Start-ServiceProcess $recognition $recognitionArgs 'recognition' 'slbridge.api_cli' $recognitionTokens
$productArgs = @('-u', '-X', 'utf8', '-m', 'slbridge.animation.scene_cli', '--archive', $product.archive, '--expected-sha256', $product.archive_sha256, '--presentation', $product.presentation, '--corpus', $product.corpus, '--composition-config', $product.composition_config, '--recognition-url', $recognitionUrl, '--port', [string]$product.port)
$productOwner = Start-ServiceProcess $product $productArgs 'product' 'slbridge.animation.scene_cli' $productTokens
$productUrl = "http://127.0.0.1:$($product.port)"
$identity = Invoke-RestMethod "$productUrl/v1/animation/presentation" -TimeoutSec 15
$library = Invoke-RestMethod "$productUrl/v1/animation/composition" -TimeoutSec 15
if ($identity.avatar_sha256 -ne $product.avatar_sha256 -or $library.bundle_id -ne $product.bundle_id) {
    throw 'The running avatar or word library differs from the selected product profile.'
}
$status = Invoke-RestMethod "$productUrl/v1/communication/recognition" -TimeoutSec 15
if (-not $status.ready) { throw 'The product is running but the recognition connection is not ready.' }
@{ recognition_pid = $recognitionOwner.ProcessId; product_pid = $productOwner.ProcessId; url = "$productUrl/communicate/"; checked_at = (Get-Date).ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputRoot 'runtime-state.json') -Encoding UTF8
Write-Output "Product: $productUrl/communicate/"
Write-Output 'The text planner may still be warming up. Readiness appears in the page header.'
