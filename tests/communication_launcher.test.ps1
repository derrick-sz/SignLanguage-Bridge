param(
    [string]$Launcher = (Join-Path $PSScriptRoot '../scripts/communication_product.ps1')
)

$ErrorActionPreference = 'Stop'

# This harness never invokes the launcher as a whole. It extracts its ownership
# helper(s) and its configuration/Stop preamble into an isolated script scope.
# OS process discovery, starts, stops, HTTP and filesystem writes are mocked.
$reviewSource = Get-Content -LiteralPath $Launcher -Raw -Encoding UTF8
$reviewParseTokens = $null
$reviewParseErrors = $null
$reviewAst = [Management.Automation.Language.Parser]::ParseInput(
    $reviewSource, [ref]$reviewParseTokens, [ref]$reviewParseErrors)
if ($reviewParseErrors.Count) { throw ($reviewParseErrors | Out-String) }

$reviewFunctions = @($reviewAst.EndBlock.Statements | Where-Object {
    $_ -is [Management.Automation.Language.FunctionDefinitionAst] -and
    $_.Name -notin @('Stop-OwnedTree', 'Start-ServiceProcess')
})
if (-not ($reviewFunctions | Where-Object Name -EQ 'Get-OwnedListener')) {
    throw 'Could not locate Get-OwnedListener in the launcher.'
}
$reviewConfiguration = @($reviewAst.EndBlock.Statements | Where-Object {
    $_ -is [Management.Automation.Language.AssignmentStatementAst] -and
    $_.Left -is [Management.Automation.Language.VariableExpressionAst] -and
    $_.Left.VariablePath.UserPath -eq 'recognition'
})
$reviewLaunchArgs = @($reviewAst.EndBlock.Statements | Where-Object {
    $_ -is [Management.Automation.Language.AssignmentStatementAst] -and
    $_.Left -is [Management.Automation.Language.VariableExpressionAst] -and
    $_.Left.VariablePath.UserPath -eq 'recognitionArgs'
})
if ($reviewConfiguration.Count -ne 1 -or $reviewLaunchArgs.Count -ne 1) {
    throw 'Could not locate the configuration/Stop preamble in the launcher.'
}
$reviewPreamble = $reviewSource.Substring(
    $reviewConfiguration[0].Extent.StartOffset,
    $reviewLaunchArgs[0].Extent.StartOffset - $reviewConfiguration[0].Extent.StartOffset)
# A broken ownership guard must not let an exit statement terminate the harness.
$reviewPreamble = $reviewPreamble -replace '\bexit\s+0\b', "throw 'HARNESS_WOULD_EXIT_WITHOUT_REFUSING_MISMATCH'"
$reviewHelperSource = ($reviewFunctions | ForEach-Object { $_.Extent.Text }) -join "`n"

function Invoke-OwnershipCase {
    param(
        [string]$Name,
        [string]$ConfiguredModel = 'unisign_pose_stage1',
        [string]$RunningModel = 'unisign_pose_stage1',
        [string]$ConfiguredDevice = 'auto',
        [string]$RunningDevice = 'auto',
        [string[]]$ExtraRecognitionArguments = @(),
        [string]$ConfiguredRealtime = '',
        [string]$ConfiguredDemo = '',
        [string]$RunningDemo = '',
        [string[]]$ExtraProductArguments = @(),
        [string]$Action = 'Start',
        [bool]$ExpectedRejection = $false,
        [switch]$Quoted
    )

    & {
        param($CaseName, $ProfileModel, $LiveModel, $ProfileDevice, $LiveDevice,
              $ExtraArgs, $CaseAction, $RejectExpected, $QuoteArguments,
              $HelperSource, $PreambleSource, $RealtimeConfig, $DemoConfig, $LiveDemo, $ProductExtras)

        $catalogPath = if ($QuoteArguments) { 'C:/review assets/catalog.json' } else { 'C:/review/catalog.json' }
        $settings = [pscustomobject]@{
            recognition = [pscustomobject]@{
                port = 8765; catalog = $catalogPath; csl_model = $ProfileModel
                libras_model = 'libras-rf'; device = $ProfileDevice
                realtime_config = $RealtimeConfig
            }
            product = [pscustomobject]@{
                port = 8879; composition_config = 'C:/review/composition.json'
                presentation = 'C:/review/presentation'
                demo_catalog = $DemoConfig
            }
        }
        $recognitionArguments = @('python.exe', '-m', 'slbridge.api_cli', '--catalog', $catalogPath,
            '--route', "csl=$LiveModel", '--route', 'libras=libras-rf', '--device', $LiveDevice,
            '--port', '8765') + @($ExtraArgs)
        $recognitionCommand = ($recognitionArguments | ForEach-Object {
            if ($QuoteArguments) { '"' + $_ + '"' } else { $_ }
        }) -join ' '
        $productArguments = @('python.exe', '-m', 'slbridge.animation.scene_cli',
            '--composition-config', 'C:/review/composition.json', '--presentation', 'C:/review/presentation',
            '--recognition-url', 'http://127.0.0.1:8765', '--port', '8879')
        if ($LiveDemo) { $productArguments += @('--demo-catalog', $LiveDemo) }
        $productArguments += @($ProductExtras)
        $productCommand = ($productArguments | ForEach-Object {
            if ($QuoteArguments) { '"' + $_ + '"' } else { $_ }
        }) -join ' '
        $mockOwners = @{
            8765 = [pscustomobject]@{ ProcessId = 4242; CommandLine = $recognitionCommand }
            8879 = [pscustomobject]@{ ProcessId = 4343; CommandLine = $productCommand }
        }
        $mockStops = [Collections.Generic.List[int]]::new()

        # The actual helper source is used, with OS entry points replaced below.
        . ([scriptblock]::Create($HelperSource))
        function Get-NetTCPConnection {
            param($State, $LocalPort, $ErrorAction)
            if ($mockOwners.ContainsKey([int]$LocalPort)) {
                [pscustomobject]@{ OwningProcess = $mockOwners[[int]$LocalPort].ProcessId }
            }
        }
        function Get-CimInstance {
            param($ClassName, $Filter, $ErrorAction)
            if (-not $Filter) { return @() } # No active generation workers.
            if ($Filter -match '^ProcessId=(\d+)$') {
                $requestedId = [int]$Matches[1]
                $mockOwners.Values | Where-Object ProcessId -EQ $requestedId
                return
            }
            throw "Unsupported mocked process lookup: $Filter"
        }
        function Stop-OwnedTree {
            param($Owner)
            if ($Owner) { $mockStops.Add([int]$Owner.ProcessId) }
        }
        function Start-ServiceProcess {
            param($Service, $Arguments, $Name, $Module, $Tokens)
            Get-OwnedListener $Service $Module $Tokens
        }
        function Start-Process { throw 'FORBIDDEN: real process start attempted by harness.' }
        function Stop-Process { throw 'FORBIDDEN: real process stop attempted by harness.' }
        function Invoke-RestMethod { throw 'FORBIDDEN: HTTP request attempted by harness.' }
        function New-Item { throw 'FORBIDDEN: filesystem write attempted by harness.' }
        function Set-Content { throw 'FORBIDDEN: filesystem write attempted by harness.' }

        $Action = $CaseAction
        $rejected = $false
        $failure = $null
        try {
            . ([scriptblock]::Create($PreambleSource))
            # This is the next ownership check after the Stop/Restart preamble.
            # Start-ServiceProcess uses the same helper before any real launch.
            $null = Get-OwnedListener $recognition 'slbridge.api_cli' $recognitionTokens
            $null = Get-OwnedListener $product 'slbridge.animation.scene_cli' $productTokens
        } catch {
            $rejected = $true
            $failure = $_.Exception.Message
        }
        $unexpectedError = $failure -like 'FORBIDDEN:*' -or
            $failure -like 'HARNESS_*' -or $failure -like 'Unsupported mocked*'
        [pscustomobject]@{
            name = $CaseName
            passed = ($rejected -eq $RejectExpected -and $mockStops.Count -eq 0 -and -not $unexpectedError)
            expected_rejection = $RejectExpected
            actual_rejection = $rejected
            stopped_pids = @($mockStops.ToArray())
            error = $failure
        }
    } $Name $ConfiguredModel $RunningModel $ConfiguredDevice $RunningDevice `
      $ExtraRecognitionArguments $Action $ExpectedRejection ([bool]$Quoted) `
      $reviewHelperSource $reviewPreamble $ConfiguredRealtime $ConfiguredDemo $RunningDemo $ExtraProductArguments
}

$reviewResults = @(
    Invoke-OwnershipCase -Name 'disabled demo refuses equals option with a quoted value' `
        -ExtraProductArguments @('--demo-catalog="C:/review assets/demos.json"') -ExpectedRejection $true
    Invoke-OwnershipCase -Name 'disabled realtime refuses equals option with a quoted value' `
        -ExtraRecognitionArguments @('--realtime-config="C:/review assets/live.json"') -ExpectedRejection $true
    Invoke-OwnershipCase -Name 'disabled demo option refuses an enabled existing product' -Quoted `
        -RunningDemo 'C:/review/demos.json' -ExpectedRejection $true
    Invoke-OwnershipCase -Name 'disabled demo StopProduct cannot stop a differently configured instance' -Quoted -Action 'StopProduct' `
        -RunningDemo 'C:/review/demos.json' -ExpectedRejection $true
    Invoke-OwnershipCase -Name 'disabled realtime option refuses a still-enabled recognition instance' -Quoted `
        -ExtraRecognitionArguments @('--realtime-config','C:/review/live.json') -ExpectedRejection $true
    Invoke-OwnershipCase -Name 'matching demo catalog with quoted path is reusable' -Quoted `
        -ConfiguredDemo 'C:/review assets/demos.json' -RunningDemo 'C:/review assets/demos.json'
    Invoke-OwnershipCase -Name 'different demo catalog refuses StopProduct without stopping anything' -Quoted -Action 'StopProduct' `
        -ConfiguredDemo 'C:/review/demos.json' -RunningDemo 'C:/review/demos-other.json' -ExpectedRejection $true
    Invoke-OwnershipCase -Name 'displaced demo path cannot satisfy adjacent catalog option' -Quoted `
        -ConfiguredDemo 'C:/review/demos.json' -RunningDemo 'C:/review/other.json' `
        -ExtraProductArguments @('--archive','C:/review/demos.json') -ExpectedRejection $true
    Invoke-OwnershipCase -Name 'matching live configuration is reusable' -Quoted `
        -ConfiguredRealtime 'C:/review/live.json' -ExtraRecognitionArguments @('--realtime-config','C:/review/live.json')
    Invoke-OwnershipCase -Name 'missing live option is rejected before a stop' -Quoted -Action 'Stop' `
        -ConfiguredRealtime 'C:/review/live.json' -ExtraRecognitionArguments @('--token-env','C:/review/live.json') -ExpectedRejection $true
    Invoke-OwnershipCase -Name 'different live configuration is rejected' -Quoted `
        -ConfiguredRealtime 'C:/review/live.json' -ExtraRecognitionArguments @('--realtime-config','C:/review/live.json-other') -ExpectedRejection $true
    Invoke-OwnershipCase -Name 'matching raw arguments are reusable'
    Invoke-OwnershipCase -Name 'matching quoted arguments and paths are reusable' -Quoted
    Invoke-OwnershipCase -Name 'prefix-only route alias is rejected' `
        -ConfiguredModel 'unisign_pose' -ExpectedRejection $true -Quoted
    Invoke-OwnershipCase -Name 'wrong device is rejected even if its text appears elsewhere' `
        -ConfiguredDevice 'cpu' -ExtraRecognitionArguments @('--token-env', 'cpu') `
        -ExpectedRejection $true -Quoted
    Invoke-OwnershipCase -Name 'wrong-route Stop leaves product alive' `
        -ConfiguredModel 'does-not-exist' -Action 'Stop' -ExpectedRejection $true -Quoted
    Invoke-OwnershipCase -Name 'wrong-route RestartProduct leaves product alive' `
        -ConfiguredModel 'does-not-exist' -Action 'RestartProduct' -ExpectedRejection $true -Quoted
)
$reviewFailures = @($reviewResults | Where-Object { -not $_.passed })
[pscustomobject]@{
    launcher = [IO.Path]::GetFullPath($Launcher)
    passed = $reviewResults.Count - $reviewFailures.Count
    failed = $reviewFailures.Count
    cases = $reviewResults
} | ConvertTo-Json -Depth 5
if ($reviewFailures.Count) { exit 1 }
