[CmdletBinding()]
param(
    [string]$GameDir = "${env:ProgramFiles(x86)}\Steam\steamapps\common\Slay the Spire 2",
    [string]$HealthUrl = "http://127.0.0.1:15526/",
    [ValidateRange(5, 300)]
    [int]$StartupTimeoutSeconds = 60
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = Split-Path -Parent $PSScriptRoot
$runtimeDir = Join-Path $root "build\headless"
$pidPath = Join-Path $runtimeDir "sts2.pid"
$game = (Resolve-Path -LiteralPath $GameDir).Path
$executable = Join-Path $game "SlayTheSpire2.exe"
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Slay the Spire 2 executable was not found at '$executable'."
}

function Test-BridgeReady {
    try {
        $health = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 1
        if ($health.status -ne "ok") { return $false }
        $stateUrl = $HealthUrl.TrimEnd('/') + "/api/v1/singleplayer?format=json"
        $state = Invoke-RestMethod -Uri $stateUrl -TimeoutSec 2
        return $null -ne $state.state_type
    } catch {
        return $false
    }
}

if (Test-Path -LiteralPath $pidPath -PathType Leaf) {
    $existingPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    $existing = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
    if ($existing -and (Test-BridgeReady)) {
        [pscustomobject]@{
            status = "already_running"
            pid = $existing.Id
            health_url = $HealthUrl
            headless = $existing.MainWindowHandle -eq 0
        }
        exit 0
    }
    Remove-Item -LiteralPath $pidPath -Force
}

$other = Get-Process -Name "SlayTheSpire2" -ErrorAction SilentlyContinue
if ($other) {
    throw "Another Slay the Spire 2 process is running (PID $($other[0].Id)). Stop it before starting the managed headless worker."
}

New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
$previousAppId = $env:SteamAppId
$previousGameId = $env:SteamGameId
try {
    # Direct Godot launches need the transient app ID for Steamworks. These
    # variables are inherited by the child only and are then restored here.
    $env:SteamAppId = "2868840"
    $env:SteamGameId = "2868840"
    $process = Start-Process `
        -FilePath $executable `
        -ArgumentList "--headless" `
        -WorkingDirectory $game `
        -WindowStyle Hidden `
        -PassThru
} finally {
    $env:SteamAppId = $previousAppId
    $env:SteamGameId = $previousGameId
}

Set-Content -LiteralPath $pidPath -Value $process.Id -Encoding ascii
$deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
do {
    $process.Refresh()
    if ($process.HasExited) {
        Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
        throw "The headless STS2 process exited during startup with code $($process.ExitCode)."
    }
    if (Test-BridgeReady) {
        # The state endpoint becomes available just before profile-scoped replay
        # paths on some starts. Give SaveManager a short quiescence window.
        Start-Sleep -Milliseconds 750
        [pscustomobject]@{
            status = "ready"
            pid = $process.Id
            health_url = $HealthUrl
            headless = $process.MainWindowHandle -eq 0
        }
        exit 0
    }
    Start-Sleep -Milliseconds 750
} while ((Get-Date) -lt $deadline)

Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
throw "Headless STS2 did not expose the bridge within $StartupTimeoutSeconds seconds."
