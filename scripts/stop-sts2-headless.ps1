[CmdletBinding()]
param(
    [string]$GameDir = "${env:ProgramFiles(x86)}\Steam\steamapps\common\Slay the Spire 2",
    [ValidateRange(1, 60)]
    [int]$TimeoutSeconds = 15
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = Split-Path -Parent $PSScriptRoot
$pidPath = Join-Path $root "build\headless\sts2.pid"
if (-not (Test-Path -LiteralPath $pidPath -PathType Leaf)) {
    [pscustomobject]@{ status = "not_running" }
    exit 0
}

$pidValue = [int](Get-Content -LiteralPath $pidPath -Raw)
$process = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
if (-not $process) {
    Remove-Item -LiteralPath $pidPath -Force
    [pscustomobject]@{ status = "not_running"; stale_pid = $pidValue }
    exit 0
}

$game = (Resolve-Path -LiteralPath $GameDir).Path
$expectedExecutable = [System.IO.Path]::GetFullPath((Join-Path $game "SlayTheSpire2.exe"))
$actualExecutable = [System.IO.Path]::GetFullPath($process.Path)
if (-not $actualExecutable.Equals($expectedExecutable, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "PID $pidValue does not belong to the configured Slay the Spire 2 executable."
}
if ($process.MainWindowHandle -ne 0) {
    throw "PID $pidValue has a visible window; refusing to stop it as a managed headless worker."
}

Stop-Process -Id $pidValue
try {
    Wait-Process -Id $pidValue -Timeout $TimeoutSeconds -ErrorAction Stop
} catch {
    throw "Headless STS2 PID $pidValue did not stop within $TimeoutSeconds seconds."
}
$releaseDeadline = (Get-Date).AddSeconds(2)
while ((Get-Process -Id $pidValue -ErrorAction SilentlyContinue) `
        -and (Get-Date) -lt $releaseDeadline) {
    Start-Sleep -Milliseconds 100
}
Remove-Item -LiteralPath $pidPath -Force
[pscustomobject]@{ status = "stopped"; pid = $pidValue }
