[CmdletBinding()]
param(
    [string]$GameDir = "${env:ProgramFiles(x86)}\Steam\steamapps\common\Slay the Spire 2",
    [string]$SourceDir = (Join-Path (Split-Path -Parent $PSScriptRoot) "build\STS2MCP"),
    [string]$OutputDir = (Join-Path (Split-Path -Parent $PSScriptRoot) "build\STS2MCP-out")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repo = "https://github.com/Gennadiyev/STS2MCP.git"
$commit = "20eadebde358a37cca41f8b38728099e6d0d19db"
$expectedVersion = "v0.107.1"
$root = Split-Path -Parent $PSScriptRoot
$patches = @(
    @{
        Path = (Join-Path $root "integrations\sts2mcp-v0.107.1.patch")
        IsApplied = { Test-Path -LiteralPath (Join-Path $SourceDir "McpMod.Compat.cs") }
    },
    @{
        Path = (Join-Path $root "integrations\sts2mcp-human-telemetry.patch")
        IsApplied = { Test-Path -LiteralPath (Join-Path $SourceDir "McpMod.Telemetry.cs") }
    },
    @{
        Path = (Join-Path $root "integrations\sts2mcp-ally-targets.patch")
        IsApplied = {
            Select-String -LiteralPath (Join-Path $SourceDir "McpMod.Actions.cs") `
                -SimpleMatch "TargetType.AnyEnemy or TargetType.AnyAlly" -Quiet
        }
    },
    @{
        Path = (Join-Path $root "integrations\sts2mcp-replay-export.patch")
        IsApplied = { Test-Path -LiteralPath (Join-Path $SourceDir "McpMod.Replay.cs") }
    },
    @{
        Path = (Join-Path $root "integrations\sts2mcp-headless-replay-loader.patch")
        IsApplied = {
            Select-String -LiteralPath (Join-Path $SourceDir "McpMod.cs") `
                -SimpleMatch "/api/v1/replay/latest/load" -Quiet
        }
    },
    @{
        Path = (Join-Path $root "integrations\sts2mcp-headless-replay-runner.patch")
        IsApplied = {
            Select-String -LiteralPath (Join-Path $SourceDir "McpMod.cs") `
                -SimpleMatch "/api/v1/replay/latest/run" -Quiet
        }
    }
)
$releaseInfoPath = Join-Path $GameDir "release_info.json"

if (-not (Test-Path -LiteralPath $releaseInfoPath -PathType Leaf)) {
    throw "Slay the Spire 2 was not found at '$GameDir'. Pass -GameDir explicitly."
}

$release = Get-Content -LiteralPath $releaseInfoPath -Raw | ConvertFrom-Json
if ($release.version -ne $expectedVersion) {
    throw "This patch targets $expectedVersion, but the installed game is '$($release.version)'."
}

if (-not (Test-Path -LiteralPath $SourceDir)) {
    git clone $repo $SourceDir
    if ($LASTEXITCODE -ne 0) { throw "Could not clone STS2MCP." }
    git -C $SourceDir checkout --detach $commit
    if ($LASTEXITCODE -ne 0) { throw "Could not check out STS2MCP commit $commit." }
} elseif (-not (Test-Path -LiteralPath (Join-Path $SourceDir ".git"))) {
    throw "SourceDir exists but is not a Git checkout: '$SourceDir'."
}

$head = (git -C $SourceDir rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $head -ne $commit) {
    throw "SourceDir must be at STS2MCP commit $commit; found '$head'."
}

$savedErrorPreference = $ErrorActionPreference
foreach ($patch in $patches) {
    if (-not (& $patch.IsApplied)) {
        $patchPath = $patch.Path
        $ErrorActionPreference = "Continue"
        git -C $SourceDir apply --check $patchPath 2>$null
        $canApply = $LASTEXITCODE -eq 0
        $ErrorActionPreference = $savedErrorPreference

        if ($canApply) {
            git -C $SourceDir apply $patchPath
            if ($LASTEXITCODE -ne 0) { throw "Could not apply '$patchPath'." }
        } else {
            $ErrorActionPreference = "Continue"
            git -C $SourceDir apply --reverse --check $patchPath 2>$null
            $isAlreadyApplied = $LASTEXITCODE -eq 0
            $ErrorActionPreference = $savedErrorPreference
            if (-not $isAlreadyApplied) {
                throw "Patch neither applies cleanly nor appears applied already: '$patchPath'."
            }
        }
    }
}

dotnet build (Join-Path $SourceDir "STS2_MCP.csproj") `
    -c Release `
    -o $OutputDir `
    -p:STS2GameDir=$GameDir
if ($LASTEXITCODE -ne 0) { throw "STS2MCP build failed." }

Copy-Item -LiteralPath (Join-Path $SourceDir "mod_manifest.json") `
    -Destination (Join-Path $OutputDir "STS2_MCP.json") `
    -Force
Copy-Item -LiteralPath (Join-Path $root "integrations\STS2_MCP.telemetry.example.conf") `
    -Destination (Join-Path $OutputDir "STS2_MCP.telemetry.example.conf") `
    -Force

Write-Host "Built the patched bridge at: $OutputDir"
Write-Host "No files in the game installation were changed."
