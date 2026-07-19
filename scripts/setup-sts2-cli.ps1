[CmdletBinding()]
param(
    [string]$GameDir = "${env:ProgramFiles(x86)}\Steam\steamapps\common\Slay the Spire 2",
    [string]$SourceDir = (Join-Path (Split-Path -Parent $PSScriptRoot) "build\sts2-cli")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = Split-Path -Parent $PSScriptRoot
$repo = "https://github.com/wuhao21/sts2-cli.git"
$commit = "d11aa883b582dd68bd39b331f3370746b30d447e"
$expectedVersion = "v0.107.1"
$game = (Resolve-Path -LiteralPath $GameDir).Path
$dataDir = Join-Path $game "data_sts2_windows_x86_64"
$releasePath = Join-Path $game "release_info.json"
if (-not (Test-Path -LiteralPath $dataDir -PathType Container)) {
    throw "STS2 Windows data directory was not found at '$dataDir'."
}
if (-not (Test-Path -LiteralPath $releasePath -PathType Leaf)) {
    throw "STS2 release metadata was not found at '$releasePath'."
}
$release = Get-Content -LiteralPath $releasePath -Raw | ConvertFrom-Json
if ($release.version -ne $expectedVersion) {
    throw "This compatibility patch targets $expectedVersion; installed game is '$($release.version)'."
}

if (-not (Test-Path -LiteralPath $SourceDir)) {
    git clone $repo $SourceDir
    if ($LASTEXITCODE -ne 0) { throw "Could not clone sts2-cli." }
    git -C $SourceDir checkout --detach $commit
    if ($LASTEXITCODE -ne 0) { throw "Could not check out sts2-cli commit $commit." }
} elseif (-not (Test-Path -LiteralPath (Join-Path $SourceDir ".git"))) {
    throw "SourceDir exists but is not a Git checkout: '$SourceDir'."
}

$head = (git -C $SourceDir rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $head -ne $commit) {
    throw "SourceDir must be at sts2-cli commit $commit; found '$head'."
}

$simulatorSource = Join-Path $SourceDir "src\Sts2Headless\RunSimulator.cs"
$compatApplied = Select-String -LiteralPath $simulatorSource `
    -SimpleMatch "SetUpSavedSingleplayer" -Quiet
if (-not $compatApplied) {
    $patchPath = Join-Path $root "integrations\sts2-cli-v0.107.1.patch"
    git -C $SourceDir apply --check $patchPath
    if ($LASTEXITCODE -ne 0) { throw "sts2-cli compatibility patch does not apply cleanly." }
    git -C $SourceDir apply $patchPath
    if ($LASTEXITCODE -ne 0) { throw "Could not apply sts2-cli compatibility patch." }
}

$libDir = Join-Path $SourceDir "lib"
New-Item -ItemType Directory -Path $libDir -Force | Out-Null
$dlls = @(
    "sts2.dll",
    "SmartFormat.dll",
    "SmartFormat.ZString.dll",
    "Sentry.dll",
    "Steamworks.NET.dll",
    "MonoMod.Backports.dll",
    "MonoMod.ILHelpers.dll",
    "0Harmony.dll",
    "System.IO.Hashing.dll"
)
foreach ($dll in $dlls) {
    $source = Join-Path $dataDir $dll
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Required game dependency is missing: '$source'."
    }
    Copy-Item -LiteralPath $source -Destination (Join-Path $libDir $dll) -Force
}
Copy-Item -LiteralPath (Join-Path $libDir "sts2.dll") `
    -Destination (Join-Path $libDir "sts2.dll.original") -Force

dotnet run --project (Join-Path $root "tools\Sts2CliPatcher\Sts2CliPatcher.csproj") `
    -- (Join-Path $libDir "sts2.dll")
if ($LASTEXITCODE -ne 0) { throw "Could not patch the isolated sts2.dll copy." }

$project = Join-Path $SourceDir "src\Sts2Headless\Sts2Headless.csproj"
dotnet build $project -c Release
if ($LASTEXITCODE -ne 0) { throw "sts2-cli build failed." }

$runtime = Join-Path $SourceDir "src\Sts2Headless\bin\Release\net9.0\Sts2Headless.dll"
[pscustomobject]@{
    status = "ready"
    upstream_commit = $commit
    game_version = $release.version
    runtime = $runtime
    modifies_game_installation = $false
}
