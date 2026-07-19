[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("milestone", "struggling", "question")]
    [string]$Trigger,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Question,

    [string]$Context = "No additional context was supplied.",

    [string]$ProposedNextAction = "No next action was proposed.",

    [string]$Model = "claude-fable-5"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$rolePath = Join-Path $projectRoot "integrations\fable\ROLE.md"
$profilePath = Join-Path $projectRoot "integrations\fable\OWNER_PROFILE.md"

foreach ($path in @($rolePath, $profilePath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required Fable prompt file is missing: $path"
    }
}

$claudeCommand = Get-Command claude -ErrorAction Stop
$authText = (& $claudeCommand.Source auth status --json 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to verify Claude CLI authentication: $authText"
}

try {
    $auth = $authText | ConvertFrom-Json
}
catch {
    throw "Claude CLI returned an unreadable authentication status."
}

if (-not $auth.loggedIn -or $auth.authMethod -ne "claude.ai") {
    throw "Fable consultations require a Claude.ai subscription login; API-key authentication is refused."
}

$role = Get-Content -LiteralPath $rolePath -Raw
$profile = Get-Content -LiteralPath $profilePath -Raw
$systemPrompt = "$role`n`n$profile"

$sha256 = [System.Security.Cryptography.SHA256]::Create()
try {
    $promptBytes = [System.Text.Encoding]::UTF8.GetBytes($systemPrompt)
    $charterHash = [System.BitConverter]::ToString($sha256.ComputeHash($promptBytes)).Replace("-", "").ToLowerInvariant()
}
finally {
    $sha256.Dispose()
}

$consultationPrompt = @"
CONSULTATION TRIGGER: $Trigger

DECISION FOR THE OWNER'S PROXY:
$Question

CURRENT EVIDENCE AND CONSTRAINTS:
$Context

CODEX'S PROPOSED NEXT ACTION:
$ProposedNextAction

Give the owner's likely verdict on this decision. Use plain language. Do not
invent missing evidence, and do not interpret this consultation as permission
to broaden the task. Return exactly one JSON object with no Markdown fence and
only these five fields:

- verdict: one of approve, revise, stop, or need_user
- role_alignment: one of aligned or uncertain
- reasoning: the plain-language decision and reasons
- assumptions: any facts you had to assume, or "none"
- caveats: concrete risks, limits, and the bounded next action

Keep the complete response concise enough for a practical engineering
checkpoint.
"@

$claudeArguments = @(
    "-p",
    "--model", $Model,
    "--system-prompt", $systemPrompt,
    "--output-format", "json",
    "--permission-mode", "plan",
    "--tools", "",
    "--strict-mcp-config",
    "--disable-slash-commands",
    "--no-session-persistence",
    "--effort", "high"
)

$rawLines = & $claudeCommand.Source @claudeArguments $consultationPrompt 2>&1
$claudeExitCode = $LASTEXITCODE
$rawResult = ($rawLines | Out-String).Trim()

try {
    $envelope = $rawResult | ConvertFrom-Json
}
catch {
    if ($claudeExitCode -ne 0) {
        throw "Fable consultation failed: $rawResult"
    }
    throw "Fable returned an unreadable response: $rawResult"
}

if ($claudeExitCode -ne 0 -or $envelope.is_error) {
    $failure = $envelope.result
    if ([string]::IsNullOrWhiteSpace($failure)) {
        $failure = $rawResult
    }
    throw "Fable consultation failed: $failure"
}

if ([string]::IsNullOrWhiteSpace($envelope.result)) {
    throw "Fable returned an empty response."
}

try {
    $advice = $envelope.result | ConvertFrom-Json
}
catch {
    throw "Fable did not honor the required JSON contract: $($envelope.result)"
}

$requiredFields = @(
    "verdict",
    "role_alignment",
    "reasoning",
    "assumptions",
    "caveats"
)
$returnedFields = @($advice.PSObject.Properties.Name)
$missingFields = @($requiredFields | Where-Object { $_ -notin $returnedFields })
if ($missingFields.Count -gt 0) {
    throw "Fable omitted required advice fields: $($missingFields -join ', ')"
}

if ($advice.verdict -notin @("approve", "revise", "stop", "need_user")) {
    throw "Fable returned an invalid verdict: $($advice.verdict)"
}
if ($advice.role_alignment -notin @("aligned", "uncertain")) {
    throw "Fable returned an invalid role alignment: $($advice.role_alignment)"
}

$actualModels = @()
if ($null -ne $envelope.modelUsage) {
    $actualModels = @($envelope.modelUsage.PSObject.Properties.Name)
}

[pscustomobject]@{
    connection = [pscustomobject]@{
        auth_method = $auth.authMethod
        subscription = $auth.subscriptionType
        requested_model = $Model
        actual_models = $actualModels
        reported_cost_usd = $envelope.total_cost_usd
    }
    charter_sha256 = $charterHash
    trigger = $Trigger
    advice = $advice
} | ConvertTo-Json -Depth 10
