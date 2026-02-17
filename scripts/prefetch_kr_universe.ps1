param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonExe = "",
    [int]$KrUniverseSize = 500,
    [int]$MaxSymbolsScan = 500,
    [switch]$ForceRefresh = $false
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ProjectRoot)) {
    throw "ProjectRoot not found: $ProjectRoot"
}

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $defaultPython = "C:\Users\heesu\anaconda3\envs\systematic-alpha\python.exe"
    if (Test-Path $defaultPython) {
        $PythonExe = $defaultPython
    } else {
        $PythonExe = "python"
    }
}

Set-Location $ProjectRoot

$kstTz = [System.TimeZoneInfo]::FindSystemTimeZoneById("Korea Standard Time")
$nowKst = [System.TimeZoneInfo]::ConvertTime((Get-Date), $kstTz)
$runDate = $nowKst.ToString("yyyyMMdd")
$stamp = $nowKst.ToString("yyyyMMdd_HHmmss")

$logDir = Join-Path (Join-Path $ProjectRoot "logs") $runDate
$null = New-Item -ItemType Directory -Force -Path $logDir
$null = New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "out")

foreach ($name in @("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy")) {
    if (Test-Path "Env:$name") {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
}

# mojito token cache path uses "~/.cache/mojito2". Use project-local home for stability.
$env:HOME = $ProjectRoot
$env:USERPROFILE = $ProjectRoot

$logFile = Join-Path $logDir ("prefetch_kr_{0}.log" -f $stamp)
$argsList = @(
    "scripts\prefetch_kr_universe.py",
    "--project-root", $ProjectRoot,
    "--kr-universe-size", "$KrUniverseSize",
    "--max-symbols-scan", "$MaxSymbolsScan"
)
if ($ForceRefresh) {
    $argsList += "--force-refresh"
}

Write-Output "[prefetch-kr] command: $PythonExe $($argsList -join ' ')"
Write-Output "[prefetch-kr] log: $logFile"

try {
    $prevErrorActionPreference = $ErrorActionPreference
    try {
        # Treat native stderr as stream output (for logging), not as terminating PowerShell errors.
        $ErrorActionPreference = "Continue"
        & $PythonExe @argsList 2>&1 | Tee-Object -FilePath $logFile
    } finally {
        $ErrorActionPreference = $prevErrorActionPreference
    }
    $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
} catch {
    $_.Exception.Message | Tee-Object -FilePath $logFile -Append
    if ($null -eq $LASTEXITCODE) {
        $exitCode = 1
    } else {
        $exitCode = [int]$LASTEXITCODE
    }
}

Write-Output "[prefetch-kr] finished (exit=$exitCode)"
exit $exitCode
