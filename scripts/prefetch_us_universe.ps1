param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonExe = "",
    [string]$UsExchange = "NASD",
    [int]$UsUniverseSize = 500,
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

$logDateDir = Join-Path (Join-Path $ProjectRoot "logs") $runDate
$logDir = Join-Path $logDateDir "us"
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

$logFile = Join-Path $logDir ("prefetch_us_{0}.log" -f $stamp)
$universeArgs = @(
    "scripts\prefetch_us_universe.py",
    "--project-root", $ProjectRoot
)
$cacheArgs = @(
    "scripts\prefetch_us_market_cache.py",
    "--project-root", $ProjectRoot,
    "--us-exchange", $UsExchange,
    "--us-universe-size", "$UsUniverseSize",
    "--max-symbols-scan", "$MaxSymbolsScan"
)
if ($ForceRefresh) {
    $cacheArgs += "--force-refresh"
}

Write-Output "[prefetch] command(universe): $PythonExe $($universeArgs -join ' ')"
Write-Output "[prefetch] command(market-cache): $PythonExe $($cacheArgs -join ' ')"
Write-Output "[prefetch] log: $logFile"

try {
    $prevErrorActionPreference = $ErrorActionPreference
    try {
        # Treat native stderr as stream output (for logging), not as terminating PowerShell errors.
        $ErrorActionPreference = "Continue"
        & $PythonExe @universeArgs 2>&1 | Tee-Object -FilePath $logFile
        $firstExit = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
        if ($firstExit -ne 0) {
            $exitCode = $firstExit
            throw "prefetch_us_universe.py failed (exit=$firstExit)"
        }
        & $PythonExe @cacheArgs 2>&1 | Tee-Object -FilePath $logFile -Append
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

Write-Output "[prefetch] finished (exit=$exitCode)"
exit $exitCode
