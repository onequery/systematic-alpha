param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonExe = ""
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

$null = New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "logs")
$null = New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "out")

foreach ($name in @("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy")) {
    if (Test-Path "Env:$name") {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $ProjectRoot ("logs\prefetch_us_{0}.log" -f $stamp)
$argsList = @(
    "scripts\prefetch_us_universe.py",
    "--project-root", $ProjectRoot
)

Write-Output "[prefetch] command: $PythonExe $($argsList -join ' ')"
Write-Output "[prefetch] log: $logFile"

try {
    & $PythonExe @argsList 2>&1 | Tee-Object -FilePath $logFile
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
