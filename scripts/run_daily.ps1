param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonExe = "",
    [int]$CollectSeconds = 600,
    [int]$FinalPicks = 3
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

foreach ($name in @("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy")) {
    if (Test-Path "Env:$name") {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
}

# mojito token cache path uses "~/.cache/mojito2". Use project-local home for stability.
$env:HOME = $ProjectRoot
$env:USERPROFILE = $ProjectRoot

$null = New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "logs")
$null = New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "out")

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputJson = Join-Path $ProjectRoot ("out\daily_{0}.json" -f $stamp)
$logFile = Join-Path $ProjectRoot ("logs\daily_{0}.log" -f $stamp)

$args = @(
    "main.py",
    "--collect-seconds", "$CollectSeconds",
    "--final-picks", "$FinalPicks",
    "--output-json", $outputJson
)

Write-Output "[run] $PythonExe $($args -join ' ')"
& $PythonExe @args 2>&1 | Tee-Object -FilePath $logFile

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
