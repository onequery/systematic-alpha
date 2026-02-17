[CmdletBinding()]
param(
    [string]$TaskName = "SystematicAlpha_KR_Open_0900",
    [string]$At = "09:00",
    [string]$RunScriptPath = "",
    [string]$PythonExe = "C:\Users\heesu\anaconda3\envs\systematic-alpha\python.exe",
    [switch]$WeekdaysOnly = $true,
    [int]$StartDelaySeconds = 5,
    [int]$MaxAttempts = 4,
    [int]$RetryDelaySeconds = 30,
    [int]$RetryBackoffMultiplier = 2,
    [int]$MaxRetryDelaySeconds = 180
)

$ErrorActionPreference = "Stop"
$target = Join-Path $PSScriptRoot "register_kr_task.ps1"
if (-not (Test-Path $target)) {
    throw "Target script not found: $target"
}

Write-Output "[alias] register_task.ps1 is kept for backward compatibility. Use register_kr_task.ps1."
& $target `
    -TaskName $TaskName `
    -At $At `
    -RunScriptPath $RunScriptPath `
    -PythonExe $PythonExe `
    -WeekdaysOnly:$($WeekdaysOnly.IsPresent) `
    -StartDelaySeconds $StartDelaySeconds `
    -MaxAttempts $MaxAttempts `
    -RetryDelaySeconds $RetryDelaySeconds `
    -RetryBackoffMultiplier $RetryBackoffMultiplier `
    -MaxRetryDelaySeconds $MaxRetryDelaySeconds
