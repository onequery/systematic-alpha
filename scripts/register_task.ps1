[CmdletBinding()]
param(
    [string]$TaskName = "SystematicAlpha_KR_Open_0900",
    [string]$At = "09:00",
    [string]$RunScriptPath = "",
    [switch]$RegisterPrefetch = $true,
    [string]$PrefetchTaskName = "SystematicAlpha_KR_Prefetch_Universe_0730",
    [string]$PrefetchAt = "07:30",
    [string]$PrefetchScriptPath = "",
    [string]$PythonExe = "C:\Users\heesu\anaconda3\envs\systematic-alpha\python.exe",
    [int]$KrUniverseSize = 500,
    [int]$MaxSymbolsScan = 500,
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
    -RegisterPrefetch:$RegisterPrefetch `
    -PrefetchTaskName $PrefetchTaskName `
    -PrefetchAt $PrefetchAt `
    -PrefetchScriptPath $PrefetchScriptPath `
    -PythonExe $PythonExe `
    -KrUniverseSize $KrUniverseSize `
    -MaxSymbolsScan $MaxSymbolsScan `
    -WeekdaysOnly:$WeekdaysOnly `
    -StartDelaySeconds $StartDelaySeconds `
    -MaxAttempts $MaxAttempts `
    -RetryDelaySeconds $RetryDelaySeconds `
    -RetryBackoffMultiplier $RetryBackoffMultiplier `
    -MaxRetryDelaySeconds $MaxRetryDelaySeconds
