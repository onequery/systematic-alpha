[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$TaskName = "SystematicAlpha_US_Open_0930ET",
    [string]$AtDst = "22:30",
    [string]$AtStd = "23:30",
    [string]$RunScriptPath = "",
    [switch]$RegisterPrefetch = $true,
    [string]$PrefetchTaskName = "SystematicAlpha_US_Prefetch_SP500_0925ET",
    [string]$PrefetchAtDst = "22:25",
    [string]$PrefetchAtStd = "23:25",
    [string]$PrefetchScriptPath = "",
    [string]$PythonExe = "C:\Users\heesu\anaconda3\envs\systematic-alpha\python.exe",
    [string]$UsExchange = "NASD",
    [int]$UsOpenWindowMinutes = 20,
    [switch]$WeekdaysOnly = $true,
    [int]$StartDelaySeconds = 5,
    [int]$MaxAttempts = 4,
    [int]$RetryDelaySeconds = 30,
    [int]$RetryBackoffMultiplier = 2,
    [int]$MaxRetryDelaySeconds = 180
)
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RunScriptPath)) {
    $RunScriptPath = Join-Path $PSScriptRoot "run_daily.ps1"
}
if ([string]::IsNullOrWhiteSpace($PrefetchScriptPath)) {
    $PrefetchScriptPath = Join-Path $PSScriptRoot "prefetch_us_universe.ps1"
}

if (-not (Test-Path $RunScriptPath)) {
    throw "Run script not found: $RunScriptPath"
}
if ($RegisterPrefetch -and -not (Test-Path $PrefetchScriptPath)) {
    throw "Prefetch script not found: $PrefetchScriptPath"
}

$scriptDir = Split-Path -Parent $RunScriptPath
$projectRoot = Split-Path -Parent $scriptDir

function Parse-TimeToDate([string]$At) {
    $parts = $At.Split(":")
    if ($parts.Count -ne 2) {
        throw "Invalid time format '$At'. Use HH:mm."
    }
    $hour = [int]$parts[0]
    $minute = [int]$parts[1]
    return (Get-Date).Date.AddHours($hour).AddMinutes($minute)
}

$atDstTime = Parse-TimeToDate -At $AtDst
$atStdTime = Parse-TimeToDate -At $AtStd
$prefetchAtDstTime = Parse-TimeToDate -At $PrefetchAtDst
$prefetchAtStdTime = Parse-TimeToDate -At $PrefetchAtStd

$psExe = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$actionArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$RunScriptPath`" -PythonExe `"$PythonExe`" -Market US -UsExchange `"$UsExchange`" -RequireUsOpen -UsOpenWindowMinutes $UsOpenWindowMinutes -StartDelaySeconds $StartDelaySeconds -MaxAttempts $MaxAttempts -RetryDelaySeconds $RetryDelaySeconds -RetryBackoffMultiplier $RetryBackoffMultiplier -MaxRetryDelaySeconds $MaxRetryDelaySeconds"
$action = New-ScheduledTaskAction -Execute $psExe -Argument $actionArgs
$prefetchActionArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$PrefetchScriptPath`" -ProjectRoot `"$projectRoot`" -PythonExe `"$PythonExe`""
$prefetchAction = New-ScheduledTaskAction -Execute $psExe -Argument $prefetchActionArgs

if ($WeekdaysOnly) {
    $triggerDst = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $atDstTime
    $triggerStd = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $atStdTime
    $prefetchTriggerDst = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $prefetchAtDstTime
    $prefetchTriggerStd = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $prefetchAtStdTime
} else {
    $triggerDst = New-ScheduledTaskTrigger -Daily -At $atDstTime
    $triggerStd = New-ScheduledTaskTrigger -Daily -At $atStdTime
    $prefetchTriggerDst = New-ScheduledTaskTrigger -Daily -At $prefetchAtDstTime
    $prefetchTriggerStd = New-ScheduledTaskTrigger -Daily -At $prefetchAtStdTime
}

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 10)

$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

$task = New-ScheduledTask -Action $action -Trigger @($triggerDst, $triggerStd) -Settings $settings -Principal $principal
if ($PSCmdlet.ShouldProcess($TaskName, "Register US scheduled task")) {
    Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
}

if ($RegisterPrefetch) {
    $prefetchTask = New-ScheduledTask -Action $prefetchAction -Trigger @($prefetchTriggerDst, $prefetchTriggerStd) -Settings $settings -Principal $principal
    if ($PSCmdlet.ShouldProcess($PrefetchTaskName, "Register US prefetch scheduled task")) {
        Register-ScheduledTask -TaskName $PrefetchTaskName -InputObject $prefetchTask -Force | Out-Null
    }
}

Write-Output "Task registered: $TaskName"
Write-Output "Market: US"
Write-Output "US exchange: $UsExchange"
Write-Output "Triggers(KST): $AtDst and $AtStd (DST/STD dual-trigger)"
if ($WeekdaysOnly) {
    Write-Output "Schedule: Weekdays (Mon-Fri)"
} else {
    Write-Output "Schedule: Daily"
}
Write-Output "Execution guard: run_daily.ps1 -RequireUsOpen -UsOpenWindowMinutes $UsOpenWindowMinutes (open-window + daily-lock)"
Write-Output "Retry config: start_delay=${StartDelaySeconds}s, max_attempts=$MaxAttempts, retry_delay=${RetryDelaySeconds}s, backoff=x$RetryBackoffMultiplier, max_retry_delay=${MaxRetryDelaySeconds}s"
Write-Output "Command: $psExe $actionArgs"
if ($RegisterPrefetch) {
    Write-Output "Prefetch task registered: $PrefetchTaskName"
    Write-Output "Prefetch triggers(KST): $PrefetchAtDst and $PrefetchAtStd (DST/STD dual-trigger)"
    Write-Output "Prefetch command: $psExe $prefetchActionArgs"
}
Write-Output ""
Write-Output "Check task:"
Write-Output "  Get-ScheduledTask -TaskName '$TaskName'"
if ($RegisterPrefetch) {
    Write-Output "  Get-ScheduledTask -TaskName '$PrefetchTaskName'"
}
Write-Output "Run immediately:"
Write-Output "  Start-ScheduledTask -TaskName '$TaskName'"
if ($RegisterPrefetch) {
    Write-Output "  Start-ScheduledTask -TaskName '$PrefetchTaskName'"
}

