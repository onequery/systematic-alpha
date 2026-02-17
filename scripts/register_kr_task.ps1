[CmdletBinding(SupportsShouldProcess = $true)]
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

if ([string]::IsNullOrWhiteSpace($RunScriptPath)) {
    $RunScriptPath = Join-Path $PSScriptRoot "run_daily.ps1"
}
if ([string]::IsNullOrWhiteSpace($PrefetchScriptPath)) {
    $PrefetchScriptPath = Join-Path $PSScriptRoot "prefetch_kr_universe.ps1"
}

if (-not (Test-Path $RunScriptPath)) {
    throw "Run script not found: $RunScriptPath"
}
if ($RegisterPrefetch -and -not (Test-Path $PrefetchScriptPath)) {
    throw "Prefetch script not found: $PrefetchScriptPath"
}

$parts = $At.Split(":")
if ($parts.Count -ne 2) {
    throw "Invalid time format. Use HH:mm, e.g. 09:00"
}

$hour = [int]$parts[0]
$minute = [int]$parts[1]
$atTime = (Get-Date).Date.AddHours($hour).AddMinutes($minute)
$prefetchAtTime = $null
if ($RegisterPrefetch) {
    $prefetchParts = $PrefetchAt.Split(":")
    if ($prefetchParts.Count -ne 2) {
        throw "Invalid prefetch time format. Use HH:mm, e.g. 08:30"
    }
    $prefetchHour = [int]$prefetchParts[0]
    $prefetchMinute = [int]$prefetchParts[1]
    $prefetchAtTime = (Get-Date).Date.AddHours($prefetchHour).AddMinutes($prefetchMinute)
}

$psExe = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$actionArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$RunScriptPath`" -PythonExe `"$PythonExe`" -Market KR -KrUniverseSize $KrUniverseSize -MaxSymbolsScan $MaxSymbolsScan -StartDelaySeconds $StartDelaySeconds -MaxAttempts $MaxAttempts -RetryDelaySeconds $RetryDelaySeconds -RetryBackoffMultiplier $RetryBackoffMultiplier -MaxRetryDelaySeconds $MaxRetryDelaySeconds"
$prefetchActionArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$PrefetchScriptPath`" -PythonExe `"$PythonExe`" -KrUniverseSize $KrUniverseSize -MaxSymbolsScan $MaxSymbolsScan"

$action = New-ScheduledTaskAction -Execute $psExe -Argument $actionArgs
$prefetchAction = New-ScheduledTaskAction -Execute $psExe -Argument $prefetchActionArgs

if ($WeekdaysOnly) {
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $atTime
    if ($RegisterPrefetch) {
        $prefetchTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $prefetchAtTime
    }
} else {
    $trigger = New-ScheduledTaskTrigger -Daily -At $atTime
    if ($RegisterPrefetch) {
        $prefetchTrigger = New-ScheduledTaskTrigger -Daily -At $prefetchAtTime
    }
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

$task = New-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -Principal $principal
if ($PSCmdlet.ShouldProcess($TaskName, "Register KR scheduled task")) {
    Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
}

if ($RegisterPrefetch) {
    $prefetchTask = New-ScheduledTask -Action $prefetchAction -Trigger $prefetchTrigger -Settings $settings -Principal $principal
    if ($PSCmdlet.ShouldProcess($PrefetchTaskName, "Register KR prefetch scheduled task")) {
        Register-ScheduledTask -TaskName $PrefetchTaskName -InputObject $prefetchTask -Force | Out-Null
    }
}

Write-Output "Task registered: $TaskName"
Write-Output "Market: KR"
Write-Output "Run time(KST): $At"
if ($WeekdaysOnly) {
    Write-Output "Schedule: Weekdays (Mon-Fri)"
} else {
    Write-Output "Schedule: Daily"
}
Write-Output "Retry config: start_delay=${StartDelaySeconds}s, max_attempts=$MaxAttempts, retry_delay=${RetryDelaySeconds}s, backoff=x$RetryBackoffMultiplier, max_retry_delay=${MaxRetryDelaySeconds}s"
Write-Output "KR universe prep: kr_universe_size=$KrUniverseSize, max_symbols_scan=$MaxSymbolsScan"
Write-Output "Command: $psExe $actionArgs"
if ($RegisterPrefetch) {
    Write-Output "Prefetch task registered: $PrefetchTaskName"
    Write-Output "Prefetch run time(KST): $PrefetchAt"
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
