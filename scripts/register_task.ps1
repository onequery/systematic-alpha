[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$TaskName = "SystematicAlpha_0900",
    [string]$At = "09:00",
    [string]$RunScriptPath = "",
    [string]$PythonExe = "C:\Users\heesu\anaconda3\envs\systematic-alpha\python.exe",
    [switch]$WeekdaysOnly = $true
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RunScriptPath)) {
    $RunScriptPath = Join-Path $PSScriptRoot "run_daily.ps1"
}

if (-not (Test-Path $RunScriptPath)) {
    throw "Run script not found: $RunScriptPath"
}

$parts = $At.Split(":")
if ($parts.Count -ne 2) {
    throw "Invalid time format. Use HH:mm, e.g. 09:00"
}

$hour = [int]$parts[0]
$minute = [int]$parts[1]
$atTime = (Get-Date).Date.AddHours($hour).AddMinutes($minute)

$psExe = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$actionArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$RunScriptPath`" -PythonExe `"$PythonExe`""

$action = New-ScheduledTaskAction -Execute $psExe -Argument $actionArgs

if ($WeekdaysOnly) {
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $atTime
} else {
    $trigger = New-ScheduledTaskTrigger -Daily -At $atTime
}

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

$task = New-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -Principal $principal
if ($PSCmdlet.ShouldProcess($TaskName, "Register scheduled task")) {
    Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
}

Write-Output "Task registered: $TaskName"
Write-Output "Run time: $At"
if ($WeekdaysOnly) {
    Write-Output "Schedule: Weekdays (Mon-Fri)"
} else {
    Write-Output "Schedule: Daily"
}
Write-Output "Command: $psExe $actionArgs"
Write-Output ""
Write-Output "Check task:"
Write-Output "  Get-ScheduledTask -TaskName '$TaskName'"
Write-Output "Run immediately:"
Write-Output "  Start-ScheduledTask -TaskName '$TaskName'"
