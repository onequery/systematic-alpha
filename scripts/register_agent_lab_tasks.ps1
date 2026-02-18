[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$RunScriptPath = "",
    [string]$PythonExe = "C:\Users\heesu\anaconda3\envs\systematic-alpha\python.exe",
    [string]$TaskPrefix = "SystematicAlpha_AgentLab",
    [string]$KrFollowAt = "09:20",
    [string]$UsFollowAtDst = "22:45",
    [string]$UsFollowAtStd = "23:45",
    [string]$DailyReviewAt = "07:10",
    [string]$WeeklyCouncilAt = "08:00",
    [switch]$RegisterTelegramChat = $true,
    [int]$ChatPollTimeoutSeconds = 25,
    [double]$ChatIdleSleepSeconds = 1.0,
    [int]$ChatMemoryLimit = 20,
    [int]$WaitTimeoutSeconds = 5400,
    [int]$PollIntervalSeconds = 15
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RunScriptPath)) {
    $RunScriptPath = Join-Path $PSScriptRoot "run_agent_lab.ps1"
}
if (-not (Test-Path $RunScriptPath)) {
    throw "Run script not found: $RunScriptPath"
}

function Parse-TimeToDate([string]$At) {
    $parts = $At.Split(":")
    if ($parts.Count -ne 2) {
        throw "Invalid time format '$At'. Use HH:mm."
    }
    $hour = [int]$parts[0]
    $minute = [int]$parts[1]
    return (Get-Date).Date.AddHours($hour).AddMinutes($minute)
}

$psExe = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"

$krTaskName = "${TaskPrefix}_KR_PostOpen_0920"
$usTaskName = "${TaskPrefix}_US_PostOpen_0930ET"
$dailyTaskName = "${TaskPrefix}_DailyReview_0710"
$weeklyTaskName = "${TaskPrefix}_WeeklyCouncil_Sat0800"
$chatTaskName = "${TaskPrefix}_TelegramChat_Logon"

$krActionArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$RunScriptPath`" -PythonExe `"$PythonExe`" -Action ingest-propose -Market KR -WaitForSessionResult -WaitTimeoutSeconds $WaitTimeoutSeconds -PollIntervalSeconds $PollIntervalSeconds -UseDailyLock"
$usActionArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$RunScriptPath`" -PythonExe `"$PythonExe`" -Action ingest-propose -Market US -WaitForSessionResult -WaitTimeoutSeconds $WaitTimeoutSeconds -PollIntervalSeconds $PollIntervalSeconds -UseDailyLock"
$dailyActionArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$RunScriptPath`" -PythonExe `"$PythonExe`" -Action daily-review"
$weeklyActionArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$RunScriptPath`" -PythonExe `"$PythonExe`" -Action weekly-council"
$chatActionArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$RunScriptPath`" -PythonExe `"$PythonExe`" -Action telegram-chat -ChatPollTimeoutSeconds $ChatPollTimeoutSeconds -ChatIdleSleepSeconds $ChatIdleSleepSeconds -ChatMemoryLimit $ChatMemoryLimit"

$krAction = New-ScheduledTaskAction -Execute $psExe -Argument $krActionArgs
$usAction = New-ScheduledTaskAction -Execute $psExe -Argument $usActionArgs
$dailyAction = New-ScheduledTaskAction -Execute $psExe -Argument $dailyActionArgs
$weeklyAction = New-ScheduledTaskAction -Execute $psExe -Argument $weeklyActionArgs
$chatAction = New-ScheduledTaskAction -Execute $psExe -Argument $chatActionArgs

$krTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At (Parse-TimeToDate $KrFollowAt)
$usTriggerDst = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At (Parse-TimeToDate $UsFollowAtDst)
$usTriggerStd = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At (Parse-TimeToDate $UsFollowAtStd)
$dailyTrigger = New-ScheduledTaskTrigger -Daily -At (Parse-TimeToDate $DailyReviewAt)
$weeklyTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At (Parse-TimeToDate $WeeklyCouncilAt)
$chatTrigger = New-ScheduledTaskTrigger -AtLogOn

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Minutes 10)

$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

$krTask = New-ScheduledTask -Action $krAction -Trigger $krTrigger -Settings $settings -Principal $principal
$usTask = New-ScheduledTask -Action $usAction -Trigger @($usTriggerDst, $usTriggerStd) -Settings $settings -Principal $principal
$dailyTask = New-ScheduledTask -Action $dailyAction -Trigger $dailyTrigger -Settings $settings -Principal $principal
$weeklyTask = New-ScheduledTask -Action $weeklyAction -Trigger $weeklyTrigger -Settings $settings -Principal $principal
$chatTask = New-ScheduledTask -Action $chatAction -Trigger $chatTrigger -Settings $settings -Principal $principal

if ($PSCmdlet.ShouldProcess($krTaskName, "Register Agent Lab KR post-open task")) {
    Register-ScheduledTask -TaskName $krTaskName -InputObject $krTask -Force | Out-Null
}
if ($PSCmdlet.ShouldProcess($usTaskName, "Register Agent Lab US post-open task")) {
    Register-ScheduledTask -TaskName $usTaskName -InputObject $usTask -Force | Out-Null
}
if ($PSCmdlet.ShouldProcess($dailyTaskName, "Register Agent Lab daily review task")) {
    Register-ScheduledTask -TaskName $dailyTaskName -InputObject $dailyTask -Force | Out-Null
}
if ($PSCmdlet.ShouldProcess($weeklyTaskName, "Register Agent Lab weekly council task")) {
    Register-ScheduledTask -TaskName $weeklyTaskName -InputObject $weeklyTask -Force | Out-Null
}
if ($RegisterTelegramChat) {
    if ($PSCmdlet.ShouldProcess($chatTaskName, "Register Agent Lab telegram chat task")) {
        Register-ScheduledTask -TaskName $chatTaskName -InputObject $chatTask -Force | Out-Null
    }
}

Write-Output "Agent Lab tasks registered."
Write-Output "KR post-open task: $krTaskName at $KrFollowAt (KST)"
Write-Output "US post-open task: $usTaskName at $UsFollowAtDst/$UsFollowAtStd (KST dual-trigger, daily lock enabled)"
Write-Output "Daily review task: $dailyTaskName at $DailyReviewAt (KST)"
Write-Output "Weekly council task: $weeklyTaskName at $WeeklyCouncilAt (KST, Saturday)"
if ($RegisterTelegramChat) {
    Write-Output "Telegram chat task: $chatTaskName (At logon, long-running worker)"
}
Write-Output ""
Write-Output "Check:"
Write-Output "  Get-ScheduledTask -TaskName '$TaskPrefix*' | Format-Table TaskName, State -AutoSize"
Write-Output "Run now:"
Write-Output "  Start-ScheduledTask -TaskName '$krTaskName'"
Write-Output "  Start-ScheduledTask -TaskName '$usTaskName'"
Write-Output "  Start-ScheduledTask -TaskName '$dailyTaskName'"
Write-Output "  Start-ScheduledTask -TaskName '$weeklyTaskName'"
if ($RegisterTelegramChat) {
    Write-Output "  Start-ScheduledTask -TaskName '$chatTaskName'"
}
