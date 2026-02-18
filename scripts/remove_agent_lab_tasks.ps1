[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$TaskPrefix = "SystematicAlpha_AgentLab"
)

$ErrorActionPreference = "Stop"

$taskNames = @(
    "${TaskPrefix}_KR_PostOpen_0920",
    "${TaskPrefix}_US_PostOpen_0930ET",
    "${TaskPrefix}_DailyReview_0710",
    "${TaskPrefix}_WeeklyCouncil_Sat0800",
    "${TaskPrefix}_TelegramChat_Logon"
)

foreach ($name in $taskNames) {
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
        if ($PSCmdlet.ShouldProcess($name, "Remove Agent Lab scheduled task")) {
            Unregister-ScheduledTask -TaskName $name -Confirm:$false
            Write-Output "Task removed: $name"
        }
    } else {
        Write-Output "Task not found: $name"
    }
}
