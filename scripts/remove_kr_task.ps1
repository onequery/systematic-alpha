[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$TaskName = "SystematicAlpha_KR_Open_0900"
)

$ErrorActionPreference = "Stop"

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    if ($PSCmdlet.ShouldProcess($TaskName, "Remove KR scheduled task")) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Output "Task removed: $TaskName"
    }
} else {
    Write-Output "Task not found: $TaskName"
}
