[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$TaskName = "SystematicAlpha_KR_Open_0900",
    [switch]$RemovePrefetch = $true,
    [string]$PrefetchTaskName = "SystematicAlpha_KR_Prefetch_Universe_0730"
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

if ($RemovePrefetch) {
    if (Get-ScheduledTask -TaskName $PrefetchTaskName -ErrorAction SilentlyContinue) {
        if ($PSCmdlet.ShouldProcess($PrefetchTaskName, "Remove KR prefetch scheduled task")) {
            Unregister-ScheduledTask -TaskName $PrefetchTaskName -Confirm:$false
            Write-Output "Task removed: $PrefetchTaskName"
        }
    } else {
        Write-Output "Task not found: $PrefetchTaskName"
    }
}
