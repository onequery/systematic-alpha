[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$TaskName = "SystematicAlpha_US_Open_0930ET",
    [string]$PrefetchTaskName = "SystematicAlpha_US_Prefetch_Setup_0830ET",
    [switch]$RemovePrefetch = $true
)

$ErrorActionPreference = "Stop"

foreach ($name in @($TaskName)) {
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
        if ($PSCmdlet.ShouldProcess($name, "Remove US scheduled task")) {
            Unregister-ScheduledTask -TaskName $name -Confirm:$false
            Write-Output "Task removed: $name"
        }
    } else {
        Write-Output "Task not found: $name"
    }
}

if ($RemovePrefetch) {
    if (Get-ScheduledTask -TaskName $PrefetchTaskName -ErrorAction SilentlyContinue) {
        if ($PSCmdlet.ShouldProcess($PrefetchTaskName, "Remove US prefetch scheduled task")) {
            Unregister-ScheduledTask -TaskName $PrefetchTaskName -Confirm:$false
            Write-Output "Task removed: $PrefetchTaskName"
        }
    } else {
        Write-Output "Task not found: $PrefetchTaskName"
    }
}
