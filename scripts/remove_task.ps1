[CmdletBinding()]
param(
    [string]$TaskName = "SystematicAlpha_KR_Open_0900"
)

$ErrorActionPreference = "Stop"
$target = Join-Path $PSScriptRoot "remove_kr_task.ps1"
if (-not (Test-Path $target)) {
    throw "Target script not found: $target"
}

Write-Output "[alias] remove_task.ps1 is kept for backward compatibility. Use remove_kr_task.ps1."
& $target -TaskName $TaskName
