[CmdletBinding()]
param(
    [switch]$RemovePrefetch = $true
)

$ErrorActionPreference = "Stop"

$removeKr = Join-Path $PSScriptRoot "remove_kr_task.ps1"
$removeUs = Join-Path $PSScriptRoot "remove_us_task.ps1"
$removeAgentLab = Join-Path $PSScriptRoot "remove_agent_lab_tasks.ps1"

foreach ($path in @($removeKr, $removeUs, $removeAgentLab)) {
    if (-not (Test-Path $path)) {
        throw "Required script not found: $path"
    }
}

Write-Output "[1/3] Removing KR tasks..."
& $removeKr -RemovePrefetch:$RemovePrefetch

Write-Output "[2/3] Removing US tasks..."
& $removeUs -RemovePrefetch:$RemovePrefetch

Write-Output "[3/3] Removing Agent Lab tasks..."
& $removeAgentLab

Write-Output ""
Write-Output "All SystematicAlpha tasks removed."

