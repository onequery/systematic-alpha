[CmdletBinding()]
param(
    [string]$PythonExe = "C:\Users\heesu\anaconda3\envs\systematic-alpha\python.exe",
    [switch]$RegisterPrefetch = $true,
    [switch]$DisableBaseSelectorTelegram = $true,
    [switch]$InitAgentLab = $true,
    [double]$CapitalKrw = 10000000,
    [int]$Agents = 3
)

$ErrorActionPreference = "Stop"

$registerKr = Join-Path $PSScriptRoot "register_kr_task.ps1"
$registerUs = Join-Path $PSScriptRoot "register_us_task.ps1"
$registerAgentLab = Join-Path $PSScriptRoot "register_agent_lab_tasks.ps1"
$runAgentLab = Join-Path $PSScriptRoot "run_agent_lab.ps1"

foreach ($path in @($registerKr, $registerUs, $registerAgentLab, $runAgentLab)) {
    if (-not (Test-Path $path)) {
        throw "Required script not found: $path"
    }
}

Write-Output "[1/4] Registering KR selector task..."
& $registerKr `
    -PythonExe $PythonExe `
    -RegisterPrefetch:$RegisterPrefetch `
    -DisableTelegram:$DisableBaseSelectorTelegram

Write-Output "[2/4] Registering US selector task..."
& $registerUs `
    -PythonExe $PythonExe `
    -RegisterPrefetch:$RegisterPrefetch `
    -DisableTelegram:$DisableBaseSelectorTelegram

Write-Output "[3/4] Registering Agent Lab follow-up/review tasks..."
& $registerAgentLab -PythonExe $PythonExe

if ($InitAgentLab) {
    Write-Output "[4/4] Initializing Agent Lab state..."
    & $runAgentLab `
        -PythonExe $PythonExe `
        -Action init `
        -CapitalKrw $CapitalKrw `
        -Agents $Agents
} else {
    Write-Output "[4/4] Skipped Agent Lab init (-InitAgentLab:\$false)."
}

Write-Output ""
Write-Output "All tasks registered with one command."
Write-Output "Recommended check:"
Write-Output "  Get-ScheduledTask -TaskName 'SystematicAlpha*' | Format-Table TaskName, State -AutoSize"

