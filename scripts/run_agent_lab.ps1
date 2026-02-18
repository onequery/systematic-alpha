[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonExe = "",
    [ValidateSet("init", "ingest-propose", "approve-orders", "daily-review", "weekly-council", "report")]
    [string]$Action = "ingest-propose",
    [ValidateSet("KR", "US")]
    [string]$Market = "KR",
    [string]$Date = "",
    [string]$Week = "",
    [double]$CapitalKrw = 10000000,
    [int]$Agents = 3,
    [string]$ProposalId = "",
    [string]$ApprovedBy = "",
    [string]$Note = "",
    [string]$DateFrom = "",
    [string]$DateTo = "",
    [switch]$WaitForSessionResult = $true,
    [int]$WaitTimeoutSeconds = 3600,
    [int]$PollIntervalSeconds = 15,
    [switch]$UseDailyLock = $true
)

$ErrorActionPreference = "Stop"
$script:RunLogFile = $null

function Get-KstNow {
    $kst = [System.TimeZoneInfo]::FindSystemTimeZoneById("Korea Standard Time")
    return [System.TimeZoneInfo]::ConvertTime((Get-Date), $kst)
}

function Get-Timestamp {
    return (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
}

function Write-MonitorLog {
    param([string]$Message)
    $line = ("[{0}] {1}" -f (Get-Timestamp), $Message)
    Write-Output $line
    if (-not [string]::IsNullOrWhiteSpace($script:RunLogFile)) {
        try {
            $line | Out-File -FilePath $script:RunLogFile -Encoding UTF8 -Append
        } catch {
            # keep console output alive
        }
    }
}

function Write-RunOutputLine {
    param([string]$Line)
    Write-Output $Line
    if (-not [string]::IsNullOrWhiteSpace($script:RunLogFile)) {
        try {
            $Line | Out-File -FilePath $script:RunLogFile -Encoding UTF8 -Append
        } catch {
            # keep console output alive
        }
    }
}

function Resolve-PythonExe {
    param([string]$Current)
    if (-not [string]::IsNullOrWhiteSpace($Current)) {
        return $Current
    }
    $defaultPython = "C:\Users\heesu\anaconda3\envs\systematic-alpha\python.exe"
    if (Test-Path $defaultPython) {
        return $defaultPython
    }
    return "python"
}

function Get-LatestSessionJsonPath {
    param(
        [string]$Root,
        [string]$MarketValue,
        [string]$RunDate
    )
    $marketTag = $MarketValue.Trim().ToLowerInvariant()
    $resultDir = Join-Path $Root ("out\{0}\{1}\results" -f $marketTag, $RunDate)
    if (-not (Test-Path $resultDir)) {
        return $null
    }
    $files = Get-ChildItem -Path $resultDir -File -Filter ("{0}_daily_*.json" -f $marketTag) `
        | Sort-Object LastWriteTime, Name
    if (-not $files -or $files.Count -eq 0) {
        return $null
    }
    return $files[-1].FullName
}

function Wait-ForSessionJson {
    param(
        [string]$Root,
        [string]$MarketValue,
        [string]$RunDate,
        [int]$TimeoutSeconds,
        [int]$PollSeconds
    )
    $timeout = [Math]::Max(1, $TimeoutSeconds)
    $poll = [Math]::Max(1, $PollSeconds)
    $elapsed = 0
    while ($elapsed -le $timeout) {
        $path = Get-LatestSessionJsonPath -Root $Root -MarketValue $MarketValue -RunDate $RunDate
        if ($path) {
            Write-MonitorLog "[session] found result json: $path"
            return $path
        }
        Write-MonitorLog "[session] waiting result json... elapsed=${elapsed}s/${timeout}s"
        Start-Sleep -Seconds $poll
        $elapsed += $poll
    }
    return $null
}

function Invoke-AgentLabCli {
    param(
        [string]$PyExe,
        [string[]]$CliArgs
    )
    Write-MonitorLog ("command: {0} -m systematic_alpha.agent_lab.cli {1}" -f $PyExe, ($CliArgs -join " "))
    $exitCode = 1
    try {
        & $PyExe -m systematic_alpha.agent_lab.cli @CliArgs 2>&1 | ForEach-Object {
            Write-RunOutputLine -Line "$_"
        }
        $nativeExit = $LASTEXITCODE
        if ($null -eq $nativeExit) {
            $exitCode = 0
        } else {
            $exitCode = [int]$nativeExit
        }
    } catch {
        Write-MonitorLog ("[error] {0}" -f $_.Exception.Message)
        $exitCode = 9009
    }
    Write-MonitorLog "[command-exit] $exitCode"
    return $exitCode
}

if (-not (Test-Path $ProjectRoot)) {
    throw "ProjectRoot not found: $ProjectRoot"
}

$PythonExe = Resolve-PythonExe -Current $PythonExe
Set-Location $ProjectRoot

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$nowKst = Get-KstNow
$runDate = if ([string]::IsNullOrWhiteSpace($Date)) { $nowKst.ToString("yyyyMMdd") } else { $Date.Trim() }
$runStamp = $nowKst.ToString("yyyyMMdd_HHmmss")
$logDir = Join-Path $ProjectRoot ("logs\agent_lab\{0}" -f $runDate)
$null = New-Item -ItemType Directory -Force -Path $logDir
$script:RunLogFile = Join-Path $logDir ("agent_lab_{0}_{1}.log" -f ($Action.Replace("-", "_")), $runStamp)

Write-MonitorLog "[start] action=$Action, market=$Market, date=$runDate"
Write-MonitorLog "[run] log_file=$script:RunLogFile"

$commonPrefix = @("--project-root", $ProjectRoot)

switch ($Action) {
    "init" {
        $args = $commonPrefix + @("init", "--capital-krw", "$CapitalKrw", "--agents", "$Agents")
        $code = Invoke-AgentLabCli -PyExe $PythonExe -CliArgs $args
        exit $code
    }

    "ingest-propose" {
        $lockPath = $null
        $lockCreated = $false
        if ($UseDailyLock) {
            $runtimeDir = Join-Path $ProjectRoot ("out\agent_lab\{0}\runtime" -f $runDate)
            $null = New-Item -ItemType Directory -Force -Path $runtimeDir
            $lockPath = Join-Path $runtimeDir ("ingest_propose_{0}.lock" -f $Market.ToLowerInvariant())
            if (Test-Path $lockPath) {
                Write-MonitorLog "[skip] daily lock exists: $lockPath"
                exit 0
            }
            $lockBody = @(
                "created_at=" + (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
                "action=ingest-propose"
                "market=" + $Market
                "date=" + $runDate
                "status=waiting_for_session_json"
            ) -join "`n"
            Set-Content -Path $lockPath -Value $lockBody -Encoding UTF8
            $lockCreated = $true
            Write-MonitorLog "[lock] created: $lockPath"
        }

        if ($WaitForSessionResult) {
            $found = Wait-ForSessionJson `
                -Root $ProjectRoot `
                -MarketValue $Market `
                -RunDate $runDate `
                -TimeoutSeconds $WaitTimeoutSeconds `
                -PollSeconds $PollIntervalSeconds
            if (-not $found) {
                Write-MonitorLog "[skip] session json not found within timeout. market=$Market date=$runDate"
                if ($lockCreated -and $lockPath -and (Test-Path $lockPath)) {
                    Remove-Item -Path $lockPath -Force -ErrorAction SilentlyContinue
                    Write-MonitorLog "[lock] removed due to timeout: $lockPath"
                }
                exit 0
            }
        }

        $ingestArgs = $commonPrefix + @("ingest-session", "--market", $Market, "--date", $runDate)
        $ingestCode = Invoke-AgentLabCli -PyExe $PythonExe -CliArgs $ingestArgs
        if ($ingestCode -ne 0) {
            if ($lockCreated -and $lockPath -and (Test-Path $lockPath)) {
                Remove-Item -Path $lockPath -Force -ErrorAction SilentlyContinue
                Write-MonitorLog "[lock] removed due to ingest failure: $lockPath"
            }
            exit $ingestCode
        }
        $proposeArgs = $commonPrefix + @("propose-orders", "--market", $Market, "--date", $runDate)
        $proposeCode = Invoke-AgentLabCli -PyExe $PythonExe -CliArgs $proposeArgs
        if ($proposeCode -ne 0) {
            if ($lockCreated -and $lockPath -and (Test-Path $lockPath)) {
                Remove-Item -Path $lockPath -Force -ErrorAction SilentlyContinue
                Write-MonitorLog "[lock] removed due to propose failure: $lockPath"
            }
        } elseif ($lockCreated -and $lockPath -and (Test-Path $lockPath)) {
            $lockBody = @(
                "created_at=" + (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
                "action=ingest-propose"
                "market=" + $Market
                "date=" + $runDate
                "status=done"
            ) -join "`n"
            Set-Content -Path $lockPath -Value $lockBody -Encoding UTF8
            Write-MonitorLog "[lock] updated status=done: $lockPath"
        }
        exit $proposeCode
    }

    "approve-orders" {
        if ([string]::IsNullOrWhiteSpace($ProposalId)) {
            throw "approve-orders requires -ProposalId"
        }
        if ([string]::IsNullOrWhiteSpace($ApprovedBy)) {
            $ApprovedBy = "$env:COMPUTERNAME/$env:USERNAME"
        }
        $args = $commonPrefix + @(
            "approve-orders",
            "--proposal-id", $ProposalId,
            "--approved-by", $ApprovedBy,
            "--note", $Note
        )
        $code = Invoke-AgentLabCli -PyExe $PythonExe -CliArgs $args
        exit $code
    }

    "daily-review" {
        $args = $commonPrefix + @("daily-review", "--date", $runDate)
        $code = Invoke-AgentLabCli -PyExe $PythonExe -CliArgs $args
        exit $code
    }

    "weekly-council" {
        $weekValue = $Week
        if ([string]::IsNullOrWhiteSpace($weekValue)) {
            $isoYear = [System.Globalization.ISOWeek]::GetYear($nowKst.DateTime)
            $isoWeek = [System.Globalization.ISOWeek]::GetWeekOfYear($nowKst.DateTime)
            $weekValue = ("{0}-W{1:D2}" -f $isoYear, $isoWeek)
        }
        $args = $commonPrefix + @("weekly-council", "--week", $weekValue)
        $code = Invoke-AgentLabCli -PyExe $PythonExe -CliArgs $args
        exit $code
    }

    "report" {
        $fromValue = if ([string]::IsNullOrWhiteSpace($DateFrom)) { $runDate } else { $DateFrom.Trim() }
        $toValue = if ([string]::IsNullOrWhiteSpace($DateTo)) { $runDate } else { $DateTo.Trim() }
        $args = $commonPrefix + @("report", "--from", $fromValue, "--to", $toValue)
        $code = Invoke-AgentLabCli -PyExe $PythonExe -CliArgs $args
        exit $code
    }
}
