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
    [switch]$UseDailyLock = $true,
    [switch]$DisableTelegram = $false
)

$ErrorActionPreference = "Stop"
$script:RunLogFile = $null
$script:TelegramEnabled = $false
$script:TelegramBotToken = ""
$script:TelegramChatId = ""
$script:TelegramThreadId = ""
$script:TelegramDisableNotification = $false

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

function Test-Truthy {
    param([string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $false
    }
    $norm = $Value.Trim().ToLowerInvariant()
    return $norm -in @("1", "true", "yes", "y", "on")
}

function Import-DotEnv {
    param(
        [string]$Path,
        [string[]]$AllowedKeys = @()
    )

    if (-not (Test-Path $Path)) {
        return
    }

    $allowLookup = $null
    if ($AllowedKeys -and $AllowedKeys.Count -gt 0) {
        $allowLookup = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
        foreach ($item in $AllowedKeys) {
            if (-not [string]::IsNullOrWhiteSpace($item)) {
                [void]$allowLookup.Add($item.Trim())
            }
        }
    }

    foreach ($raw in Get-Content -Path $Path -Encoding UTF8) {
        $line = $raw.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            continue
        }
        if ($line.StartsWith("export ")) {
            $line = $line.Substring(7).Trim()
        }
        $eqIdx = $line.IndexOf("=")
        if ($eqIdx -lt 1) {
            continue
        }
        $key = $line.Substring(0, $eqIdx).Trim()
        $value = $line.Substring($eqIdx + 1).Trim()
        if ([string]::IsNullOrWhiteSpace($key)) {
            continue
        }
        if ($allowLookup -ne $null -and -not $allowLookup.Contains($key)) {
            continue
        }
        if ($value.Length -ge 2) {
            $first = $value[0]
            $last = $value[$value.Length - 1]
            if (($first -eq '"' -and $last -eq '"') -or ($first -eq "'" -and $last -eq "'")) {
                $value = $value.Substring(1, $value.Length - 2)
            }
        }
        if (-not (Test-Path "Env:$key")) {
            Set-Item -Path "Env:$key" -Value $value
        }
    }
}

function Normalize-TelegramToken {
    param([string]$Token)
    if ([string]::IsNullOrWhiteSpace($Token)) {
        return ""
    }
    $clean = $Token.Trim()
    if ($clean.StartsWith("bot", [System.StringComparison]::OrdinalIgnoreCase)) {
        $clean = $clean.Substring(3)
    }
    return $clean
}

function Truncate-Text {
    param(
        [string]$Text,
        [int]$MaxChars = 3000
    )
    if ([string]::IsNullOrWhiteSpace($Text)) {
        return $Text
    }
    if ($Text.Length -le $MaxChars) {
        return $Text
    }
    return $Text.Substring(0, $MaxChars) + "`n...(truncated)..."
}

function Send-TelegramMessage {
    param([string]$Text)
    if (-not $script:TelegramEnabled) {
        return
    }
    if ([string]::IsNullOrWhiteSpace($Text)) {
        return
    }

    $uri = "https://api.telegram.org/bot$($script:TelegramBotToken)/sendMessage"
    $payload = @{
        chat_id = $script:TelegramChatId
        text = (Truncate-Text -Text $Text)
        disable_web_page_preview = $true
    }
    if ($script:TelegramDisableNotification) {
        $payload["disable_notification"] = $true
    }
    if (-not [string]::IsNullOrWhiteSpace($script:TelegramThreadId)) {
        $threadValue = 0L
        if ([int64]::TryParse($script:TelegramThreadId, [ref]$threadValue)) {
            $payload["message_thread_id"] = $threadValue
        }
    }

    try {
        $resp = Invoke-RestMethod `
            -Method Post `
            -Uri $uri `
            -ContentType "application/json; charset=utf-8" `
            -Body ($payload | ConvertTo-Json -Compress) `
            -TimeoutSec 15
        if ($null -ne $resp -and $null -ne $resp.ok -and (-not $resp.ok)) {
            [void](Write-MonitorLog "[telegram] api error: code=$($resp.error_code) desc=$($resp.description)")
        }
    } catch {
        [void](Write-MonitorLog "[telegram] send failed: $($_.Exception.Message)")
    }
}

function Resolve-ExitCode {
    param([object]$Result)
    if ($null -eq $Result) {
        return 9009
    }
    $last = $Result
    if ($Result -is [System.Array]) {
        if ($Result.Count -gt 0) {
            $last = $Result[$Result.Count - 1]
        }
    }
    try {
        return [int]$last
    } catch {
        return 9009
    }
}

function Build-ProposalSummaryMessage {
    param(
        [string]$ProjectRootPath,
        [string]$RunDate,
        [string]$MarketValue
    )
    $marketTag = $MarketValue.Trim().ToLowerInvariant()
    $path = Join-Path $ProjectRootPath ("out\agent_lab\{0}\proposals_{1}_{0}.json" -f $RunDate, $marketTag)
    if (-not (Test-Path $path)) {
        return "[AgentLab] propose result not found: $path"
    }
    try {
        $obj = Get-Content -Path $path -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        return "[AgentLab] propose result parse failed: $path"
    }
    $lines = @(
        "[AgentLab] propose"
        "market=$MarketValue"
        "date=$RunDate"
    )
    $rows = @()
    if ($null -ne $obj.proposals) {
        $rows = @($obj.proposals)
    }
    if ($rows.Count -eq 0) {
        $lines += "proposals=0"
        return ($lines -join "`n")
    }
    foreach ($row in $rows) {
        $agent = [string]$row.agent_id
        $status = [string]$row.status
        $blocked = [string]$row.blocked_reason
        $orders = @()
        if ($null -ne $row.orders) {
            $orders = @($row.orders)
        }
        if ($orders.Count -eq 0) {
            $detail = "orders=0"
        } else {
            $chunks = @()
            foreach ($order in $orders) {
                $side = [string]$order.side
                $symbol = [string]$order.symbol
                $qty = [double]($order.quantity)
                $chunks += ("{0} {1} x{2}" -f $side, $symbol, [math]::Round($qty, 4))
            }
            $detail = ($chunks -join ", ")
        }
        if ([string]::IsNullOrWhiteSpace($blocked)) {
            $lines += ("{0}: {1} | {2}" -f $agent, $status, $detail)
        } else {
            $lines += ("{0}: {1} | {2} | blocked={3}" -f $agent, $status, $detail, $blocked)
        }
    }
    return ($lines -join "`n")
}

function Get-WeeklyCouncilPayload {
    param(
        [string]$ProjectRootPath,
        [string]$WeekId
    )
    $name = ("weekly_council_{0}.json" -f $WeekId.Replace("-", "_"))
    $root = Join-Path $ProjectRootPath "out\agent_lab"
    if (-not (Test-Path $root)) {
        return $null
    }
    $files = Get-ChildItem -Path $root -Recurse -File -Filter $name | Sort-Object LastWriteTime, FullName
    if (-not $files -or $files.Count -eq 0) {
        return $null
    }
    $path = $files[-1].FullName
    try {
        $obj = Get-Content -Path $path -Raw -Encoding UTF8 | ConvertFrom-Json
        return @{
            path = $path
            data = $obj
        }
    } catch {
        return $null
    }
}

function Build-WeeklyCouncilSummaryMessage {
    param(
        [string]$ProjectRootPath,
        [string]$WeekId
    )
    $bundle = Get-WeeklyCouncilPayload -ProjectRootPath $ProjectRootPath -WeekId $WeekId
    if ($null -eq $bundle) {
        return "[AgentLab] weekly-council complete`nweek=$WeekId`n(summary file not found)"
    }

    $obj = $bundle.data
    $lines = @(
        "[AgentLab] weekly-council"
        ("week={0}" -f [string]$obj.week_id)
        ("champion={0}" -f [string]$obj.champion_agent_id)
    )

    $promoted = @()
    if ($null -ne $obj.promoted_versions) {
        $promoted = @($obj.promoted_versions.PSObject.Properties | ForEach-Object { "{0}:{1}" -f $_.Name, $_.Value })
    }
    if ($promoted.Count -gt 0) {
        $lines += ("promoted=" + ($promoted -join ", "))
    } else {
        $lines += "promoted=(none)"
    }

    $moderatorSummary = ""
    if ($null -ne $obj.discussion -and $null -ne $obj.discussion.moderator) {
        $moderatorSummary = [string]$obj.discussion.moderator.summary
    }
    if (-not [string]::IsNullOrWhiteSpace($moderatorSummary)) {
        $lines += ("moderator=" + $moderatorSummary)
    }

    if ($null -ne $obj.discussion -and $null -ne $obj.discussion.rounds) {
        $rounds = @($obj.discussion.rounds)
        if ($rounds.Count -gt 0) {
            foreach ($round in $rounds | Select-Object -First 2) {
                $roundNo = [string]$round.round
                $phase = [string]$round.phase
                $lines += ("round{0}_{1}" -f $roundNo, $phase)
                $speeches = @()
                if ($null -ne $round.speeches) {
                    $speeches = @($round.speeches)
                }
                foreach ($speech in $speeches | Select-Object -First 3) {
                    $aid = [string]$speech.agent_id
                    $text = ""
                    if (-not [string]::IsNullOrWhiteSpace([string]$speech.thesis)) {
                        $text = [string]$speech.thesis
                    } elseif (-not [string]::IsNullOrWhiteSpace([string]$speech.rebuttal)) {
                        $text = [string]$speech.rebuttal
                    }
                    if ($text.Length -gt 160) {
                        $text = $text.Substring(0, 160) + "..."
                    }
                    if (-not [string]::IsNullOrWhiteSpace($text)) {
                        $lines += ("{0}: {1}" -f $aid, $text)
                    }
                }
            }
        }
    }

    $alerts = @()
    if ($null -ne $obj.llm_alerts) {
        $alerts = @($obj.llm_alerts)
    }
    if ($alerts.Count -gt 0) {
        $lines += ("llm_alerts=" + $alerts.Count)
        $topAlerts = $alerts | Select-Object -First 3
        foreach ($a in $topAlerts) {
            $lines += ("alert: agent={0}, phase={1}, reason={2}" -f [string]$a.agent_id, [string]$a.phase, [string]$a.reason)
        }
    } else {
        $lines += "llm_alerts=0"
    }

    return ($lines -join "`n")
}

function Get-WeeklyCouncilTokenAlerts {
    param(
        [string]$ProjectRootPath,
        [string]$WeekId
    )
    $bundle = Get-WeeklyCouncilPayload -ProjectRootPath $ProjectRootPath -WeekId $WeekId
    if ($null -eq $bundle) {
        return @()
    }
    $obj = $bundle.data
    if ($null -eq $obj.llm_alerts) {
        return @()
    }
    $alerts = @($obj.llm_alerts)
    $hits = New-Object System.Collections.Generic.List[string]
    foreach ($a in $alerts) {
        $reason = [string]$a.reason
        if ($reason -match "daily_budget_exceeded|insufficient_quota|quota|billing|context_length_exceeded|max_tokens|token") {
            $hits.Add($reason)
        }
    }
    return @($hits | Select-Object -Unique)
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

Import-DotEnv -Path (Join-Path $ProjectRoot ".env") -AllowedKeys @(
    "TELEGRAM_ENABLED",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "TELEGRAM_THREAD_ID",
    "TELEGRAM_DISABLE_NOTIFICATION"
)

$script:TelegramBotToken = Normalize-TelegramToken ([string]$env:TELEGRAM_BOT_TOKEN)
$script:TelegramChatId = [string]$env:TELEGRAM_CHAT_ID
$script:TelegramThreadId = [string]$env:TELEGRAM_THREAD_ID
$script:TelegramDisableNotification = Test-Truthy $env:TELEGRAM_DISABLE_NOTIFICATION

$enabledByKeys = -not (
    [string]::IsNullOrWhiteSpace($script:TelegramBotToken) -or
    [string]::IsNullOrWhiteSpace($script:TelegramChatId)
)
if ([string]::IsNullOrWhiteSpace($env:TELEGRAM_ENABLED)) {
    $script:TelegramEnabled = $enabledByKeys
} else {
    $script:TelegramEnabled = (Test-Truthy $env:TELEGRAM_ENABLED) -and $enabledByKeys
}
if ($DisableTelegram) {
    $script:TelegramEnabled = $false
}

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
        $result = Invoke-AgentLabCli -PyExe $PythonExe -CliArgs $args
        $code = Resolve-ExitCode -Result $result
        if ($code -eq 0) {
            Send-TelegramMessage (
                "[AgentLab] init success`n" +
                "capital_krw=$CapitalKrw`n" +
                "agents=$Agents"
            )
        }
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
        $ingestResult = Invoke-AgentLabCli -PyExe $PythonExe -CliArgs $ingestArgs
        $ingestCode = Resolve-ExitCode -Result $ingestResult
        if ($ingestCode -ne 0) {
            if ($lockCreated -and $lockPath -and (Test-Path $lockPath)) {
                Remove-Item -Path $lockPath -Force -ErrorAction SilentlyContinue
                Write-MonitorLog "[lock] removed due to ingest failure: $lockPath"
            }
            Send-TelegramMessage (
                "[AgentLab] ingest failed`n" +
                "market=$Market`n" +
                "date=$runDate`n" +
                "exit_code=$ingestCode"
            )
            exit $ingestCode
        }
        $proposeArgs = $commonPrefix + @("propose-orders", "--market", $Market, "--date", $runDate)
        $proposeResult = Invoke-AgentLabCli -PyExe $PythonExe -CliArgs $proposeArgs
        $proposeCode = Resolve-ExitCode -Result $proposeResult
        if ($proposeCode -ne 0) {
            if ($lockCreated -and $lockPath -and (Test-Path $lockPath)) {
                Remove-Item -Path $lockPath -Force -ErrorAction SilentlyContinue
                Write-MonitorLog "[lock] removed due to propose failure: $lockPath"
            }
            Send-TelegramMessage (
                "[AgentLab] propose failed`n" +
                "market=$Market`n" +
                "date=$runDate`n" +
                "exit_code=$proposeCode"
            )
        } else {
            if ($lockCreated -and $lockPath -and (Test-Path $lockPath)) {
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
            Send-TelegramMessage (Build-ProposalSummaryMessage -ProjectRootPath $ProjectRoot -RunDate $runDate -MarketValue $Market)
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
        $result = Invoke-AgentLabCli -PyExe $PythonExe -CliArgs $args
        $code = Resolve-ExitCode -Result $result
        if ($code -eq 0) {
            Send-TelegramMessage (
                "[AgentLab] approval executed`n" +
                "proposal_id=$ProposalId`n" +
                "approved_by=$ApprovedBy"
            )
        }
        exit $code
    }

    "daily-review" {
        $args = $commonPrefix + @("daily-review", "--date", $runDate)
        $result = Invoke-AgentLabCli -PyExe $PythonExe -CliArgs $args
        $code = Resolve-ExitCode -Result $result
        if ($code -eq 0) {
            Send-TelegramMessage (
                "[AgentLab] daily-review complete`n" +
                "date=$runDate"
            )
        }
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
        $result = Invoke-AgentLabCli -PyExe $PythonExe -CliArgs $args
        $code = Resolve-ExitCode -Result $result
        if ($code -eq 0) {
            Send-TelegramMessage (Build-WeeklyCouncilSummaryMessage -ProjectRootPath $ProjectRoot -WeekId $weekValue)
            $tokenAlerts = @(Get-WeeklyCouncilTokenAlerts -ProjectRootPath $ProjectRoot -WeekId $weekValue)
            if ($tokenAlerts.Count -gt 0) {
                $topAlerts = @($tokenAlerts | Select-Object -First 3)
                Send-TelegramMessage (
                    "[AgentLab] alert`n" +
                    "OpenAI token/quota limit issue detected during weekly-council.`n" +
                    "week=$weekValue`n" +
                    "reasons=$($topAlerts -join '; ')`n" +
                    "action=Increase OPENAI_MAX_DAILY_COST, check OpenAI quota/billing, or reduce meeting frequency."
                )
            }
        }
        exit $code
    }

    "report" {
        $fromValue = if ([string]::IsNullOrWhiteSpace($DateFrom)) { $runDate } else { $DateFrom.Trim() }
        $toValue = if ([string]::IsNullOrWhiteSpace($DateTo)) { $runDate } else { $DateTo.Trim() }
        $args = $commonPrefix + @("report", "--from", $fromValue, "--to", $toValue)
        $result = Invoke-AgentLabCli -PyExe $PythonExe -CliArgs $args
        $code = Resolve-ExitCode -Result $result
        if ($code -eq 0) {
            Send-TelegramMessage (
                "[AgentLab] report complete`n" +
                "from=$fromValue`n" +
                "to=$toValue"
            )
        }
        exit $code
    }
}
