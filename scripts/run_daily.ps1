param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonExe = "",
    [ValidateSet("KR", "US")]
    [string]$Market = "KR",
    [string]$UsExchange = "NASD",
    [double]$UsPollInterval = 2.0,
    [string]$UniverseFile = "",
    [switch]$RequireUsOpen = $false,
    [switch]$AssumeOpenForTest = $false,
    [int]$UsOpenWindowMinutes = 20,
    [int]$CollectSeconds = 600,
    [int]$FinalPicks = 3,
    [int]$PreCandidates = 40,
    [int]$MaxSymbolsScan = 500,
    [int]$KrUniverseSize = 500,
    [int]$UsUniverseSize = 500,
    [double]$MinChangePct = 3.0,
    [double]$MinGapPct = 2.0,
    [double]$MinPrevTurnover = 10000000000,
    [double]$MinStrength = 100.0,
    [double]$MinVolRatio = 0.10,
    [double]$MinBidAskRatio = 1.2,
    [int]$MinPassConditions = 5,
    [double]$MinMaintainRatio = 0.6,
    [double]$RestSleepSec = 0.03,
    [switch]$AllowShortBias = $false,
    [int]$MinExecTicks = 30,
    [int]$MinOrderbookTicks = 30,
    [double]$MinRealtimeCumVolume = 1.0,
    [double]$MinRealtimeCoverageRatio = 0.8,
    [switch]$AllowLowCoverage = $false,
    [int]$Stage1LogInterval = 20,
    [int]$RealtimeLogInterval = 10,
    [string]$OvernightReportPath = "",
    [switch]$Mock = $false,
    [int]$StartDelaySeconds = 5,
    [int]$MaxAttempts = 4,
    [int]$RetryDelaySeconds = 30,
    [int]$RetryBackoffMultiplier = 2,
    [int]$MaxRetryDelaySeconds = 180,
    [int]$NotifyTailLines = 20,
    [switch]$NotifyStart = $true
)
$ErrorActionPreference = "Stop"
$script:RunLogFile = $null

function Ensure-Tls12 {
    try {
        $current = [Net.ServicePointManager]::SecurityProtocol
        if (($current -band [Net.SecurityProtocolType]::Tls12) -eq 0) {
            [Net.ServicePointManager]::SecurityProtocol = $current -bor [Net.SecurityProtocolType]::Tls12
        }
    } catch {
        # Keep silent; runtime will surface HTTPS errors if this fails.
    }
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

function Truncate-Text {
    param(
        [string]$Text,
        [int]$MaxChars = 3200
    )

    if ([string]::IsNullOrEmpty($Text)) {
        return $Text
    }
    if ($Text.Length -le $MaxChars) {
        return $Text
    }
    return $Text.Substring(0, $MaxChars) + "`n...(truncated)..."
}

function Get-OutputTail {
    param(
        [object[]]$Lines,
        [int]$TailLines = 20
    )

    if (-not $Lines -or $Lines.Count -eq 0) {
        return "(no output)"
    }

    $textLines = @($Lines | ForEach-Object { "$_" })
    $start = [Math]::Max(0, $textLines.Count - [Math]::Max(1, $TailLines))
    return ($textLines[$start..($textLines.Count - 1)] -join "`n")
}

function Get-SelectionSummary {
    param(
        [string]$JsonPath,
        [int]$TopN = 3
    )

    if (-not (Test-Path $JsonPath)) {
        return "result json not found: $JsonPath"
    }

    try {
        $obj = Get-Content -Path $JsonPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        return "failed to parse json: $($_.Exception.Message)"
    }

    $rows = @()
    if ($null -ne $obj.final) {
        $rows = @($obj.final)
    } elseif ($null -ne $obj.all_ranked) {
        $rows = @($obj.all_ranked)
    }
    if ($rows.Count -eq 0) {
        return "no ranked symbols in output json."
    }

    $lines = @()
    $limit = [Math]::Min([Math]::Max(1, $TopN), $rows.Count)
    for ($i = 0; $i -lt $limit; $i++) {
        $row = $rows[$i]
        $rank = if ($null -ne $row.rank) { $row.rank } else { $i + 1 }
        $name = if ($row.name) { [string]$row.name } else { "-" }
        $code = if ($row.code) { [string]$row.code } else { "-" }
        $rec = if ($null -ne $row.recommendation_score) { "{0:N2}" -f [double]$row.recommendation_score } else { "-" }
        $score = if ($null -ne $row.score -and $null -ne $row.max_score) { "$($row.score)/$($row.max_score)" } else { "-" }
        $lines += ("#{0} {1}({2}) rec={3}% score={4}" -f $rank, $name, $code, $rec, $score)
    }

    return ($lines -join "`n")
}

function Send-TelegramViaPython {
    param([string]$Text)

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return $false
    }
    if ([string]::IsNullOrWhiteSpace($PythonExe)) {
        return $false
    }

    $pyScript = @'
import json
import os
import sys
import requests

token = os.environ.get("SA_TELEGRAM_BOT_TOKEN", "").strip()
chat_id = os.environ.get("SA_TELEGRAM_CHAT_ID", "").strip()
thread_id = os.environ.get("SA_TELEGRAM_THREAD_ID", "").strip()
disable_notification = os.environ.get("SA_TELEGRAM_DISABLE_NOTIFICATION", "").strip().lower() in {"1","true","yes","y","on"}
text = os.environ.get("SA_TELEGRAM_TEXT", "")

if not token or not chat_id or not text:
    sys.exit(2)

for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(key, None)

url = f"https://api.telegram.org/bot{token}/sendMessage"
payload = {
    "chat_id": chat_id,
    "text": text,
    "disable_web_page_preview": True,
}
if disable_notification:
    payload["disable_notification"] = True
if thread_id:
    try:
        payload["message_thread_id"] = int(thread_id)
    except Exception:
        pass

resp = requests.post(url, json=payload, timeout=15)
resp.raise_for_status()
obj = resp.json()
if not obj.get("ok"):
    raise RuntimeError(f"telegram_api_error:{obj}")
sys.exit(0)
'@

    $env:SA_TELEGRAM_BOT_TOKEN = $script:TelegramBotToken
    $env:SA_TELEGRAM_CHAT_ID = $script:TelegramChatId
    $env:SA_TELEGRAM_THREAD_ID = $script:TelegramThreadId
    $env:SA_TELEGRAM_DISABLE_NOTIFICATION = if ($script:TelegramDisableNotification) { "1" } else { "0" }
    $env:SA_TELEGRAM_TEXT = $Text

    try {
        $pyScript | & $PythonExe - 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    } finally {
        Remove-Item Env:SA_TELEGRAM_BOT_TOKEN -ErrorAction SilentlyContinue
        Remove-Item Env:SA_TELEGRAM_CHAT_ID -ErrorAction SilentlyContinue
        Remove-Item Env:SA_TELEGRAM_THREAD_ID -ErrorAction SilentlyContinue
        Remove-Item Env:SA_TELEGRAM_DISABLE_NOTIFICATION -ErrorAction SilentlyContinue
        Remove-Item Env:SA_TELEGRAM_TEXT -ErrorAction SilentlyContinue
    }
}

function Send-TelegramMessage {
    param([string]$Text)

    if (-not $script:TelegramEnabled) {
        return
    }
    if ([string]::IsNullOrWhiteSpace($Text)) {
        return
    }

    Ensure-Tls12
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
            $errCode = $resp.error_code
            $errDesc = $resp.description
            Write-MonitorLog "[telegram] api error: code=$errCode desc=$errDesc"
            if ($errCode -eq 401) {
                Write-MonitorLog "[telegram] token unauthorized. disabling telegram notifications for this run."
                $script:TelegramEnabled = $false
            }
        }
    } catch {
        $statusCode = $null
        try {
            if ($_.Exception.Response -and $_.Exception.Response.StatusCode) {
                $statusCode = [int]$_.Exception.Response.StatusCode
            }
        } catch {
            $statusCode = $null
        }
        if ($statusCode -eq 401) {
            Write-MonitorLog "[telegram] send failed: HTTP 401 Unauthorized. check TELEGRAM_BOT_TOKEN (do not include leading 'bot'). disabling telegram notifications for this run."
            $script:TelegramEnabled = $false
            return
        }
        Write-MonitorLog "[telegram] send failed: $($_.Exception.Message)"
        if (Send-TelegramViaPython -Text $Text) {
            Write-MonitorLog "[telegram] fallback send succeeded (python requests)."
            return
        }
    }
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
            # Ignore file logging failures and keep console output alive.
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

function To-InvariantString {
    param([double]$Value)
    return $Value.ToString([System.Globalization.CultureInfo]::InvariantCulture)
}

function Wait-WithProgress {
    param(
        [int]$TotalSeconds,
        [string]$Label = "wait"
    )

    $total = [Math]::Max(0, $TotalSeconds)
    if ($total -le 0) {
        return
    }

    for ($elapsed = 0; $elapsed -lt $total; $elapsed++) {
        $remain = $total - $elapsed
        Write-MonitorLog "[$Label] elapsed=${elapsed}s/${total}s, remaining=${remain}s"
        Start-Sleep -Seconds 1
    }
    Write-MonitorLog "[$Label] elapsed=${total}s/${total}s, remaining=0s"
}

function Get-UsMarketState {
    $tz = [System.TimeZoneInfo]::FindSystemTimeZoneById("Eastern Standard Time")
    $nowLocal = Get-Date
    $nowEt = [System.TimeZoneInfo]::ConvertTime($nowLocal, $tz)
    $isWeekday = $nowEt.DayOfWeek -in @(
        [DayOfWeek]::Monday,
        [DayOfWeek]::Tuesday,
        [DayOfWeek]::Wednesday,
        [DayOfWeek]::Thursday,
        [DayOfWeek]::Friday
    )
    $openEt = Get-Date -Date $nowEt.Date -Hour 9 -Minute 30 -Second 0
    $closeEt = Get-Date -Date $nowEt.Date -Hour 16 -Minute 0 -Second 0
    $isOpen = $isWeekday -and ($nowEt -ge $openEt) -and ($nowEt -lt $closeEt)
    $minutesFromOpen = [int][Math]::Round(($nowEt - $openEt).TotalMinutes)
    return @{
        now_et = $nowEt
        is_weekday = $isWeekday
        open_et = $openEt
        close_et = $closeEt
        is_regular_open = $isOpen
        minutes_from_open = $minutesFromOpen
    }
}

if (-not (Test-Path $ProjectRoot)) {
    throw "ProjectRoot not found: $ProjectRoot"
}

$Market = $Market.Trim().ToUpperInvariant()

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $defaultPython = "C:\Users\heesu\anaconda3\envs\systematic-alpha\python.exe"
    if (Test-Path $defaultPython) {
        $PythonExe = $defaultPython
    } else {
        $PythonExe = "python"
    }
}

Set-Location $ProjectRoot

Import-DotEnv -Path (Join-Path $ProjectRoot ".env") -AllowedKeys @(
    "KIS_APP_KEY",
    "KIS_APP_SECRET",
    "KIS_ACC_NO",
    "KIS_USER_ID",
    "TELEGRAM_ENABLED",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "TELEGRAM_THREAD_ID",
    "TELEGRAM_DISABLE_NOTIFICATION"
)
Ensure-Tls12
Write-MonitorLog "[config] .env secrets loaded (whitelist). strategy/runtime params are sourced from run_daily.ps1 arguments."

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

if ($script:TelegramEnabled) {
    Write-MonitorLog "[notify] telegram enabled (chat_id=$script:TelegramChatId)."
} else {
    Write-MonitorLog "[notify] telegram disabled."
}

foreach ($name in @("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy")) {
    if (Test-Path "Env:$name") {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
}

# mojito token cache path uses "~/.cache/mojito2". Use project-local home for stability.
$env:HOME = $ProjectRoot
$env:USERPROFILE = $ProjectRoot

$null = New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "logs")
$null = New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "out")

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$marketTag = if ($Market -eq "US") { "us" } else { "kr" }
$script:RunLogFile = Join-Path $ProjectRoot ("logs\runner_{0}_{1}.log" -f $marketTag, $stamp)
$outputJson = Join-Path $ProjectRoot ("out\{0}_daily_{1}.json" -f $marketTag, $stamp)
$resolvedOvernightReportPath = if ([string]::IsNullOrWhiteSpace($OvernightReportPath)) {
    Join-Path $ProjectRoot "out\selection_overnight_report.csv"
} else {
    $OvernightReportPath
}
$baseArgs = @(
    "main.py",
    "--market", $Market.ToLowerInvariant(),
    "--collect-seconds", "$CollectSeconds",
    "--final-picks", "$FinalPicks",
    "--pre-candidates", "$PreCandidates",
    "--max-symbols-scan", "$MaxSymbolsScan",
    "--kr-universe-size", "$KrUniverseSize",
    "--us-universe-size", "$UsUniverseSize",
    "--min-change-pct", (To-InvariantString -Value $MinChangePct),
    "--min-gap-pct", (To-InvariantString -Value $MinGapPct),
    "--min-prev-turnover", (To-InvariantString -Value $MinPrevTurnover),
    "--min-strength", (To-InvariantString -Value $MinStrength),
    "--min-vol-ratio", (To-InvariantString -Value $MinVolRatio),
    "--min-bid-ask-ratio", (To-InvariantString -Value $MinBidAskRatio),
    "--min-pass-conditions", "$MinPassConditions",
    "--min-maintain-ratio", (To-InvariantString -Value $MinMaintainRatio),
    "--rest-sleep", (To-InvariantString -Value $RestSleepSec),
    "--min-exec-ticks", "$MinExecTicks",
    "--min-orderbook-ticks", "$MinOrderbookTicks",
    "--min-realtime-cum-volume", (To-InvariantString -Value $MinRealtimeCumVolume),
    "--min-realtime-coverage-ratio", (To-InvariantString -Value $MinRealtimeCoverageRatio),
    "--stage1-log-interval", "$Stage1LogInterval",
    "--realtime-log-interval", "$RealtimeLogInterval",
    "--overnight-report-path", $resolvedOvernightReportPath,
    "--output-json", $outputJson
)
if ($Market -eq "US") {
    if (-not [string]::IsNullOrWhiteSpace($UsExchange)) {
        $baseArgs += @("--exchange", $UsExchange)
    }
    $baseArgs += @("--us-poll-interval", (To-InvariantString -Value $UsPollInterval))
}
if (-not [string]::IsNullOrWhiteSpace($UniverseFile)) {
    $baseArgs += @("--universe-file", $UniverseFile)
}
if ($AssumeOpenForTest) {
    $baseArgs += @("--test-assume-open")
}
if ($AllowShortBias) {
    $baseArgs += @("--allow-short-bias")
} else {
    $baseArgs += @("--long-only")
}
if ($AllowLowCoverage) {
    $baseArgs += @("--allow-low-coverage")
} else {
    $baseArgs += @("--invalidate-on-low-coverage")
}
if ($Mock) {
    $baseArgs += @("--mock")
}

Write-MonitorLog "[run] monitor_log: $script:RunLogFile"
Write-MonitorLog (
    "[strategy] market=$Market, collect=$CollectSeconds, final_picks=$FinalPicks, pre_candidates=$PreCandidates, " +
    "max_scan=$MaxSymbolsScan, kr_pool=$KrUniverseSize, us_pool=$UsUniverseSize, " +
    "change=$MinChangePct, gap=$MinGapPct, prev_turnover=$MinPrevTurnover, " +
    "strength=$MinStrength, vol_ratio=$MinVolRatio, bid_ask_ratio=$MinBidAskRatio, " +
    "pass_conditions=$MinPassConditions, maintain_ratio=$MinMaintainRatio, " +
    "long_only=$(-not $AllowShortBias), allow_low_coverage=$AllowLowCoverage"
)

if ($Market -eq "US" -and $RequireUsOpen -and -not $AssumeOpenForTest) {
    $state = Get-UsMarketState
    $openWindow = [Math]::Max(1, $UsOpenWindowMinutes)
    $withinOpenWindow = $state.is_weekday -and ($state.minutes_from_open -ge 0) -and ($state.minutes_from_open -lt $openWindow)
    Write-MonitorLog (
        "[market-check] US now(et)={0:yyyy-MM-dd HH:mm:ss}, open={1:HH:mm}, close={2:HH:mm}, weekday={3}, regular_open={4}, minutes_from_open={5}, open_window={6}m, within_open_window={7}" -f `
            $state.now_et, $state.open_et, $state.close_et, $state.is_weekday, $state.is_regular_open, $state.minutes_from_open, $openWindow, $withinOpenWindow
    )
    if (-not $withinOpenWindow) {
        Write-MonitorLog "[market-check] outside US open window. skipping run."
        if ($script:TelegramEnabled) {
            Send-TelegramMessage (
                "[SystematicAlpha] skipped`n" +
                "reason=outside US open window`n" +
                ("now_et={0:yyyy-MM-dd HH:mm:ss}" -f $state.now_et) + "`n" +
                ("minutes_from_open={0}, window={1}" -f $state.minutes_from_open, $openWindow)
            )
        }
        exit 0
    }

    $usEtDate = $state.now_et.ToString("yyyyMMdd")
    $usDayLock = Join-Path $ProjectRoot ("out\us_run_lock_{0}.txt" -f $usEtDate)
    if (Test-Path $usDayLock) {
        Write-MonitorLog "[market-check] US daily lock exists ($usDayLock). skipping duplicate run."
        if ($script:TelegramEnabled) {
            Send-TelegramMessage (
                "[SystematicAlpha] skipped`n" +
                "reason=US daily lock exists`n" +
                ("lock={0}" -f $usDayLock)
            )
        }
        exit 0
    }

    $lockText = @(
        "created_at=" + (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
        "now_et=" + ($state.now_et.ToString("yyyy-MM-dd HH:mm:ss"))
        "market=US"
    ) -join "`n"
    Set-Content -Path $usDayLock -Value $lockText -Encoding UTF8
    Write-MonitorLog "[market-check] created US daily lock: $usDayLock"
}
elseif ($Market -eq "US" -and $RequireUsOpen -and $AssumeOpenForTest) {
    Write-MonitorLog "[market-check] RequireUsOpen bypassed by AssumeOpenForTest."
}

$hostTag = "$env:COMPUTERNAME/$env:USERNAME"
$exchangeLabel = if ($Market -eq "US") { $UsExchange } else { "KRX" }
if ($NotifyStart) {
    Send-TelegramMessage (
        "[SystematicAlpha] start`n" +
        "host=$hostTag`n" +
        "market=$Market`n" +
        "exchange=$exchangeLabel`n" +
        "assume_open_test=$AssumeOpenForTest`n" +
        "collect_seconds=$CollectSeconds`n" +
        "final_picks=$FinalPicks`n" +
        "max_attempts=$MaxAttempts"
    )
}

if ($StartDelaySeconds -gt 0) {
    Wait-WithProgress -TotalSeconds $StartDelaySeconds -Label "startup"
}

$delay = [Math]::Max(1, $RetryDelaySeconds)
$multiplier = [Math]::Max(1, $RetryBackoffMultiplier)
$maxDelay = [Math]::Max(1, $MaxRetryDelaySeconds)

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    $attemptLog = Join-Path $ProjectRoot ("logs\{0}_daily_{1}_try{2}.log" -f $marketTag, $stamp, $attempt)
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    Write-MonitorLog "[attempt $attempt/$MaxAttempts] starting"
    Write-MonitorLog "command: $PythonExe -u $($baseArgs -join ' ')"
    Write-MonitorLog "live_log: $attemptLog"

    $exitCode = 1
    $combinedOutput = @()

    try {
        $combinedOutputList = New-Object System.Collections.Generic.List[string]
        $prevErrorActionPreference = $ErrorActionPreference
        try {
            # Native stderr should be captured as log lines, not treated as terminating PowerShell errors.
            $ErrorActionPreference = "Continue"
            & $PythonExe -u @baseArgs 2>&1 | ForEach-Object {
                $line = "$_"
                $combinedOutputList.Add($line) | Out-Null
                $line | Tee-Object -FilePath $attemptLog -Append
            }
        } finally {
            $ErrorActionPreference = $prevErrorActionPreference
        }
        $combinedOutput = $combinedOutputList.ToArray()
        $nativeExit = $LASTEXITCODE
        if ($null -eq $nativeExit) {
            $exitCode = 0
        } else {
            $exitCode = [int]$nativeExit
        }
    } catch {
        $errText = $_ | Out-String
        $combinedOutput = @($errText)
        $combinedOutput | Tee-Object -FilePath $attemptLog
        Write-MonitorLog "[attempt $attempt/$MaxAttempts] pipeline exception: $($_.Exception.Message)"
        $exitCode = 9009
    }
    $stopwatch.Stop()
    Write-MonitorLog ("[attempt $attempt/$MaxAttempts] finished (exit=$exitCode, elapsed={0:n1}s)" -f $stopwatch.Elapsed.TotalSeconds)

    if ($exitCode -eq 0) {
        Write-MonitorLog "[success] completed on attempt $attempt."
        $summary = Get-SelectionSummary -JsonPath $outputJson -TopN $FinalPicks
        Send-TelegramMessage (
            "[SystematicAlpha] success`n" +
            "host=$hostTag`n" +
            "attempt=$attempt/$MaxAttempts`n" +
            "output=$outputJson`n" +
            "log=$attemptLog`n" +
            "`n$summary"
        )
        exit 0
    }

    if ($attempt -ge $MaxAttempts) {
        Write-MonitorLog "[failed] all attempts exhausted. last_exit_code=$exitCode"
        $tail = Get-OutputTail -Lines $combinedOutput -TailLines $NotifyTailLines
        Send-TelegramMessage (
            "[SystematicAlpha] failed`n" +
            "host=$hostTag`n" +
            "attempt=$attempt/$MaxAttempts`n" +
            "exit_code=$exitCode`n" +
            "last_log=$attemptLog`n" +
            "`n--- log tail ---`n" +
            (Truncate-Text -Text $tail -MaxChars 2000)
        )
        exit $exitCode
    }

    $rawText = ($combinedOutput | Out-String)
    $nextDelay = $delay
    if ($rawText -match "EGW00133") {
        # KIS token endpoint rate limit: at most once per minute.
        $nextDelay = [Math]::Max($nextDelay, 65)
    }

    Write-MonitorLog "[retry] attempt failed (exit=$exitCode). waiting $nextDelay sec."
    $retryTail = Get-OutputTail -Lines $combinedOutput -TailLines $NotifyTailLines
    Send-TelegramMessage (
        "[SystematicAlpha] retry`n" +
        "host=$hostTag`n" +
        "attempt=$attempt/$MaxAttempts`n" +
        "exit_code=$exitCode`n" +
        "next_delay=${nextDelay}s`n" +
        "log=$attemptLog`n" +
        "`n--- log tail ---`n" +
        (Truncate-Text -Text $retryTail -MaxChars 1500)
    )
    Wait-WithProgress -TotalSeconds $nextDelay -Label "retry-delay"
    $delay = [Math]::Min($delay * $multiplier, $maxDelay)
}

