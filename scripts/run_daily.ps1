param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonExe = "",
    [int]$CollectSeconds = 600,
    [int]$FinalPicks = 3,
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
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return
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

if (-not (Test-Path $ProjectRoot)) {
    throw "ProjectRoot not found: $ProjectRoot"
}

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $defaultPython = "C:\Users\heesu\anaconda3\envs\systematic-alpha\python.exe"
    if (Test-Path $defaultPython) {
        $PythonExe = $defaultPython
    } else {
        $PythonExe = "python"
    }
}

Set-Location $ProjectRoot

Import-DotEnv -Path (Join-Path $ProjectRoot ".env")
Ensure-Tls12

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
$script:RunLogFile = Join-Path $ProjectRoot ("logs\runner_{0}.log" -f $stamp)
$outputJson = Join-Path $ProjectRoot ("out\daily_{0}.json" -f $stamp)
$baseArgs = @(
    "main.py",
    "--collect-seconds", "$CollectSeconds",
    "--final-picks", "$FinalPicks",
    "--output-json", $outputJson
)

Write-MonitorLog "[run] monitor_log: $script:RunLogFile"

$hostTag = "$env:COMPUTERNAME/$env:USERNAME"
if ($NotifyStart) {
    Send-TelegramMessage (
        "[SystematicAlpha] start`n" +
        "host=$hostTag`n" +
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
    $attemptLog = Join-Path $ProjectRoot ("logs\daily_{0}_try{1}.log" -f $stamp, $attempt)
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    Write-MonitorLog "[attempt $attempt/$MaxAttempts] starting"
    Write-MonitorLog "command: $PythonExe -u $($baseArgs -join ' ')"
    Write-MonitorLog "live_log: $attemptLog"

    $exitCode = 1
    $combinedOutput = @()

    try {
        $combinedOutputList = New-Object System.Collections.Generic.List[string]
        & $PythonExe -u @baseArgs 2>&1 | ForEach-Object {
            $line = "$_"
            $combinedOutputList.Add($line) | Out-Null
            $line | Tee-Object -FilePath $attemptLog -Append
        }
        $combinedOutput = $combinedOutputList.ToArray()
        $nativeExit = $LASTEXITCODE
        if ($null -eq $nativeExit) {
            $exitCode = 0
        } else {
            $exitCode = [int]$nativeExit
        }
    } catch {
        $combinedOutput = @($_.Exception.Message)
        $combinedOutput | Tee-Object -FilePath $attemptLog
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
