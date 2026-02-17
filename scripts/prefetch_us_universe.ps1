param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonExe = "",
    [string]$UsExchange = "NASD",
    [int]$UsUniverseSize = 500,
    [int]$MaxSymbolsScan = 500,
    [switch]$ForceRefresh = $false,
    [switch]$RequireUsPrefetchWindow = $true,
    [int]$UsPrefetchLeadMinutes = 60,
    [int]$UsPrefetchWindowMinutes = 45,
    [int]$PrefetchMinSuccessCount = 20,
    [double]$PrefetchMinSuccessRatio = 0.2,
    [switch]$AssumeOpenForTest = $false,
    [switch]$NotifySkips = $false
)

$ErrorActionPreference = "Stop"
$script:LogFile = $null
$script:TelegramEnabled = $false
$script:TelegramBotToken = ""
$script:TelegramChatId = ""
$script:TelegramThreadId = ""
$script:TelegramDisableNotification = $false

function Ensure-Tls12 {
    try {
        $current = [Net.ServicePointManager]::SecurityProtocol
        if (($current -band [Net.SecurityProtocolType]::Tls12) -eq 0) {
            [Net.ServicePointManager]::SecurityProtocol = $current -bor [Net.SecurityProtocolType]::Tls12
        }
    } catch {
        # ignore
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

function Write-LogLine {
    param([string]$Message)
    Write-Output $Message
    if (-not [string]::IsNullOrWhiteSpace($script:LogFile)) {
        try {
            $Message | Out-File -FilePath $script:LogFile -Encoding UTF8 -Append
        } catch {
            # ignore
        }
    }
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
        text = $Text
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
            Write-LogLine "[telegram] api error: code=$errCode desc=$errDesc"
            if ($errCode -eq 401) {
                Write-LogLine "[telegram] token unauthorized. disabling telegram notifications for this run."
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
            Write-LogLine "[telegram] send failed: HTTP 401 Unauthorized. disabling telegram notifications."
            $script:TelegramEnabled = $false
            return
        }
        Write-LogLine "[telegram] send failed: $($_.Exception.Message)"
        if (Send-TelegramViaPython -Text $Text) {
            Write-LogLine "[telegram] fallback send succeeded (python requests)."
            return
        }
    }
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
    $minutesFromOpen = [int][Math]::Round(($nowEt - $openEt).TotalMinutes)
    return @{
        now_et = $nowEt
        is_weekday = $isWeekday
        open_et = $openEt
        minutes_from_open = $minutesFromOpen
    }
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

$kstTz = [System.TimeZoneInfo]::FindSystemTimeZoneById("Korea Standard Time")
$nowKst = [System.TimeZoneInfo]::ConvertTime((Get-Date), $kstTz)
$runDate = $nowKst.ToString("yyyyMMdd")
$stamp = $nowKst.ToString("yyyyMMdd_HHmmss")

$logMarketDir = Join-Path (Join-Path $ProjectRoot "logs") "us"
$logDir = Join-Path $logMarketDir $runDate
$null = New-Item -ItemType Directory -Force -Path $logDir
$null = New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "out")
$runOutMarketDir = Join-Path (Join-Path (Join-Path $ProjectRoot "out") "us") $runDate
$runtimeDir = Join-Path $runOutMarketDir "runtime"
$null = New-Item -ItemType Directory -Force -Path $runtimeDir

foreach ($name in @("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy")) {
    if (Test-Path "Env:$name") {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
}

# mojito token cache path uses "~/.cache/mojito2". Use project-local home for stability.
$env:HOME = $ProjectRoot
$env:USERPROFILE = $ProjectRoot
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$script:LogFile = Join-Path $logDir ("prefetch_us_{0}.log" -f $stamp)
$prefetchMinSuccessRatioText = $PrefetchMinSuccessRatio.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$universeArgs = @(
    "scripts\prefetch_us_universe.py",
    "--project-root", $ProjectRoot
)
$cacheArgs = @(
    "scripts\prefetch_us_market_cache.py",
    "--project-root", $ProjectRoot,
    "--us-exchange", $UsExchange,
    "--us-universe-size", "$UsUniverseSize",
    "--max-symbols-scan", "$MaxSymbolsScan",
    "--min-success-count", "$PrefetchMinSuccessCount",
    "--min-success-ratio", "$prefetchMinSuccessRatioText"
)
if ($ForceRefresh) {
    $cacheArgs += "--force-refresh"
}

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

$prefetchLockPath = ""
$prefetchLockAcquired = $false
if ($RequireUsPrefetchWindow -and -not $AssumeOpenForTest) {
    $state = Get-UsMarketState
    $leadMinutes = [Math]::Abs($UsPrefetchLeadMinutes)
    $windowMinutes = [Math]::Max(1, $UsPrefetchWindowMinutes)
    $targetMinutesFromOpen = -$leadMinutes
    $deltaMinutes = [Math]::Abs($state.minutes_from_open - $targetMinutesFromOpen)
    $withinWindow = $state.is_weekday -and ($deltaMinutes -lt $windowMinutes)
    Write-LogLine (
        "[prefetch-check] now(et)={0:yyyy-MM-dd HH:mm:ss}, weekday={1}, minutes_from_open={2}, target={3}, " +
        "window=+/-{4}, within_window={5}" -f `
            $state.now_et, $state.is_weekday, $state.minutes_from_open, $targetMinutesFromOpen, $windowMinutes, $withinWindow
    )
    if (-not $withinWindow) {
        Write-LogLine "[prefetch-check] outside prefetch window. skipping."
        if ($script:TelegramEnabled -and $NotifySkips) {
            Send-TelegramMessage (
                "[SystematicAlpha] skipped`n" +
                "reason=outside US prefetch window`n" +
                ("now_et={0:yyyy-MM-dd HH:mm:ss}" -f $state.now_et) + "`n" +
                ("minutes_from_open={0}, target={1}, tolerance=+/-{2}" -f $state.minutes_from_open, $targetMinutesFromOpen, $windowMinutes)
            )
        }
        exit 0
    }

    $etDate = $state.now_et.ToString("yyyyMMdd")
    $prefetchLockPath = Join-Path $runtimeDir ("us_prefetch_lock_{0}.txt" -f $etDate)
    if (Test-Path $prefetchLockPath) {
        Write-LogLine "[prefetch-check] prefetch lock exists ($prefetchLockPath). skipping duplicate run."
        if ($script:TelegramEnabled -and $NotifySkips) {
            Send-TelegramMessage (
                "[SystematicAlpha] skipped`n" +
                "reason=US prefetch lock exists`n" +
                ("lock={0}" -f $prefetchLockPath)
            )
        }
        exit 0
    }

    $lockText = @(
        "created_at=" + (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
        "now_et=" + ($state.now_et.ToString("yyyy-MM-dd HH:mm:ss"))
        "market=US"
    ) -join "`n"
    Set-Content -Path $prefetchLockPath -Value $lockText -Encoding UTF8
    $prefetchLockAcquired = $true
    Write-LogLine "[prefetch-check] created US prefetch lock: $prefetchLockPath"
}
elseif ($RequireUsPrefetchWindow -and $AssumeOpenForTest) {
    Write-LogLine "[prefetch-check] prefetch window guard bypassed by AssumeOpenForTest."
}

Write-LogLine "[prefetch] command(universe): $PythonExe $($universeArgs -join ' ')"
Write-LogLine "[prefetch] command(market-cache): $PythonExe $($cacheArgs -join ' ')"
Write-LogLine "[prefetch] log: $script:LogFile"
Send-TelegramMessage (
    "[SystematicAlpha] prefetch-start`n" +
    "market=US`n" +
    "step=universe+prev_stats`n" +
    "us_exchange=$UsExchange`n" +
    "us_universe_size=$UsUniverseSize`n" +
    "max_symbols_scan=$MaxSymbolsScan`n" +
    "log=$script:LogFile"
)

try {
    $prevErrorActionPreference = $ErrorActionPreference
    try {
        # Treat native stderr as stream output (for logging), not as terminating PowerShell errors.
        $ErrorActionPreference = "Continue"

        Send-TelegramMessage (
            "[SystematicAlpha] prefetch-stage`n" +
            "market=US`n" +
            "stage=1/2 universe"
        )
        & $PythonExe @universeArgs 2>&1 | ForEach-Object {
            $line = "$_"
            Write-LogLine $line
        }
        $firstExit = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
        if ($firstExit -ne 0) {
            $exitCode = $firstExit
            throw "prefetch_us_universe.py failed (exit=$firstExit)"
        }
        Send-TelegramMessage (
            "[SystematicAlpha] prefetch-stage`n" +
            "market=US`n" +
            "stage=1/2 done"
        )

        Send-TelegramMessage (
            "[SystematicAlpha] prefetch-stage`n" +
            "market=US`n" +
            "stage=2/2 prev-day-cache"
        )
        & $PythonExe @cacheArgs 2>&1 | ForEach-Object {
            $line = "$_"
            Write-LogLine $line
        }
    } finally {
        $ErrorActionPreference = $prevErrorActionPreference
    }
    $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
} catch {
    Write-LogLine "$($_.Exception.Message)"
    if ($null -eq $LASTEXITCODE) {
        $exitCode = 1
    } else {
        $exitCode = [int]$LASTEXITCODE
    }
}

Write-LogLine "[prefetch] finished (exit=$exitCode)"
if ($exitCode -ne 0 -and $prefetchLockAcquired -and -not [string]::IsNullOrWhiteSpace($prefetchLockPath)) {
    if (Test-Path $prefetchLockPath) {
        Remove-Item -Path $prefetchLockPath -Force -ErrorAction SilentlyContinue
        Write-LogLine "[prefetch-check] removed prefetch lock due failure: $prefetchLockPath"
    }
}
if ($exitCode -eq 0) {
    Send-TelegramMessage (
        "[SystematicAlpha] prefetch-success`n" +
        "market=US`n" +
        "log=$script:LogFile"
    )
} else {
    Send-TelegramMessage (
        "[SystematicAlpha] prefetch-failed`n" +
        "market=US`n" +
        "exit_code=$exitCode`n" +
        "log=$script:LogFile"
    )
}
exit $exitCode
