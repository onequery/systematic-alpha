param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonExe = "",
    [int]$KrUniverseSize = 500,
    [int]$MaxSymbolsScan = 500,
    [switch]$ForceRefresh = $false
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

$logMarketDir = Join-Path (Join-Path $ProjectRoot "logs") "kr"
$logDir = Join-Path $logMarketDir $runDate
$null = New-Item -ItemType Directory -Force -Path $logDir
$null = New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "out")

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

$script:LogFile = Join-Path $logDir ("prefetch_kr_{0}.log" -f $stamp)
$argsList = @(
    "scripts\prefetch_kr_universe.py",
    "--project-root", $ProjectRoot,
    "--kr-universe-size", "$KrUniverseSize",
    "--max-symbols-scan", "$MaxSymbolsScan"
)
if ($ForceRefresh) {
    $argsList += "--force-refresh"
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

Write-LogLine "[prefetch-kr] command: $PythonExe $($argsList -join ' ')"
Write-LogLine "[prefetch-kr] log: $script:LogFile"
Send-TelegramMessage (
    "[SystematicAlpha] prefetch-start`n" +
    "market=KR`n" +
    "step=universe+prev_stats`n" +
    "kr_universe_size=$KrUniverseSize`n" +
    "max_symbols_scan=$MaxSymbolsScan`n" +
    "log=$script:LogFile"
)

try {
    $prevErrorActionPreference = $ErrorActionPreference
    $stageNotified = @{}
    try {
        # Treat native stderr as stream output (for logging), not as terminating PowerShell errors.
        $ErrorActionPreference = "Continue"
        & $PythonExe @argsList 2>&1 | ForEach-Object {
            $line = "$_"
            Write-LogLine $line

            if ($line -match "^\[prefetch-kr\] start" -and -not $stageNotified.ContainsKey("start")) {
                $stageNotified["start"] = $true
                Send-TelegramMessage (
                    "[SystematicAlpha] prefetch-stage`n" +
                    "market=KR`n" +
                    "stage=start`n" +
                    "event=$line"
                )
            }
            if ($line -match "^\[prefetch-kr\] prev-day cache" -and -not $stageNotified.ContainsKey("prev")) {
                $stageNotified["prev"] = $true
                Send-TelegramMessage (
                    "[SystematicAlpha] prefetch-stage`n" +
                    "market=KR`n" +
                    "stage=prev-day-cache`n" +
                    "event=$line"
                )
            }
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

Write-LogLine "[prefetch-kr] finished (exit=$exitCode)"
if ($exitCode -eq 0) {
    Send-TelegramMessage (
        "[SystematicAlpha] prefetch-success`n" +
        "market=KR`n" +
        "log=$script:LogFile"
    )
} else {
    Send-TelegramMessage (
        "[SystematicAlpha] prefetch-failed`n" +
        "market=KR`n" +
        "exit_code=$exitCode`n" +
        "log=$script:LogFile"
    )
}
exit $exitCode
