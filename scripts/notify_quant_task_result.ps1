param(
    [Parameter(Mandatory=$true)][string]$TaskName,
    [Parameter(Mandatory=$true)][string]$Status,
    [Parameter(Mandatory=$true)][string]$Message,
    [string]$DetailPath = "",
    [string]$LogPath = ""
)

$ErrorActionPreference = "Continue"
$ProjectRoot = "C:\dev\quant"
$LogDir = Join-Path $ProjectRoot "logs\quant_automation"
$OutDir = Join-Path $ProjectRoot "output\automation_reliability_v1"
$LatestText = Join-Path $LogDir "latest_notification.txt"
$HistoryLog = Join-Path $LogDir "notification_history.log"
$LatestJson = Join-Path $OutDir "latest_task_notification.json"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$LocalTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$PopupAttempted = $true
$PopupSuccess = $false
$PopupError = ""
$ShortMessage = "[$Status] $TaskName - $Message"

try {
    $msgCmd = Get-Command msg.exe -ErrorAction SilentlyContinue
    if ($null -ne $msgCmd) {
        & msg.exe $env:USERNAME /TIME:20 $ShortMessage 2>$null
        if ($LASTEXITCODE -eq 0) {
            $PopupSuccess = $true
        } else {
            $PopupError = "msg.exe exit_code=$LASTEXITCODE"
        }
    } else {
        $shell = New-Object -ComObject WScript.Shell
        $null = $shell.Popup($ShortMessage, 20, "Quant Automation", 0x40)
        $PopupSuccess = $true
    }
} catch {
    $PopupError = $_.Exception.Message
}

$Payload = [ordered]@{
    task_name = $TaskName
    status = $Status
    message = $Message
    local_time = $LocalTime
    detail_path = $DetailPath
    log_path = $LogPath
    popup_attempted = $PopupAttempted
    popup_success = $PopupSuccess
    popup_error = $PopupError
}

$Text = @"
task_name: $TaskName
status: $Status
message: $Message
local_time: $LocalTime
detail_path: $DetailPath
log_path: $LogPath
popup_attempted: $PopupAttempted
popup_success: $PopupSuccess
popup_error: $PopupError
"@

Set-Content -LiteralPath $LatestText -Value $Text -Encoding UTF8
Add-Content -LiteralPath $HistoryLog -Value ($Payload | ConvertTo-Json -Compress) -Encoding UTF8
$Payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $LatestJson -Encoding UTF8

Write-Host $Text
exit 0
