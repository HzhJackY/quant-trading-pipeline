param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$TaskName = "QuantPaperTradingDailyMonitor"
$ProjectRoot = "C:\dev\quant"
$BatPath = Join-Path $ProjectRoot "scripts\run_paper_trading_monitor_with_notification.bat"
$SafeMonitor = Join-Path $ProjectRoot "monitoring\project_daily_monitor.py"
$RunTime = "22:15"

if (-not (Test-Path -LiteralPath $SafeMonitor)) {
    throw "Safe paper trading monitor entry not found: $SafeMonitor"
}
if (-not (Test-Path -LiteralPath $BatPath)) {
    throw "Wrapper BAT not found: $BatPath"
}

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing -and -not $Force) {
    Write-Host "Task already exists. Re-run with -Force to replace the same-name task."
    Write-Host "manual_trigger: schtasks /Run /TN $TaskName"
    exit 0
}
if ($existing -and $Force) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$Action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$BatPath`"" -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $RunTime
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 10) `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 75) `
    -MultipleInstances IgnoreNew
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Description "Paper trading monitor wrapper with notification. No real orders." | Out-Null

Write-Host "Task registered successfully: $TaskName"
Write-Host "Schedule: Monday-Friday $RunTime local computer time"
Write-Host "Execution time limit: 75 minutes"
Write-Host "Action: cmd.exe /c `"$BatPath`""
Write-Host "Start in: $ProjectRoot"
Write-Host "manual_trigger: schtasks /Run /TN $TaskName"
