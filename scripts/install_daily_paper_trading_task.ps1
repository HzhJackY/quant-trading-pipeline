param()

$ErrorActionPreference = "Stop"

$TaskName = "QuantDailyPaperTradingMonitor"
$ProjectRoot = "C:\dev\quant"
$BatPath = Join-Path $ProjectRoot "scripts\run_daily_paper_trading_monitor.bat"
$RunTime = "22:15"

# The default schedule is Monday-Friday 22:15 in this computer's local time.
# If this computer is in a US time zone but you want A-share after-close timing,
# manually change $RunTime to the corresponding local computer time before running.

if (-not (Test-Path -LiteralPath $BatPath)) {
    throw "Batch file not found: $BatPath"
}

$Action = New-ScheduledTaskAction `
    -Execute $BatPath `
    -WorkingDirectory $ProjectRoot

$Trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At $RunTime

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 60) `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Minutes 10)

$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Force | Out-Null

Write-Host "Task name: $TaskName"
Write-Host "Run time: Monday-Friday $RunTime local computer time"
Write-Host "Command: $BatPath"
Write-Host "Manual test command: python monitoring\project_daily_monitor.py --dry-run"
Write-Host "View task status: schtasks /Query /TN $TaskName /V /FO LIST"
Write-Host "Manual trigger: schtasks /Run /TN $TaskName"
