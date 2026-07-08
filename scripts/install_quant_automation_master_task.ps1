param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$TaskName = "QuantAutomationMaster2200"
$ProjectRoot = "C:\dev\quant"
$BatPath = Join-Path $ProjectRoot "scripts\run_quant_automation_master_2200.bat"
$RunTime = "22:00"

if (-not (Test-Path -LiteralPath $BatPath)) {
    throw "Master BAT not found: $BatPath"
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
    -ExecutionTimeLimit (New-TimeSpan -Minutes 60) `
    -MultipleInstances IgnoreNew
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Description "Unified quant automation master. Consider disabling separate tasks manually to avoid duplicate runs." | Out-Null

Write-Host "Task registered successfully: $TaskName"
Write-Host "Schedule: Monday-Friday $RunTime local computer time"
Write-Host "Execution time limit: 60 minutes"
Write-Host "Action: cmd.exe /c `"$BatPath`""
Write-Host "Start in: $ProjectRoot"
Write-Host "manual_trigger: schtasks /Run /TN $TaskName"
Write-Host "Note: If using this master task, consider manually disabling separate shadow/paper tasks later to avoid duplicate runs."
