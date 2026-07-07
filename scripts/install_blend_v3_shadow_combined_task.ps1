param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$TaskName = "QuantBlendV3ShadowCombinedDaily"
$ProjectRoot = "C:\dev\quant"
$BatPath = Join-Path $ProjectRoot "scripts\run_blend_v3_shadow_refresh_then_monitor.bat"
$RunTime = "22:00"

Write-Host "============================================================"
Write-Host "Blend V3 Shadow Combined Task Installer"
Write-Host "============================================================"
Write-Host "task_name: $TaskName"
Write-Host "schedule: Monday-Friday $RunTime local computer time"
Write-Host "action: cmd.exe /c `"$BatPath`""
Write-Host "start_in: $ProjectRoot"

if (-not (Test-Path -LiteralPath $BatPath)) {
    throw "BAT not found: $BatPath"
}

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing -and -not $Force) {
    Write-Host "Task already exists. Re-run with -Force to replace it."
    Write-Host "No old task was deleted or disabled."
    Write-Host "manual_trigger: schtasks /Run /TN $TaskName"
    exit 0
}

if ($existing -and $Force) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Existing same-name task replaced because -Force was supplied."
}

$Action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$BatPath`"" -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $RunTime
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 10) `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 45) `
    -MultipleInstances IgnoreNew
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Shadow-only Blend V3 combined price refresh and monitor. Not production. Not trading instruction." | Out-Null

Write-Host "Task registered successfully."
Write-Host "manual_trigger: schtasks /Run /TN $TaskName"
Write-Host "status_check: powershell -ExecutionPolicy Bypass -File scripts\check_quant_automation_status.ps1"
Write-Host "============================================================"
