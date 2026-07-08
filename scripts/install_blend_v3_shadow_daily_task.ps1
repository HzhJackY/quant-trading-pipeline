param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$TaskName = "QuantBlendV3ShadowDailyMonitor"
$ProjectRoot = "C:\Users\HzhJa\Desktop\quant"
$BatPath = Join-Path $ProjectRoot "scripts\run_blend_v3_shadow_live_update.bat"
$StatusCommand = "powershell -ExecutionPolicy Bypass -File scripts\check_blend_v3_shadow_daily_status.ps1"

Write-Host "============================================================"
Write-Host "Blend V3 Shadow Daily Task Installer"
Write-Host "============================================================"
Write-Host "task_name: $TaskName"
Write-Host "project_root: $ProjectRoot"
Write-Host "bat_path: $BatPath"
Write-Host "schedule: Monday-Friday 18:25 local computer time"
Write-Host "note: If this PC is not using China local time, manually adjust the scheduled time."
Write-Host "------------------------------------------------------------"

if (-not (Test-Path -LiteralPath $BatPath)) {
    throw "BAT entry not found: $BatPath"
}

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing -and -not $Force) {
    Write-Host "Task already exists. Re-run with -Force to replace it."
    Write-Host "manual_run_command: schtasks /Run /TN `"$TaskName`""
    Write-Host "status_check_command: $StatusCommand"
    exit 0
}

if ($existing -and $Force) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Existing task removed because -Force was supplied."
}

$action = New-ScheduledTaskAction `
    -Execute $BatPath `
    -WorkingDirectory $ProjectRoot

$trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At 18:25

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -StartWhenAvailable `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Minutes 10) `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 60) `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Shadow-only Blend V3 daily monitor. Not production. Not trading instruction." | Out-Null

Write-Host "Task registered successfully."
Write-Host "manual_run_command: schtasks /Run /TN `"$TaskName`""
Write-Host "status_check_command: $StatusCommand"
Write-Host "dashboard_command: streamlit run monitoring\blend_v3_shadow_report.py"
Write-Host "============================================================"
