param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$TaskName = "QuantBlendV3ShadowPriceRefresh"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BatPath = Join-Path $ProjectRoot "scripts\run_blend_v3_shadow_price_refresh.bat"

if (-not (Test-Path -LiteralPath $BatPath)) {
    throw "BAT not found: $BatPath"
}

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing -and -not $Force) {
    Write-Host "Task already exists: $TaskName"
    Write-Host "Use -Force to replace it."
    exit 1
}
if ($existing -and $Force) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$BatPath`"" -WorkingDirectory $ProjectRoot
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 18:05
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Minutes 10)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal | Out-Null

Write-Host "Installed task: $TaskName"
Write-Host "Schedule: Monday-Friday 18:05 local time"
Write-Host "Runs before QuantBlendV3ShadowDailyMonitor 18:25"
Write-Host "Manual trigger:"
Write-Host "  Start-ScheduledTask -TaskName $TaskName"
Write-Host "Manual run without scheduler:"
Write-Host "  cmd /c scripts\run_blend_v3_shadow_price_refresh.bat"
