param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$TaskName = "QuantBlendV3ShadowDailyMonitor"

Write-Host "============================================================"
Write-Host "Blend V3 Shadow Daily Task Uninstaller"
Write-Host "============================================================"
Write-Host "task_name: $TaskName"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $existing) {
    Write-Host "Task not found. Nothing to remove."
    exit 0
}

if (-not $Force) {
    $answer = Read-Host "Delete scheduled task '$TaskName'? Type YES to confirm"
    if ($answer -ne "YES") {
        Write-Host "Cancelled."
        exit 0
    }
}

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
$remaining = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($remaining) {
    Write-Host "Removal attempted, but task still exists."
    exit 1
}

Write-Host "Task removed successfully."
Write-Host "============================================================"
