$ErrorActionPreference = "Stop"
$TaskName = "QuantBlendV3ShadowPriceRefresh"
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Uninstalled task: $TaskName"
} else {
    Write-Host "Task not found: $TaskName"
}
