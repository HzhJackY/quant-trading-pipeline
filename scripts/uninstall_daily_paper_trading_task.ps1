param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$TaskName = "QuantDailyPaperTradingMonitor"

if (-not $Force) {
    $Answer = Read-Host "Delete scheduled task '$TaskName'? Type YES to confirm"
    if ($Answer -ne "YES") {
        Write-Host "Cancelled."
        exit 0
    }
}

$Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -eq $Existing) {
    Write-Host "Task not found: $TaskName"
    exit 0
}

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
Write-Host "Deleted task: $TaskName"
