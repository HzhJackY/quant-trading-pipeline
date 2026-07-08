$ErrorActionPreference = "Continue"

$TaskName = "QuantBlendV3ShadowDailyMonitor"
$PriceRefreshTaskName = "QuantBlendV3ShadowPriceRefresh"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BatPath = Join-Path $ProjectRoot "scripts\run_blend_v3_shadow_live_update.bat"
$PriceRefreshBatPath = Join-Path $ProjectRoot "scripts\run_blend_v3_shadow_price_refresh.bat"
$HoldingsPath = Join-Path $ProjectRoot "output\blend_v3_shadow_live\latest_shadow_holdings_live.csv"
$StatusPath = Join-Path $ProjectRoot "output\blend_v3_shadow_monitoring\shadow_monitor_latest_status.json"
$NavPath = Join-Path $ProjectRoot "output\blend_v3_shadow_monitoring\shadow_daily_nav.csv"
$PriceCachePath = Join-Path $ProjectRoot "output\blend_v3_shadow_monitoring\price_cache\shadow_daily_prices.csv"
$PriceRefreshStatusPath = Join-Path $ProjectRoot "output\blend_v3_shadow_monitoring\price_cache\shadow_price_refresh_status.json"
$LogPath = Join-Path $ProjectRoot "logs\blend_v3_shadow\shadow_live_update.log"
$OutPath = Join-Path $ProjectRoot "output\blend_v3_shadow_monitoring\automation\shadow_daily_status_check.md"

New-Item -ItemType Directory -Force -Path (Split-Path $OutPath) | Out-Null

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
$info = if ($task) { Get-ScheduledTaskInfo -TaskName $TaskName -ErrorAction SilentlyContinue } else { $null }
$priceTask = Get-ScheduledTask -TaskName $PriceRefreshTaskName -ErrorAction SilentlyContinue
$priceInfo = if ($priceTask) { Get-ScheduledTaskInfo -TaskName $PriceRefreshTaskName -ErrorAction SilentlyContinue } else { $null }
$statusJson = if (Test-Path -LiteralPath $StatusPath) { Get-Content -LiteralPath $StatusPath -Raw } else { "" }
$status = if ($statusJson) { $statusJson | ConvertFrom-Json } else { $null }
$priceRefreshJson = if (Test-Path -LiteralPath $PriceRefreshStatusPath) { Get-Content -LiteralPath $PriceRefreshStatusPath -Raw } else { "" }
$priceRefreshStatus = if ($priceRefreshJson) { $priceRefreshJson | ConvertFrom-Json } else { $null }
$priceCacheLatestDate = "n/a"
if (Test-Path -LiteralPath $PriceCachePath) {
    try {
        $priceCacheLatestDate = (Import-Csv -LiteralPath $PriceCachePath | Sort-Object date -Descending | Select-Object -First 1).date
    } catch {
        $priceCacheLatestDate = "read_error: $($_.Exception.Message)"
    }
}
$logTail = if (Test-Path -LiteralPath $LogPath) { Get-Content -LiteralPath $LogPath -Tail 40 } else { @("log file missing: $LogPath") }

$lines = @()
$lines += "# Blend V3 Shadow Daily Status Check"
$lines += ""
$lines += "- task_name: $TaskName"
$lines += "- task_exists: $([bool]$task)"
$lines += "- task_state: $(if ($task) { $task.State } else { 'missing' })"
$lines += "- last_run_time: $(if ($info) { $info.LastRunTime } else { 'n/a' })"
$lines += "- last_task_result: $(if ($info) { $info.LastTaskResult } else { 'n/a' })"
$lines += "- next_run_time: $(if ($info) { $info.NextRunTime } else { 'n/a' })"
$lines += "- bat_exists: $(Test-Path -LiteralPath $BatPath)"
$lines += "- price_refresh_bat_exists: $(Test-Path -LiteralPath $PriceRefreshBatPath)"
$lines += "- latest_shadow_holdings_exists: $(Test-Path -LiteralPath $HoldingsPath)"
$lines += "- shadow_status_exists: $(Test-Path -LiteralPath $StatusPath)"
$lines += "- shadow_daily_nav_exists: $(Test-Path -LiteralPath $NavPath)"
$lines += "- price_refresh_task_exists: $([bool]$priceTask)"
$lines += "- price_refresh_last_run_time: $(if ($priceInfo) { $priceInfo.LastRunTime } else { 'n/a' })"
$lines += "- price_refresh_last_task_result: $(if ($priceInfo) { $priceInfo.LastTaskResult } else { 'n/a' })"
$lines += "- shadow_price_cache_exists: $(Test-Path -LiteralPath $PriceCachePath)"
$lines += "- shadow_price_cache_latest_date: $priceCacheLatestDate"
$lines += "- shadow_price_refresh_decision: $(if ($priceRefreshStatus) { $priceRefreshStatus.decision } else { 'n/a' })"
$lines += "- current_run_date: $(if ($status) { $status.current_run_date } else { 'n/a' })"
$lines += "- latest_feature_month: $(if ($status) { $status.latest_feature_month } else { 'n/a' })"
$lines += "- latest_price_date: $(if ($status) { $status.latest_price_date } else { 'n/a' })"
$lines += "- latest_nav_date: $(if ($status) { $status.latest_nav_date } else { 'n/a' })"
$lines += "- stale_price_warning: $(if ($status) { $status.stale_price_warning } else { 'n/a' })"
$lines += "- stale_price_days: $(if ($status) { $status.stale_price_days } else { 'n/a' })"
$lines += "- nav_update_blocked_by_stale_price: $(if ($status) { $status.nav_update_blocked_by_stale_price } else { 'n/a' })"
$lines += "- price_source: $(if ($status) { $status.price_source } else { 'n/a' })"
$lines += "- decision: $(if ($status) { $status.decision } else { 'n/a' })"
$lines += ""
$lines += "## Latest Shadow Status"
$lines += "~~~json"
$lines += $statusJson
$lines += "~~~"
$lines += ""
$lines += "## Log Tail"
$lines += "~~~text"
$lines += $logTail
$lines += "~~~"

$lines | Set-Content -LiteralPath $OutPath -Encoding UTF8
$lines | ForEach-Object { Write-Host $_ }
