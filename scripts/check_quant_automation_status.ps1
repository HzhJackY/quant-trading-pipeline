$ErrorActionPreference = "Continue"
$ProjectRoot = "C:\dev\quant"
$Paths = [ordered]@{
    "统一自动化状态" = Join-Path $ProjectRoot "output\automation_reliability_v1\latest_automation_status.json"
    "最新通知" = Join-Path $ProjectRoot "logs\quant_automation\latest_notification.txt"
    "Shadow price refresh 状态" = Join-Path $ProjectRoot "output\blend_v3_shadow_monitoring\price_cache\shadow_price_refresh_status.json"
    "Shadow monitor 状态" = Join-Path $ProjectRoot "output\blend_v3_shadow_monitoring\shadow_monitor_latest_status.json"
    "Paper trading monitor 状态" = Join-Path $ProjectRoot "output\project_monitoring\latest_status.json"
}
$Tasks = @(
    "QuantBlendV3ShadowCombinedDaily",
    "QuantBlendV3ShadowPriceRefresh",
    "QuantBlendV3ShadowDailyMonitor",
    "QuantDailyPaperTradingMonitor",
    "QuantPaperTradingDailyMonitor",
    "QuantAutomationMaster2200"
)

function Show-JsonSummary {
    param(
        [string]$Path,
        [string]$Label
    )
    try {
        $j = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        if ($Label -eq "统一自动化状态") {
            Write-Host "    last_updated: $($j.last_updated)"
            foreach ($name in $j.tasks.PSObject.Properties.Name) {
                $t = $j.tasks.$name
                Write-Host "    task=$name status=$($t.status) success=$($t.success) exit_code=$($t.exit_code) latest_price=$($t.latest_price_date) latest_nav=$($t.latest_nav_date)"
            }
            $shadow = $j.tasks.QuantBlendV3ShadowCombinedDaily
            $paper = $j.tasks.QuantPaperTradingDailyMonitor
            if ($shadow -and $paper) {
                $shadowDay = ([datetime]$shadow.local_run_time).ToString("yyyy-MM-dd")
                $paperDay = ([datetime]$paper.local_run_time).ToString("yyyy-MM-dd")
                if ($paperDay -lt $shadowDay) {
                    Write-Host "    警告: Paper trading 最新运行日 $paperDay 早于 Shadow 最新运行日 $shadowDay，paper 自动化可能漏跑。"
                }
            }
        } elseif ($Label -eq "Paper trading monitor 状态") {
            Write-Host "    run_id: $($j.run_id)"
            Write-Host "    success: $($j.success)"
            Write-Host "    return_code: $($j.return_code)"
            Write-Host "    duration_seconds: $($j.duration_seconds)"
            Write-Host "    working_directory: $($j.working_directory)"
            Write-Host "    log_path: $($j.log_path)"
            Write-Host "    report_path: $($j.report_path)"
            Write-Host "    failure_reason: $($j.failure_reason)"
        } elseif ($Label -eq "Shadow price refresh 状态") {
            Write-Host "    refresh_run_time: $($j.refresh_run_time)"
            Write-Host "    latest_price_date_before: $($j.latest_price_date_before)"
            Write-Host "    latest_price_date_after: $($j.latest_price_date_after)"
            Write-Host "    stale_price_warning_after_refresh: $($j.stale_price_warning_after_refresh)"
            Write-Host "    success_count: $($j.success_count)"
            Write-Host "    failed_count: $($j.failed_count)"
            Write-Host "    decision: $($j.decision)"
        } elseif ($Label -eq "Shadow monitor 状态") {
            Write-Host "    current_run_date: $($j.current_run_date)"
            Write-Host "    latest_feature_month: $($j.latest_feature_month)"
            Write-Host "    latest_price_date: $($j.latest_price_date)"
            Write-Host "    latest_nav_date: $($j.latest_nav_date)"
            Write-Host "    stale_price_warning: $($j.stale_price_warning)"
            Write-Host "    decision: $($j.decision)"
        } else {
            Get-Content -LiteralPath $Path -TotalCount 40
        }
    } catch {
        Write-Host "    无法解析 JSON，显示前 40 行:"
        Get-Content -LiteralPath $Path -TotalCount 40
    }
}

Write-Host "============================================================"
Write-Host "Quant 自动化状态检查"
Write-Host "============================================================"
Write-Host "项目目录: $ProjectRoot"
Write-Host ""

Write-Host "计划任务状态:"
foreach ($TaskName in $Tasks) {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -eq $task) {
        Write-Host "  - $TaskName : 不存在"
        continue
    }
    $info = Get-ScheduledTaskInfo -TaskName $TaskName -ErrorAction SilentlyContinue
    Write-Host "  - $TaskName : 存在 / State=$($task.State) / LastRun=$($info.LastRunTime) / LastResult=$($info.LastTaskResult) / NextRun=$($info.NextRunTime)"
}

Write-Host ""
Write-Host "状态文件:"
foreach ($label in $Paths.Keys) {
    $path = $Paths[$label]
    if (Test-Path -LiteralPath $path) {
        Write-Host "  - $label : $path"
        if ($path -like "*.json") {
            Show-JsonSummary -Path $path -Label $label
        } else {
            Get-Content -LiteralPath $path -TotalCount 40
        }
    } else {
        Write-Host "  - $label : 不存在 ($path)"
    }
    Write-Host ""
}

$Logs = @(
    (Join-Path $ProjectRoot "logs\quant_automation\shadow_combined_2200.log"),
    (Join-Path $ProjectRoot "logs\quant_automation\paper_trading_monitor.log"),
    (Join-Path $ProjectRoot "logs\quant_automation\automation_master_2200.log")
)

Write-Host "最近日志最后 40 行:"
foreach ($log in $Logs) {
    if (Test-Path -LiteralPath $log) {
        Write-Host "------------------------------------------------------------"
        Write-Host $log
        Get-Content -LiteralPath $log -Tail 40
    } else {
        Write-Host "  - 日志不存在: $log"
    }
}
Write-Host "============================================================"
