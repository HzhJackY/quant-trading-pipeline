@echo off
setlocal enabledelayedexpansion
set "PROJECT_ROOT=C:\dev\quant"
set "LOG_DIR=%PROJECT_ROOT%\logs\quant_automation"
set "LOG_FILE=%LOG_DIR%\shadow_combined_2200.log"
set "PRICE_STATUS=%PROJECT_ROOT%\output\blend_v3_shadow_monitoring\price_cache\shadow_price_refresh_status.json"
set "SHADOW_STATUS=%PROJECT_ROOT%\output\blend_v3_shadow_monitoring\shadow_monitor_latest_status.json"
set "NOTIFY=%PROJECT_ROOT%\scripts\notify_quant_task_result.ps1"
set "STATUS_WRITER=%PROJECT_ROOT%\scripts\write_quant_automation_status.py"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
cd /d "%PROJECT_ROOT%"

echo ============================================================
echo Quant Blend V3 Shadow Combined 22:00
echo ============================================================
echo started_at: %DATE% %TIME%
echo log_file: %LOG_FILE%
echo ------------------------------------------------------------
>> "%LOG_FILE%" echo ============================================================
>> "%LOG_FILE%" echo started_at: %DATE% %TIME%

echo [1/2] Running price refresh...
call scripts\run_blend_v3_shadow_price_refresh.bat > "%TEMP%\quant_shadow_price_stage.out" 2>&1
set "PRICE_EXIT=%ERRORLEVEL%"
type "%TEMP%\quant_shadow_price_stage.out"
type "%TEMP%\quant_shadow_price_stage.out" >> "%LOG_FILE%"
echo price_refresh_exit_code: !PRICE_EXIT!
>> "%LOG_FILE%" echo price_refresh_exit_code: !PRICE_EXIT!

for /f "tokens=1,* delims==" %%A in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Test-Path -LiteralPath '%PRICE_STATUS%') { $j=Get-Content -LiteralPath '%PRICE_STATUS%' -Raw | ConvertFrom-Json; 'PRICE_LATEST_PRICE_DATE_AFTER=' + $j.latest_price_date_after; 'PRICE_STALE_PRICE_WARNING_AFTER=' + $j.stale_price_warning_after_refresh; 'PRICE_DECISION=' + $j.decision } else { 'PRICE_DECISION=PRICE_STATUS_MISSING' }"') do set "%%A=%%B"
echo price_latest_price_date_after: !PRICE_LATEST_PRICE_DATE_AFTER!
echo price_stale_price_warning_after: !PRICE_STALE_PRICE_WARNING_AFTER!
echo price_decision: !PRICE_DECISION!
>> "%LOG_FILE%" echo price_latest_price_date_after: !PRICE_LATEST_PRICE_DATE_AFTER!
>> "%LOG_FILE%" echo price_stale_price_warning_after: !PRICE_STALE_PRICE_WARNING_AFTER!
>> "%LOG_FILE%" echo price_decision: !PRICE_DECISION!

echo [2/2] Running shadow monitor...
call scripts\run_blend_v3_shadow_live_update.bat > "%TEMP%\quant_shadow_monitor_stage.out" 2>&1
set "MONITOR_EXIT=%ERRORLEVEL%"
type "%TEMP%\quant_shadow_monitor_stage.out"
type "%TEMP%\quant_shadow_monitor_stage.out" >> "%LOG_FILE%"
echo shadow_monitor_exit_code: !MONITOR_EXIT!
>> "%LOG_FILE%" echo shadow_monitor_exit_code: !MONITOR_EXIT!

for /f "tokens=1,* delims==" %%A in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Test-Path -LiteralPath '%SHADOW_STATUS%') { $j=Get-Content -LiteralPath '%SHADOW_STATUS%' -Raw | ConvertFrom-Json; $runDate=$j.current_run_date_local; if (-not $runDate) { $runDate=$j.current_run_date }; 'SHADOW_CURRENT_RUN_DATE=' + $runDate; 'SHADOW_LATEST_PRICE_DATE=' + $j.latest_price_date; 'SHADOW_LATEST_NAV_DATE=' + $j.latest_nav_date; 'SHADOW_STALE_PRICE_WARNING=' + $j.stale_price_warning; 'SHADOW_DECISION=' + $j.decision } else { 'SHADOW_DECISION=SHADOW_STATUS_MISSING' }"') do set "%%A=%%B"
echo shadow_current_run_date: !SHADOW_CURRENT_RUN_DATE!
echo shadow_latest_price_date: !SHADOW_LATEST_PRICE_DATE!
echo shadow_latest_nav_date: !SHADOW_LATEST_NAV_DATE!
echo shadow_stale_price_warning: !SHADOW_STALE_PRICE_WARNING!
echo shadow_decision: !SHADOW_DECISION!
>> "%LOG_FILE%" echo shadow_current_run_date: !SHADOW_CURRENT_RUN_DATE!
>> "%LOG_FILE%" echo shadow_latest_price_date: !SHADOW_LATEST_PRICE_DATE!
>> "%LOG_FILE%" echo shadow_latest_nav_date: !SHADOW_LATEST_NAV_DATE!
>> "%LOG_FILE%" echo shadow_stale_price_warning: !SHADOW_STALE_PRICE_WARNING!
>> "%LOG_FILE%" echo shadow_decision: !SHADOW_DECISION!

set "FINAL_EXIT=0"
set "STATUS=SUCCESS"
set "FAILED_STAGE="
set "FAILURE_REASON="
if not "!PRICE_EXIT!"=="0" (
  set "FINAL_EXIT=!PRICE_EXIT!"
  set "STATUS=FAILED"
  set "FAILED_STAGE=price_refresh"
  set "FAILURE_REASON=price refresh exit_code=!PRICE_EXIT!"
)
if "!STATUS!"=="SUCCESS" if not "!MONITOR_EXIT!"=="0" (
  set "FINAL_EXIT=!MONITOR_EXIT!"
  set "STATUS=FAILED"
  set "FAILED_STAGE=shadow_monitor"
  set "FAILURE_REASON=shadow monitor exit_code=!MONITOR_EXIT!"
)
if "!STATUS!"=="SUCCESS" if not exist "%SHADOW_STATUS%" (
  set "FINAL_EXIT=2"
  set "STATUS=FAILED"
  set "FAILED_STAGE=shadow_monitor_status"
  set "FAILURE_REASON=shadow status file missing"
)

if "!STATUS!"=="SUCCESS" (
  set "MESSAGE=Shadow combined finished. price_date=!SHADOW_LATEST_PRICE_DATE!, nav_date=!SHADOW_LATEST_NAV_DATE!, decision=!SHADOW_DECISION!"
  set "SUCCESS_ARG=true"
) else (
  set "MESSAGE=Shadow combined failed at !FAILED_STAGE!: !FAILURE_REASON!"
  set "SUCCESS_ARG=false"
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%NOTIFY%" -TaskName "QuantBlendV3ShadowCombinedDaily" -Status "!STATUS!" -Message "!MESSAGE!" -DetailPath "%SHADOW_STATUS%" -LogPath "%LOG_FILE%" >> "%LOG_FILE%" 2>&1
python "%STATUS_WRITER%" --task-name "QuantBlendV3ShadowCombinedDaily" --category "shadow" --status "!STATUS!" --exit-code !FINAL_EXIT! --stage "!FAILED_STAGE!" --success !SUCCESS_ARG! --failure-reason "!FAILURE_REASON!" --latest-price-date "!SHADOW_LATEST_PRICE_DATE!" --latest-nav-date "!SHADOW_LATEST_NAV_DATE!" --stale-price-warning "!SHADOW_STALE_PRICE_WARNING!" --log-path "%LOG_FILE%" --detail-path "%SHADOW_STATUS%" --notes "price_decision=!PRICE_DECISION!; shadow_decision=!SHADOW_DECISION!" >> "%LOG_FILE%" 2>&1

echo ------------------------------------------------------------
echo status: !STATUS!
echo failure_reason: !FAILURE_REASON!
echo latest_price_date: !SHADOW_LATEST_PRICE_DATE!
echo latest_nav_date: !SHADOW_LATEST_NAV_DATE!
echo notification_log: logs\quant_automation\latest_notification.txt
echo automation_status: output\automation_reliability_v1\latest_automation_status.json
echo ============================================================
>> "%LOG_FILE%" echo finished_at: %DATE% %TIME% status=!STATUS! exit_code=!FINAL_EXIT!
exit /b !FINAL_EXIT!
