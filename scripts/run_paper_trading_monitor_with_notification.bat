@echo off
setlocal enabledelayedexpansion
set "PROJECT_ROOT=C:\dev\quant"
set "LOG_DIR=%PROJECT_ROOT%\logs\quant_automation"
set "LOG_FILE=%LOG_DIR%\paper_trading_monitor.log"
set "PAPER_STATUS=%PROJECT_ROOT%\output\project_monitoring\latest_status.json"
set "NOTIFY=%PROJECT_ROOT%\scripts\notify_quant_task_result.ps1"
set "STATUS_WRITER=%PROJECT_ROOT%\scripts\write_quant_automation_status.py"
set "MONITOR_TIMEOUT_MINUTES=50"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
cd /d "%PROJECT_ROOT%"

echo ============================================================
echo Quant Paper Trading Monitor With Notification
echo ============================================================
echo started_at: %DATE% %TIME%
echo log_file: %LOG_FILE%
>> "%LOG_FILE%" echo ============================================================
>> "%LOG_FILE%" echo started_at: %DATE% %TIME%

echo entry: python monitoring\project_daily_monitor.py --timeout-minutes %MONITOR_TIMEOUT_MINUTES% --no-notify
echo monitor_timeout_minutes: %MONITOR_TIMEOUT_MINUTES%
>> "%LOG_FILE%" echo entry: python monitoring\project_daily_monitor.py --timeout-minutes %MONITOR_TIMEOUT_MINUTES% --no-notify
python monitoring\project_daily_monitor.py --timeout-minutes %MONITOR_TIMEOUT_MINUTES% --no-notify > "%TEMP%\quant_paper_monitor.out" 2>&1
set "PAPER_EXIT=%ERRORLEVEL%"
type "%TEMP%\quant_paper_monitor.out"
type "%TEMP%\quant_paper_monitor.out" >> "%LOG_FILE%"
>> "%LOG_FILE%" echo paper_monitor_exit_code: !PAPER_EXIT!

for /f "tokens=1,* delims==" %%A in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Test-Path -LiteralPath '%PAPER_STATUS%') { $j=Get-Content -LiteralPath '%PAPER_STATUS%' -Raw | ConvertFrom-Json; 'PAPER_STATUS_SUCCESS=' + $j.success; 'PAPER_STATUS_RETURN_CODE=' + $j.return_code; 'PAPER_FAILURE_REASON=' + $j.failure_reason; 'PAPER_REPORT_PATH=' + $j.report_path; 'PAPER_LOG_PATH=' + $j.log_path } else { 'PAPER_STATUS_SUCCESS=False'; 'PAPER_FAILURE_REASON=paper status file missing' }"') do set "%%A=%%B"

set "FINAL_EXIT=!PAPER_EXIT!"
set "STATUS=SUCCESS"
set "FAILURE_REASON="
if not "!PAPER_EXIT!"=="0" (
  set "STATUS=FAILED"
  set "FAILURE_REASON=paper monitor exit_code=!PAPER_EXIT! !PAPER_FAILURE_REASON!"
)
if "!PAPER_EXIT!"=="124" (
  set "STATUS=FAILED"
  set "FAILURE_REASON=paper monitor timed out after %MONITOR_TIMEOUT_MINUTES% minutes"
)
if "!STATUS!"=="SUCCESS" if /I not "!PAPER_STATUS_SUCCESS!"=="True" (
  set "STATUS=FAILED"
  set "FINAL_EXIT=2"
  set "FAILURE_REASON=paper latest_status success was not true: !PAPER_FAILURE_REASON!"
)

if "!STATUS!"=="SUCCESS" (
  set "MESSAGE=Paper trading monitor finished successfully."
  set "SUCCESS_ARG=true"
) else (
  set "MESSAGE=Paper trading monitor failed: !FAILURE_REASON!"
  set "SUCCESS_ARG=false"
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%NOTIFY%" -TaskName "QuantPaperTradingDailyMonitor" -Status "!STATUS!" -Message "!MESSAGE!" -DetailPath "%PAPER_STATUS%" -LogPath "%LOG_FILE%" >> "%LOG_FILE%" 2>&1
python "%STATUS_WRITER%" --task-name "QuantPaperTradingDailyMonitor" --category "paper_trading" --status "!STATUS!" --exit-code !FINAL_EXIT! --stage "paper_monitor" --success !SUCCESS_ARG! --failure-reason "!FAILURE_REASON!" --log-path "%LOG_FILE%" --detail-path "%PAPER_STATUS%" --notes "report_path=!PAPER_REPORT_PATH!; monitor_log_path=!PAPER_LOG_PATH!" >> "%LOG_FILE%" 2>&1

echo ------------------------------------------------------------
echo status: !STATUS!
echo failure_reason: !FAILURE_REASON!
echo latest_status: output\project_monitoring\latest_status.json
echo notification_log: logs\quant_automation\latest_notification.txt
echo automation_status: output\automation_reliability_v1\latest_automation_status.json
echo ============================================================
>> "%LOG_FILE%" echo finished_at: %DATE% %TIME% status=!STATUS! exit_code=!FINAL_EXIT!
exit /b !FINAL_EXIT!
