@echo off
setlocal enabledelayedexpansion
set "PROJECT_ROOT=C:\dev\quant"
set "LOG_DIR=%PROJECT_ROOT%\logs\quant_automation"
set "LOG_FILE=%LOG_DIR%\automation_master_2200.log"
set "NOTIFY=%PROJECT_ROOT%\scripts\notify_quant_task_result.ps1"
set "STATUS_WRITER=%PROJECT_ROOT%\scripts\write_quant_automation_status.py"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
cd /d "%PROJECT_ROOT%"

echo ============================================================
echo Quant Automation Master 22:00
echo ============================================================
echo started_at: %DATE% %TIME%
echo log_file: %LOG_FILE%
>> "%LOG_FILE%" echo ============================================================
>> "%LOG_FILE%" echo started_at: %DATE% %TIME%

call scripts\run_blend_v3_shadow_refresh_then_monitor.bat > "%TEMP%\quant_master_shadow.out" 2>&1
set "SHADOW_EXIT=%ERRORLEVEL%"
type "%TEMP%\quant_master_shadow.out"
type "%TEMP%\quant_master_shadow.out" >> "%LOG_FILE%"

set "PAPER_EXIT=0"
set "PAPER_STATUS=SKIPPED"
set "PAPER_STANDALONE_ENABLED=false"
for /f "tokens=1,* delims==" %%A in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "$t=Get-ScheduledTask -TaskName 'QuantPaperTradingDailyMonitor' -ErrorAction SilentlyContinue; if ($t -and $t.State -ne 'Disabled') { 'PAPER_STANDALONE_ENABLED=true' }"') do set "%%A=%%B"
set "RECENT_PAPER_SUCCESS=false"
for /f "tokens=1,* delims==" %%A in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "$p='C:\dev\quant\output\automation_reliability_v1\latest_automation_status.json'; if (Test-Path -LiteralPath $p) { $j=Get-Content -LiteralPath $p -Raw | ConvertFrom-Json; $r=$j.tasks.QuantPaperTradingDailyMonitor; if ($r -and $r.success -eq $true) { $age=((Get-Date) - [datetime]$r.local_run_time).TotalMinutes; if ($age -le 60) { 'RECENT_PAPER_SUCCESS=true'; 'RECENT_PAPER_AGE_MINUTES=' + [math]::Round($age,1) } } }"') do set "%%A=%%B"
if exist "scripts\run_paper_trading_monitor_with_notification.bat" (
  if /I "!PAPER_STANDALONE_ENABLED!"=="true" (
    set "PAPER_STATUS=STANDALONE_TASK_ENABLED_SKIPPED"
    echo paper_trading_monitor_wrapper: skipped because QuantPaperTradingDailyMonitor is installed/enabled as independent 22:15 task
    >> "%LOG_FILE%" echo paper_trading_monitor_wrapper: skipped because QuantPaperTradingDailyMonitor is installed/enabled as independent 22:15 task
  ) else if /I "!RECENT_PAPER_SUCCESS!"=="true" (
    set "PAPER_STATUS=RECENT_SUCCESS_SKIPPED"
    echo paper_trading_monitor_wrapper: skipped because recent success exists within 60 minutes; age_minutes=!RECENT_PAPER_AGE_MINUTES!
    >> "%LOG_FILE%" echo paper_trading_monitor_wrapper: skipped because recent success exists within 60 minutes; age_minutes=!RECENT_PAPER_AGE_MINUTES!
  ) else (
    call scripts\run_paper_trading_monitor_with_notification.bat > "%TEMP%\quant_master_paper.out" 2>&1
    set "PAPER_EXIT=!ERRORLEVEL!"
    set "PAPER_STATUS=RAN"
    type "%TEMP%\quant_master_paper.out"
    type "%TEMP%\quant_master_paper.out" >> "%LOG_FILE%"
  )
) else (
  echo paper_trading_monitor_wrapper: not found, skipped
  >> "%LOG_FILE%" echo paper_trading_monitor_wrapper: not found, skipped
)

set "FINAL_EXIT=0"
set "STATUS=SUCCESS"
set "FAILURE_REASON="
if not "!SHADOW_EXIT!"=="0" (
  set "FINAL_EXIT=!SHADOW_EXIT!"
  set "STATUS=FAILED"
  set "FAILURE_REASON=shadow combined exit_code=!SHADOW_EXIT!"
)
if "!STATUS!"=="SUCCESS" if not "!PAPER_EXIT!"=="0" (
  set "FINAL_EXIT=!PAPER_EXIT!"
  set "STATUS=FAILED"
  set "FAILURE_REASON=paper trading monitor exit_code=!PAPER_EXIT!"
)

if "!STATUS!"=="SUCCESS" (
  set "MESSAGE=Automation master finished. shadow_exit=!SHADOW_EXIT!, paper_status=!PAPER_STATUS!, paper_exit=!PAPER_EXIT!"
  set "SUCCESS_ARG=true"
) else (
  set "MESSAGE=Automation master failed: !FAILURE_REASON!"
  set "SUCCESS_ARG=false"
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%NOTIFY%" -TaskName "QuantAutomationMaster2200" -Status "!STATUS!" -Message "!MESSAGE!" -DetailPath "%PROJECT_ROOT%\output\automation_reliability_v1\latest_automation_status.json" -LogPath "%LOG_FILE%" >> "%LOG_FILE%" 2>&1
python "%STATUS_WRITER%" --task-name "QuantAutomationMaster2200" --category "master" --status "!STATUS!" --exit-code !FINAL_EXIT! --stage "master" --success !SUCCESS_ARG! --failure-reason "!FAILURE_REASON!" --log-path "%LOG_FILE%" --detail-path "%PROJECT_ROOT%\output\automation_reliability_v1\latest_automation_status.json" --notes "shadow_exit=!SHADOW_EXIT!; paper_status=!PAPER_STATUS!; paper_exit=!PAPER_EXIT!" >> "%LOG_FILE%" 2>&1

echo ------------------------------------------------------------
echo status: !STATUS!
echo failure_reason: !FAILURE_REASON!
echo shadow_exit: !SHADOW_EXIT!
echo paper_status: !PAPER_STATUS!
echo paper_exit: !PAPER_EXIT!
echo latest_notification: logs\quant_automation\latest_notification.txt
echo latest_automation_status: output\automation_reliability_v1\latest_automation_status.json
echo ============================================================
>> "%LOG_FILE%" echo finished_at: %DATE% %TIME% status=!STATUS! exit_code=!FINAL_EXIT!
exit /b !FINAL_EXIT!
