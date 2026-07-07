@echo off
setlocal
set "PROJECT_ROOT=C:\dev\quant"
set "WRAPPER_LOG=%PROJECT_ROOT%\logs\daily_monitor\task_scheduler_wrapper.log"

if not exist "%PROJECT_ROOT%\logs\daily_monitor" mkdir "%PROJECT_ROOT%\logs\daily_monitor"

cd /d "%PROJECT_ROOT%"
echo [%date% %time%] Starting QuantDailyPaperTradingMonitor >> "%WRAPPER_LOG%"
python monitoring\project_daily_monitor.py >> "%WRAPPER_LOG%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"
echo [%date% %time%] Finished QuantDailyPaperTradingMonitor exit_code=%EXIT_CODE% >> "%WRAPPER_LOG%"
exit /b %EXIT_CODE%
