@echo off
setlocal enabledelayedexpansion
set PROJECT_ROOT=%~dp0..
cd /d "%PROJECT_ROOT%"
set SCRIPT=scripts\run_blend_v3_shadow_live_inference_v1.py
set LOG_FILE=logs\blend_v3_shadow\shadow_live_update.log
set STATUS_FILE=output\blend_v3_shadow_monitoring\shadow_monitor_latest_status.json
if not exist logs\blend_v3_shadow mkdir logs\blend_v3_shadow
echo ============================================================
echo Blend V3 Shadow Live Update
echo ===========================
echo.
echo project_root: %PROJECT_ROOT%
echo script: %SCRIPT%
echo log_file: %LOG_FILE%
echo status_file: %STATUS_FILE%
echo started_at: %DATE% %TIME%
echo ---------------
echo.
echo ## Running shadow live update...
echo.
python %SCRIPT% > "%TEMP%\blend_v3_shadow_update.out" 2>&1
set EXIT_CODE=%ERRORLEVEL%
type "%TEMP%\blend_v3_shadow_update.out"
type "%TEMP%\blend_v3_shadow_update.out" >> "%LOG_FILE%"
echo.>> "%LOG_FILE%"
echo exit_code: %EXIT_CODE%
if not "%EXIT_CODE%"=="0" (
  echo decision: FAILED
  echo 请查看日志：%LOG_FILE%
  echo ============================================================
  exit /b %EXIT_CODE%
)
for /f "tokens=1,* delims==" %%A in ('findstr /b "decision= latest_feature_month= latest_price_date= latest_nav_date= stale_price_warning= shadow_holding_count=" "%TEMP%\blend_v3_shadow_update.out"') do echo %%A: %%B
echo dashboard:
echo streamlit run monitoring\blend_v3_shadow_report.py
echo ============================================================
exit /b %EXIT_CODE%
