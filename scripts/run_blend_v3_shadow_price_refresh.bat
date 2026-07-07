@echo off
setlocal enabledelayedexpansion
set PROJECT_ROOT=%~dp0..
cd /d "%PROJECT_ROOT%"
set SCRIPT=scripts\refresh_blend_v3_shadow_prices_v1.py
set LOG_FILE=logs\blend_v3_shadow\shadow_price_refresh.log
if not exist logs\blend_v3_shadow mkdir logs\blend_v3_shadow
echo ============================================================
echo Blend V3 Shadow Price Refresh
echo =============================
echo project_root: %PROJECT_ROOT%
echo script: %SCRIPT%
echo log_file: %LOG_FILE%
echo started_at: %DATE% %TIME%
echo ------------------------------------------------------------
python %SCRIPT% > "%TEMP%\blend_v3_shadow_price_refresh.out" 2>&1
set EXIT_CODE=%ERRORLEVEL%
type "%TEMP%\blend_v3_shadow_price_refresh.out"
type "%TEMP%\blend_v3_shadow_price_refresh.out" >> "%LOG_FILE%"
echo.>> "%LOG_FILE%"
echo exit_code: %EXIT_CODE%>> "%LOG_FILE%"
if not "%EXIT_CODE%"=="0" (
  echo decision: FAILED
  echo log_file: %LOG_FILE%
  echo ============================================================
  exit /b %EXIT_CODE%
)
for /f "tokens=1,* delims==" %%A in ('findstr /b "refresh_universe_count= latest_price_date_before= latest_price_date_after= success_count= failed_count= decision=" "%TEMP%\blend_v3_shadow_price_refresh.out"') do echo %%A: %%B
echo ============================================================
exit /b %EXIT_CODE%
