@echo off
chcp 65001 >nul
title LSTM 每日交易计划

set MODEL_DIR=.\saved_models\production_v1
set DATA_PATH=D:\zhw\A股数据
set N_HOLD=10
set K_REBALANCE=2

echo ============================================================
echo LSTM 每日交易计划系统
echo ============================================================
echo 运行时间: %date% %time%
echo 模型目录: %MODEL_DIR%
echo 数据路径: %DATA_PATH%
echo 持仓数量: %N_HOLD%  调仓上限: %K_REBALANCE%
echo ============================================================
echo.

echo [STEP 1/2] 运行每日预测...
python daily_trader.py --model_dir %MODEL_DIR% --data_path %DATA_PATH%
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] 每日预测失败，错误码: %errorlevel%
    echo [提示] 请确保已完成模型训练，且模型目录包含:
    echo         best_model.pth, scaler.pkl, model_meta.pkl
    pause
    exit /b %errorlevel%
)

echo.

set PRED_FILE=
for /f "delims=" %%f in ('dir /b /o-d daily_predictions_*.csv 2^>nul') do (
    set PRED_FILE=%%f
    goto :found
)
:found
if "%PRED_FILE%"=="" (
    echo [ERROR] 未找到预测文件 daily_predictions_*.csv
    pause
    exit /b 1
)
echo [INFO] 预测文件: %PRED_FILE%

echo.
echo [STEP 2/2] 生成交易计划...
python generate_trade_plan.py --predictions %PRED_FILE% --n %N_HOLD% --k %K_REBALANCE%
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] 交易计划生成失败，错误码: %errorlevel%
    pause
    exit /b %errorlevel%
)

echo.
echo ============================================================
echo 完成！请在交易计划 CSV 中查看买卖清单：
echo   trade_plans\trade_plan_YYYYMMDD.csv
echo ============================================================
echo 下一步: 打开同花顺APP，根据清单手动买卖
echo ============================================================
pause
