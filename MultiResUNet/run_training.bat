@echo off
REM MultiResUNet Training Script with Logging and Backup for Windows
REM This script trains MultiResUNet and saves all outputs with timestamps

setlocal enabledelayedexpansion

REM Get current directory
set "PROJECT_DIR=%~dp0"
set "RUNS_DIR=%PROJECT_DIR%runs"

REM Get timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set "dt=%%I"
set "TIMESTAMP=%dt:~0,8%_%dt:~8,6%"

REM Default training parameters (OPTIMIZED for 3000-4000 samples @ 640x640)
set "EPOCHS=150"
set "BATCH_SIZE=4"              REM Reduced for large datasets
set "LEARNING_RATE=1e-4"
set "DATA_LIMIT="
set "VALIDATION_SPLIT=0.1"
set "INPUT_CHANNELS=3"
set "OUTPUT_CHANNELS=4"
set "GRADIENT_CLIP=1.0"         REM Prevent gradient explosion
set "DEVICE=cuda"
set "NUM_WORKERS=8"             REM Optimized for 32-core CPU
set "PREFETCH_FACTOR=4"         REM Increased prefetch for large datasets
set "TENSORBOARD=true"          REM Enable TensorBoard by default
set "MEMORY_SAFE_MODE=true"     REM Enable automatic memory management

REM Scale optimization for large datasets
set "SCALE_ENABLED=false"
set "SCALE_FACTOR=0.5"          REM 50% reduction (640x640 -> 320x320)

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :end_parse_args
if /i "%~1"=="--epochs" (set "EPOCHS=%~2" & shift & shift & goto :parse_args)
if /i "%~1"=="--batch-size" (set "BATCH_SIZE=%~2" & shift & shift & goto :parse_args)
if /i "%~1"=="--learning-rate" (set "LEARNING_RATE=%~2" & shift & shift & goto :parse_args)
if /i "%~1"=="--data-limit" (set "DATA_LIMIT=%~2" & shift & shift & goto :parse_args)
if /i "%~1"=="--validation-split" (set "VALIDATION_SPLIT=%~2" & shift & shift & goto :parse_args)
if /i "%~1"=="--input-channels" (set "INPUT_CHANNELS=%~2" & shift & shift & goto :parse_args)
if /i "%~1"=="--output-channels" (set "OUTPUT_CHANNELS=%~2" & shift & shift & goto :parse_args)
if /i "%~1"=="--gradient-clip" (set "GRADIENT_CLIP=%~2" & shift & shift & goto :parse_args)
if /i "%~1"=="--device" (set "DEVICE=%~2" & shift & shift & goto :parse_args)
if /i "%~1"=="--tensorboard" (set "TENSORBOARD=true" & shift & goto :parse_args)
if /i "%~1"=="--no-tensorboard" (set "TENSORBOARD=false" & shift & goto :parse_args)
if /i "%~1"=="--help" goto :show_help
echo Unknown parameter passed: %~1
exit /b 1

:show_help
echo Usage: %~nx0 [OPTIONS]
echo.
echo Options:
echo   --epochs NUM           Number of training epochs (default: 50)
echo   --batch-size NUM       Batch size for training (default: 2)
echo   --learning-rate FLOAT  Learning rate (default: 1e-4)
echo   --data-limit NUM       Number of training samples (default: 100)
echo   --validation-split FLOAT Validation split ratio (default: 0.2)
echo   --input-channels NUM   Number of input channels (default: 3)
echo   --output-channels NUM  Number of output channels (default: 4)
echo   --gradient-clip FLOAT  Gradient clipping threshold (default: 1.0)
echo   --device DEVICE        Training device: cuda or cpu (default: cuda)
echo   --tensorboard          Enable TensorBoard logging (default: true)
echo   --no-tensorboard       Disable TensorBoard logging
echo   --help                 Show this help message
echo.
echo Example:
echo   %~nx0 --epochs 100 --batch-size 4 --data-limit 200 --tensorboard
exit /b 0

:end_parse_args

REM Create directories
if not exist "%RUNS_DIR%" mkdir "%RUNS_DIR%"
if not exist "%RUNS_DIR%\models" mkdir "%RUNS_DIR%\models"
if not exist "%RUNS_DIR%\logs" mkdir "%RUNS_DIR%\logs"
if not exist "%RUNS_DIR%\histories" mkdir "%RUNS_DIR%\histories"

REM Create log file with timestamp
set "LOG_FILE=%RUNS_DIR%\logs\training_%TIMESTAMP%.log"

echo ======================================== | tee "%LOG_FILE%"
echo MultiResUNet Training Log | tee -a "%LOG_FILE%"
echo ======================================== | tee -a "%LOG_FILE%"
echo Timestamp: %TIMESTAMP% | tee -a "%LOG_FILE%"
echo Project Directory: %PROJECT_DIR% | tee -a "%LOG_FILE%"
echo. | tee -a "%LOG_FILE%"

REM Print configuration
echo Training Configuration: | tee -a "%LOG_FILE%"
echo   Epochs: %EPOCHS% | tee -a "%LOG_FILE%"
echo   Batch Size: %BATCH_SIZE% | tee -a "%LOG_FILE%"
echo   Learning Rate: %LEARNING_RATE% | tee -a "%LOG_FILE%"
echo   Data Limit: %DATA_LIMIT% | tee -a "%LOG_FILE%"
echo   Validation Split: %VALIDATION_SPLIT% | tee -a "%LOG_FILE%"
echo   Input Channels: %INPUT_CHANNELS% | tee -a "%LOG_FILE%"
echo   Output Channels: %OUTPUT_CHANNELS% | tee -a "%LOG_FILE%"
echo   Gradient Clip: %GRADIENT_CLIP% | tee -a "%LOG_FILE%"
echo   Device: %DEVICE% | tee -a "%LOG_FILE%"
echo   TensorBoard: %TENSORBOARD% | tee -a "%LOG_FILE%"
echo. | tee -a "%LOG_FILE%"

REM Navigate to project directory
cd /d "%PROJECT_DIR%"

REM Run training and log output
echo Starting training... | tee -a "%LOG_FILE%"
echo ======================================== | tee -a "%LOG_FILE%"

REM Build training command dynamically
set "TRAIN_CMD=python train.py"
set "TRAIN_CMD=%TRAIN_CMD% --epochs %EPOCHS%"
set "TRAIN_CMD=%TRAIN_CMD% --batch-size %BATCH_SIZE%"
set "TRAIN_CMD=%TRAIN_CMD% --learning-rate %LEARNING_RATE%"
set "TRAIN_CMD=%TRAIN_CMD% --data-limit %DATA_LIMIT%"
set "TRAIN_CMD=%TRAIN_CMD% --validation-split %VALIDATION_SPLIT%"
set "TRAIN_CMD=%TRAIN_CMD% --input-channels %INPUT_CHANNELS%"
set "TRAIN_CMD=%TRAIN_CMD% --output-channels %OUTPUT_CHANNELS%"
set "TRAIN_CMD=%TRAIN_CMD% --gradient-clip %GRADIENT_CLIP%"
set "TRAIN_CMD=%TRAIN_CMD% --device %DEVICE%"
set "TRAIN_CMD=%TRAIN_CMD% --verbose"
set "TRAIN_CMD=%TRAIN_CMD% --save-model"
set "TRAIN_CMD=%TRAIN_CMD% --save-dir %RUNS_DIR%\models"
set "TRAIN_CMD=%TRAIN_CMD% --debug"

REM Add TensorBoard arguments if enabled
if /i "%TENSORBOARD%"=="true" (
    set "TRAIN_CMD=%TRAIN_CMD% --tensorboard"
    set "TRAIN_CMD=%TRAIN_CMD% --log-dir %RUNS_DIR%\tensorboard"
    echo TensorBoard logging enabled | tee -a "%LOG_FILE%"
    echo   Log directory: %RUNS_DIR%\tensorboard | tee -a "%LOG_FILE%"
)

REM Execute training command
%TRAIN_CMD% 2>&1 | tee -a "%LOG_FILE%"

set "TRAIN_STATUS=%ERRORLEVEL%"

echo. | tee -a "%LOG_FILE%"
echo ======================================== | tee -a "%LOG_FILE%"
if %TRAIN_STATUS% EQU 0 (
    echo Training completed successfully! | tee -a "%LOG_FILE%"
) else (
    echo Training failed with status: %TRAIN_STATUS% | tee -a "%LOG_FILE%"
)
echo ======================================== | tee -a "%LOG_FILE%"

REM Copy training history if it exists
if exist "training_history.npy" (
    copy /Y "training_history.npy" "%RUNS_DIR%\histories\history_%TIMESTAMP%.npy"
    echo Training history saved to: %RUNS_DIR%\histories\history_%TIMESTAMP%.npy | tee -a "%LOG_FILE%"
)

REM Copy model files if they exist
if exist "models" (
    xcopy /E /I /Y "models\*.*" "%RUNS_DIR%\models\" 2>nul
    echo Model files copied to: %RUNS_DIR%\models\ | tee -a "%LOG_FILE%"
)

REM Create manifest file
set "MANIFEST_FILE=%RUNS_DIR%\manifest_%TIMESTAMP%.txt"
(
    echo MultiResUNet Training Run Manifest
    echo ===================================
    echo Timestamp: %TIMESTAMP%
    echo Log File: %LOG_FILE%
    echo.
    echo Configuration:
    echo   Epochs: %EPOCHS%
    echo   Batch Size: %BATCH_SIZE%
    echo   Learning Rate: %LEARNING_RATE%
    echo   Data Limit: %DATA_LIMIT%
    echo   Validation Split: %VALIDATION_SPLIT%
    echo   Input Channels: %INPUT_CHANNELS%
    echo   Output Channels: %OUTPUT_CHANNELS%
    echo   Gradient Clip: %GRADIENT_CLIP%
    echo   Device: %DEVICE%
    echo.
    echo Files:
    echo   Log: logs\training_%TIMESTAMP%.log
    echo   Models: models\
    echo   History: histories\history_%TIMESTAMP%.npy
    echo   Backup: backup_%TIMESTAMP%.tar.gz
    echo.
    echo Training Status: %TRAIN_STATUS%
) > "%MANIFEST_FILE%"

echo Manifest saved to: %MANIFEST_FILE% | tee -a "%LOG_FILE%"

REM Create backup using tar (available in Windows 10+)
set "BACKUP_FILE=%RUNS_DIR%\backup_%TIMESTAMP%.tar.gz"
echo. | tee -a "%LOG_FILE%"
echo Creating backup: %BACKUP_FILE% | tee -a "%LOG_FILE%"

REM Check if tar is available
where tar >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    tar -czf "%BACKUP_FILE%" -C "%RUNS_DIR%" "logs\training_%TIMESTAMP%.log" "models" "histories\history_%TIMESTAMP%.npy" "manifest_%TIMESTAMP%.txt" 2>nul
    if exist "%BACKUP_FILE%" (
        for %%A in ("%BACKUP_FILE%") do set "BACKUP_SIZE=%%~zA"
        echo Backup created successfully: %BACKUP_FILE% | tee -a "%LOG_FILE%"
    ) else (
        echo Warning: Backup creation failed, some files may be missing | tee -a "%LOG_FILE%"
    )
) else (
    echo Warning: tar command not found. Skipping backup creation. | tee -a "%LOG_FILE%"
    echo You can manually compress the runs folder later. | tee -a "%LOG_FILE%"
)

REM Clean up old backups (keep last 10)
echo. | tee -a "%LOG_FILE%"
echo Cleaning up old backups (keeping last 10)... | tee -a "%LOG_FILE%"
cd /d "%RUNS_DIR%"
for /f "skip=10 delims=" %%F in ('dir /b /o-d backup_*.tar.gz 2^>nul') do del "%%F" 2>nul
cd /d "%PROJECT_DIR%"

REM Print summary
echo. | tee -a "%LOG_FILE%"
echo ======================================== | tee -a "%LOG_FILE%"
echo Training Run Summary | tee -a "%LOG_FILE%"
echo ======================================== | tee -a "%LOG_FILE%"
echo Log File: %LOG_FILE% | tee -a "%LOG_FILE%"
echo Models Directory: %RUNS_DIR%\models\ | tee -a "%LOG_FILE%"
echo Training History: %RUNS_DIR%\histories\history_%TIMESTAMP%.npy | tee -a "%LOG_FILE%"
if exist "%BACKUP_FILE%" echo Backup File: %BACKUP_FILE% | tee -a "%LOG_FILE%"
echo Manifest: %MANIFEST_FILE% | tee -a "%LOG_FILE%"
echo. | tee -a "%LOG_FILE%"
echo To view the log file: | tee -a "%LOG_FILE%"
echo   type %LOG_FILE% | tee -a "%LOG_FILE%"
echo. | tee -a "%LOG_FILE%"
if exist "%BACKUP_FILE%" (
    echo To extract the backup: | tee -a "%LOG_FILE%"
    echo   tar -xzf %BACKUP_FILE% -C your_destination | tee -a "%LOG_FILE%"
)
echo ======================================== | tee -a "%LOG_FILE%"

endlocal
exit /b %TRAIN_STATUS%
