@echo off
setlocal EnableDelayedExpansion

REM "https://devblogs.microsoft.com/oldnewthing/20060823-00/?p=29993"

REM ae
set task=%1

REM laptop_pt
set bert=%2

REM laptop
set domain=%3

REM pt_ae
set run_dir=%4

REM 9
set runs=%5

REM 4
set epochs=%6
if [%epochs%]==[] set epochs=4
echo Number of epochs: %epochs%

REM set OUTPUT_DIR=run\%run_dir%\%domain%\1
set DATA_DIR=%task%\%domain%
set TRAIN_LOG_FILE=run\%run_dir%\%domain%\train_log.txt

for /l %%r in (1, 1, %runs%) do (
    set OUTPUT_DIR=run\%run_dir%\%domain%\%%r
    echo "Run !OUTPUT_DIR!"
    REM echo [!OUTPUT_DIR!]

    if not exist !OUTPUT_DIR! mkdir !OUTPUT_DIR!

    python src\run_%task%.py --bert_model %bert% --num_train_epochs %epochs% ^
        --output_dir !OUTPUT_DIR! --data_dir %DATA_DIR% --log_file %TRAIN_LOG_FILE% --seed %%r --do_save
)

REM script\run_ae.bat ae laptop_pt laptop pt_ae 9