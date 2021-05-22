@echo off
setlocal EnableDelayedExpansion

set domain=%1

set runs=%2

for /l %%r in (1, 1, %runs%) do (
    set prediction=debug\pt_ae\%domain%\%%r\predictions.json

    if exist !prediction! (
        python eval\evaluate_ae.py --pred_json=!prediction!
    )
)

REM eval\run_ae_eval.bat laptop 9
