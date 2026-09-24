@echo off
setlocal
cd /d "%~dp0"
python build_final.py || goto :error
python train_final.py --epochs 150 --seq-len 64 --batch-size 4 --dim 48 --heads 4 --layers 2 --ff-dim 192 --lr 0.0005 || goto :error
python validate_final.py || goto :error
echo.
echo DORI AI FINAL BUILD COMPLETE.
pause
exit /b 0
:error
echo.
echo BUILD FAILED. Read the error above.
pause
exit /b 1
