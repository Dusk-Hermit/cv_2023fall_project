@echo off

set "PROJECT_BASE=C:\Users\Dusk_Hermit\Desktop\test\tensorflow_test\train"

cd /d %PROJECT_BASE%\src_repo

python preprocess.py
@REM python train.py
@REM python infer_test.py

@REM tensorboard --logdir=%PROJECT_BASE%\tensorboard