@echo off

set "PROJECT_BASE=C:\Users\Dusk_Hermit\Desktop\test\tensorflow_test\train"

cd /d %PROJECT_BASE%\src_repo

python preprocess.py
python train.py
python infer_test.py
