@echo off
set "PROJECT_BASE=C:\Users\Dusk_Hermit\Desktop\test\tensorflow_test\train"
set "SRC_REPO=%PROJECT_BASE%\src_repo"

if exist "%SRC_REPO%\infer_test_output" rd /s /q "%SRC_REPO%\infer_test_output"
if exist "%SRC_REPO%\datasets" rd /s /q "%SRC_REPO%\datasets"
if exist "%PROJECT_BASE%\models" rd /s /q "%PROJECT_BASE%\models"
if exist "%PROJECT_BASE%\tensorboard" rd /s /q "%PROJECT_BASE%\tensorboard"
