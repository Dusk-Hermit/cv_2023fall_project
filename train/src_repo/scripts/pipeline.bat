set PROJECT_BASE=C:/Users/Dusk_Hermit/Desktop/test/tensorflow_test/train
@REM 全流程

@REM 数据集处理
python %~dp0/preprocess.py

@REM 训练
%~dp0/train.bat

@REM 推理测试
%~dp0/predict_testing.bat