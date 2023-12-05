set PROJECT_BASE=C:/Users/Dusk_Hermit/Desktop/test/tensorflow_test/train
@REM 使用tensorboard查看训练过程

@REM 三分类训练
tensorboard --logdir %PROJECT_BASE%\models\yolov8_class3\train
@REM 一分类训练
@REM tensorboard --logdir %PROJECT_BASE%\models\yolov8_class1\train
@REM @REM 栏杆seg任务
@REM tensorboard --logdir %PROJECT_BASE%\models\yolov8_segmentation\train