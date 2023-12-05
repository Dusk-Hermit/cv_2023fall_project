set PROJECT_BASE=C:\Users\Dusk_Hermit\Desktop\test\tensorflow_test\train

@REM 需要适时修改推理所需要的模型的路径，如从`/train/`修改到`/train2/`等
@REM 推理测试
@REM 这里的测试输出内容到训练的项目文件夹下，可修改

@REM 三分类训练
yolo predict ^
    model=%PROJECT_BASE%\models\yolov8_class3\train\weights\best.pt ^
    source=%PROJECT_BASE%\src_repo\datasets\class3\images\val ^
    project=%PROJECT_BASE%\models\yolov8_class3

@REM 一分类训练
yolo predict ^
    model=%PROJECT_BASE%\models\yolov8_class1\train\weights\best.pt ^
    source=%PROJECT_BASE%\src_repo\datasets\class1\images\val ^
    project=%PROJECT_BASE%\models\yolov8_class1

@REM 栏杆seg任务
yolo predict ^
    model=%PROJECT_BASE%\models\yolov8_segmentation\train\weights\best.pt ^
    source=%PROJECT_BASE%\src_repo\datasets\segmentation\images\val ^
    project=%PROJECT_BASE%\models\yolov8_segmentation

