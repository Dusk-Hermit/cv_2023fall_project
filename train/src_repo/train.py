from utils import yolo_python_api
import os
from config import *
import shutil
static_pose_model_path=os.path.join(PROJECT_BASE,r"src_repo\yolov8n-pose.pt").replace("\\","/")
static_seg_model_path=os.path.join(PROJECT_BASE,r"src_repo\yolov8n-seg.pt").replace("\\","/")

tensorboard_dir=os.path.join(PROJECT_BASE,'tensorboard').replace("\\","/")
class3_tensorboard_dir=os.path.join(tensorboard_dir,'class3').replace("\\","/")
class1_tensorboard_dir=os.path.join(tensorboard_dir,'class1').replace("\\","/")
seg_tensorboard_dir=os.path.join(tensorboard_dir,'segmentation').replace("\\","/")



if __name__ == "__main__":
    os.makedirs(tensorboard_dir,exist_ok=True)
    os.makedirs(class3_tensorboard_dir,exist_ok=True)
    os.makedirs(class1_tensorboard_dir,exist_ok=True)
    os.makedirs(seg_tensorboard_dir,exist_ok=True)
    
    for i in range(10):
        # 重新更新训练集
        if os.path.exists(os.path.join(PROJECT_BASE,r"src_repo/datasets").replace("\\","/")):
            shutil.rmtree(os.path.join(PROJECT_BASE,r"src_repo/datasets").replace("\\","/"))
        os.system('python'+' '+os.path.join(PROJECT_BASE,r"src_repo/preprocess.py").replace("\\","/"))

        yolo_python_api.yolo_train(
            pt_dir_path=os.path.join(PROJECT_BASE,r"src_repo\models\yolov8_segmentation").replace("\\","/"),
            alternate_pt= static_seg_model_path,
            epochs=20,
            batch_size=32,
            project_name='yolov8_segmentation',
            yaml=os.path.join(PROJECT_BASE,r"src_repo\yaml_files\segmentation.yaml").replace("\\","/"),
            imgsz=640,
            patience=20,
        )
        
        # clear all folders in seg_tensorboard_dir
        for x in os.listdir(seg_tensorboard_dir):
            shutil.rmtree(os.path.join(seg_tensorboard_dir,x).replace("\\","/"))
        # move newest train folder to seg_tensorboard_dir
        newest_train=yolo_python_api.get_newest_train(os.path.join(PROJECT_BASE,r"models/yolov8_segmentation").replace("\\","/"))
        shutil.copytree(newest_train,os.path.join(seg_tensorboard_dir,os.path.basename(newest_train)).replace("\\","/"))
        print(f'Finished {i}th training of segmentation model. Updated tensorboard with train folder { newest_train}.')
        
        yolo_python_api.yolo_train(
            pt_dir_path=os.path.join(PROJECT_BASE,r"src_repo\models\yolov8_class1").replace("\\","/"),
            alternate_pt= static_pose_model_path,
            epochs=20,
            batch_size=32,
            project_name='yolov8_class1',
            yaml=os.path.join(PROJECT_BASE,r"src_repo\yaml_files\class1.yaml").replace("\\","/"),
            imgsz=640,
            patience=20,
        )
        
        # clear all folders in class1_tensorboard_dir
        for x in os.listdir(class1_tensorboard_dir):
            shutil.rmtree(os.path.join(class1_tensorboard_dir,x).replace("\\","/"))
        # move newest train folder to class1_tensorboard_dir
        newest_train=yolo_python_api.get_newest_train(os.path.join(PROJECT_BASE,r"models/yolov8_class1").replace("\\","/"))
        shutil.copytree(newest_train,os.path.join(class1_tensorboard_dir,os.path.basename(newest_train)).replace("\\","/"))
        print(f'Finished {i}th training of class1 model. Updated tensorboard with train folder { newest_train}.')
        
        yolo_python_api.yolo_train(
            pt_dir_path=os.path.join(PROJECT_BASE,r"src_repo\models\yolov8_class3").replace("\\","/"),
            alternate_pt= static_pose_model_path,
            epochs=20, 
            batch_size=32,
            project_name='yolov8_class3',
            yaml=os.path.join(PROJECT_BASE,r"src_repo\yaml_files\class3.yaml").replace("\\","/"),
            imgsz=640,
            patience=20,
        )
        # clear all folders in class3_tensorboard_dir
        for x in os.listdir(class3_tensorboard_dir):
            shutil.rmtree(os.path.join(class3_tensorboard_dir,x).replace("\\","/"))
        # move newest train folder to class3_tensorboard_dir
        newest_train=yolo_python_api.get_newest_train(os.path.join(PROJECT_BASE,r"models/yolov8_class3").replace("\\","/"))
        shutil.copytree(newest_train,os.path.join(class3_tensorboard_dir,os.path.basename(newest_train)).replace("\\","/"))
        print(f'Finished {i}th training of class3 model. Updated tensorboard with train folder { newest_train}.')
        