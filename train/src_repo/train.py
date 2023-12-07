from utils import yolo_python_api
import os
from config import *
import shutil
from utils import tensorflow_utils
import time
static_pose_model_path=os.path.join(PROJECT_BASE,r"src_repo\yolov8n-pose.pt").replace("\\","/")
static_seg_model_path=os.path.join(PROJECT_BASE,r"src_repo\yolov8n-seg.pt").replace("\\","/")

tensorboard_dir=os.path.join(PROJECT_BASE,'tensorboard').replace("\\","/")
class3_tensorboard_dir=os.path.join(tensorboard_dir,'class3').replace("\\","/")
class1_tensorboard_dir=os.path.join(tensorboard_dir,'class1').replace("\\","/")
seg_tensorboard_dir=os.path.join(tensorboard_dir,'segmentation').replace("\\","/")

model_dir=os.path.join(PROJECT_BASE,'models').replace("\\","/")

LOOPS=1
EPOCHS=10
BATCH_SIZE=32

def log_content_of_dir_recursively(dirname):
    for x in os.listdir(dirname):
        print(os.path.join(dirname,x).replace("\\","/"))
        if os.path.isdir(os.path.join(dirname,x)):
            log_content_of_dir_recursively(os.path.join(dirname,x).replace("\\","/"))

if __name__ == "__main__":
    os.makedirs(tensorboard_dir,exist_ok=True)
    os.makedirs(class3_tensorboard_dir,exist_ok=True)
    os.makedirs(class1_tensorboard_dir,exist_ok=True)
    os.makedirs(seg_tensorboard_dir,exist_ok=True)
    
    for i in range(LOOPS):
        # 重新更新训练集
        
        if os.path.exists(os.path.join(PROJECT_BASE,r"src_repo/datasets").replace("\\","/")):
            shutil.rmtree(os.path.join(PROJECT_BASE,r"src_repo/datasets").replace("\\","/"))
        os.system('python'+' '+os.path.join(PROJECT_BASE,r"src_repo/preprocess.py").replace("\\","/"))

        print('-'*80)
        print(f'Start {i}th loop. Training segmentation model.')
        yolo_python_api.yolo_train(
            pt_dir_path=os.path.join(PROJECT_BASE,r"models\yolov8_segmentation").replace("\\","/"),
            alternate_pt= static_seg_model_path,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
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
        target_train=os.path.join(seg_tensorboard_dir,os.path.basename(newest_train)).replace("\\","/")
        shutil.copytree(newest_train,target_train)
        tensorflow_utils.write_things_to_tensorboard(target_train)
        print('-'*80)
        print('Loging model_dir\n')
        log_content_of_dir_recursively(model_dir)
        print('-'*80)
        print('Loging tensorboard_dir\n')
        log_content_of_dir_recursively(tensorboard_dir)
        print('-'*80)
        time.sleep(5)
        
        print('-'*80)
        print(f'Start {i}th loop. Training class1 model.')
        yolo_python_api.yolo_train(
            pt_dir_path=os.path.join(PROJECT_BASE,r"models\yolov8_class1").replace("\\","/"),
            alternate_pt= static_pose_model_path,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
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
        target_train=os.path.join(class1_tensorboard_dir,os.path.basename(newest_train)).replace("\\","/")
        shutil.copytree(newest_train,target_train)
        tensorflow_utils.write_things_to_tensorboard(target_train)
        print('-'*80)
        print('Loging model_dir\n')
        log_content_of_dir_recursively(model_dir)
        print('-'*80)
        print('Loging tensorboard_dir\n')
        log_content_of_dir_recursively(tensorboard_dir)
        print('-'*80)
        time.sleep(5)
        
        print('-'*80)
        print(f'Start {i}th loop. Training class3 model.')
        yolo_python_api.yolo_train(
            pt_dir_path=os.path.join(PROJECT_BASE,r"models\yolov8_class3").replace("\\","/"),
            alternate_pt= static_pose_model_path,
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE,
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
        target_train=os.path.join(class3_tensorboard_dir,os.path.basename(newest_train)).replace("\\","/")
        shutil.copytree(newest_train,target_train)
        tensorflow_utils.write_things_to_tensorboard(target_train)
        print('-'*80)
        print('Loging model_dir\n')
        log_content_of_dir_recursively(model_dir)
        print('-'*80)
        print('Loging tensorboard_dir\n')
        log_content_of_dir_recursively(tensorboard_dir)
        print('-'*80)
        time.sleep(5)
        