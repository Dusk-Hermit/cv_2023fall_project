# 测试模型输出结果
# 会用一些模型，用里面指定的图片路径，生成一个mask文件和json文件，到src_repo/test_output中

from postprocess import *
from config import *
import os
import numpy as np
import glob
from ultralytics import YOLO
import time
import time
from datetime import datetime
import json
from utils import yolo_python_api
import shutil

test_output_dir=os.path.join(PROJECT_BASE,'src_repo','infer_test_output').replace("\\","/")
# 清除test_output_dir
if os.path.exists(test_output_dir):
    shutil.rmtree(test_output_dir)
os.makedirs(test_output_dir,exist_ok=True)


# 编码阶段使用，测试图片文件夹路径
# val_jpg_folder_path=SOURCE_DATA # 测试环境使用
val_jpg_folder_path=os.path.join(PROJECT_BASE,"src_repo/datasets/class3/images/val").replace("\\","/")
jpg_files = glob.glob(f"{val_jpg_folder_path}/*.jpg")

# 测试模型路径，需要适时修改
# class3_trained_model_path=os.path.join(PROJECT_BASE,"models/yolov8_class3/train/weights/best.pt").replace("\\","/")
class3_trained_model_path=yolo_python_api.get_best_pt_path(os.path.join(PROJECT_BASE,"models/yolov8_class3").replace("\\","/"),if_last=False)
# class1_trained_model_path=os.path.join(PROJECT_BASE,"models/yolov8_class1/train/weights/best.pt").replace("\\","/")
class1_trained_model_path=yolo_python_api.get_best_pt_path(os.path.join(PROJECT_BASE,"models/yolov8_class1").replace("\\","/"),if_last=False)
# seg_trained_model_path=os.path.join(PROJECT_BASE,"models/yolov8_segmentation/train/weights/best.pt").replace("\\","/")
seg_trained_model_path=yolo_python_api.get_best_pt_path(os.path.join(PROJECT_BASE,"models/yolov8_segmentation").replace("\\","/"),if_last=False)
# seg_trained_model_path=os.path.join("yolov8n-seg.pt").replace("\\","/")
# 使用原始的yolov8n-seg.pt，梦搞出更多的mask，可以让生成的mask不是全黑的

if __name__ == "__main__":
    print('Infer test start')
    print(f'class3_trained_model_path: {class3_trained_model_path}')
    print(f'class1_trained_model_path: {class1_trained_model_path}')
    print(f'seg_trained_model_path: {seg_trained_model_path}')
    
    
    # 测试一个pipeline
    # 仅仅结合了3分类模型和seg模型，就拼凑出结果
    class3_model=YOLO(class3_trained_model_path)
    seg_model=YOLO(seg_trained_model_path)

    for img_path in jpg_files:
        class3_results = class3_model(img_path)
        seg_results = seg_model(img_path)
        
        # with open('test3.txt','w') as f:
        #     sys.stdout = f  # 将标准输出重定向到文件对象
        #     print(seg_results)
        #     sys.stdout = sys.__stdout__
        
        
        assert len(class3_results)==1
        assert len(seg_results)==1
        class3_result=class3_results[0]
        seg_result=seg_results[0]
        
        # 生成mask图片和json文件
        ## 保存的文件名后缀为时间戳
        timestamp = time.time()
        readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d-%H-%M-%S')
        ## 保存的文件路径
        save_path_name=os.path.join(test_output_dir,readable_time+'-'+os.path.basename(img_path)).replace("\\","/")
        mask_path=save_path_name.replace(".jpg",".png")
        json_path=save_path_name.replace(".jpg",".json")
        ## 保存
        obj = generate_handin_obj_v1(class3_result,seg_result,mask_path,color=(255,255,255))
        with open(json_path, 'w') as f:
            json.dump(obj, f, indent=4)