import os 
import sys

# Add the parent directory to the sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from config import *
import shutil
from ultralytics import YOLO

static_pose_model_path=os.path.join(PROJECT_BASE,r"src_repo\yolov8n-pose.pt").replace("\\","/")
static_seg_model_path=os.path.join(PROJECT_BASE,r"src_repo\yolov8n-seg.pt").replace("\\","/")

class3_yaml_path=os.path.join(PROJECT_BASE,r"src_repo\yaml_files\class3.yaml").replace("\\","/")
class1_yaml_path=os.path.join(PROJECT_BASE,r"src_repo\yaml_files\class1.yaml").replace("\\","/")
seg_yaml_path=os.path.join(PROJECT_BASE,r"src_repo\yaml_files\segmentation.yaml").replace("\\","/")

train_project_base=os.path.join(PROJECT_BASE,r"models").replace("\\","/")

import os

def get_best_pt_path(dirname,if_last=False):
    # 传入的dirname是比如说./models/yolov8_class3
    if not os.path.exists(dirname):
        return None
    # 从dirname中找到最新的best.pt文件
    train_dir_list=os.listdir(dirname)
    train_dir_list=[x for x in train_dir_list if not x.startswith('predict')]
    if len(train_dir_list)==0:
        return None
    # dir_list是形如train, train2, train3, ...的列表，需要将他们排序
    
    temp_list=[]
    for i,x in enumerate(train_dir_list):
        if x=='train':
            temp_list.append((1,x))
        else:
            temp_list.append((int(float(x[5:])),x))
    temp_list.sort(key=lambda x:x[0])
    train_dir_list=[x[1] for x in temp_list]
    # 数字最大的文件夹，就是最新的训练文件夹
    
    # 从后往前找，找到第一个best.pt文件
    target='best.pt'
    if if_last:
        target='last.pt'
    for train_dir in reversed(train_dir_list):
        best_pt_path=os.path.join(dirname,train_dir,'weights',target).replace("\\","/")
        if os.path.exists(best_pt_path):
            return best_pt_path
    return None

def get_newest_train(dirname):
    # # 对于'train'文件夹可能有些小bug
    # if not os.path.exists(dirname):
    #     return None
    # # 从dirname中找到最新的best.pt文件
    # train_dir_list=os.listdir(dirname)
    # train_dir_list=[x for x in train_dir_list if not x.startswith('predict')]
    # if len(train_dir_list)==0:
    #     return None
    # # dir_list是形如train, train2, train3, ...的列表，需要将他们排序
    # for i,x in enumerate(train_dir_list):
    #     if x=='train':
    #         train_dir_list[i]='train1'
    # train_dir_list.sort(key=lambda x:int(float(x[5:])))
    # return os.path.join(dirname,train_dir_list[-1]).replace("\\","/")
    newest_pt_path=get_best_pt_path(dirname,if_last=True)
    if newest_pt_path is None:
        print('yolo_python_api.py --- get_newest_train: Impossible to get here.')
        return None
    return os.path.dirname(os.path.dirname(newest_pt_path)).replace("\\","/")

def yolo_train(
    pt_dir_path,
    alternate_pt,
    epochs,
    batch_size,
    project_name,
    yaml=seg_yaml_path,
    imgsz=1080,
    patience=20,
):
    best_pt=get_best_pt_path(pt_dir_path,if_last=True)
    if best_pt is None:
        best_pt = alternate_pt
    else:
        # 即存在best_pt，它放置在pt_dir_path下的某个文件夹中
        # 在这里执行：清楚pt_dir_path下的所有文件夹，只保留best_pt所在的文件夹
        # 并且把best_pt所在的文件夹之前的一些序号的文件夹，建立空的文件夹
        # 目的是1、使得yolo生成的文件夹序号是连续的；2、为了清理空间，满足极市平台的models文件夹下的文件数量限制
        best_pt_dir=os.path.dirname(os.path.dirname(best_pt)) # 如./models/yolov8_class3/train2
        best_pt_dir_basename=os.path.basename(best_pt_dir)
        for x in os.listdir(pt_dir_path):
            if x != best_pt_dir_basename:
                shutil.rmtree(os.path.join(pt_dir_path,x).replace("\\","/"))
        
        if best_pt_dir_basename!='train':
            best_pt_dir_basename_num=int(float(best_pt_dir_basename[5:]))
            for i in range(1,best_pt_dir_basename_num):
                os.makedirs(os.path.join(pt_dir_path,f'train{i if i !=1 else ""}').replace("\\","/"),exist_ok=True) 
    
    print(f'yolo_python_api.py --- pt used to train: {best_pt}')
        
    model = YOLO(best_pt)
    results = model.train(
        data=yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        patience=patience,
        project=os.path.join(train_project_base,project_name).replace("\\","/"),
    )




if __name__ == "__main__":
    sample_path=r'C:\Users\Dusk_Hermit\Desktop\test\tensorflow_test\train\models\yolov8_segmentation'.replace("\\","/")
    print(get_best_pt_path(sample_path))