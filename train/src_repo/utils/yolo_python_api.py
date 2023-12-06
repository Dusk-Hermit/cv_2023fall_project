from ultralytics import YOLO
from config import *
import os 
static_pose_model_path=os.path.join(PROJECT_BASE,r"src_repo\yolov8n-pose.pt").replace("\\","/")
static_seg_model_path=os.path.join(PROJECT_BASE,r"src_repo\yolov8n-seg.pt").replace("\\","/")

class3_yaml_path=os.path.join(PROJECT_BASE,r"src_repo\yaml_files\class3.yaml").replace("\\","/")
class1_yaml_path=os.path.join(PROJECT_BASE,r"src_repo\yaml_files\class1.yaml").replace("\\","/")
seg_yaml_path=os.path.join(PROJECT_BASE,r"src_repo\yaml_files\segmentation.yaml").replace("\\","/")

train_project_base=os.path.join(PROJECT_BASE,r"models").replace("\\","/")

import os

def get_newest_train(dirname):
    # 对于'train'文件夹可能有些小bug
    if not os.path.exists(dirname):
        return None
    # 从dirname中找到最新的best.pt文件
    train_dir_list=os.listdir(dirname)
    if len(train_dir_list)==0:
        return None
    # dir_list是形如train, train2, train3, ...的列表，需要将他们排序
    for i,x in enumerate(train_dir_list):
        if x=='train':
            train_dir_list[i]='train1'
    train_dir_list=[x for x in train_dir_list if not x.startswith('predict')]
    train_dir_list.sort(key=lambda x:int(float(x[5:])))
    return os.path.join(dirname,train_dir_list[-1]).replace("\\","/")

def get_best_pt_path(dirname,if_last=False):
    if not os.path.exists(dirname):
        return None
    # 从dirname中找到最新的best.pt文件
    train_dir_list=os.listdir(dirname)
    if len(train_dir_list)==0:
        return None
    # dir_list是形如train, train2, train3, ...的列表，需要将他们排序
    for i,x in enumerate(train_dir_list):
        if x=='train':
            train_dir_list[i]='train1'
    train_dir_list=[x for x in train_dir_list if not x.startswith('predict')]
    train_dir_list.sort(key=lambda x:int(float(x[5:])))
    
    # 从后往前找，找到第一个best.pt文件
    target='best.pt'
    if if_last:
        target='last.pt'
    for train_dir in reversed(train_dir_list):
        best_pt_path=os.path.join(dirname,train_dir,'weights',target).replace("\\","/")
        if os.path.exists(best_pt_path):
            return best_pt_path
    return None

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
    sample_path=r'C:\Users\Dusk_Hermit\Desktop\test\tensorflow_test\train\models\1'.replace("\\","/")
    print(get_best_pt_path(sample_path))