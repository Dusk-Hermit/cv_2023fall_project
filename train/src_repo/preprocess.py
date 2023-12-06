from config import *
import os
import sys
sys.path.insert(0, os.path.join(PROJECT_BASE, "src_repo",'utils').replace("\\","/"))
from utils import preprocess_class3
from utils import preprocess_class1_post
from utils import preprocess_seg_1_split
from utils import preprocess_seg_2_mask_to_yolo
# 该脚本的执行，会将原始数据集，转换为多个模型需要的训练数据集，并储存在src_repo/datasets文件夹下

DATASETS_PATH= os.path.join(PROJECT_BASE, "src_repo/datasets").replace("\\","/")

# 生成3分类的yolov8所需要的数据集

# yaml文件使用的数据集路径
class3_datasets_path=os.path.join(DATASETS_PATH, "class3").replace("\\","/")

class_names = ['jump_over_railing', 'climb_over_railing', 'person']
class_id_map = {class_name: i for i, class_name in enumerate(class_names)}

preprocess_class3.convert_and_split_dataset(SOURCE_DATA, class3_datasets_path, class_names,test_size=0.2,random_state=42)

# 生成1分类的yolov8所需要的数据集

# yaml文件使用数据集路径
class1_datasets_path=os.path.join(DATASETS_PATH, "class1").replace("\\","/")

preprocess_class1_post.copy_and_modify_text_files(class3_datasets_path, class1_datasets_path)

# 生成栏杆seg所需要的数据集

# yaml文件使用数据集路径
seg_datasets_path=os.path.join(DATASETS_PATH, "segmentation").replace("\\","/")

seg_images_datasets_path=os.path.join(DATASETS_PATH,'segmentation', "images").replace("\\","/")
seg_masks_datasets_path=os.path.join(DATASETS_PATH,'segmentation', "masks").replace("\\","/")
seg_labels_datasets_path=os.path.join(DATASETS_PATH,'segmentation', "labels").replace("\\","/")
preprocess_seg_1_split.split_custom_dataset(SOURCE_DATA, seg_images_datasets_path, seg_masks_datasets_path, train_ratio=0.8, val_ratio=0.15, test_ratio=0.05)

for split in ['train', 'val','test']:
    mask_path_of_spilt=os.path.join(seg_masks_datasets_path,split).replace("\\","/")
    label_path_of_spilt=os.path.join(seg_labels_datasets_path,split).replace("\\","/")
    preprocess_seg_2_mask_to_yolo.mask_to_yolo(mask_path_of_spilt, label_path_of_spilt)
    
print('Preprocessing Finished')