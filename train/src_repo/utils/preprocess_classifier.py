### 看v2版本的，这个文件没写完弃了

import numpy as np
import torch
import os
import glob
import sys
# Add the parent directory to the sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from config import *
import json

# 写的有点乱……

static_mask_train_dataset_path=os.path.join(PROJECT_BASE,r"src_repo\datasets\segmentation\masks\train").replace("\\","/")

def read_processed_class3_dataset(dataset_path):
    # 根据dataset_path，读取所有的label文件的绝对路径
    # dataset_path = os.path.join(PROJECT_BASE, 'src_repo', 'datasets', 'class3').replace("\\", "/")
    train_label_path=os.path.join(dataset_path,'labels','train').replace("\\","/")
    label_files=glob.glob(f"{train_label_path}/*.txt")
    return label_files

def read_processed_class3_label(filepath,static_mask_train_dataset_path):
    # filepath是一个label文件的绝对路径
    # print(f'filepath: {filepath}')
    filepath=filepath.replace("\\","/")
    with open(filepath, 'r') as f:
        lines = f.read()
    line_list = lines.split('\n')
    line_list = [line.strip() for line in line_list if line.strip()]
    num_samples = len(line_list)
    num_attributes = len(line_list[0].split())

    data=[]
    
    for i, line in enumerate(line_list):
        # print(line)
        line_data = line.split(' ')
        line_data = [x.strip() for x in line_data if x.strip()]
        line_data = [float(x) for x in line_data]
        # 输出的data是一个list，每个元素是一个tuple，tuple的第一个元素是一个tensor，第二个元素是一个mask的路径
        a_piece_of_data=(torch.tensor(line_data),get_mask_path_from_label_filepath(filepath,static_mask_train_dataset_path))
        data.append(a_piece_of_data)

    return data

def get_mask_path_from_label_filepath(label_filepath,static_mask_train_dataset_path):
    # 输入的label_filepath，一个txt文件的绝对路径，输出对应的同名mask文件的绝对路径
    basename=os.path.basename(label_filepath)
    basename=basename.replace('.txt','.png')
    mask_path=os.path.join(static_mask_train_dataset_path,basename).replace("\\","/")
    return mask_path

def generate_train_dataset_for_binary_classifier(dataset_path,static_mask_train_dataset_path):
    # 传入的是末端classifier所需要的数据集的路径：classifier所需要的数据集从class3数据集中生成
    label_files=read_processed_class3_dataset(dataset_path)
    data=[]
    for label_file in label_files:
        t_0=read_processed_class3_label(label_file,static_mask_train_dataset_path)
        data.extend(t_0)
    return data

# 创建torch dataset 类



if __name__ == '__main__':
    # label_files=read_processed_class3_dataset(os.path.join(PROJECT_BASE, 'src_repo', 'datasets', 'class3').replace("\\", "/"))
    # for label_file in label_files:
    #     t_0=read_processed_class3_label(label_file，generate_train_dataset_for_binary_classifier)
    #     # print(t_0)
        
    #     # shape= (n,56)
    #     # n为一个label文件中的line num
    #     # 56 = 1 分类 + 4 bbox + 17 keypoints*3
    #     print(t_0.shape)
    data = generate_train_dataset_for_binary_classifier(os.path.join(PROJECT_BASE, 'src_repo', 'datasets', 'class3').replace("\\", "/"),static_mask_train_dataset_path)
    for data_element in data:
        class_label=data_element[0][0]
        