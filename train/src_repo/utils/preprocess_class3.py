import json
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import argparse

class_names = ['jump_over_railing', 'climb_over_railing', 'person']
class_id_map = {class_name: i for i, class_name in enumerate(class_names)}

def convert_to_yolo(json_data, output_path, class_names):
    with open(output_path, 'w') as file:
        for annotation in json_data['annotations']:
            image_id = annotation['image_id']
        
            keypoints = annotation['keypoints']

            # 获取类别名称并确保类别名称在 class_names 列表中
            # class_name = class_names[category_id - 1]
            class_name = annotation['category_name']
            category_id = class_id_map[class_name]
            # category_id = 0

            # 计算包围框中心和宽高
            bbox = annotation['bbox']
            x_center = (bbox[0] + bbox[2] / 2) / json_data['images'][0]['width']
            y_center = (bbox[1] + bbox[3] / 2) / json_data['images'][0]['height']
            width = bbox[2] / json_data['images'][0]['width']
            height = bbox[3] / json_data['images'][0]['height']

            # 构建 YOLO 行
            yolo_line = f'{category_id} {x_center} {y_center} {width} {height}'

            # 添加关键点信息
            for i in range(0, len(keypoints), 3):
                px, py, visibility = keypoints[i] / json_data['images'][0]['width'], keypoints[i + 1]/ json_data['images'][0]['height'], keypoints[i + 2]
                yolo_line += f' {px} {py} {visibility}'

            file.write(yolo_line + '\n')

def convert_and_split_dataset(data_folder, dest_folder, class_names, test_size=0.2, random_state=42):
                # 数据集划分
    json_files = list(Path(data_folder).glob('*.json'))
    train_files, val_files = train_test_split(json_files, test_size=test_size, random_state=random_state)
    os.makedirs(dest_folder, exist_ok=True)
    
    output_folder_images_train = os.path.join(dest_folder, 'images', 'train')
    output_folder_images_val = os.path.join(dest_folder, 'images', 'val')
    output_folder_labels_train = os.path.join(dest_folder, 'labels', 'train')
    output_folder_labels_val = os.path.join(dest_folder, 'labels', 'val')
    os.makedirs(output_folder_images_train, exist_ok=True)
    os.makedirs(output_folder_images_val, exist_ok=True)
    os.makedirs(output_folder_labels_train, exist_ok=True)
    os.makedirs(output_folder_labels_val, exist_ok=True)
    # 转换训练集
    for json_file in tqdm(train_files, desc='Converting training set'):
        with open(json_file,encoding='utf-8') as f:
            json_data = json.load(f)

        src_path = os.path.join(data_folder, json_file.stem + '.jpg')
        shutil.copy(src_path, output_folder_images_train)
        # 构建输出文件路径
        output_path = os.path.join(output_folder_labels_train, json_file.stem + '.txt')
        # 执行转换
        convert_to_yolo(json_data, output_path, class_names)

    # 转换验证集
    for json_file in tqdm(val_files, desc='Converting validation set'):
        with open(json_file,encoding='utf-8') as f:
            json_data = json.load(f)

        
        src_path = os.path.join(data_folder, json_file.stem + '.jpg')
        shutil.copy(src_path, output_folder_images_val)

        output_path = os.path.join(output_folder_labels_val, json_file.stem + '.txt')

        # 执行转换
        convert_to_yolo(json_data, output_path, class_names)

if '__main__' == __name__:
    # 设置参数
    # data_folder = '/home/data/2788'
    data_folder = r'C:\Users\Dusk_Hermit\Desktop\test\tensorflow_test\archive'.replace("\\","/")
    # output_folder = '/project/train/src_repo/datasets'
    output_folder = r'C:\Users\Dusk_Hermit\Desktop\test\tensorflow_test\train/src_repo/datasets'.replace("\\","/")

    class_names = ['jump_over_railing', 'climb_over_railing', 'person']
    class_id_map = {class_name: i for i, class_name in enumerate(class_names)}
    convert_and_split_dataset(data_folder, output_folder, class_names,test_size=0.2,random_state=42)
