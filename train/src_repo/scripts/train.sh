#!/bin/bash

PROJECT_BASE="/project/train"

# 人体关键点识别任务

# 训练：yolo class3 三分类训练
yolo pose train \
    data="$PROJECT_BASE/src_repo/yaml_files/class3.yaml" \
    model="$PROJECT_BASE/src_repo/yolov8n-pose.pt" \
    epochs=10 \
    imgsz=640 \
    project="$PROJECT_BASE/models/yolov8_class1"

# 训练：yolo只预测人
yolo pose train \
    data="$PROJECT_BASE/src_repo/yaml_files/class1.yaml" \
    model="$PROJECT_BASE/src_repo/yolov8n-pose.pt" \
    epochs=10 \
    imgsz=640 \
    project="$PROJECT_BASE/models/yolov8_class3"

# 栏杆seg任务
yolo segment train \
    data="$PROJECT_BASE/src_repo/yaml_files/segmentation.yaml" \
    model="$PROJECT_BASE/src_repo/yolov8n-seg.pt" \
    epochs=10 \
    imgsz=640 \
    project="$PROJECT_BASE/models/yolov8_segmentation"
