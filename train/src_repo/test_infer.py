from config import *
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
import glob
import cv2
from ultralytics.utils.plotting import Annotator
from matplotlib import pyplot as plt
import os

# 推理，三分类模型
# 路径需要更换就更换
class3_trained_model_path=os.path.join(PROJECT_BASE,"models/yolov8_class3/train/weights/best.pt").replace("\\","/")
model=YOLO(class3_trained_model_path)

val_jpg_folder_path=os.path.join(PROJECT_BASE,"src_repo/datasets/class3/images/val").replace("\\","/")
jpg_files = glob.glob(f"{val_jpg_folder_path}/*.jpg")

# https://stackoverflow.com/questions/75324341/yolov8-get-predicted-bounding-box
for img_path in jpg_files:
    results = model(img_path)
    for i, r in enumerate(results):
        print(vars(r))
        # print(r.boxes)
        # print(r.keypoints)
    break



