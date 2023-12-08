import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
from config import *
from PIL import Image
import base64
from PIL import Image
from io import BytesIO
import glob
pt_encoded_path=r'C:\Users\Dusk_Hermit\Desktop\test\tensorflow_test\train\src_repo\temp.txt'
output_path=r'C:\Users\Dusk_Hermit\Desktop\test\tensorflow_test\train\src_repo\temp.pt'

def decode_pt(pt_encoded_path,output_path):
    with open(pt_encoded_path) as f:
        pt_encoded=f.read()
    decoded_image_binary = base64.b64decode(pt_encoded)
    with open(output_path, 'wb') as f:
        f.write(decoded_image_binary)

decode_pt(pt_encoded_path,output_path)

from ultralytics import YOLO
model=YOLO(output_path)

val_jpg_folder_path=os.path.join(PROJECT_BASE,"src_repo/datasets/class3/images/val").replace("\\","/")
jpg_files = glob.glob(f"{val_jpg_folder_path}/*.jpg")
for img_path in jpg_files:
    results = model(img_path)

