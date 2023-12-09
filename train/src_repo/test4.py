import os
import sys
import tensorflow as tf
import base64
from PIL import Image
from io import BytesIO
import time

pt1_file=''
pt2_file=''

def encode_img_with_filepath_write_to_tensorboard(img_path,file_writer):
    with open(img_path, 'rb') as image_file:
        image_binary = image_file.read()
  
    encoded_image = base64.b64encode(image_binary).decode('utf-8')
    with file_writer.as_default():
        time_str=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        tf.summary.text('Encoded___'+ time_str +img_path, encoded_image, step=0) 
        # 这里第一个参数是tensorboard中显示该写入的变量的名字，可以修改使得它readable

tensorboard_path='/project/train/tensorboard/logs'

for i in range(30):
    file_writer = tf.summary.create_file_writer(tensorboard_path)
    encode_img_with_filepath_write_to_tensorboard(pt1_file,file_writer)
    encode_img_with_filepath_write_to_tensorboard(pt2_file,file_writer)