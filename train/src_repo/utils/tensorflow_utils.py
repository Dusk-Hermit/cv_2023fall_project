import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import base64
from PIL import Image
from io import BytesIO

def encode_img_with_filepath_write_to_tensorboard(img_path,file_writer):
	with open(img_path, 'rb') as image_file:
		image_binary = image_file.read()
  
	encoded_image = base64.b64encode(image_binary).decode('utf-8')
	with file_writer.as_default():
		tf.summary.text('Encoded___'+img_path, encoded_image, step=0) 
        # 这里第一个参数是tensorboard中显示该写入的变量的名字，可以修改使得它readable

def write_things_to_tensorboard(target_train):
    # 将target_train文件夹中的png jpg csv 全部写入该文件夹的tensorboard下面一个子文件夹
    log_dir=os.path.join(target_train,'logs').replace("\\","/")
    
    file_writer = tf.summary.create_file_writer(log_dir)
    
    # 读取png文件，并写入tensorboard
    for file in os.listdir(target_train):
        if file.endswith('.png'):
            img_path=os.path.join(target_train,file).replace("\\","/")
            # read png
            image = tf.io.read_file(img_path)
            image = tf.image.decode_png(image, channels=4)

            # 添加batch维度
            image = tf.expand_dims(image, 0)

            with file_writer.as_default():
                tf.summary.image(file, image, step=0)
            encode_img_with_filepath_write_to_tensorboard(img_path,file_writer)
    
    # 读取jpg文件，并写入tensorboard
    for file in os.listdir(target_train):
        if file.endswith('.jpg'):
            img_path=os.path.join(target_train,file).replace("\\","/")
            # read jpg
            image = tf.io.read_file(img_path)
            image = tf.image.decode_jpeg(image, channels=3)

            # 添加batch维度
            image = tf.expand_dims(image, 0)
            with file_writer.as_default():
                tf.summary.image(file, image, step=0)
            encode_img_with_filepath_write_to_tensorboard(img_path,file_writer)
    
    weight_path=os.path.join(target_train,'weights').replace("\\","/")
    for file in os.listdir(weight_path.replace("\\","/")):
        if file.endswith('.pt'):
            pt_path=os.path.join(weight_path,file).replace("\\","/")
            encode_img_with_filepath_write_to_tensorboard(pt_path,file_writer)
    
if __name__ == "__main__":
    write_things_to_tensorboard(r"C:\Users\Dusk_Hermit\Desktop\test\tensorflow_test\train\tensorboard\segmentation\train7")