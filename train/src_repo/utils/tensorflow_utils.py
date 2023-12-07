import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
from PIL import Image
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
    
if __name__ == "__main__":
    write_things_to_tensorboard(r"C:\Users\Dusk_Hermit\Desktop\test\tensorflow_test\train\tensorboard\segmentation\train7")