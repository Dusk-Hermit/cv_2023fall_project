import os
import shutil
from sklearn.model_selection import train_test_split

def split_custom_dataset(dataset_dir, output_images_dir, output_masks_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # 获取所有jpg图像文件路径
    jpg_images = [f for f in os.listdir(dataset_dir) if f.lower().endswith('.jpg')]
    total_images = len(jpg_images)

    # 划分数据集
    train_images, test_images = train_test_split(jpg_images, test_size=test_ratio, random_state=42)
    train_images, val_images = train_test_split(train_images, test_size=val_ratio / (1 - test_ratio), random_state=42)

    # 创建输出目录
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)

    # 将图像复制到相应的目录
    for split, images in [('train', train_images), ('val', val_images), ('test', test_images)]:
        split_images_dir = os.path.join(output_images_dir, split)
        os.makedirs(split_images_dir, exist_ok=True)
        split_masks_dir = os.path.join(output_masks_dir, split)
        os.makedirs(split_masks_dir, exist_ok=True)

        for image in images:
            # 处理相同命名的 jpg 和 png 文件
            image_name, _ = os.path.splitext(image)
            png_filename = image_name + '.png'

            # 复制 jpg 图像
            src_jpg_path = os.path.join(dataset_dir, image)
            dst_jpg_path = os.path.join(split_images_dir, image)
            shutil.copy(src_jpg_path, dst_jpg_path)

            # 复制 PNG 文件（如果存在）
            src_png_path = os.path.join(dataset_dir, png_filename)
            dst_png_path = os.path.join(split_masks_dir, png_filename)
            if os.path.exists(src_png_path):
                shutil.copy(src_png_path, dst_png_path)

    print(f"Dataset split into {len(train_images)} training, {len(val_images)} validation, and {len(test_images)} test images.")

if __name__ == "__main__":
    # 自建数据集的目录 (update with your path)
    # dataset_directory = './2788'
    dataset_directory = r'C:\Users\Dusk_Hermit\Desktop\test\tensorflow_test\archive'.replace("\\","/")

    # 输出目录
    # output_images_directory = '/project/train/src_repo/yolov8/datasets/images'
    output_images_directory = r'C:\Users\Dusk_Hermit\Desktop\test\tensorflow_test\train/src_repo/datasets_mask/images'.replace("\\","/")
    # output_masks_directory = '/project/train/src_repo/yolov8/datasets/masks'
    output_masks_directory = r'C:\Users\Dusk_Hermit\Desktop\test\tensorflow_test\train/src_repo/datasets_mask/masks'.replace("\\","/")

    split_custom_dataset(dataset_directory, output_images_directory, output_masks_directory)
