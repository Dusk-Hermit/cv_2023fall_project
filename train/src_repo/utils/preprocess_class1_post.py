# 修改datasets文件夹下的所有txt文件，将每行第一个字符（类别）替换为0，已进行1分类的pose识别

import shutil
import glob
import os

def copy_and_modify_text_files(source_folder, destination_folder):
    # 创建目标文件夹
    os.makedirs(destination_folder, exist_ok=True)

    # 复制文件夹中的所有内容
    shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)

    # 查找并修改所有 .txt 文件
    txt_files = glob.glob(os.path.join(destination_folder, '**/*.txt'), recursive=True)
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            content = f.read()

        # 修改第一个字符
        if content:
            txt_lines=content.split("\n")
            txt_lines=['0'+line[1:] for line in txt_lines if line]
            modified_content = "\n".join(txt_lines)

            # 写回文件
            with open(txt_file, 'w') as f:
                f.write(modified_content)

if __name__ == "__main__":
    source_folder = r"C:\Users\Dusk_Hermit\Desktop\test\tensorflow_test\train\src_repo\datasets"  # 替换为你的源文件夹路径
    destination_folder = r"C:\Users\Dusk_Hermit\Desktop\test\tensorflow_test\train\src_repo\datasets_1"  # 替换为你的目标文件夹路径

    copy_and_modify_text_files(source_folder, destination_folder)
