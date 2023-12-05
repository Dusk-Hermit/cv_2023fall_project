# 测试ji.py的输出结果

from ji import *
from config import *
import os

sample_img_path=os.path.join(PROJECT_BASE,r"src_repo\datasets\class3\images\train\street_1_040335.jpg").replace("\\","/")

init_handle=init()
sample_img=cv2.imread(sample_img_path)
sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

output_folder_path=os.path.join(os.path.dirname(PROJECT_BASE),'ev_sdk',"output").replace("\\","/")
os.makedirs(output_folder_path,exist_ok=True)

args={
    "mask_output_path": os.path.join(output_folder_path,"mask_output.png").replace("\\","/")
}

json_str = process_image(handle=init_handle, input_image=sample_img, args=json.dumps(args))

output_json_path=os.path.join(output_folder_path,"output.json").replace("\\","/")
with open(output_json_path,'w') as f:
    f.write(json_str)