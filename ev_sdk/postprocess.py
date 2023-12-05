import os
import numpy as np
from PIL import Image
import cv2
from config import class_names
import torch
import sys
# 经验感觉results返回的都是只有一项，但没有找到文档说明

# 传入的result，是for result in results，这里的results是分割模型model(img_path)的返回值
# 该函数将result中的mask信息保存为png图片，其中将轮廓填充为color颜色
def seg_predict_result_to_png(
    result,
    save_path,
    image_shape,
    color=(1,1,1),
):
    image_array = np.zeros(image_shape,dtype=np.uint8)
    if result.masks :
        xy=result.masks.xy
        cv2.fillPoly(image_array, [np.int32(elem) for elem in xy], color=color)
    image = Image.fromarray(image_array)
    image.save(save_path)

# 这里传入的result也是results的一个元素，但它是检测模型model(img_path)的返回值
# alert logic v1 version
def create_ji_result_with_mask_path_v1(
    result,
    mask_path,
):
    boxes = result.boxes if hasattr(result, 'boxes') else []
    keypoints = result.keypoints if hasattr(result, 'keypoints') else []
    confs = boxes.conf if hasattr(boxes, 'conf') else []
    cls_list = boxes.cls if hasattr(boxes, 'cls') else []

    # Create a dictionary to store the processed results
    processed_result = {
        "algorithm_data": {
            "is_alert": False,  # You can set this based on your own logic
            "target_count": len(boxes) if boxes else 0,
            "target_info": []
        },
        "model_data": {
            "mask": mask_path,
            "objects": []
        }
    }

    for j in range(len(boxes)):
        box = boxes.xywh[j] 
        keypoint = keypoints[j] 
        conf = confs[j] 
        cls = cls_list[j] 
        # Example: Extracting keypoint data
        keypoint_data = keypoint.data if keypoint else []
        keypoint_data_copy = keypoint_data.detach().clone()
        keypoint_data_copy[:, :, 2] = torch.round(keypoint_data_copy[:, :, 2] * 2)
        keypoint_data_copy_list=keypoint_data_copy.squeeze().tolist()

        # Example: Creating object information
        object_info = {
            "x": int(box[0]),
            "y": int(box[1]),
            "width": int(box[2]),
            "height": int(box[3]),
            "confidence": float(conf) if conf else 0.0,
            "name": class_names[int(cls)],
        }
        target_info = object_info.copy()
        processed_result["algorithm_data"]["target_info"].append(target_info)
        object_info["keypoints"] = {
            "keypoints": keypoint_data_copy_list,
            "score": float(torch.mean(keypoint.conf)) if keypoint else 0.0,
        }
        
        # Append object information to the processed result
        processed_result["model_data"]["objects"].append(object_info)
        
        # Alert logic: 
        # v1: if any object detected is classified as class_names[1], then alert
        if object_info["name"]==class_names[1]:
            processed_result["algorithm_data"]["is_alert"]=True


    # Append the processed result to a list or use it as needed
    return processed_result

def generate_handin_obj_v1(
    class3_result,
    seg_result,
    save_path,
    image_shape=None, # 加入了image_shape参数，不使用也行
    color=(1,1,1),
):
    '''
    仅仅结合了3分类模型和seg模型，就拼凑出结果，这是第一版而已
    生成提交的结果文件，格式为json
    '''
    if image_shape is None:
        image_shape = class3_result.boxes.orig_shape
    seg_predict_result_to_png(seg_result,save_path,image_shape,color=color)
    return create_ji_result_with_mask_path_v1(class3_result,save_path)
    