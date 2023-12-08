import json
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import os
from config import *
from postprocess import *
import yolo_python_api
from utils.preprocess_classifier_v2 import *

class3_trained_model_path=yolo_python_api.get_best_pt_path(os.path.join(PROJECT_BASE,"models/yolov8_class3").replace("\\","/"),if_last=False)
class1_trained_model_path=yolo_python_api.get_best_pt_path(os.path.join(PROJECT_BASE,"models/yolov8_class1").replace("\\","/"),if_last=False)
seg_trained_model_path=yolo_python_api.get_best_pt_path(os.path.join(PROJECT_BASE,"models/yolov8_segmentation").replace("\\","/"),if_last=False)
# class3_trained_model_path=os.path.join(PROJECT_BASE,"models/yolov8_class3/train/weights/best.pt").replace("\\","/")
# class1_trained_model_path=os.path.join(PROJECT_BASE,"models/yolov8_class1/train/weights/best.pt").replace("\\","/")
# seg_trained_model_path=os.path.join(PROJECT_BASE,"models/yolov8_segmentation/train/weights/best.pt").replace("\\","/")

# hard link path
net_model_path=os.path.join(PROJECT_BASE,"models/classifier",'classifier_20231209033526.pth').replace("\\","/")


def init():
    """Initialize model
    Returns: model
    """
    class3_model=YOLO(class3_trained_model_path)
    seg_model=YOLO(seg_trained_model_path)
    classifier = CustomNet()
    classifier.load_state_dict(torch.load(net_model_path))
    
    return {
        "class3_model": class3_model,
        "seg_model": seg_model,
        "classifier": classifier,
    }

def process_image(handle=None, input_image=None, args=None, ** kwargs):
    """
        Do inference to analysis input_image and get output
            Attributes:
            handle: algorithm handle returned by init()
            input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
            args: string in JSON format, format: {
                "mask_output_path": "/path/to/output/mask.png"
            }
        Returns: process result
    """
    # Process image here
    rgb_image_numpy = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    # Convert the RGB image array to a PIL Image
    pil_image = Image.fromarray(rgb_image_numpy)
    # Save the PIL Image to a temporary file
    temp_image_path = 'temp.jpg'
    pil_image.save(temp_image_path)
    
    # Use model to process the image
    class3_model = handle["class3_model"]
    seg_model = handle["seg_model"]
    classifier = handle["classifier"]
    
    class3_results = class3_model(temp_image_path)
    seg_results = seg_model(temp_image_path)
    class3_result=class3_results[0]
    seg_result=seg_results[0]
    
    # mask output path
    args = json.loads(args)
    mask_output_path = args['mask_output_path']
    
    # Generate dummy mask data
    h, w, _ = input_image.shape
    

    obj = generate_handin_obj_v1(class3_result,seg_result,mask_output_path,image_shape=(h,w,3))

    object_num = len(obj["model_data"]["objects"])
    for i in range(object_num):
        keypoints = obj["model_data"]["objects"][i]["keypoints"]["keypoints"]
        # print(keypoints)
        # print('-'*80)
        keypoints = torch.tensor(keypoints).float()
        mask_tensor = process_mask(mask_output_path).squeeze()
        
        # print(classifier)
        # print(type(classifier))
        # print(keypoints)
        # print(type(keypoints))
        # print(keypoints.shape)
        # print(mask_tensor)
        # print(type(mask_tensor))
        # print(mask_tensor.shape)
        keypoints = keypoints.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)
        output = classifier((keypoints, mask_tensor))
        # binary classification
        _, prediction = torch.max(output, 1)
        if prediction == 0:
            obj["model_data"]["objects"][i]["name"] = "person"
            obj['algorithm_data']['target_info'][i]['name'] = 'person'
        else:
            obj["model_data"]["objects"][i]["name"] = "climb_over_railing"
            obj['algorithm_data']['target_info'][i]['name'] = 'climb_over_railing'

    ### kept for reference
    fake_result = {}
    fake_result["algorithm_data"] = {
        "is_alert": True,
        "target_count": 0,
        "target_info": []
    }
    fake_result["model_data"] = {
        "mask": "mask_output_path",
        "objects": [
            {
                "x": 1805,
                "y": 886,
                "width": 468,
                "height": 595,
                "confidence": 0.7937217950820923,
                "name": "person",
                "keypoints": {
                    "keypoints": [
                        2161.423828125,
                        990.58984375,
                        1.0,
                        2161.423828125,
                        981.29296875,
                        1.0,
                        2161.423828125,
                        981.29296875,
                        1.0,
                        2093.238525390625,
                        967.34765625,
                        1.0,
                        2124.231689453125,
                        985.94140625,
                        1.0,
                        2031.251708984375,
                        995.23828125,
                        1.0,
                        2068.443603515625,
                        1069.61328125,
                        1.0,
                        2093.238525390625,
                        1074.26171875,
                        1.0,
                        2124.231689453125,
                        1190.47265625,
                        1.0,
                        2161.423828125,
                        1088.20703125,
                        1.0,
                        2198.615966796875,
                        1130.04296875,
                        1.0,
                        1944.47021484375,
                        1185.82421875,
                        1.0,
                        2012.6556396484375,
                        1236.95703125,
                        1.0,
                        2124.231689453125,
                        1106.80078125,
                        0.0,
                        2186.218505859375,
                        1195.12109375,
                        1.0,
                        2130.430419921875,
                        1232.30859375,
                        0.0,
                        2173.8212890625,
                        1246.25390625,
                        1.0
                    ],
                    "score": 0.7535077333450317
                }
            }
        ]
    }
    
    # temp img clean up
    os.remove(temp_image_path)
    
    return json.dumps(obj, indent=4)