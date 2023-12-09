import os
import json
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import glob
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
import shutil
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
# Add the parent directory to the sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
import time
from utils.tensorflow_utils import *

# 这一行可以注释掉，注释掉需要修改下面的原始数据集路径
from config import *

FIXED_IMG_SIZE=(224,224)
# 或者是(1080,1920)？ 但是感觉太大，而且就算变换成正方形也无所谓吧
# preprocess就需要resize成正方形，否则不支持batch


def get_newest_classifier_model_path():
    classifier_model_dir_path=os.path.join(PROJECT_BASE,'models','classifier').replace("\\","/")
    list=os.listdir(classifier_model_dir_path)
    # sort and get the newest model
    list.sort(key=lambda fn: os.path.getmtime(os.path.join(classifier_model_dir_path, fn)))
    return os.path.join(classifier_model_dir_path,list[-1]).replace("\\","/")
    

def return_ready_datapiece(img_path):
    # 输入的img_path，是一个jpg文件的绝对路径，同名的json和png文件也应当存在
    # 给dataset init 做preprocess用的
    keypoint_path=img_path.replace('.jpg','.json')
    mask_path=img_path.replace('.jpg','.png')
    data=[]
    with open(keypoint_path, 'r',encoding='utf-8') as f:
        text=f.read()
        keypoint_data = json.loads(text)

    for annotation in keypoint_data['annotations']:
        image_id = annotation['image_id']
        keypoints = annotation['keypoints']
        keypoints = torch.tensor(keypoints).float().squeeze()

        class_name = annotation['category_name']
        if(class_name == "person"):
            category_id = 0
        else:
            category_id = 1
        # data 的每行格式如下
        data.append((category_id,keypoints,mask_path))
    return data

def process_mask(mask_path,keypoints):
    mask = Image.open(mask_path)
    width, height = mask.size
    assert len(keypoints) == 17*3
    for i in range(17*3):
        if i % 3 == 0:
            keypoints[i] = keypoints[i] / width
        elif i % 3 == 1:
            keypoints[i] = keypoints[i] / height
        else:
            keypoints[i] = keypoints[i] /2 # visibility = 0, 1, 2
    
    # 先禁用transform，transform在这里能干嘛？
    # mask_tensor = transforms.ToTensor()(mask)
    # mask_tensor = transforms.ToTensor()(mask).squeeze()
    mask_tensor = torch.from_numpy(np.array(mask)).float().squeeze()
    
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
    # print(mask_tensor.unsqueeze(0).shape)
    resized_mask_tensor = transforms.Resize(FIXED_IMG_SIZE)(mask_tensor.unsqueeze(0))
    
    # Standardize the mask tensor
    # 但是本身取值就是0.0和1.0，不需要standardize
    # min_value = mask_tensor.min()
    # max_value = mask_tensor.max()
    # print(f'min_value: {min_value}')
    # print(f'max_value: {max_value}')

    # resized_mask_tensor = (mask_tensor - min_value) / (max_value - min_value)
    return resized_mask_tensor,keypoints

class CustomDataset(Dataset):
    def __init__(self, raw_folder, transform=None):
        self.raw_folder = raw_folder
        self.transform = transform

        # preprocess stage
        # Get only filenames with .jpg extension
        self.filenames = glob.glob(f"{raw_folder}/*.jpg")
        self.data=[]
        for filename in self.filenames:
            self.data.extend(return_ready_datapiece(filename))
        print(f'len(self.data): {len(self.data)}')
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        category_id,keypoints,mask_path=self.data[idx]
        resized_mask_tensor,keypoints = process_mask(mask_path,keypoints)

        return category_id, keypoints, resized_mask_tensor

class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 待定，用卷积也行，卷积的话就是考虑局部性，可以分析一下需求
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.fc1 = nn.Linear(224*224, 128)
        self.fc2 = nn.Linear(128+ 51, 2)

    def forward(self, x):
        keypoints, mask_tensor = x
        # mask_tensor: (batch_size, 1, 224, 224)
        # keypoints: (batch_size, 51)
        # 待定，不过输出的肯定是一个(batch_size, 2)的tensor，是二分类嘛
        
        mask_tensor = self.flatten1(mask_tensor)
        keypoints = self.flatten2(keypoints)
        mask_tensor = self.fc1(mask_tensor)
        x=torch.cat((mask_tensor,keypoints),1)
        x = self.fc2(x)
        return x

def generate_train_tools(net):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    return criterion,optimizer,scheduler

def train(net,config_dict,tools):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    net.to(device)
    
    criterion,optimizer,scheduler = tools
    nepoch=config_dict['nepoch']
    batch_size=config_dict['batch_size']
    log_freq=config_dict['log_freq']
    train_dataset=CustomDataset(config_dict['train_dataset_path'])
    train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    
    start_time=time.time()
    for epoch in range(nepoch):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_start_time=time.time()
        loss_list=[]
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            
            category_id, keypoints, mask_tensor = data
            category_id = category_id.to(device)
            keypoints = keypoints.to(device)
            mask_tensor = mask_tensor.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net((keypoints,mask_tensor))
            loss = criterion(outputs, category_id)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % log_freq == log_freq-1:    # print every `log_freq` mini-batches
                print(f'epoch: {epoch+1}, batch: {i+1}, loss: {running_loss / log_freq}')
                loss_list.append(running_loss / log_freq)
                running_loss = 0.0
        scheduler.step()
        print(f'epoch {epoch+1} finished, time elapsed: {time.time()-epoch_start_time}, epoch mean loss: {np.mean(loss_list)}')
    print(f'Finished Training, time elapsed: {time.time()-start_time}')

def save_model(net):
    time_str=time.strftime("%Y%m%d%H%M%S", time.localtime())
    model_dir=os.path.join(PROJECT_BASE,'models','classifier').replace("\\","/")
    os.makedirs(model_dir,exist_ok=True)
    save_model_path=os.path.join(model_dir,f'classifier_{time_str}.pth').replace("\\","/")
    torch.save(net.state_dict(), save_model_path)
    
    tensorboard_dir=os.path.join(PROJECT_BASE,'tensorboard').replace("\\","/")
    log_dir=os.path.join(tensorboard_dir,'classifier').replace("\\","/")
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    file_writer = tf.summary.create_file_writer(log_dir)
    encode_img_with_filepath_write_to_tensorboard(save_model_path,file_writer,)
    
    
    

if __name__ == '__main__':
    TEST_STAGE=2
    raw_folder = SOURCE_DATA
    # raw_folder = os.path.join('/home/data/2788').replace("\\", "/")
    if TEST_STAGE==1:

        # Example usage
        transform = transforms.Compose([
            # Add your desired transformations here
            
        ])
        dataset = CustomDataset(raw_folder, transform=transform)

        # Adjust batch_size, shuffle, and other DataLoader parameters based on your requirements
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        # Iterate through the dataloader to get batches of data
        for batch in dataloader:
            category_id, keypoints, mask_tensor = batch

            print(category_id)
            print(keypoints)
            print(keypoints.shape)
            print(mask_tensor)
            print(mask_tensor.shape)
            print('-'*80)
            
            # # 可以看一下mask的histogram，值分布
            # flat_tensor = mask_tensor.view(-1)
            # # Compute the histogram
            # histogram = torch.histc(flat_tensor, bins=100, min=flat_tensor.min(), max=flat_tensor.max())
            # # Plot the histogram
            # plt.bar(range(len(histogram)), histogram)
            # plt.xlabel('Value')
            # plt.ylabel('Frequency')
            # plt.title('Histogram')
            # plt.show()
            
            break
    elif TEST_STAGE==2:
        net = CustomNet()
        tools = generate_train_tools(net)
        train(
            net,
            {
                'nepoch':10,
                'batch_size':2,
                'log_freq':4, # 样例集的样本太少了
                'train_dataset_path':raw_folder,
            },
            tools,
        )
        save_model(net)
    elif TEST_STAGE==3:
        net_path=os.path.join(PROJECT_BASE,'models','classifier','classifier_20231209033526.pth').replace("\\","/")
        net = CustomNet()
        print(net)
        print(type(net))
        ################# WARNING
        # net = net.load_state_dict(torch.load(net_path)) # ERROR!!!!!
        net.load_state_dict(torch.load(net_path))
