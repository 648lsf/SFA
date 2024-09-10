"""Implementation of evaluate attack result."""
import os
import torch
from torch.autograd import Variable as V
from torch import nn
# from torch.autograd.gradcheck import zero_gradients # 
from torchvision import transforms as T
from Normalize import Normalize, TfNormalize
from PIL import Image
from torch.utils import data
import pandas as pd
import csv  
from loader import ImageNet,ImageNet_ssa
from torch.utils.data import DataLoader
from torch_nets import (
    tf_inception_v3,
    tf_inception_v4,
    tf_resnet_v2_50,
    tf_resnet_v2_101,
    tf_resnet_v2_152,
    tf_inc_res_v2,
    tf_adv_inception_v3,
    tf_ens3_adv_inc_v3,
    tf_ens4_adv_inc_v3,
    tf_ens_adv_inc_res_v2,
    )

batch_size = 20

input_csv = './dataset/images.csv'

input_dir = './dataset/images'

adv_dir = ''

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'tf2torch_inception_v3':
        net = tf_inception_v3
    elif net_name == 'tf2torch_inception_v4':
        net = tf_inception_v4
    elif net_name == 'tf2torch_resnet_v2_50':
        net = tf_resnet_v2_50
    elif net_name == 'tf2torch_resnet_v2_101':
        net = tf_resnet_v2_101
    elif net_name == 'tf2torch_resnet_v2_152':
        net = tf_resnet_v2_152
    elif net_name == 'tf2torch_inc_res_v2':
        net = tf_inc_res_v2
    elif net_name == 'tf2torch_adv_inception_v3':
        net = tf_adv_inception_v3
    elif net_name == 'tf2torch_ens3_adv_inc_v3':
        net = tf_ens3_adv_inc_v3
    elif net_name == 'tf2torch_ens4_adv_inc_v3':
        net = tf_ens4_adv_inc_v3
    elif net_name == 'tf2torch_ens_adv_inc_res_v2':
        net = tf_ens_adv_inc_res_v2
    else:
        print('Wrong model name!')

    model = nn.Sequential(
        
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        TfNormalize('tensorflow'),
        
        net.KitModel(model_path).eval().to(device))  ##  net.KitModel(model_path).eval().cuda(),)
    
    return model


class ImageNet_ssa(data.Dataset):
    def __init__(self, dir, csv_path, transforms = None):
        self.dir = dir   
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        # ImageID = img_obj['ImageId'] + '.png'
        ImageID = img_obj['ImageId'] + '.JPEG'
        Truelabel = img_obj['TrueLabel'] 
        TargetClass = img_obj['TargetClass'] - 1
        img_path = os.path.join(self.dir, ImageID)
        pil_img = Image.open(img_path).convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)
        return data, ImageID, Truelabel,img_path

    def __len__(self):
        return len(self.csv)
    
    
def verify(model_name, path):

    model = get_model(model_name, path)

    # X = ImageNet(adv_dir, input_csv, T.Compose([T.ToTensor()]))
    X = ImageNet_ssa(adv_dir, input_csv, T.Compose([T.ToTensor()]))
    data_loader = DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)  
    sum = 0
    record=[]
    for images, _, gt_cpu,path in data_loader: # gt_cpu 
        gt = gt_cpu.to(device)     # gt = gt_cpu.cuda()
        images = images.to(device) # images = images.cuda()
        with torch.no_grad():
            sum += (model(images)[0].argmax(1) != (gt)).detach().sum().cpu() # 
            incorrect_indices = (model(images)[0].argmax(1) != gt).nonzero(as_tuple=False).squeeze(1)  
            for idx in incorrect_indices:  
                # 假设path是一个列表，与数据集中的图像一一对应  
                record.append(path[idx.item()])  
            

    with open('error.csv', 'w', newline='') as file:  
        writer = csv.writer(file)  
        # 遍历列表，将每个元素作为一行写入CSV文件  
        for item in record:  
            writer.writerow([item])  # 注意这里要将元素包装成列表，因为writerow期望一个列表参数
    print(model_name + ' Attack success rate = {:.2%}'.format(sum / 1000.0))

def main():
    # model_names = ['tf2torch_inception_v3','tf2torch_adv_inception_v3','tf2torch_ens3_adv_inc_v3','tf2torch_inc_res_v2','tf2torch_resnet_v2_101']
    # model_names = ['tf2torch_inception_v3']
    # model_names = ['tf2torch_adv_inception_v3','tf2torch_ens3_adv_inc_v3','tf2torch_ens4_adv_inc_v3','tf2torch_ens_adv_inc_res_v2']
    # model_names = ['tf2torch_inception_v3','tf2torch_inception_v4']
    # model_names = ['tf2torch_inception_v3','tf2torch_inc_res_v2','tf2torch_resnet_v2_101']
    model_names = ['tf2torch_inception_v3','tf2torch_inception_v4','tf2torch_inc_res_v2','tf2torch_resnet_v2_101']
    # model_names = ['tf2torch_adv_inception_v3','tf2torch_ens3_adv_inc_v3','tf2torch_ens4_adv_inc_v3','tf2torch_ens_adv_inc_res_v2']
    models_path = './models/'
    for model_name in model_names:
        print(model_name, ' ======= start =======')
        verify(model_name, models_path)

if __name__ == '__main__':
    main()
    
    
