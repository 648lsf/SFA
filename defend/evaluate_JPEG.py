import numpy as np
import torch
from torchvision import transforms as T
from loader import ImageNet
from torch.utils.data import DataLoader
import utils
from torch.utils import data
import pandas as pd
from PIL import Image
import os
import advertorch.defenses as defenses
import torchvision 


class ImageNet_ssa(data.Dataset):
    def __init__(self, dir, csv_path, transforms = None):
        self.dir = dir   
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        ImageID = img_obj['ImageId'] + '.png'
        # ImageID = img_obj['ImageId'] + '.JPEG'
        Truelabel = img_obj['TrueLabel'] 
        TargetClass = img_obj['TargetClass'] - 1
        img_path = os.path.join(self.dir, ImageID)
        pil_img = Image.open(img_path).convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)
        return data, ImageID, Truelabel

    def __len__(self):
        return len(self.csv)

batch_size = 10
input_csv = '/MinierData/tangjiawei/SSA-c955cf07c8372bfc4e9f17e647042e027f9f3b1d/dataset/images.csv'
# adv_dir = '../outputs/our2_tf2torch_inception_v3_iter5'
adv_dir = './data/SSA'
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def evaluate(model_name, path):
    model = utils.get_model(model_name, path).eval().to(device)
    X = ImageNet_ssa(adv_dir, input_csv, T.Compose([T.ToTensor()]))
    data_loader = DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    sum = 0
    for x, _, y in data_loader:
        x_org = x.clone()
        x_min = torch.min(x)
        x_max = torch.max(x)
        x = (x - x_min) / (x_max - x_min) * 255  # 先(0,1),再(0,255)
        x = x.to(dtype=torch.uint8)
        temp = []
        for i in range(batch_size):
            temp.append(torchvision.io.decode_jpeg(torchvision.io.encode_jpeg(x[i], 95)))
            # temp.append(defenses.JPEGFilter(quality=70)(x[i]))
        x = torch.stack(temp)  # 张量list合并为一个张量
        x = x.reshape(x_org.shape)
        x = x.to(dtype=x_org.dtype)
        x = x / 255.0 * (x_max - x_min) + x_min
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            sum += (model(x)[0].argmax(1) != y).detach().sum().cpu()
    print(model_name + '  rate = {:.2%}'.format(sum / 1000.0))


def main():
    model_names = ['tf2torch_ens3_adv_inc_v3']
    models_path = '/MinierData/tangjiawei/FSD-MIM-and-NPGA/models/'
    for model_name in model_names:
        evaluate(model_name, models_path)
        print("===================================================")


if __name__ == '__main__':
    main()


