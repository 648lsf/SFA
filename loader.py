import os
from PIL import Image
from torch.utils import data
import pandas as pd
from torchvision import transforms as T
# cuda

class ImageNet(data.Dataset):                              # 
    def __init__(self, dir, csv_path, transforms = None):  # ImageNet(adv_dir, input_csv, T.Compose([T.ToTensor()]))
        self.dir = dir                                     # 
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        ImageID = img_obj['filename']
        Truelabel = img_obj['label']             # 
        # TargetClass = img_obj['TargetClass'] 
        
        img_path = os.path.join(self.dir, ImageID)      
        img_path = os.path.splitext(img_path)[0] + '.png'  #如果是png格式
            
        pil_img = Image.open(img_path).convert('RGB')      
        if self.transforms:
            data = self.transforms(pil_img)
        return data, ImageID, Truelabel

    def __len__(self):
        return len(self.csv)


class ImageNet_SFA(data.Dataset):
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