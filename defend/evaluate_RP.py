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
import feature_distillation

batch_size = 10
input_csv = '/MinierData/tangjiawei/SSA-c955cf07c8372bfc4e9f17e647042e027f9f3b1d/dataset/images.csv'
adv_dir = './data/OUR'
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

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
        return data, ImageID, Truelabel

    def __len__(self):
        return len(self.csv)
    

def evaluate(model_name, path):
    model = utils.get_model(model_name, path).eval().to(device)
    X = ImageNet_ssa(adv_dir, input_csv, T.Compose([T.ToTensor()]))
    data_loader = DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)

    sum = 0
    for x, _, y in data_loader:
        resize = np.random.randint(310, 331)
        x = torch.nn.functional.interpolate(x, size=(resize, resize), mode='nearest')
        a = np.random.randint(0, 331 - resize)
        b = np.random.randint(0, 331 - resize)
        x = torch.nn.functional.pad(x, (a, 331 - resize - a, b, 331 - resize - b))
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
